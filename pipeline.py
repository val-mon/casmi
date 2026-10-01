"""
Background
- A molecule is written as text with a SMILES (ex: CCO for ethanol)
    - Its key is the inchikey14: a 14-letter code for the molecule's skeleton
- A spectrum is a measurement
    - The same molecule is measured many times (different energies, adducts, instruments)
    - The parquet has 2.5M rows (one per spectrum) but far fewer molecules

Images
Built locally with Docker, pushed to the class registry, pulled by the cluster.
The code is shipped separately as a code bundle, so changing code does not rebuild the image.

Pipeline (the `main` task)
    distinct_structures ──┬── rdkit shards ──── merge ──┐
                          ├── mordred shards ── merge ──┤
                          ├── cdk shards ────── merge ──┼── evaluate x 9 ── report
                          ├── chemeleon shards  merge ──┤
                          └── split_spectra ────────────┘

Run
    uv run python pipeline.py          # development run (1% of the molecules)
    uv run python pipeline.py --full   # full run
"""

import asyncio
import os
import sys
from pathlib import Path

import flyte
import numpy as np
import pyarrow as pa
from flyte.io import File

from casmi_flyte.baseline import evaluate_representation
from casmi_flyte.config import TEST_ADDUCTS
from casmi_flyte.tables import (
    matrix_column,
    merge_tables,
    read_table,
    stable_fraction,
    write_table,
)

# Imported here (not inside the tasks) so that the code bundle contains them.
# They only import numpy at load time, so every image can load this file.
from snippets import cdk_jpype, chemeleon, mordred_desc, rdkit_fp

REGISTRY = "registry.86.119.83.247.sslip.io/fancy-eagle"
CASMI_TRAIN_URI = "s3://302-data/kaggle_CASMI2026/train.parquet"
CASMI_ENDPOINT = "https://zhw-a.s3.cloud.switch.ch"

CDK_JAR = "/opt/cdk-2.13.jar"
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
TIMSTOF_LIBS = [
    "enveda-180",
    "enveda-np-examples",
]

REPRESENTATIONS = {
    "morgan2": "rdkit",
    "maccs": "rdkit",
    "atompair": "rdkit",
    "torsion": "rdkit",
    "pubchem": "cdk",
    "klekota_roth": "cdk",
    "cdk_substructure": "cdk",
    "mordred": "mordred",
    "chemeleon": "chemeleon",
}

# region Environments
# One per featurizer, because their dependencies conflict
#   - rdkit needs rdkit==2026.3.3 (the metric's version)
#   - mordred==1.2.0 (2019) needs an old numpy and an old rdkit
#   - cdk needs Java
#   - chemeleon needs torch (big image)


def image(name: str, python_version: tuple[int, int] | None = None) -> flyte.Image:
    """Default Flyte base image, pushed to our group's registry project."""
    return flyte.Image.from_debian_base(
        registry=REGISTRY, name=name, python_version=python_version
    )


data_env = flyte.TaskEnvironment(
    name="data",
    image=image("data").with_pip_packages("polars", "pyarrow", "numpy"),
    secrets=[
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="CASMI_S3_KEY_ID"),
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="CASMI_S3_SECRET"),
    ],
    resources=flyte.Resources(cpu=1, memory="4Gi"),
)

rdkit_env = flyte.TaskEnvironment(
    name="rdkit",
    image=image("rdkit").with_pip_packages("rdkit==2026.3.3", "numpy", "pyarrow"),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
)

# Hypothesis to confirm on the cluster: mordred 1.2.0 breaks with numpy >= 1.24 and networkx 3,
# so we use Python 3.11 with an old numpy and an rdkit built for numpy 1.x.
mordred_env = flyte.TaskEnvironment(
    name="mordred",
    image=image("mordred", python_version=(3, 11)).with_pip_packages(
        "mordred==1.2.0", "numpy<1.24", "networkx<3", "rdkit==2023.9.6", "pyarrow"
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
)

cdk_env = flyte.TaskEnvironment(
    name="cdk",
    image=image("cdk")
    .with_apt_packages("default-jre-headless", "curl")
    .with_pip_packages("jpype1", "numpy", "pyarrow")
    .with_commands(
        [
            f"curl -L -o {CDK_JAR} https://github.com/cdk/cdk/releases/download/cdk-2.13/cdk-2.13.jar"
        ]
    ),
    resources=flyte.Resources(cpu=1, memory="3Gi"),
)

chemeleon_env = flyte.TaskEnvironment(
    name="chemeleon",
    image=image("chemeleon")
    .with_apt_packages(
        "curl",
        # system libraries needed by cuik_molmaker (imported by chemprop)
        "libxrender1",
        "libxext6",
        "libexpat1",
        "libfontconfig1",
        "libfreetype6",
    )
    .with_pip_packages(
        "torch", index_url="https://download.pytorch.org/whl/cpu"
    )  # CPU-only torch: much smaller
    .with_pip_packages("chemprop", "numpy", "pyarrow")
    .with_commands([f"curl -L -o {chemeleon.CHEMELEON_WEIGHTS} {CHEMELEON_URL}"]),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

eval_env = flyte.TaskEnvironment(
    name="eval",
    image=image("eval").with_pip_packages("numpy", "pyarrow", "polars", "scikit-learn"),
    resources=flyte.Resources(cpu=2, memory="8Gi"),
)

# The driver only orchestrates: it waits for the other tasks, so it needs few resources.
# depends_on: deploying the driver also deploys (and builds) every environment it calls.
driver_env = flyte.TaskEnvironment(
    name="driver",
    image=image("driver").with_pip_packages("numpy", "pyarrow"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    depends_on=[data_env, rdkit_env, mordred_env, cdk_env, chemeleon_env, eval_env],
)


def casmi_storage_options() -> dict[str, str]:
    return {
        "aws_access_key_id": os.environ["CASMI_S3_KEY_ID"],
        "aws_secret_access_key": os.environ["CASMI_S3_SECRET"],
        "aws_endpoint_url": CASMI_ENDPOINT,
        "aws_region": "ch",
    }


# endregion

# region Data
# counts spectra and distinct inchikey14
# shows that features must be computed per molecule, not per spectrum
@data_env.task
async def explore() -> dict[str, int]:
    import polars as pl

    lf = pl.scan_parquet(CASMI_TRAIN_URI, storage_options=casmi_storage_options())
    print(lf.collect_schema())
    print(lf.head(5).collect())
    stats = lf.select(
        pl.len().alias("n_spectra"),
        pl.col("inchikey14").n_unique().alias("n_structures"),
    ).collect()
    print(stats)
    return stats.to_dicts()[0]


# one row per inchikey14 with its most measured SMILES
# this table is the input of every featurizer
@data_env.task(cache="auto")
async def distinct_structures(fraction: float = 1.0) -> File:
    """One row per inchikey14, with its most measured SMILES (ties: smallest SMILES)."""
    import polars as pl

    lf = pl.scan_parquet(CASMI_TRAIN_URI, storage_options=casmi_storage_options())
    df = (
        lf.group_by("inchikey14", "normalized_smiles")
        .agg(pl.len().alias("n_spectra"))
        .sort(
            ["inchikey14", "n_spectra", "normalized_smiles"],
            descending=[False, True, False],
        )
        .group_by("inchikey14", maintain_order=True)
        .first()
        .sort("inchikey14")
        .collect()
    )
    table = df.to_arrow()

    # fraction = % of molecules kept for next operations, helps reduce computing time
    if fraction < 1.0:
        table = table.filter(pa.array(stable_fraction(table["inchikey14"]) < fraction))
    print(f"{table.num_rows:,} structures (fraction={fraction})")
    return await write_table(table, "structures.parquet")


@data_env.task(cache="auto")
async def split_spectra(
    structures: File, n_holdout: int = 400, max_train: int = 200_000
) -> tuple[File, File]:
    """Train / hold-out spectra. The hold-out looks like the Kaggle test set:
    timsTOF spectra, test adducts only, and none of its molecules in train."""
    import polars as pl

    keys = (await read_table(structures, ["inchikey14"]))["inchikey14"].to_pylist()

    # only the test adducts, only our molecules, only the columns the evaluation needs
    spectra = (
        pl.scan_parquet(CASMI_TRAIN_URI, storage_options=casmi_storage_options())
        .filter(pl.col("adduct").is_in(list(TEST_ADDUCTS)))
        .filter(pl.col("inchikey14").is_in(keys))
        .select(
            "inchikey14",
            "adduct",
            "precursor_mz",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "collision_energy_ev",
            "ingest_lib",
        )
    )
    on_timstof = pl.col("ingest_lib").is_in(TIMSTOF_LIBS)

    # choose the hold-out molecules among those measured on the timsTOF
    # (stable_fraction: always the same choice, on every machine and every run)
    candidates = pa.array(
        spectra.filter(on_timstof)
        .select("inchikey14")
        .unique()
        .collect()["inchikey14"]
        .to_list()
    )
    chosen = candidates.take(
        np.argsort(stable_fraction(candidates))[:n_holdout]
    ).to_pylist()
    in_holdout = pl.col("inchikey14").is_in(chosen)

    holdout = spectra.filter(in_holdout & on_timstof).collect()
    # train = every spectrum of the other molecules, thinned to about `max_train` spectra
    # (the evaluation uses 100k spectra at most, no need to move the whole 3 GB)
    train = spectra.filter(~in_holdout)
    n_train = train.select(pl.len()).collect().item()
    train = train.gather_every(max(1, -(-n_train // max_train))).collect(
        engine="streaming"
    )

    print(
        f"hold-out: {len(chosen)} molecules, {holdout.height:,} spectra; train: {train.height:,} spectra"
    )
    # oldest compat level: plain arrow types (string, list) that pyarrow.compute in baseline.py handles
    old = pl.CompatLevel.oldest()
    return (
        await write_table(train.to_arrow(compat_level=old), "train.parquet"),
        await write_table(holdout.to_arrow(compat_level=old), "holdout.parquet"),
    )


@data_env.task(cache="auto")
async def merge(parts: list[File], name: str) -> File:
    """Concatenate the shards of one featurizer into one parquet file."""
    return await merge_tables(parts, name)


# endregion

# region Featurizers
# One task per shard (= rows start..end of the structures table)
# The snippets return valid=False for a SMILES they can't parse: one bad molecule never fails the run.


async def read_shard(structures: File, start: int, end: int) -> pa.Table:
    """Rows start..end of the structures table."""
    table = await read_table(structures, ["inchikey14", "normalized_smiles"])
    return table.slice(start, end - start)


def features_table(
    shard: pa.Table, valid: np.ndarray, matrices: dict[str, np.ndarray]
) -> pa.Table:
    """R1 layout: inchikey14, valid, then one column per representation."""
    columns = {"inchikey14": shard["inchikey14"], "valid": pa.array(valid)}
    for name, x in matrices.items():
        columns[name] = matrix_column(x)  # (n, d) matrix -> one fixed-size-list column
    return pa.table(columns)


@rdkit_env.task(cache="auto")
async def rdkit_shard(structures: File, start: int, end: int) -> File:
    shard = await read_shard(structures, start, end)
    fps, mass, valid = rdkit_fp.featurize_smiles(shard["normalized_smiles"].to_pylist())
    table = features_table(shard, valid, fps).append_column(
        "exact_mass", pa.array(mass)
    )
    return await write_table(table, f"rdkit_{start}.parquet")


@mordred_env.task(cache="auto")
async def mordred_shard(structures: File, start: int, end: int) -> File:
    shard = await read_shard(structures, start, end)
    x, valid = mordred_desc.featurize_smiles(shard["normalized_smiles"].to_pylist())
    return await write_table(
        features_table(shard, valid, {"mordred": x}), f"mordred_{start}.parquet"
    )


@cdk_env.task(cache="auto")
async def cdk_shard(structures: File, start: int, end: int) -> File:
    shard = await read_shard(structures, start, end)
    fps, valid = cdk_jpype.featurize_smiles(
        shard["normalized_smiles"].to_pylist(), jar=CDK_JAR
    )
    return await write_table(features_table(shard, valid, fps), f"cdk_{start}.parquet")


@chemeleon_env.task(cache="auto")
async def chemeleon_shard(structures: File, start: int, end: int) -> File:
    shard = await read_shard(structures, start, end)
    x, valid = chemeleon.featurize_smiles(shard["normalized_smiles"].to_pylist())
    return await write_table(
        features_table(shard, valid, {"chemeleon": x}), f"chemeleon_{start}.parquet"
    )


# endregion

# region Evaluation


@eval_env.task(cache="auto")
async def evaluate(
    representation: str, features: File, masses: File, train: File, holdout: File
) -> dict[str, float]:
    """Train spectrum -> representation, then rank the hold-out candidates (see baseline.py)."""
    return await evaluate_representation(
        representation, features, masses, train, holdout
    )


# endregion

# region Driver


def bounds(n_rows: int, n_shards: int) -> list[tuple[int, int]]:
    """Cut 0..n_rows into n_shards (start, end) slices of (almost) equal size."""
    n_shards = max(1, min(n_shards, n_rows))
    return [
        (i * n_rows // n_shards, (i + 1) * n_rows // n_shards) for i in range(n_shards)
    ]


async def featurize(
    shard_task, name: str, structures: File, n_rows: int, n_shards: int
) -> File:
    """Fan out: one `shard_task` per slice, all at the same time, then merge them into one file."""
    parts = await asyncio.gather(
        *(shard_task(structures, start, end) for start, end in bounds(n_rows, n_shards))
    )
    return await merge(list(parts), f"{name}.parquet")


def report_html(
    metrics: dict[str, dict[str, float]], coverage: dict[str, float]
) -> str:
    """HTML table, best representation first."""
    rows = sorted(metrics.items(), key=lambda item: -item[1]["mrr@25"])
    lines = [
        "<h2>Representations ranked by MRR@25</h2>",
        "<table><tr><th>#</th><th>representation</th><th>MRR@25</th><th>top1</th>"
        "<th>random MRR@25</th><th>coverage</th></tr>",
    ]
    for rank, (name, m) in enumerate(rows, start=1):
        lines.append(
            f"<tr><td>{rank}</td><td>{name}</td><td>{m['mrr@25']:.4f}</td><td>{m['top1']:.3f}</td>"
            f"<td>{m['random_mrr@25']:.4f}</td><td>{coverage[REPRESENTATIONS[name]]:.0%}</td></tr>"
        )
    lines.append("</table>")
    first = next(iter(metrics.values()))
    lines.append(
        f"<p>{first['n_holdout_molecules']:.0f} hold-out molecules, {first['train_spectra']:.0f} training spectra.</p>"
    )
    return "\n".join(lines)


@driver_env.task(report=True)
async def main(
    fraction: float = 0.01,
    n_shards: int = 4,
    mordred_max_molecules: int = 2_000,
    n_holdout: int = 400,
) -> dict[str, float]:
    """The whole pipeline. Returns {representation: MRR@25} and writes the report."""
    # 1. the molecules
    structures = await distinct_structures(fraction)
    n = len(await read_table(structures, ["inchikey14"]))
    n_mordred = min(
        n, mordred_max_molecules
    )  # mordred is slow: we cover only the first molecules
    coverage = {"rdkit": 1.0, "cdk": 1.0, "chemeleon": 1.0, "mordred": n_mordred / n}

    # 2. the 4 featurizers and the split don't depend on each other: they all run at the same time
    rdkit, cdk, chemeleon_fp, mordred, (train, holdout) = await asyncio.gather(
        featurize(rdkit_shard, "rdkit", structures, n, n_shards),
        featurize(cdk_shard, "cdk", structures, n, n_shards),
        featurize(chemeleon_shard, "chemeleon", structures, n, n_shards),
        featurize(mordred_shard, "mordred", structures, n_mordred, n_shards),
        split_spectra(structures, n_holdout),
    )
    features = {
        "rdkit": rdkit,
        "cdk": cdk,
        "chemeleon": chemeleon_fp,
        "mordred": mordred,
    }

    # 3. one evaluation per representation, in parallel
    # the RDKit file also holds exact_mass: it is the "masses" table used to find the candidates
    results = await asyncio.gather(
        *(
            evaluate(name, features[featurizer], rdkit, train, holdout)
            for name, featurizer in REPRESENTATIONS.items()
        )
    )
    metrics = dict(zip(REPRESENTATIONS, results))

    # 4. the report (task's "Report" tab in the UI)
    await flyte.report.replace.aio(report_html(metrics, coverage), do_flush=True)
    return {name: m["mrr@25"] for name, m in metrics.items()}


# endregion

if __name__ == "__main__":
    # root_dir: the code bundle contains casmi_flyte/ and snippets/
    flyte.init_from_config(root_dir=Path(__file__).parent)

    full = "--full" in sys.argv
    if full:  # full run: every molecule
        r = flyte.run(main, fraction=1.0, n_shards=16, mordred_max_molecules=50_000)
    else:  # development run: 1% of the molecules, must take < 5 minutes
        r = flyte.run(main, fraction=0.01, n_shards=4, mordred_max_molecules=2_000)
    print(r.url)
    r.wait()
