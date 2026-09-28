import os

import flyte
from flyte.io import File

import pyarrow as pa
import numpy as np
from flyte.io import File
from casmi_flyte.tables import matrix_column, stable_fraction, write_table
from snippets.rdkit_fp import featurize_smiles


REGISTRY = "registry.86.119.83.247.sslip.io/fancy-eagle"
CASMI_TRAIN_URI = "s3://302-data/kaggle_CASMI2026/train.parquet"
CASMI_ENDPOINT = "https://zhw-a.s3.cloud.switch.ch"

data_env = flyte.TaskEnvironment(
    name="data",
    image=flyte.Image.from_debian_base(
        registry=REGISTRY, name="data"
    ).with_pip_packages("polars", "pyarrow", "numpy"),
    secrets=[
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="CASMI_S3_KEY_ID"),
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="CASMI_S3_SECRET"),
    ],
    resources=flyte.Resources(cpu=1, memory="4Gi"),
)

rdkit_env = flyte.TaskEnvironment(
    name="rdkit",
    image=flyte.Image.from_debian_base(
        registry=REGISTRY,
        name="rdkit",
    ).with_pip_packages(
        "polars",
        "pyarrow",
        "numpy",
        "rdkit==2026.3.3",
    ),
    secrets=[
        flyte.Secret(
            key="casmi-s3-access-key-id",
            as_env_var="CASMI_S3_KEY_ID",
        ),
        flyte.Secret(
            key="casmi-s3-secret-access-key",
            as_env_var="CASMI_S3_SECRET",
        ),
    ],
    resources=flyte.Resources(
        cpu=2,
        memory="4Gi",
    ),
)


def casmi_storage_options() -> dict[str, str]:
    return {
        "aws_access_key_id": os.environ["CASMI_S3_KEY_ID"],
        "aws_secret_access_key": os.environ["CASMI_S3_SECRET"],
        "aws_endpoint_url": CASMI_ENDPOINT,
        "aws_region": "ch",
    }


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
        # Stable order -> same bytes -> same content hash -> downstream caches stay valid.
        .sort("inchikey14")
        .collect()
    )
    table = df.to_arrow()
    if fraction < 1.0:
        table = table.filter(pa.array(stable_fraction(table["inchikey14"]) < fraction))
    print(f"{table.num_rows:,} structures (fraction={fraction})")
    return await write_table(table, "structures.parquet")

@rdkit_env.task
async def rdkit_feature(n_structures: int) -> File:
    import polars as pl

    spectra = pl.scan_parquet(
        CASMI_TRAIN_URI,
        storage_options=casmi_storage_options(),
    )

    structures = (
        spectra
        .select("inchikey14", "normalized_smiles")
        .drop_nulls()
        .sort("inchikey14")
        .unique(
            subset=["inchikey14"],
            keep="first",
        )
        .collect()
    )

    keys = pa.array(structures["inchikey14"].to_list())
    fractions = stable_fraction(keys)
    selected_indices = np.argsort(fractions)[:n_structures]
    selected = structures[selected_indices]

    smiles = selected["normalized_smiles"].to_list()
    features, exact_mass, valid = featurize_smiles(smiles)

    result = pa.table(
        {
            "inchikey14": selected["inchikey14"].to_list(),
            "valid": pa.array(valid),
            "morgan2": matrix_column(features["morgan2"]),
            "maccs": matrix_column(features["maccs"]),
            "atompair": matrix_column(features["atompair"]),
            "torsion": matrix_column(features["torsion"]),
            "exact_mass": pa.array(exact_mass),
        }
    )

    return await write_table(result, "rdkit.parquet")


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(distinct_structures, fraction=0.01)
    print(r.url)
    r.wait()
