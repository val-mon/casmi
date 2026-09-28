import os

import flyte
from flyte.io import File

REGISTRY = "registry.86.119.83.247.sslip.io/fancy-eagle"
CASMI_TRAIN_URI = "s3://302-data/kaggle_CASMI2026/train.parquet"
CASMI_ENDPOINT = "https://zhw-a.s3.cloud.switch.ch"

data_env = flyte.TaskEnvironment(
    name="data",
    image=flyte.Image.from_debian_base(
        registry=REGISTRY, name="data"
    ).with_pip_packages("polars", "pyarrow"),
    secrets=[
        flyte.Secret(key="casmi-s3-access-key-id", as_env_var="CASMI_S3_KEY_ID"),
        flyte.Secret(key="casmi-s3-secret-access-key", as_env_var="CASMI_S3_SECRET"),
    ],
    resources=flyte.Resources(cpu=1, memory="2Gi"),
)

rdkit_env = flyte.TaskEnvironment(
    name="rdkit",
    image=flyte.Image.from_debian_base(
        registry=REGISTRY, name="rdkit"
    ).with_pip_packages("polars", "pyarrow", "rdkit==2026.3.3", "numpy"),
    resources=flyte.Resources(cpu=1, memory="4Gi"),
)

# Env du driver : doit dépendre des envs qu'il appelle
driver_env = flyte.TaskEnvironment(
    name="driver",
    image=flyte.Image.from_debian_base(registry=REGISTRY, name="driver"),
    depends_on=[data_env, rdkit_env],
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


@data_env.task
async def prepare_structures() -> File:
    import polars as pl

    lf = pl.scan_parquet(CASMI_TRAIN_URI, storage_options=casmi_storage_options())
    smiles = (
        lf.group_by("inchikey14", "normalized_smiles")
        .agg(pl.len().alias("n_spectra"))
        .sort(
            ["inchikey14", "n_spectra", "normalized_smiles"],
            descending=[False, True, False],
        )
        .unique("inchikey14", keep="first", maintain_order=True)
        .collect()
    )
    print(f"{smiles.height} structures uniques")
    smiles.write_parquet("structures.parquet")
    return await File.from_local("structures.parquet")


@rdkit_env.task
async def rdkit_features(structures: File) -> File:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from snippets.rdkit_fp import featurize_smiles

    BATCH = 20_000

    path = await structures.download()
    pf = pq.ParquetFile(path)

    writer = None
    n_total = n_invalid = 0
    for rb in pf.iter_batches(
        batch_size=BATCH, columns=["inchikey14", "normalized_smiles"]
    ):
        smiles = rb.column("normalized_smiles").to_pylist()
        fps, mass, valid = featurize_smiles(smiles)
        n_total += len(valid)
        n_invalid += int((~valid).sum())

        cols = {
            "inchikey14": rb.column("inchikey14"),
            "normalized_smiles": rb.column("normalized_smiles"),
        }
        for name, arr in fps.items():
            flat = pa.array(np.ascontiguousarray(arr).ravel(), type=pa.uint8())
            cols[name] = pa.FixedSizeListArray.from_arrays(flat, arr.shape[1])
        cols["exact_mass"] = pa.array(mass, type=pa.float64())
        cols["valid"] = pa.array(valid, type=pa.bool_())

        table = pa.table(cols)
        if writer is None:
            writer = pq.ParquetWriter("features.parquet", table.schema)
        writer.write_table(table)
        del fps, cols, table

    writer.close()
    print(f"{n_total} structures, {n_invalid} SMILES non parsables")
    return await File.from_local("features.parquet")


@driver_env.task
async def pipeline() -> File:
    structures = await prepare_structures()
    return await rdkit_features(structures)


if __name__ == "__main__":
    from pathlib import Path

    flyte.init_from_config(root_dir=Path(__file__).parent)
    r = flyte.with_runcontext(copy_style="all").run(pipeline)
    print(r.url)
    r.wait()
