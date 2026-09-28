"""
Background
- A molecule is written as text with a SMILES
    - Its key is the inchikey14: a 14-letter code for the molecule's skeleton
- A spectrum is a measurement
    - The same molecule is measured many times (different energies, adducts, instruments)
    - The parquet has 2.5M rows (one per spectrum) but far fewer molecules

Images
Built locally with Docker, pushed to the class registry, pulled by the cluster.
The code is shipped separately as a code bundle, so changing code does not rebuild the image.
"""

import os

import flyte
import pyarrow as pa
from flyte.io import File

from casmi_flyte.tables import stable_fraction, write_table

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


def casmi_storage_options() -> dict[str, str]:
    return {
        "aws_access_key_id": os.environ["CASMI_S3_KEY_ID"],
        "aws_secret_access_key": os.environ["CASMI_S3_SECRET"],
        "aws_endpoint_url": CASMI_ENDPOINT,
        "aws_region": "ch",
    }


# checkpoint 1
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


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(distinct_structures, fraction=0.01)
    print(r.url)
    r.wait()
