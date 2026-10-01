import os

import flyte
from flyte.io import File

REGISTRY = "registry.86.119.83.247.sslip.io/fancy-eagle"
CASMI_TRAIN_URI = "s3://302-data/kaggle_CASMI2026/train.parquet"
CASMI_ENDPOINT = "https://zhw-a.s3.cloud.switch.ch"
N_SHARDS = 32

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
    resources=flyte.Resources(cpu=1, memory="3Gi"),
)

mordred_env = flyte.TaskEnvironment(
    name="mordred",
    image=flyte.Image.from_debian_base(
        registry=REGISTRY, name="mordred"
    ).with_pip_packages(
        "mordredcommunity", "rdkit==2026.3.3", "numpy", "pandas", "pyarrow", "polars"
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
)

chemeleon_env = flyte.TaskEnvironment(
    name="chemeleon",
    image=flyte.Image.from_debian_base(registry=REGISTRY, name="chemeleon")
    .with_apt_packages(
        "curl",
        "libxrender1",
        "libxext6",
        "libexpat1",
        "libfontconfig1",
        "libfreetype6",
    )
    .with_pip_packages("torch", index_url="https://download.pytorch.org/whl/cpu")
    .with_pip_packages("chemprop", "numpy", "pyarrow")
    .with_commands(
        [
            "curl -L -o /opt/chemeleon_mp.pt https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
        ]
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# Env du driver : doit dépendre des envs qu'il appelle
driver_env = flyte.TaskEnvironment(
    name="driver",
    image=flyte.Image.from_debian_base(
        registry=REGISTRY, name="driver"
    ).with_pip_packages("pyarrow"),
    depends_on=[data_env, rdkit_env, mordred_env, chemeleon_env],
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


@mordred_env.task(cache="auto", retries=2)
async def mordred_shard(structures: File, start: int, end: int) -> File:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from snippets.mordred_desc import featurize_smiles

    path = await structures.download()
    tbl = pq.read_table(path, columns=["inchikey14", "normalized_smiles"]).slice(
        start, end - start
    )

    X, valid = featurize_smiles(tbl.column("normalized_smiles").to_pylist())

    flat = pa.array(X.ravel(), type=pa.float32())
    out = pa.table(
        {
            "inchikey14": tbl.column("inchikey14"),
            "mordred": pa.FixedSizeListArray.from_arrays(flat, X.shape[1]),
            "valid": pa.array(valid),
        }
    )
    name = f"mordred_{start}.parquet"
    pq.write_table(out, name)
    print(f"[{start}:{end}] {int(valid.sum())}/{len(valid)} valides")
    return await File.from_local(name)


@mordred_env.task(cache="auto")
async def mordred_merge(shards: list[File]) -> File:
    import pyarrow.parquet as pq

    writer = None
    for f in shards:
        tbl = pq.read_table(await f.download())
        if writer is None:
            writer = pq.ParquetWriter("mordred.parquet", tbl.schema)
        writer.write_table(tbl)
        del tbl
    writer.close()
    return await File.from_local("mordred.parquet")


@driver_env.task
async def mordred_features(structures: File) -> File:
    import asyncio
    import pyarrow.parquet as pq

    n_rows = pq.ParquetFile(await structures.download()).metadata.num_rows
    bounds = [
        (i * n_rows // N_SHARDS, (i + 1) * n_rows // N_SHARDS) for i in range(N_SHARDS)
    ]
    print(f"{n_rows} structures en {N_SHARDS} tranches")

    shards = await asyncio.gather(*(mordred_shard(structures, s, e) for s, e in bounds))
    return await mordred_merge(list(shards))


@mordred_env.task
async def performance_test(structures: File) -> dict[str, float]:
    import time
    import polars as pl
    from snippets.mordred_desc import featurize_smiles

    path = await structures.download()
    df = pl.read_parquet(path)
    sample = df.sample(500, seed=0)["normalized_smiles"].to_list()

    featurize_smiles(sample[:5])  # échauffement

    t0 = time.perf_counter()
    X, valid = featurize_smiles(sample)
    dt = time.perf_counter() - t0

    per_mol = dt / len(sample)
    total_h = per_mol * df.height / 3600
    print(f"{per_mol * 1000:.1f} ms/molécule, {int(valid.sum())}/{len(sample)} valides")
    print(f"estimation totale : {total_h:.1f} h pour {df.height} structures")
    return {
        "ms_per_mol": per_mol * 1000,
        "estimated_hours": total_h,
        "n_structures": float(df.height),
    }


@chemeleon_env.task(cache="auto")
async def chemeleon_features(structures: File) -> File:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from snippets.chemeleon import featurize_smiles

    torch.set_num_threads(2)  # = cpu du pod, sinon torch voit tous les cœurs du nœud

    pf = pq.ParquetFile(await structures.download())
    writer = None
    for rb in pf.iter_batches(
        batch_size=20_000, columns=["inchikey14", "normalized_smiles"]
    ):
        X, valid = featurize_smiles(rb.column("normalized_smiles").to_pylist())

        flat = pa.array(X.ravel(), type=pa.float32())
        table = pa.table(
            {
                "inchikey14": rb.column("inchikey14"),
                "chemeleon": pa.FixedSizeListArray.from_arrays(flat, X.shape[1]),
                "valid": pa.array(valid),
            }
        )
        if writer is None:
            writer = pq.ParquetWriter("chemeleon.parquet", table.schema)
        writer.write_table(table)
        del X, flat, table

    writer.close()
    return await File.from_local("chemeleon.parquet")


@driver_env.task
async def pipeline() -> File:
    structures = await prepare_structures()
    return await chemeleon_features(structures)


if __name__ == "__main__":
    from pathlib import Path

    flyte.init_from_config(root_dir=Path(__file__).parent)
    r = flyte.with_runcontext(copy_style="all").run(pipeline)
    print(r.url)
    r.wait()
