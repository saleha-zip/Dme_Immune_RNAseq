"""
01_load_data.py
---------------
Stage 1 — Ingestion
Loads the full count matrix (17,736 genes x 41 samples) and metadata
into Spark DataFrames, joins them, and caches for downstream use.

Inputs:
  - data/count_matrix/raw_counts.csv      (genes x samples, wide format)
  - data/metadata/metadata_table.csv      (42 rows, columns: sample, replicate,
                                           time, hours, time_label, sample_ID)

Outputs (written to data/processed/):
  - spark_counts_long.parquet             long-format (gene, sample, count)
  - spark_metadata.parquet                cleaned metadata for all 41 samples

Note: 4B is absent from the count matrix (excluded during QC in original paper).
      The join handles this automatically — 41 samples result after inner join.

Key Spark concepts demonstrated:
  - Lazy evaluation (transformations only execute on actions like .count())
  - Wide-to-long reshape using stack() for partition-friendly layout
  - Biologically meaningful repartitioning by time_point
  - Parquet persistence for fast columnar reads in downstream stages
  - Cache placement after full transformation chain
"""

import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)

# ── Spark session ─────────────────────────────────────────────────────────────
# Driver memory set to 2g for portability across systems (8GB RAM laptop safe)
# Increase to 4g on HPC
spark = (
    SparkSession.builder
    .appName("DmelImmune_Ingestion")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

t0 = time.time()

# ── 1. Load metadata ──────────────────────────────────────────────────────────
# inferSchema=False: safe for metadata — avoids accidental type coercion
# Columns: sample, replicate, time, hours, time_label, sample_ID
# sample_ID (e.g. "1A", "14B") is the join key to count matrix column headers
metadata = (
    spark.read.csv(
        "data/metadata/metadata_table.csv",
        header=True,
        inferSchema=False
    )
    .select(
        F.col("sample").alias("full_sample_name"),
        F.col("replicate"),
        F.col("time").cast("integer").alias("time_point"),
        F.col("hours").cast("integer").alias("hours"),
        F.col("time_label"),
        F.col("sample_ID").alias("sample_id")   # e.g. "1A", "14B"
    )
)

# .count() is the first ACTION — this is where Spark executes the lazy plan
n_meta = metadata.count()
print(f"\n── Metadata loaded: {n_meta} samples ──")
metadata.orderBy("hours", "replicate").show(10, truncate=False)

metadata.write.mode("overwrite").parquet("data/processed/spark_metadata.parquet")

# ── 2. Load count matrix (wide format: genes x samples) ──────────────────────
# inferSchema=False then explicit cast: avoids mixed-type columns on wide CSVs
counts_raw = spark.read.csv(
    "data/count_matrix/raw_counts.csv",
    header=True,
    inferSchema=False
)

# Rename unnamed first column (GEO matrices often export it as "" or _c0)
first_col  = counts_raw.columns[0]
counts_raw = counts_raw.withColumnRenamed(first_col, "gene_id")

n_genes   = counts_raw.count()
n_samples = len(counts_raw.columns) - 1
print(f"\n── Count matrix loaded: {n_genes} genes x {n_samples} samples ──")
print(f"   Columns (first 5): {counts_raw.columns[1:6]}")
print(f"   Columns (last 3) : {counts_raw.columns[-3:]}")

# Identify sample columns present in both matrix AND metadata
# .collect() is acceptable here — only 41 sample IDs brought to driver
meta_sample_ids = [r["sample_id"] for r in metadata.select("sample_id").collect()]
sample_cols     = [c for c in counts_raw.columns if c in meta_sample_ids]
missing         = set(meta_sample_ids) - set(sample_cols)
if missing:
    print(f"   Note: {len(missing)} metadata sample(s) absent from matrix "
          f"(expected — 4B excluded in original paper): {sorted(missing)}")

keep_cols     = ["gene_id"] + sample_cols
counts_subset = counts_raw.select(keep_cols)

# ── 3. Melt wide → long format ────────────────────────────────────────────────
# Reshape genes x samples → (gene_id, sample_id, raw_count)
# Long format enables efficient groupBy aggregations across partitions
n_keep = len(sample_cols)
counts_long = counts_subset.select(
    "gene_id",
    F.expr(
        f"stack({n_keep}, " +
        ", ".join([f"'{c}', `{c}`" for c in sample_cols]) +
        ") as (sample_id, raw_count)"
    )
)

# Null filter: drop any malformed rows (missing counts)
counts_long = counts_long.filter(F.col("raw_count").isNotNull())

# Explicit cast: ensure raw_count is long integer, not string (inferSchema=False)
counts_long = counts_long.withColumn("raw_count", F.col("raw_count").cast("long"))

# ── 4. Join with metadata on sample_id ───────────────────────────────────────
counts_joined = counts_long.join(
    metadata.select("sample_id", "hours", "time_point",
                    "replicate", "time_label"),
    on="sample_id",
    how="inner"
)

# Repartition by time_point — 21 unique time points = natural distributed unit
# Capped at defaultParallelism for portability across local and HPC environments
n_partitions = min(21, spark.sparkContext.defaultParallelism)
counts_partitioned = counts_joined.repartition(n_partitions, "time_point")

# Cache AFTER full transformation chain — optimal placement
counts_partitioned.cache()

# .count() materialises the cache (first action on this DataFrame)
n_rows = counts_partitioned.count()
print(f"\n── Long-format DataFrame ──")
print(f"   Rows      : {n_rows:,}  ({n_genes} genes x {n_keep} samples)")
print(f"   Partitions: {counts_partitioned.rdd.getNumPartitions()}")
counts_partitioned.orderBy("hours", "gene_id").show(5)

# ── 5. Write out ──────────────────────────────────────────────────────────────
counts_partitioned.write.mode("overwrite").parquet(
    "data/processed/spark_counts_long.parquet"
)

elapsed = time.time() - t0
print(f"\n✓ Ingestion complete in {elapsed:.1f}s")
print("  → data/processed/spark_counts_long.parquet")
print("  → data/processed/spark_metadata.parquet")

# Release cache and stop session cleanly
counts_partitioned.unpersist()
spark.stop()
