"""
02_preprocess.py
----------------
Stage 2 — Preprocessing
Reads the long-format counts from Stage 1, applies:
  1. Low-count gene filtering  (total raw count < 10 across all samples)
  2. CPM normalization         (parallelised per sample partition)
  3. log2(CPM + 1) transform   (variance stabilisation)

Also runs the same pipeline sequentially with pandas as the benchmark baseline.

Outputs (written to data/processed/):
  - spark_counts_normalized.parquet
  - benchmark_preprocessing.csv   (Spark vs pandas timing)
"""

import os
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.makedirs("results/benchmarks", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("DmelImmune_Preprocessing")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ═══════════════════════════════════════════════════════════════════════════════
# SPARK PATH
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [SPARK] Preprocessing ──")
t_spark_start = time.time()

counts = spark.read.parquet("data/processed/spark_counts_long.parquet")
counts.cache()
counts.count()   # materialize cache immediately

# Step 1 — Filter low-count genes
# Keep genes whose total raw count across all samples is >= 10
gene_totals = counts.groupBy("gene_id").agg(
    F.sum("raw_count").alias("total_count")
)
passing_genes = gene_totals.filter(F.col("total_count") >= 10).select("gene_id")
counts_filtered = counts.join(passing_genes, on="gene_id", how="inner")

n_before = counts.select("gene_id").distinct().count()
n_after  = counts_filtered.select("gene_id").distinct().count()
print(f"  Genes before filtering : {n_before}")
print(f"  Genes after filtering  : {n_after} (removed {n_before - n_after})")

# Step 2 — CPM normalisation (per sample, parallelised over partitions)
# library_size = total raw counts per sample
lib_sizes = counts_filtered.groupBy("sample_id").agg(
    F.sum("raw_count").alias("library_size")
)
counts_with_lib = counts_filtered.join(lib_sizes, on="sample_id", how="inner")

counts_cpm = counts_with_lib.withColumn(
    "cpm",
    (F.col("raw_count") / F.col("library_size")) * 1e6
)

# Step 3 — log2(CPM + 1) transform
counts_norm = counts_cpm.withColumn(
    "log2_cpm",
    F.log2(F.col("cpm") + 1)
)

# Drop intermediate columns we don't need downstream
counts_norm = counts_norm.select(
    "gene_id", "sample_id", "hours", "time_point", "replicate",
    "raw_count", "cpm", "log2_cpm"
)

counts_norm.cache()
counts_norm.write.mode("overwrite").parquet(
    "data/processed/spark_counts_normalized.parquet"
)

t_spark_end = time.time()
spark_time  = t_spark_end - t_spark_start

print(f"\n  Sample output:")
counts_norm.orderBy("hours", "gene_id").show(5)
print(f"  [SPARK] Time: {spark_time:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# PANDAS BASELINE (sequential benchmark)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [PANDAS] Preprocessing (sequential baseline) ──")
t_pandas_start = time.time()

df = pd.read_csv("data/count_matrix/raw_counts.csv", index_col=0)

# Filter low-count genes
df = df[df.sum(axis=1) >= 10]

# CPM normalisation
lib_sizes_pd = df.sum(axis=0)
df_cpm = (df.div(lib_sizes_pd, axis=1)) * 1e6

# log2 transform
df_log2 = np.log2(df_cpm + 1)

t_pandas_end  = time.time()
pandas_time   = t_pandas_end - t_pandas_start
print(f"  [PANDAS] Time: {pandas_time:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# Save benchmark results
# ═══════════════════════════════════════════════════════════════════════════════
benchmark = pd.DataFrame([
    {"stage": "preprocessing", "method": "pandas_sequential", "time_seconds": round(pandas_time, 3)},
    {"stage": "preprocessing", "method": "spark_distributed",  "time_seconds": round(spark_time, 3)},
])
benchmark["speedup"] = round(pandas_time / spark_time, 2)
benchmark.to_csv("results/benchmarks/benchmark_preprocessing.csv", index=False)

print("\n── Benchmark summary ──")
print(benchmark.to_string(index=False))
print(f"\n✓ Preprocessing complete")
print("  Outputs → data/processed/spark_counts_normalized.parquet")
print("            results/benchmarks/benchmark_preprocessing.csv")

counts.unpersist()
counts_norm.unpersist()
spark.stop()
