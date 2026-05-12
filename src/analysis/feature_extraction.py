"""
03_feature_extraction.py
------------------------
Stage 3 — Distributed Feature Extraction
Computes per-gene expression dynamics across time points using Spark:
  1. Per-gene variance across all samples        (identifies variable genes)
  2. Mean log2-CPM per gene per time point       (expression profiles)
  3. Fold change vs control (t=0)                (immune response magnitude)
  4. Immune gene overlay                         (join with curated gene list)

Outputs (written to results/tables/):
  - top500_variable_genes.csv
  - fold_change_all_genes.csv
  - immune_gene_dynamics.csv
"""

import os
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.makedirs("results/tables", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("DmelImmune_FeatureExtraction")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

t0 = time.time()

# ── Load normalized counts ────────────────────────────────────────────────────
counts = spark.read.parquet("data/processed/spark_counts_normalized.parquet")
counts.cache()
print(f"\nLoaded: {counts.count()} rows")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Per-gene variance (distributed groupBy over gene partitions)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [1] Per-gene variance ──")
t1 = time.time()

gene_variance = (
    counts
    .groupBy("gene_id")
    .agg(
        F.variance("log2_cpm").alias("variance"),
        F.mean("log2_cpm").alias("mean_expr"),
        F.count("sample_id").alias("n_samples")
    )
    .orderBy(F.desc("variance"))
)

top500 = gene_variance.limit(500)
top500_pd = top500.toPandas()
top500_pd.to_csv("results/tables/top500_variable_genes.csv", index=False)
print(f"  Top variable gene: {top500_pd.iloc[0]['gene_id']} "
      f"(variance={top500_pd.iloc[0]['variance']:.2f})")
print(f"  Time: {time.time()-t1:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Mean expression per gene per time point
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [2] Mean expression per time point ──")
t2 = time.time()

mean_by_tp = (
    counts
    .groupBy("gene_id", "hours", "time_point")
    .agg(F.mean("log2_cpm").alias("mean_log2_cpm"))
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Fold change vs control (hours == 0)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [3] Fold change vs t=0 ──")

# Extract control expression per gene
control_expr = (
    mean_by_tp
    .filter(F.col("hours") == 0)
    .select(
        F.col("gene_id"),
        F.col("mean_log2_cpm").alias("ctrl_expr")
    )
)

# Join and compute log2 fold change: treated - control (in log2 space)
fc = (
    mean_by_tp
    .filter(F.col("hours") > 0)
    .join(control_expr, on="gene_id", how="inner")
    .withColumn(
        "log2fc",
        F.col("mean_log2_cpm") - F.col("ctrl_expr")
    )
)

# Max absolute fold change across all time points per gene
max_fc = (
    fc
    .groupBy("gene_id")
    .agg(F.max(F.abs(F.col("log2fc"))).alias("max_abs_log2fc"))
)

# Keep genes with max |log2FC| > 1 (i.e. > 2-fold change)
de_genes = max_fc.filter(F.col("max_abs_log2fc") > 1)
n_de = de_genes.count()
print(f"  DE candidate genes (|log2FC| > 1 at any time point): {n_de}")

# Save full fold change table for top 500 variable genes only (manageable size)
top500_ids = [r["gene_id"] for r in top500.select("gene_id").collect()]
fc_top500  = fc.filter(F.col("gene_id").isin(top500_ids))
fc_top500_pd = fc_top500.orderBy("gene_id", "hours").toPandas()
fc_top500_pd.to_csv("results/tables/fold_change_top500.csv", index=False)
print(f"  Time: {time.time()-t2:.2f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Immune gene overlay
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [4] Immune gene dynamics ──")
t4 = time.time()

# Load curated immune gene list
immune_pd = pd.read_csv("data/metadata/List_of_immune_genes_updated.csv")
# Columns: Name, Symbol, ID, FlyBaseID, Chromosome, Start, End,
#          Functional Class, Immune Process, Reference
# FlyBaseID is our join key (FBgn format)

immune_spark = spark.createDataFrame(
    immune_pd[["Symbol", "FlyBaseID", "Functional Class", "Immune Process"]]
    .rename(columns={
        "FlyBaseID": "gene_id",
        "Functional Class": "functional_class",
        "Immune Process": "immune_process"
    })
)

# Join normalized counts with immune gene annotations
immune_counts = counts.join(immune_spark, on="gene_id", how="inner")
n_immune_found = immune_counts.select("gene_id").distinct().count()
print(f"  Immune genes found in count matrix: {n_immune_found} / {len(immune_pd)}")

# Mean expression trajectory per immune process category
immune_dynamics = (
    immune_counts
    .groupBy("immune_process", "functional_class", "hours")
    .agg(
        F.mean("log2_cpm").alias("mean_log2_cpm"),
        F.countDistinct("gene_id").alias("n_genes")
    )
    .orderBy("immune_process", "hours")
)

immune_dynamics_pd = immune_dynamics.toPandas()
immune_dynamics_pd.to_csv("results/tables/immune_gene_dynamics.csv", index=False)

print(f"  Immune process categories: "
      f"{immune_dynamics_pd['immune_process'].nunique()}")
print(f"  Time: {time.time()-t4:.2f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
total = time.time() - t0
print(f"\n✓ Feature extraction complete in {total:.1f}s")
print("  Outputs → results/tables/top500_variable_genes.csv")
print("            results/tables/fold_change_top500.csv")
print("            results/tables/immune_gene_dynamics.csv")

counts.unpersist()
spark.stop()
