"""
04_analysis.py
--------------
Stage 4 — Distributed Analysis (PySpark MLlib)
  1. PCA on top 500 variable genes across samples
     → samples coloured by time point in 2D PCA space
  2. KMeans clustering of genes by expression profile
     → assigns each gene to a temporal expression pattern cluster
  3. Cluster annotation — maps clusters to biological labels

Outputs:
  - results/tables/pca_sample_coordinates.csv
  - results/tables/gene_clusters.csv
  - results/plots/pca_samples.png
  - results/plots/cluster_profiles.png
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import PCA, VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("DmelImmune_Analysis")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── Load data ─────────────────────────────────────────────────────────────────
counts     = spark.read.parquet("data/processed/spark_counts_normalized.parquet")
counts.cache()
counts.count()   # materialize cache — counts reused in PCA, clustering, and plotting
top500_pd  = pd.read_csv("results/tables/top500_variable_genes.csv")
top500_ids = top500_pd["gene_id"].tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PCA — samples in gene expression space
#    Pivot: one row per sample, one column per gene (top 500)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [1] PCA on samples ──")
t1 = time.time()

# Filter to top 500 variable genes
counts_top = counts.filter(F.col("gene_id").isin(top500_ids))

# Pivot to wide format: rows = samples, columns = genes
pivot = (
    counts_top
    .groupBy("sample_id", "hours", "replicate")
    .pivot("gene_id", top500_ids)
    .agg(F.first("log2_cpm"))
    .fillna(0)
)

# Assemble feature vector
assembler = VectorAssembler(inputCols=top500_ids, outputCol="features")
pivot_vec = assembler.transform(pivot)

# Scale features (zero mean, unit variance) — important before PCA
scaler    = StandardScaler(inputCol="features", outputCol="scaled_features",
                           withMean=True, withStd=True)
scaler_model = scaler.fit(pivot_vec)
pivot_scaled = scaler_model.transform(pivot_vec)

# Run PCA (k=5 components, use first 2 for plotting)
pca       = PCA(k=5, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(pivot_scaled)
pca_result = pca_model.transform(pivot_scaled)

explained = pca_model.explainedVariance.toArray()
print(f"  PC1 variance explained: {explained[0]*100:.1f}%")
print(f"  PC2 variance explained: {explained[1]*100:.1f}%")

# Extract PC1 and PC2 using vector_to_array (JVM-native, avoids Python UDF overhead)
pca_coords = (
    pca_result
    .withColumn("pcs", vector_to_array("pca_features"))
    .withColumn("PC1", F.col("pcs")[0])
    .withColumn("PC2", F.col("pcs")[1])
    .select("sample_id", "hours", "replicate", "PC1", "PC2")
)

pca_pd = pca_coords.toPandas()
pca_pd.to_csv("results/tables/pca_sample_coordinates.csv", index=False)
print(f"  PCA time: {time.time()-t1:.2f}s")

# Plot PCA
fig, ax = plt.subplots(figsize=(7, 5.5))
time_points = sorted(pca_pd["hours"].unique())
cmap   = cm.get_cmap("plasma", len(time_points))
colors = {tp: cmap(i) for i, tp in enumerate(time_points)}

for _, row in pca_pd.iterrows():
    marker = "o" if row["replicate"] == "A" else "^"
    ax.scatter(row["PC1"], row["PC2"],
               color=colors[row["hours"]], marker=marker,
               s=70, edgecolors="white", linewidths=0.4, zorder=3)

sm = plt.cm.ScalarMappable(cmap="plasma",
     norm=plt.Normalize(vmin=min(time_points), vmax=max(time_points)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Hours post-infection")
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
ax.set_title("PCA of D. melanogaster immune response samples\n"
             "(top 500 variable genes, circles=rep A, triangles=rep B)",
             fontsize=10)
ax.grid(True, alpha=0.3, linewidth=0.5)
plt.tight_layout()
plt.savefig("results/plots/pca_samples.png", dpi=150)
plt.close()
print("  Saved → results/plots/pca_samples.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. KMeans — cluster genes by temporal expression profile
#    Feature vector: mean log2-CPM at each time point (one value per time point)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [2] KMeans gene clustering ──")
t2 = time.time()

# Build gene × time_point matrix: mean log2-CPM per gene per time point
time_points_ordered = sorted(
    [r["hours"] for r in counts.select("hours").distinct().collect()]
)

gene_time_matrix = (
    counts.filter(F.col("gene_id").isin(top500_ids))
    .groupBy("gene_id", "hours")
    .agg(F.mean("log2_cpm").alias("mean_expr"))
    .groupBy("gene_id")
    .pivot("hours", time_points_ordered)
    .agg(F.first("mean_expr"))
    .fillna(0)
)

# Assemble into feature vector
# Use actual column names from pivot output — safer than str-casting integers,
# since Spark pivot column names from numeric values are version-dependent
tp_cols = [c for c in gene_time_matrix.columns if c != "gene_id"]
gene_assembler = VectorAssembler(inputCols=tp_cols, outputCol="features")
gene_vec = gene_assembler.transform(gene_time_matrix).fillna(0)

# KMeans with k=5 clusters (biologically: up-early, up-late, down, transient, flat)
k = 5
kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
kmeans_model = kmeans.fit(gene_vec)
gene_clusters = kmeans_model.transform(gene_vec)

# Silhouette score
evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster")
silhouette = evaluator.evaluate(gene_clusters)
print(f"  Silhouette score (k={k}): {silhouette:.3f}")

clusters_pd = gene_clusters.select("gene_id", "cluster").toPandas()
clusters_pd.to_csv("results/tables/gene_clusters.csv", index=False)
print(f"  Cluster sizes:\n{clusters_pd['cluster'].value_counts().sort_index().to_string()}")
print(f"  KMeans time: {time.time()-t2:.2f}s")

# ── Plot cluster mean expression profiles ─────────────────────────────────────
# Collect mean expression per cluster per time point
counts_top_pd = (
    counts.filter(F.col("gene_id").isin(top500_ids))
    .groupBy("gene_id", "hours")
    .agg(F.mean("log2_cpm").alias("mean_expr"))
    .toPandas()
)
counts_top_pd = counts_top_pd.merge(clusters_pd, on="gene_id")

cluster_colors = ["#E8593C", "#3B8BD4", "#1D9E75", "#EF9F27", "#888780"]

fig, axes = plt.subplots(1, k, figsize=(14, 3.5), sharey=True)
for i, ax in enumerate(axes):
    subset = counts_top_pd[counts_top_pd["cluster"] == i]
    profile = subset.groupby("hours")["mean_expr"].mean()
    ax.plot(profile.index, profile.values,
            color=cluster_colors[i], linewidth=2)
    ax.fill_between(profile.index, profile.values,
                    alpha=0.15, color=cluster_colors[i])
    n_genes = clusters_pd[clusters_pd["cluster"] == i].shape[0]
    ax.set_title(f"Cluster {i}\n({n_genes} genes)", fontsize=9)
    ax.set_xlabel("Hours post-infection", fontsize=8)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.25, linewidth=0.5)

axes[0].set_ylabel("Mean log₂(CPM + 1)", fontsize=9)
fig.suptitle("Gene expression clusters — D. melanogaster immune response",
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig("results/plots/cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved → results/plots/cluster_profiles.png")

print(f"\n✓ Analysis complete")
print("  Outputs → results/tables/pca_sample_coordinates.csv")
print("            results/tables/gene_clusters.csv")
print("            results/plots/pca_samples.png")
print("            results/plots/cluster_profiles.png")

counts.unpersist()
spark.stop()
