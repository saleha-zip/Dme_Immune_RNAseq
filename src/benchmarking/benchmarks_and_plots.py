"""
05_benchmarks_and_plots.py
--------------------------
Stage 5 — Benchmarking + Final Results Plots
  1. Partition scaling benchmark  (normalization at 4 / 8 / 21 / 42 partitions)
  2. Speedup chart                (pandas sequential vs Spark distributed)
  3. Heatmap of top 50 DE genes   across time points
  4. Immune gene dynamics plot    (Imd vs Toll vs Cellular trajectories)

Outputs → results/plots/ and results/benchmarks/
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/benchmarks", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("DmelImmune_Benchmarking")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Partition scaling benchmark
#    Re-run the CPM normalization step at different partition counts
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [1] Partition scaling benchmark ──")

counts_raw = spark.read.parquet("data/processed/spark_counts_long.parquet")

# Cache because the same DataFrame is reused across multiple benchmark runs
counts_raw.cache()
counts_raw.count()   # materialize cache

def run_normalization(n_partitions):
    df = counts_raw.repartition(n_partitions, "time_point")

    lib = df.groupBy("sample_id").agg(
        F.sum("raw_count").alias("lib_size")
    )

    df2 = (
        df.join(lib, on="sample_id")
        .withColumn(
            "log2_cpm",
            F.log2((F.col("raw_count") / F.col("lib_size")) * 1e6 + 1)
        )
    )

    # Force evaluation
    _ = df2.agg(F.sum("log2_cpm")).collect()

partition_counts = [4, 8, 21, 42]
partition_times  = []

for n in partition_counts:
    start = time.time()
    run_normalization(n)
    elapsed = time.time() - start

    partition_times.append({
        "n_partitions": n,
        "time_seconds": round(elapsed, 3)
    })

    print(f"  {n:3d} partitions → {elapsed:.2f}s")

partition_df = pd.DataFrame(partition_times)

partition_df.to_csv(
    "results/benchmarks/partition_scaling.csv",
    index=False
)

# Release cache after benchmark loop
counts_raw.unpersist()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load preprocessing benchmark and build speedup chart
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [2] Speedup chart ──")

bench_pd = pd.read_csv(
    "results/benchmarks/benchmark_preprocessing.csv"
)

pandas_time = bench_pd.loc[
    bench_pd["method"] == "pandas_sequential",
    "time_seconds"
].values[0]

spark_time = bench_pd.loc[
    bench_pd["method"] == "spark_distributed",
    "time_seconds"
].values[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Left: absolute execution times ────────────────────────────────────────────
ax = axes[0]

methods = [
    "Pandas\n(sequential)",
    "PySpark\n(distributed)"
]

times = [pandas_time, spark_time]

colors = [
    "#888780",
    "#3B8BD4"
]

bars = ax.bar(
    methods,
    times,
    color=colors,
    width=0.45,
    edgecolor="white",
    linewidth=0.5
)

for bar, t in zip(bars, times):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.05,
        f"{t:.2f}s",
        ha="center",
        va="bottom",
        fontsize=10
    )

ax.set_ylabel("Time (seconds)")
ax.set_title("Preprocessing time: sequential vs distributed")
ax.set_ylim(0, max(times) * 1.25)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)

# ── Right: partition scaling ──────────────────────────────────────────────────
ax = axes[1]

ax.plot(
    partition_df["n_partitions"],
    partition_df["time_seconds"],
    marker="o",
    color="#1D9E75",
    linewidth=2,
    markersize=6
)

ax.set_xlabel("Number of partitions")
ax.set_ylabel("Time (seconds)")
ax.set_title("Normalization time vs partition count")
ax.set_xticks(partition_counts)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

plt.savefig(
    "results/plots/benchmarks.png",
    dpi=150
)

plt.close()

print(f"  Speedup: {pandas_time / spark_time:.2f}×")
print("  Saved → results/plots/benchmarks.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Heatmap — top 50 DE genes across time points
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [3] DE gene heatmap ──")

fc_pd = pd.read_csv(
    "results/tables/fold_change_top500.csv"
)

# Select top 50 genes by max absolute fold change
top50_genes = (
    fc_pd.groupby("gene_id")["log2fc"]
    .apply(lambda x: x.abs().max())
    .sort_values(ascending=False)
    .head(50)
    .index.tolist()
)

heat_data = (
    fc_pd[fc_pd["gene_id"].isin(top50_genes)]
    .pivot_table(
        index="gene_id",
        columns="hours",
        values="log2fc"
    )
    .reindex(columns=sorted(fc_pd["hours"].unique()))
)

fig, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(
    heat_data,
    cmap="RdBu_r",
    center=0,
    vmin=-4,
    vmax=4,
    linewidths=0,
    ax=ax,
    cbar_kws={
        "label": "log₂ fold change vs t=0",
        "shrink": 0.6
    }
)

ax.set_xlabel("Hours post-infection")
ax.set_ylabel("Gene (FlyBase ID)")

ax.set_title(
    "Top 50 differentially expressed genes\n"
    "D. melanogaster innate immune response",
    fontsize=11
)

ax.tick_params(axis="y", labelsize=7)

plt.tight_layout()

plt.savefig(
    "results/plots/heatmap_top50_DE_genes.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("  Saved → results/plots/heatmap_top50_DE_genes.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Immune process dynamics — Imd vs Toll vs Cellular
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── [4] Immune process dynamics ──")

imm_pd = pd.read_csv(
    "results/tables/immune_gene_dynamics.csv"
)

process_colors = {
    "IMD":        "#E8593C",
    "Humoral":    "#EF9F27",
    "Toll":       "#3B8BD4",
    "Cellular":   "#1D9E75",
    "Anti-viral": "#888780",
}

fig, ax = plt.subplots(figsize=(9, 5))

for process, group in imm_pd.groupby("immune_process"):

    profile = group.groupby("hours")["mean_log2_cpm"].mean()

    color = process_colors.get(process, "#AAAAAA")

    ax.plot(
        profile.index,
        profile.values,
        label=f"{process} (n={group['n_genes'].iloc[0]})",
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=4
    )

ax.axvline(
    0,
    color="gray",
    linewidth=1,
    linestyle="--",
    alpha=0.5,
    label="Infection (t=0)"
)

ax.set_xlabel("Hours post-infection")
ax.set_ylabel("Mean log₂(CPM + 1)")

ax.set_title(
    "Immune gene expression dynamics by functional category\n"
    "D. melanogaster innate immune response",
    fontsize=10
)

ax.legend(fontsize=8, framealpha=0.7)

ax.grid(True, alpha=0.25, linewidth=0.5)

ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()

plt.savefig(
    "results/plots/immune_process_dynamics.png",
    dpi=150
)

plt.close()

print("  Saved → results/plots/immune_process_dynamics.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Final benchmark summary table
# ═══════════════════════════════════════════════════════════════════════════════
summary = pd.DataFrame([
    {
        "stage": "preprocessing",
        "method": "pandas_sequential",
        "time_s": pandas_time,
        "notes": "full 41-sample matrix"
    },

    {
        "stage": "preprocessing",
        "method": "spark_distributed",
        "time_s": spark_time,
        "notes": "full 41-sample distributed pipeline"
    },

    {
        "stage": "partition_scaling_4",
        "method": "spark",
        "time_s": partition_times[0]["time_seconds"],
        "notes": ""
    },

    {
        "stage": "partition_scaling_8",
        "method": "spark",
        "time_s": partition_times[1]["time_seconds"],
        "notes": ""
    },

    {
        "stage": "partition_scaling_21",
        "method": "spark",
        "time_s": partition_times[2]["time_seconds"],
        "notes": ""
    },

    {
        "stage": "partition_scaling_42",
        "method": "spark",
        "time_s": partition_times[3]["time_seconds"],
        "notes": ""
    },
])

summary.to_csv(
    "results/benchmarks/full_benchmark_summary.csv",
    index=False
)

print(f"\n✓ All benchmarks and plots complete")

print("  results/plots/       → benchmarks.png,")
print("                          heatmap_top50_DE_genes.png,")
print("                          immune_process_dynamics.png")

print("  results/benchmarks/  → full_benchmark_summary.csv,")
print("                          partition_scaling.csv")

spark.stop()