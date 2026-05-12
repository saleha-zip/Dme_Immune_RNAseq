"""
06_scaling_benchmark.py
-----------------------
Scalability Demonstration — Synthetic Data Scaling Experiment

PURPOSE:
    This script is NOT part of the biological analysis pipeline.
    It is a standalone computational benchmark that answers one question:

        "At what dataset size does Spark outperform pandas?"

    The real pipeline (stages 01-05) runs on 41 biological samples.
    That dataset is intentionally small — Spark overhead dominates there,
    which is expected and well-documented for sub-memory datasets.

    Here we synthetically scale the count matrix by replication to simulate
    larger RNA-seq cohorts (e.g. GTEx, recount3, ENCODE) and find the
    crossover point where distributed computing becomes beneficial.

WHAT IS BEING BENCHMARKED:
    The normalization step (CPM + log2 transform) — the most compute-intensive
    purely numerical operation in the pipeline. Identical logic, both methods.

SCALE FACTORS:
    1x   →   41 samples  (our real dataset)
    5x   →  205 samples
    10x  →  410 samples
    25x  →  1025 samples
    50x  →  2050 samples
    100x →  4100 samples  (large cohort scale)

NOTE:
    Replicated rows are computationally equivalent to new samples for
    benchmarking normalization throughput. No biological conclusions
    should be drawn from this synthetic data.
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

os.makedirs("results/benchmarks", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# ── Spark session ─────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("DmelImmune_ScalingBenchmark")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")   # tuned for local mode
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── Load real long-format data as base ───────────────────────────────────────
print("\n── Loading base dataset ──")
base_spark = spark.read.parquet("data/processed/spark_counts_long.parquet")
base_count = base_spark.count()
print(f"   Base rows: {base_count:,}  (41 samples × ~16,800 genes)")

# Load same data into pandas as base
base_pd = pd.read_csv("data/count_matrix/raw_counts.csv", index_col=0)
base_samples = base_pd.shape[1]
print(f"   Pandas base: {base_pd.shape[0]:,} genes × {base_samples} samples")

# ── Define scale factors ──────────────────────────────────────────────────────
scale_factors = [1, 5, 10, 25, 50, 100]
results = []

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── Running scaling benchmark ──")
print(f"{'Scale':>6} {'Samples':>8} {'Rows':>12} {'Pandas (s)':>12} {'Spark (s)':>11} {'Speedup':>9}")
print("─" * 65)

for scale in scale_factors:
    n_samples = base_samples * scale
    n_rows    = base_count * scale

    # ── Pandas benchmark ─────────────────────────────────────────────────────
    # Replicate columns to simulate more samples
    df_scaled = pd.concat([base_pd] * scale, axis=1)
    # Deduplicate column names (required for library size computation)
    df_scaled.columns = [f"s{i}" for i in range(df_scaled.shape[1])]

    t_pd_start = time.time()
    lib_sizes  = df_scaled.sum(axis=0)
    df_cpm     = df_scaled.div(lib_sizes, axis=1) * 1e6
    df_log2    = np.log2(df_cpm + 1)
    _ = df_log2.values.sum()   # force full evaluation
    t_pd_end   = time.time()
    pandas_time = round(t_pd_end - t_pd_start, 3)

    # ── Spark benchmark ──────────────────────────────────────────────────────
    # Replicate rows to simulate more samples (union is O(1) in the plan)
    df_spark = base_spark
    for _ in range(scale - 1):
        df_spark = df_spark.union(base_spark)

    # Use a number of partitions proportional to scale, capped sensibly
    n_partitions = min(scale * 4, 64)
    df_spark = df_spark.repartition(n_partitions)

    t_sp_start = time.time()
    lib_sizes_sp = df_spark.groupBy("sample_id").agg(
        F.sum("raw_count").alias("lib_size")
    )
    df_norm = (
        df_spark
        .join(lib_sizes_sp, on="sample_id")
        .withColumn("log2_cpm",
                    F.log2((F.col("raw_count") / F.col("lib_size")) * 1e6 + 1))
    )
    # Force full DAG execution
    _ = df_norm.agg(F.sum("log2_cpm")).collect()
    t_sp_end   = time.time()
    spark_time = round(t_sp_end - t_sp_start, 3)

    speedup = round(pandas_time / spark_time, 2)

    results.append({
        "scale_factor":  scale,
        "n_samples":     n_samples,
        "n_rows":        n_rows,
        "pandas_seconds": pandas_time,
        "spark_seconds":  spark_time,
        "speedup":        speedup
    })

    marker = " ← crossover" if speedup > 1.0 and results[-2]["speedup"] <= 1.0 \
             else (" ✓ Spark wins" if speedup > 1.0 else "")
    print(f"{scale:>6}x {n_samples:>8,} {n_rows:>12,} "
          f"{pandas_time:>12.3f} {spark_time:>11.3f} {speedup:>8.2f}×{marker}")

# ── Save results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("results/benchmarks/scaling_benchmark.csv", index=False)
print(f"\n   Saved → results/benchmarks/scaling_benchmark.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Scaling curves: pandas vs Spark wall-clock time
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(results_df["n_samples"], results_df["pandas_seconds"],
        marker="o", color="#888780", linewidth=2, markersize=6,
        label="Pandas (sequential)")
ax.plot(results_df["n_samples"], results_df["spark_seconds"],
        marker="s", color="#3B8BD4", linewidth=2, markersize=6,
        label="PySpark (distributed, local mode)")

# Mark crossover if it exists
crossover = results_df[results_df["speedup"] >= 1.0]
if not crossover.empty:
    cx = crossover.iloc[0]
    ax.axvline(cx["n_samples"], color="#1D9E75", linewidth=1.2,
               linestyle="--", alpha=0.7)
    ax.text(cx["n_samples"] * 1.03, ax.get_ylim()[1] * 0.9,
            f"Crossover\n~{int(cx['n_samples'])} samples",
            color="#1D9E75", fontsize=8.5, va="top")

ax.set_xlabel("Number of samples (simulated)")
ax.set_ylabel("Wall-clock time (seconds)")
ax.set_title("Normalization runtime: pandas vs PySpark\nas dataset size increases",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)

# ── PLOT 2 — Speedup ratio across scale factors ───────────────────────────────
ax = axes[1]
colors = ["#E8593C" if s < 1.0 else "#1D9E75"
          for s in results_df["speedup"]]
bars = ax.bar(results_df["n_samples"].astype(str),
              results_df["speedup"],
              color=colors, edgecolor="white", linewidth=0.5)

ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.6,
           label="Break-even (speedup = 1×)")

for bar, val in zip(bars, results_df["speedup"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}×", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Number of samples (simulated)")
ax.set_ylabel("Speedup (pandas time / Spark time)")
ax.set_title("PySpark speedup over pandas\nRed = Spark slower, Green = Spark faster",
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=30)

plt.suptitle(
    "Synthetic scaling experiment — CPM normalization throughput\n"
    "NOTE: Simulated data only. No biological conclusions.",
    fontsize=9, color="#666666", y=1.02
)
plt.tight_layout()
plt.savefig("results/plots/scaling_benchmark.png", dpi=150, bbox_inches="tight")
plt.close()
print("   Saved → results/plots/scaling_benchmark.png")

# ── Print clean summary ───────────────────────────────────────────────────────
spark_wins = results_df[results_df["speedup"] >= 1.0]
if not spark_wins.empty:
    threshold = spark_wins.iloc[0]
    print(f"\n── Summary ──")
    print(f"   Spark breaks even at ~{int(threshold['n_samples'])} samples "
          f"({threshold['scale_factor']}× our real dataset)")
    print(f"   At {int(results_df.iloc[-1]['n_samples'])} samples: "
          f"{results_df.iloc[-1]['speedup']}× speedup")
else:
    print("\n── Summary ──")
    print("   Spark did not outperform pandas at any tested scale in local mode.")
    print("   This is expected — true gains require a multi-node cluster.")
    print("   Framing: 'crossover exists at cluster scale, not demonstrated in local mode'")

print(f"\n✓ Scaling benchmark complete")
print("  → results/benchmarks/scaling_benchmark.csv")
print("  → results/plots/scaling_benchmark.png")

spark.stop()
