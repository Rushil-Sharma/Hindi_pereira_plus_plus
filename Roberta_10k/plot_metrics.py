"""
Plot clustering metrics (WCSS, Silhouette, Davies-Bouldin, Calinski-Harabasz)
for quantized RoBERTa-Hindi embeddings.

Reads:  clustering_metrics_roberta.csv (from evaluate_clusters.py)
Writes: clustering_comparison.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_PATH = SCRIPT_DIR / "clustering_metrics_roberta.csv"

# ---------------------------------------------------------------------------
# 1. Load the metrics
# ---------------------------------------------------------------------------
if not METRICS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {METRICS_PATH}. Please run evaluate_clusters.py first."
    )

df = pd.read_csv(METRICS_PATH)

# ---------------------------------------------------------------------------
# 2. Set up the 2x2 subplot grid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Clustering Algorithm Comparison: Quantized RoBERTa-Hindi Embedding Space",
    fontsize=18, fontweight="bold", y=0.95
)

axes = axes.flatten()

metrics = [
    ("WCSS",                "Elbow Method: WCSS / Inertia vs. k",  "Within-Cluster Sum of Squares (WCSS)", False),
    ("Silhouette",          "Silhouette Score vs. k",               "Mean Silhouette Score",                True),
    ("Davies_Bouldin",      "Davies-Bouldin Index vs. k",           "Davies-Bouldin Index",                 False),
    ("Calinski_Harabasz",   "Calinski-Harabasz Index vs. k",        "Calinski-Harabasz Score",              True),
]

# ---------------------------------------------------------------------------
# 3. Plot each metric
# ---------------------------------------------------------------------------
for i, (col, title, ylabel, higher_is_better) in enumerate(metrics):
    ax = axes[i]

    sns.lineplot(
        data=df,
        x="k",
        y=col,
        hue="Algorithm",
        marker="o",
        ax=ax,
        palette="tab10",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    note = "(Higher is better)" if higher_is_better else "(Lower is better)"
    ax.text(
        0.03, 0.08, note, transform=ax.transAxes,
        fontsize=9, verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.legend(title="Algorithm", fontsize=8, title_fontsize=9,
              loc="best", framealpha=0.8)

# ---------------------------------------------------------------------------
# 4. Adjust layout and save
# ---------------------------------------------------------------------------
plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88)

out_path = SCRIPT_DIR / "clustering_comparison.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Plot saved successfully to {out_path}")

plt.show()