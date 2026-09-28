"""
Plot clustering metrics (WCSS, Silhouette, Davies-Bouldin, Calinski-Harabasz)
for Word2Vec Hindi embeddings on Shabd dataset.

Reads:  clustering_metrics_word2vec.csv (or clustering_metrics.csv)
Writes: clustering_comparison.png and clustering_comparison_word2vec.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths (Kaggle & local friendly)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
KAGGLE_WORKING = Path("/kaggle/working")

if KAGGLE_WORKING.exists():
    OUT_DIR = KAGGLE_WORKING
else:
    OUT_DIR = SCRIPT_DIR

METRICS_PATH = OUT_DIR / "clustering_metrics_word2vec.csv"
if not METRICS_PATH.exists():
    METRICS_PATH = OUT_DIR / "clustering_metrics.csv"

# ---------------------------------------------------------------------------
# 1. Load the metrics
# ---------------------------------------------------------------------------
if not METRICS_PATH.exists():
    raise FileNotFoundError(
        f"Could not find metrics CSV at {METRICS_PATH}. Please run evaluate_cluster.py first."
    )

df = pd.read_csv(METRICS_PATH)
print("Loaded clustering metrics:")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Set up the 2x2 subplot grid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Clustering Algorithm Comparison: Word2Vec-Hindi Embedding Space (Shabd)",
    fontsize=18, fontweight="bold", y=0.95
)

axes = axes.flatten()

metrics = [
    ("WCSS",              "Elbow Method: WCSS / Inertia vs. k",  "Within-Cluster Sum of Squares (WCSS)", False),
    ("Silhouette",        "Silhouette Score vs. k",              "Mean Silhouette Score",                True),
    ("Davies_Bouldin",    "Davies-Bouldin Index vs. k",          "Davies-Bouldin Index",                 False),
    ("Calinski_Harabasz", "Calinski-Harabasz Index vs. k",       "Calinski-Harabasz Score",              True),
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
        linewidth=2,
        markersize=6,
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

out_path = OUT_DIR / "clustering_comparison.png"
out_path_named = OUT_DIR / "clustering_comparison_word2vec.png"

plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.savefig(out_path_named, dpi=300, bbox_inches="tight")
print(f"Plot saved successfully to:\n  - {out_path}\n  - {out_path_named}")

plt.show()
