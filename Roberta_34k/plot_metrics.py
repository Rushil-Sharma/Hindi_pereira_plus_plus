"""
Plot clustering metrics from an existing CSV on Kaggle.
Reads:  /kaggle/working/clustering_metrics.csv
Writes: /kaggle/working/clustering_comparison.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = Path("/kaggle/working")
METRICS_PATH = OUT_DIR / "clustering_metrics.csv"

if not METRICS_PATH.exists():
    raise FileNotFoundError(f"{METRICS_PATH} not found. Run the metrics script first.")

df = pd.read_csv(METRICS_PATH)
print(df.head())

# ---------------------------------------------------------------------------
# 2x2 plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Clustering Algorithm Comparison: Hindi FastText Embedding Space",
    fontsize=18, fontweight="bold", y=0.95,
)
axes = axes.flatten()

metrics = [
    ("WCSS",              "Elbow Method: WCSS / Inertia vs. k", "WCSS",              False),
    ("Silhouette",        "Silhouette Score vs. k",             "Mean Silhouette",   True),
    ("Davies_Bouldin",    "Davies-Bouldin Index vs. k",         "Davies-Bouldin",    False),
    ("Calinski_Harabasz", "Calinski-Harabasz Index vs. k",      "Calinski-Harabasz", True),
]

for i, (col, title, ylabel, higher_better) in enumerate(metrics):
    ax = axes[i]
    sns.lineplot(
        data=df, x="k", y=col, hue="Algorithm",
        marker="o", ax=ax, palette="tab10",
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    note = "(Higher is better)" if higher_better else "(Lower is better)"
    ax.text(
        0.03, 0.08, note, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(title="Algorithm", fontsize=8, title_fontsize=9,
              loc="best", framealpha=0.8)

plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88)
out_path = OUT_DIR / "clustering_comparison.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {out_path}")
plt.show()