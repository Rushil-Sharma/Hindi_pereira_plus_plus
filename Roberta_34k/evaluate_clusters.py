from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import (
    MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, KMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# ---------------------------------------------------------------------------
# Paths (Kaggle-friendly — no __file__)
# ---------------------------------------------------------------------------
KAGGLE_WORKING = Path("/kaggle/working")

if KAGGLE_WORKING.exists():
    OUT_DIR = KAGGLE_WORKING
elif "__file__" in globals():
    OUT_DIR = Path(__file__).resolve().parent
else:
    OUT_DIR = Path.cwd()

X_PATH = OUT_DIR / "X_normalized.npy"
WORDS_PATH = OUT_DIR / "words.txt"

# ---------------------------------------------------------------------------
# 1. Load the saved embeddings and words
# ---------------------------------------------------------------------------
print(f"Loading embeddings from {X_PATH} ...")
X = np.load(X_PATH)
with open(WORDS_PATH, "r", encoding="utf-8") as f:
    words = [line.strip() for line in f.readlines()]

print(f"Embedding matrix: {X.shape}, words: {len(words)}")

# ---------------------------------------------------------------------------
# 2. k values to test
# ---------------------------------------------------------------------------
k_values = [50, 100, 150, 200, 250, 300]
metrics_data = []

# ---------------------------------------------------------------------------
# 3. Evaluate each algorithm for every k
# ---------------------------------------------------------------------------
for k in k_values:
    print(f"\n=== Evaluating k={k} ===")

    algorithms = {
        "MiniBatchKMeans": MiniBatchKMeans(
            n_clusters=k, random_state=42, n_init=10
        ),
        "Agglomerative_Cosine": AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ),
        "Spectral": SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=42,
        ),
        "Voronoi_Partition": KMeans(
            n_clusters=k, random_state=42, n_init=10
        ),
        "Gaussian_Mixture": GaussianMixture(
            n_components=k, covariance_type="diag", random_state=42
        ),
    }

    for name, algo in algorithms.items():
        print(f"  Running {name} ...")

        labels = algo.fit_predict(X)
        labels = np.asarray(labels).ravel()

        # WCSS: use inertia_ if available, otherwise compute manually
        if hasattr(algo, "inertia_"):
            wcss = float(algo.inertia_)
        else:
            wcss = 0.0
            for cid in np.unique(labels):
                pts = X[labels == cid]
                if len(pts) == 0:
                    continue
                centroid = pts.mean(axis=0)
                wcss += float(np.sum((pts - centroid) ** 2))

        sil_score = silhouette_score(X, labels, metric="euclidean")
        db_index = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)

        metrics_data.append({
            "k": k,
            "Algorithm": name,
            "WCSS": wcss,
            "Silhouette": sil_score,
            "Davies_Bouldin": db_index,
            "Calinski_Harabasz": ch_score,
        })

# ---------------------------------------------------------------------------
# 4. Save metrics
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(metrics_data)
out_path = OUT_DIR / "clustering_metrics.csv"
results_df.to_csv(out_path, index=False)
print(f"\nMetrics saved to {out_path}")

# ---------------------------------------------------------------------------
# 5. Preview
# ---------------------------------------------------------------------------
print("\nPreview of the metrics:")
print(results_df.head(10).to_string(index=False))