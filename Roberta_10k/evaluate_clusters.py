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
# Paths  ← ONLY CHANGE: point to RoBERTa outputs
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
X_PATH = SCRIPT_DIR / "X_roberta_normalized.npy"
WORDS_PATH = SCRIPT_DIR / "words_roberta.txt"

# ---------------------------------------------------------------------------
# 1. Load the saved embeddings and words
# ---------------------------------------------------------------------------
print("Loading saved embeddings...")
X = np.load(X_PATH)
with open(WORDS_PATH, "r", encoding="utf-8") as f:
    words = [line.strip() for line in f.readlines()]

print(f"Embeddings: {X.shape}  |  Words: {len(words)}")

# ---------------------------------------------------------------------------
# 2. Define the k values and algorithms to test
# ---------------------------------------------------------------------------
k_values = [50, 100, 150, 200, 250, 300]
metrics_data = []

# ---------------------------------------------------------------------------
# 3. Evaluate each algorithm for every k
# ---------------------------------------------------------------------------
for k in k_values:
    print(f"\nEvaluating k={k}...")

    algorithms = {
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10),
        "Agglomerative_Cosine": AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average'),
        "Spectral": SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', random_state=42),
        "Voronoi_Partition": KMeans(n_clusters=k, random_state=42, n_init=10),
        "Gaussian_Mixture": GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
    }

    for name, algo in algorithms.items():
        print(f"  Running {name}...")

        # Fit the model
        labels = algo.fit_predict(X)

        # Sanity check: how many distinct clusters did we actually get?
        n_unique = len(np.unique(labels))
        if n_unique < k:
            print(f"    ⚠ {name} produced only {n_unique} unique clusters (expected {k})")

        # --- Metric 1: WCSS (Inertia) ---
        if hasattr(algo, "inertia_"):
            wcss = algo.inertia_
        else:
            # Manual WCSS using Euclidean distance to centroid
            wcss = 0.0
            for cluster_id in np.unique(labels):
                cluster_points = X[labels == cluster_id]
                if len(cluster_points) == 0:
                    continue
                centroid = cluster_points.mean(axis=0)
                wcss += np.sum((cluster_points - centroid) ** 2)

        # --- Metric 2: Silhouette (higher is better) ---
        sil_score = silhouette_score(X, labels, metric='euclidean')

        # --- Metric 3: Davies-Bouldin (lower is better) ---
        db_index = davies_bouldin_score(X, labels)

        # --- Metric 4: Calinski-Harabasz (higher is better) ---
        ch_score = calinski_harabasz_score(X, labels)

        metrics_data.append({
            "k": k,
            "Algorithm": name,
            "WCSS": wcss,
            "Silhouette": sil_score,
            "Davies_Bouldin": db_index,
            "Calinski_Harabasz": ch_score
        })

        print(f"    WCSS={wcss:.2f}  Sil={sil_score:.4f}  "
              f"DB={db_index:.4f}  CH={ch_score:.2f}")

# ---------------------------------------------------------------------------
# 4. Save the metrics to a CSV
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(metrics_data)
out_path = SCRIPT_DIR / "clustering_metrics_roberta.csv"
results_df.to_csv(out_path, index=False)
print(f"\nMetrics saved to {out_path}")

# ---------------------------------------------------------------------------
# 5. Print a preview
# ---------------------------------------------------------------------------
print("\nPreview of the metrics:")
print(results_df.head(10).to_string(index=False))