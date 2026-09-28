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
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
X_PATH = SCRIPT_DIR / "X_normalized.npy"
WORDS_PATH = SCRIPT_DIR / "words.txt"

# ---------------------------------------------------------------------------
# 1. Load the saved embeddings and words
# ---------------------------------------------------------------------------
print("Loading saved embeddings...")
X = np.load(X_PATH)
with open(WORDS_PATH, "r", encoding="utf-8") as f:
    words = [line.strip() for line in f.readlines()]

# ---------------------------------------------------------------------------
# 2. Define the k values and algorithms to test
# ---------------------------------------------------------------------------
k_values = [50, 100, 150, 200, 250, 300]

# We'll store the results in a list of dictionaries
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
        if name == "Gaussian_Mixture":
            labels = algo.fit_predict(X)
        else:
            labels = algo.fit_predict(X)
        
        # Calculate Metrics
        # 1. WCSS (Inertia) - only available for centroid-based models
        if hasattr(algo, "inertia_"):
            wcss = algo.inertia_
        else:
            # For algorithms without inertia_, we compute it manually
            # (This approximates WCSS using Euclidean distance to centroid)
            wcss = 0
            for cluster_id in np.unique(labels):
                cluster_points = X[labels == cluster_id]
                centroid = cluster_points.mean(axis=0)
                wcss += np.sum((cluster_points - centroid) ** 2)
        
        # 2. Silhouette Score (higher is better, -1 to 1)
        sil_score = silhouette_score(X, labels, metric='euclidean')
        
        # 3. Davies-Bouldin Index (lower is better)
        db_index = davies_bouldin_score(X, labels)
        
        # 4. Calinski-Harabasz Score (higher is better)
        ch_score = calinski_harabasz_score(X, labels)
        
        # Store results
        metrics_data.append({
            "k": k,
            "Algorithm": name,
            "WCSS": wcss,
            "Silhouette": sil_score,
            "Davies_Bouldin": db_index,
            "Calinski_Harabasz": ch_score
        })

# ---------------------------------------------------------------------------
# 4. Save the metrics to a CSV
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(metrics_data)
out_path = SCRIPT_DIR / "clustering_metrics.csv"
results_df.to_csv(out_path, index=False)
print(f"\nMetrics saved to {out_path}")

# ---------------------------------------------------------------------------
# 5. Print a preview of the results
# ---------------------------------------------------------------------------
print("\nPreview of the metrics:")
print(results_df.head(10).to_string(index=False))