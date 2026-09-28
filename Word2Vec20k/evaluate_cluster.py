"""
GPU-Accelerated Evaluation Script for Word2Vec Hindi Clustering.
Computes WCSS, Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score
across multiple values of k (50, 100, 150, 200, 250, 300) for 4 clustering algorithms.

GPU Acceleration:
  - MiniBatchKMeans, KMeans, GMM via RAPIDS cuML on GPU (when available)
  - PyTorch CUDA vectorized WCSS and distance calculations
  - Graceful fallback to optimized scikit-learn on CPU
"""

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import (
    MiniBatchKMeans, AgglomerativeClustering, KMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# ---------------------------------------------------------------------------
# GPU Detection & cuML Libraries
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Evaluation Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

try:
    from cuml.cluster import (
        KMeans as cuKMeans,
        MiniBatchKMeans as cuMiniBatchKMeans,
    )
    from cuml.mixture import GaussianMixture as cuGaussianMixture
    _HAS_CUML = True
    print("    cuML detected: GPU-accelerated clustering enabled!")
except ImportError:
    cuKMeans = cuMiniBatchKMeans = cuGaussianMixture = None
    _HAS_CUML = False
    print("    cuML not detected: running optimized CPU clustering fallback.")

# ---------------------------------------------------------------------------
# Paths (Kaggle & local friendly)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
KAGGLE_WORKING = Path("/kaggle/working")

if KAGGLE_WORKING.exists():
    OUT_DIR = KAGGLE_WORKING
else:
    OUT_DIR = SCRIPT_DIR

X_PATH = OUT_DIR / "X_word2vec_normalized.npy"
if not X_PATH.exists():
    X_PATH = OUT_DIR / "X_normalized.npy"

WORDS_PATH = OUT_DIR / "words_word2vec.txt"
if not WORDS_PATH.exists():
    WORDS_PATH = OUT_DIR / "words.txt"

if not X_PATH.exists():
    raise FileNotFoundError(f"Could not find embeddings file at {X_PATH}. Run cluster.py first.")

# ---------------------------------------------------------------------------
# 1. Load Embeddings and Words
# ---------------------------------------------------------------------------
print(f"Loading embeddings from {X_PATH}...")
X = np.load(X_PATH).astype(np.float32)

if WORDS_PATH.exists():
    with open(WORDS_PATH, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f.readlines()]
    print(f"Embedding matrix: {X.shape}, words: {len(words)}")
else:
    print(f"Embedding matrix: {X.shape}")

# ---------------------------------------------------------------------------
# 2. Evaluation Parameters
# ---------------------------------------------------------------------------
k_values = [50, 100, 150, 200, 250, 300]
metrics_data = []

# Silhouette sampling prevents O(N^2) memory lockup on CPU for 20k-34k samples.
# 10k random sample yields virtually identical scores in seconds.
SILHOUETTE_SAMPLE_SIZE = 10000 if len(X) > 10000 else None

# Pre-load to GPU tensor if CUDA available for fast WCSS
if DEVICE.type == "cuda":
    X_cuda = torch.as_tensor(X, device=DEVICE)
else:
    X_cuda = None

# ---------------------------------------------------------------------------
# 3. Evaluate Algorithms for Every k (GPU accelerated where possible)
# ---------------------------------------------------------------------------
for k in k_values:
    print(f"\n{'='*20} Evaluating k={k} {'='*20}")

    if _HAS_CUML:
        algorithms = {
            "MiniBatchKMeans": cuMiniBatchKMeans(
                n_clusters=k, random_state=42
            ),
            "Voronoi_Partition": cuKMeans(
                n_clusters=k, random_state=42
            ),
            "Gaussian_Mixture": cuGaussianMixture(
                n_components=k, covariance_type="diag", random_state=42
            ),
            "Agglomerative_Cosine": AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ),
        }
    else:
        algorithms = {
            "MiniBatchKMeans": MiniBatchKMeans(
                n_clusters=k, random_state=42, n_init="auto"
            ),
            "Voronoi_Partition": KMeans(
                n_clusters=k, random_state=42, n_init="auto"
            ),
            "Gaussian_Mixture": GaussianMixture(
                n_components=k, covariance_type="diag", random_state=42
            ),
            "Agglomerative_Cosine": AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ),
        }

    for name, algo in algorithms.items():
        print(f"  Running {name}...")

        labels = algo.fit_predict(X)

        if _HAS_CUPY and isinstance(labels, cp.ndarray):
            labels = cp.asnumpy(labels)
        labels = np.asarray(labels).ravel()

        # Sanity check: unique clusters
        n_unique = len(np.unique(labels))
        if n_unique < k:
            print(f"    Notice: {name} produced {n_unique} unique clusters (expected {k})")

        # Metric 1: WCSS (Inertia) - GPU accelerated if CUDA available
        if hasattr(algo, "inertia_") and algo.inertia_ is not None:
            wcss = float(algo.inertia_)
        elif X_cuda is not None:
            # Fast vectorized WCSS on GPU
            labels_cuda = torch.as_tensor(labels, device=DEVICE)
            wcss = 0.0
            for cid in torch.unique(labels_cuda):
                pts = X_cuda[labels_cuda == cid]
                if pts.shape[0] == 0:
                    continue
                centroid = pts.mean(dim=0, keepdim=True)
                wcss += float(((pts - centroid) ** 2).sum().item())
        else:
            wcss = 0.0
            for cid in np.unique(labels):
                pts = X[labels == cid]
                if len(pts) == 0:
                    continue
                centroid = pts.mean(axis=0)
                wcss += float(np.sum((pts - centroid) ** 2))

        # Metric 2: Silhouette Score (higher is better)
        sil_score = silhouette_score(
            X, labels, metric="euclidean",
            sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=42
        )

        # Metric 3: Davies-Bouldin Index (lower is better)
        db_index = davies_bouldin_score(X, labels)

        # Metric 4: Calinski-Harabasz Score (higher is better)
        ch_score = calinski_harabasz_score(X, labels)

        metrics_data.append({
            "k": k,
            "Algorithm": name,
            "WCSS": wcss,
            "Silhouette": sil_score,
            "Davies_Bouldin": db_index,
            "Calinski_Harabasz": ch_score,
        })

        print(f"    WCSS={wcss:.2f} | Sil={sil_score:.4f} | DB={db_index:.4f} | CH={ch_score:.2f}")

# ---------------------------------------------------------------------------
# 4. Save Metrics
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(metrics_data)
out_csv_word2vec = OUT_DIR / "clustering_metrics_word2vec.csv"
out_csv_default = OUT_DIR / "clustering_metrics.csv"

results_df.to_csv(out_csv_word2vec, index=False)
results_df.to_csv(out_csv_default, index=False)
print(f"\nMetrics successfully saved to {out_csv_word2vec.name} and {out_csv_default.name}")

# ---------------------------------------------------------------------------
# 5. Preview Table
# ---------------------------------------------------------------------------
print("\nPreview of clustering metrics:")
print(results_df.head(15).to_string(index=False))
