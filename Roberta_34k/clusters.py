"""
GPU-accelerated clustering pipeline for Hindi FastText embeddings.

What runs on GPU:
  - L2 normalization (CuPy)
  - KMeans, MiniBatchKMeans, GaussianMixture (RAPIDS cuML)

What stays on CPU:
  - FastText vector extraction (no GPU backend exists)
  - Agglomerative, Spectral clustering (no mature GPU impl.)

Falls back to sklearn if cuML / CuPy are not installed.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os

import fasttext
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional GPU libraries
# ---------------------------------------------------------------------------
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
except ImportError:
    cuKMeans = cuMiniBatchKMeans = cuGaussianMixture = None
    _HAS_CUML = False

# CPU fallbacks
from sklearn.cluster import (
    MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, KMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize as sk_normalize


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KAGGLE_WORKING = Path("/kaggle/working")

if KAGGLE_WORKING.exists():
    OUT_DIR = KAGGLE_WORKING
elif "__file__" in globals():
    OUT_DIR = Path(__file__).resolve().parent
else:
    OUT_DIR = Path.cwd()

# Your dataset directory
DATASET_DIR = Path("/kaggle/input/datasets/rushilsharma13/34-hindi")

# Fallback roots to search
SEARCH_ROOTS = [
    DATASET_DIR,
    Path("/kaggle/input"),
]


def _find_file(name_patterns, roots=SEARCH_ROOTS):
    """Search for the first file matching any of the given glob patterns."""
    for root in roots:
        if not root.exists():
            continue
        for pattern in name_patterns:
            matches = list(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def _find_shabd_csv():
    patterns = [
        "Shabd Psycholinguistic database_34k Words List.csv",
        "*Shabd*.csv",
        "*shabd*.csv",
        "*34k*.csv",
    ]
    found = _find_file(patterns)
    if found:
        return found

    # Nothing found — print what IS there to help diagnose
    print("\n>>> Could not find the Shabd CSV. Listing available files:")
    for root in SEARCH_ROOTS:
        if root.exists():
            print(f"--- {root} ---")
            for r, _, fs in os.walk(root):
                for f in fs:
                    print("   ", os.path.join(r, f))
    raise FileNotFoundError("Shabd CSV not found.")


SHABD_PATH = _find_shabd_csv()
print(f"Using Shabd CSV: {SHABD_PATH}")
print(f"Output directory: {OUT_DIR}")

RANDOM_STATE = 42
N_CLUSTERS = 200

# ---------------------------------------------------------------------------
# 1. Load Shabd dataset
# ---------------------------------------------------------------------------
print("Loading Shabd dataset...")
shabd_data = pd.read_csv(SHABD_PATH, encoding="utf-8-sig", low_memory=False)
shabd_data.columns = shabd_data.columns.str.strip()
shabd_data = shabd_data.sort_values(by="Frequency", ascending=False).head(10000)
unique_words = (
    shabd_data["Word"].dropna().astype(str).str.strip().unique().tolist()
)
print(f"Total unique words selected for clustering: {len(unique_words)}")

# ---------------------------------------------------------------------------
# 2. Load FastText model (CPU-only)
# ---------------------------------------------------------------------------
INDICFT_PATH = _find_file(["indicnlp.ft.hi.300.bin"])
if INDICFT_PATH is None:
    INDICFT_PATH = OUT_DIR / "indicnlp.ft.hi.300.bin"  # placeholder

if INDICFT_PATH.exists():
    print(f"Loading local IndicFT model from {INDICFT_PATH}...")
    ft_model = fasttext.load_model(str(INDICFT_PATH))
else:
    print("Local IndicFT model not found. Falling back to Hugging Face...")
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-hi-vectors", filename="model.bin"
    )
    ft_model = fasttext.load_model(model_path)

# ---------------------------------------------------------------------------
# 3. Extract FastText vectors
# ---------------------------------------------------------------------------
print("Extracting FastText vectors...")

def _get_vec(w):
    return ft_model.get_word_vector(w)

n_threads = min(8, os.cpu_count() or 4)
with ThreadPoolExecutor(max_workers=n_threads) as ex:
    vectors = list(ex.map(_get_vec, unique_words))

X = np.asarray(vectors, dtype=np.float32)   # (N, 300)
words = unique_words
print(f"Raw embedding matrix: {X.shape}  (threads used: {n_threads})")

# ---------------------------------------------------------------------------
# 4. L2 normalization
# ---------------------------------------------------------------------------
if _HAS_CUPY:
    print("L2-normalizing on GPU (CuPy)...")
    X_gpu = cp.asarray(X)
    norms = cp.linalg.norm(X_gpu, axis=1, keepdims=True)
    norms = cp.where(norms == 0, cp.float32(1.0), norms)
    X_normalized = cp.asnumpy(X_gpu / norms)
else:
    print("L2-normalizing on CPU (sklearn)...")
    X_normalized = sk_normalize(X, norm="l2")

print(f"Normalized embedding matrix: {X_normalized.shape}")

np.save(OUT_DIR / "X_normalized.npy", X_normalized)
with open(OUT_DIR / "words.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))

# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------
if _HAS_CUML:
    print("\n>>> RAPIDS cuML detected — KMeans / MiniBatchKMeans / GMM on GPU.")
    algorithms = {
        "MiniBatchKMeans": cuMiniBatchKMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
        ),
        "Voronoi_Partition": cuKMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
        ),
        "Gaussian_Mixture": cuGaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="diag",
            random_state=RANDOM_STATE,
        ),
        "Agglomerative_Cosine": AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="cosine", linkage="average"
        ),
        "Spectral": SpectralClustering(
            n_clusters=N_CLUSTERS,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=RANDOM_STATE,
        ),
    }
else:
    print("\n>>> cuML NOT found — falling back to CPU scikit-learn.")
    print("    For GPU speedup install RAPIDS: https://rapids.ai/start.html")
    algorithms = {
        "MiniBatchKMeans": MiniBatchKMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
        ),
        "Agglomerative_Cosine": AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="cosine", linkage="average"
        ),
        "Spectral": SpectralClustering(
            n_clusters=N_CLUSTERS,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=RANDOM_STATE,
        ),
        "Voronoi_Partition": KMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
        ),
        "Gaussian_Mixture": GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="diag",
            random_state=RANDOM_STATE,
        ),
    }

# ---------------------------------------------------------------------------
# 6. Fit + save
# ---------------------------------------------------------------------------
for name, algo in algorithms.items():
    print(f"\nRunning {name}...")

    labels = algo.fit_predict(X_normalized)

    if _HAS_CUPY and isinstance(labels, cp.ndarray):
        labels = cp.asnumpy(labels)
    labels = np.asarray(labels).ravel()

    results_df = pd.DataFrame({"word": words, "cluster": labels})
    out_path = OUT_DIR / f"{name}_clusters.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {name} results to {out_path.name}")

print("\nClustering complete for all 5 algorithms.")
print(f"All outputs written to: {OUT_DIR}")