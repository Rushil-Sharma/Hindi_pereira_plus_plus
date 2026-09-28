# =============================================================================
# KAGGLE NOTEBOOK: Hindi FastText Clustering on Full 34k Shabd Dataset
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Install dependencies
# -----------------------------------------------------------------------------
# !pip install fasttext huggingface_hub -q/

# -----------------------------------------------------------------------------
# 1. Imports & Setup
# -----------------------------------------------------------------------------
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import warnings
warnings.filterwarnings("ignore")

import fasttext
import numpy as np
import pandas as pd

# Optional GPU libraries
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

from sklearn.cluster import (
    MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, KMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 2. Locate the Shabd dataset on Kaggle
# -----------------------------------------------------------------------------
print("Searching for Shabd dataset...")

# Common Kaggle mount points
possible_roots = [
    Path("/kaggle/input/shabd-psycholinguistic-database-34k-words-list"),
    Path("/kaggle/input/shabd-psycholinguistic-database-34k-words-list/"),
    Path("/kaggle/input/datasets/rushilsharma13/shabd-psycholinguistic-database-34k-words-list"),
]

KAGGLE_INPUT = None
for root in possible_roots:
    if root.exists():
        KAGGLE_INPUT = root
        print(f"Found dataset root: {KAGGLE_INPUT}")
        break

if KAGGLE_INPUT is None:
    # Fallback: search all of /kaggle/input for a matching CSV
    print("Searching /kaggle/input recursively for Shabd CSV...")
    for csv_file in Path("/kaggle/input").rglob("*.csv"):
        if "Shabd" in csv_file.name or "Psycholinguistic" in csv_file.name:
            KAGGLE_INPUT = csv_file.parent
            print(f"Found CSV at: {csv_file}")
            break

if KAGGLE_INPUT is None:
    # Last resort: print directory tree to help debug
    print("\n--- Contents of /kaggle/input ---")
    for p in Path("/kaggle/input").iterdir():
        print(p)
    raise FileNotFoundError(
        "Could not find Shabd dataset. Please make sure you added the dataset "
        "to this notebook via the 'Add Data' button."
    )

# Now find the CSV file inside KAGGLE_INPUT
csv_files = list(KAGGLE_INPUT.rglob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {KAGGLE_INPUT}")

SHABD_PATH = csv_files[0]
print(f"Using dataset: {SHABD_PATH}")

# Output directory
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS = 200

# -----------------------------------------------------------------------------
# 3. Load Shabd dataset (ALL 34k words)
# -----------------------------------------------------------------------------
print("Loading Shabd dataset (full 34k)...")
shabd_data = pd.read_csv(SHABD_PATH, encoding="utf-8-sig", low_memory=False)
shabd_data.columns = shabd_data.columns.str.strip()

shabd_data = shabd_data.sort_values(by="Frequency", ascending=False)
unique_words = (
    shabd_data["Word"].dropna().astype(str).str.strip().unique().tolist()
)
print(f"Total unique words selected for clustering: {len(unique_words)}")

# -----------------------------------------------------------------------------
# 4. Load FastText model
# -----------------------------------------------------------------------------
indicft_candidates = list(Path("/kaggle/input").rglob("indicnlp.ft.hi.300.bin"))
if indicft_candidates:
    INDICFT_PATH = indicft_candidates[0]
    print(f"Loading local IndicFT model from {INDICFT_PATH}...")
    ft_model = fasttext.load_model(str(INDICFT_PATH))
else:
    print("Local IndicFT model not found. Falling back to Hugging Face...")
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(
        repo_id="facebook/fasttext-hi-vectors", filename="model.bin"
    )
    ft_model = fasttext.load_model(model_path)

# -----------------------------------------------------------------------------
# 5. Extract FastText vectors
# -----------------------------------------------------------------------------
print("Extracting FastText vectors...")

def _get_vec(w):
    return ft_model.get_word_vector(w)

n_threads = min(8, os.cpu_count() or 4)
with ThreadPoolExecutor(max_workers=n_threads) as ex:
    vectors = list(ex.map(_get_vec, unique_words))

X = np.asarray(vectors, dtype=np.float32)
words = unique_words
print(f"Raw embedding matrix: {X.shape}  (threads used: {n_threads})")

# -----------------------------------------------------------------------------
# 6. L2 normalization
# -----------------------------------------------------------------------------
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

np.save(OUTPUT_DIR / "X_normalized.npy", X_normalized)
with open(OUTPUT_DIR / "words.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))

# -----------------------------------------------------------------------------
# 7. Clustering at k=200
# -----------------------------------------------------------------------------
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
            n_components=N_CLUSTERS, covariance_type="diag",
            random_state=RANDOM_STATE,
        ),
        "Agglomerative_Cosine": AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="cosine", linkage="average"
        ),
        "Spectral": SpectralClustering(
            n_clusters=N_CLUSTERS, affinity="nearest_neighbors",
            n_neighbors=10, assign_labels="kmeans",
            random_state=RANDOM_STATE,
        ),
    }
else:
    print("\n>>> cuML NOT found — falling back to CPU scikit-learn.")
    algorithms = {
        "MiniBatchKMeans": MiniBatchKMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
        ),
        "Voronoi_Partition": KMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10
        ),
        "Gaussian_Mixture": GaussianMixture(
            n_components=N_CLUSTERS, covariance_type="diag",
            random_state=RANDOM_STATE,
        ),
        "Agglomerative_Cosine": AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="cosine", linkage="average"
        ),
        "Spectral": SpectralClustering(
            n_clusters=N_CLUSTERS, affinity="nearest_neighbors",
            n_neighbors=10, assign_labels="kmeans",
            random_state=RANDOM_STATE,
        ),
    }

for name, algo in algorithms.items():
    print(f"\nRunning {name}...")
    labels = algo.fit_predict(X_normalized)

    if _HAS_CUPY and isinstance(labels, cp.ndarray):
        labels = cp.asnumpy(labels)
    labels = np.asarray(labels).ravel()

    results_df = pd.DataFrame({"word": words, "cluster": labels})
    out_path = OUTPUT_DIR / f"{name}_clusters.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {name} results to {out_path.name}")

print("\nClustering complete for all 5 algorithms.")

# -----------------------------------------------------------------------------
# 8. Multi-k Evaluation
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("STARTING MULTI-K EVALUATION")
print("="*60)

k_values = [50, 100, 150, 200, 250, 300]
metrics_data = []

for k in k_values:
    print(f"\nEvaluating k={k}...")
    algorithms_eval = {
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10),
        "Agglomerative_Cosine": AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average'),
        "Spectral": SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=10, assign_labels='kmeans', random_state=42),
        "Voronoi_Partition": KMeans(n_clusters=k, random_state=42, n_init=10),
        "Gaussian_Mixture": GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
    }

    for name, algo in algorithms_eval.items():
        print(f"  Running {name}...")
        labels = algo.fit_predict(X_normalized)

        if hasattr(algo, "inertia_"):
            wcss = algo.inertia_
        else:
            wcss = 0
            for cluster_id in np.unique(labels):
                cluster_points = X_normalized[labels == cluster_id]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0)
                    wcss += np.sum((cluster_points - centroid) ** 2)

        if len(X_normalized) > 20000:
            sil_score = silhouette_score(X_normalized, labels, metric='euclidean', sample_size=10000, random_state=42)
        else:
            sil_score = silhouette_score(X_normalized, labels, metric='euclidean')

        db_index = davies_bouldin_score(X_normalized, labels)
        ch_score = calinski_harabasz_score(X_normalized, labels)

        metrics_data.append({
            "k": k,
            "Algorithm": name,
            "WCSS": wcss,
            "Silhouette": sil_score,
            "Davies_Bouldin": db_index,
            "Calinski_Harabasz": ch_score
        })

results_df = pd.DataFrame(metrics_data)
metrics_out = OUTPUT_DIR / "clustering_metrics.csv"
results_df.to_csv(metrics_out, index=False)
print(f"\nMetrics saved to {metrics_out}")

# -----------------------------------------------------------------------------
# 9. Plot Metrics
# -----------------------------------------------------------------------------
if not metrics_out.exists():
    raise FileNotFoundError(f"Could not find {metrics_out}. Run evaluation first.")

df = pd.read_csv(metrics_out)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Clustering Algorithm Comparison: FastText Embedding Space (34k Words)",
             fontsize=18, fontweight='bold', y=0.95)
axes = axes.flatten()

metrics = [
    ("WCSS", "Elbow Method: WCSS / Inertia vs. k", "Within-Cluster Sum of Squares (WCSS)", False),
    ("Silhouette", "Silhouette Score vs. k", "Mean Silhouette Score", True),
    ("Davies_Bouldin", "Davies-Bouldin Index vs. k", "Davies-Bouldin Index", False),
    ("Calinski_Harabasz", "Calinski-Harabasz Index vs. k", "Calinski-Harabasz Score", True)
]

for i, (col, title, ylabel, higher_is_better) in enumerate(metrics):
    ax = axes[i]
    sns.lineplot(data=df, x="k", y=col, hue="Algorithm", marker="o", ax=ax, palette="tab10")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)

    note = "(Higher is better)" if higher_is_better else "(Lower is better)"
    ax.text(0.03, 0.08, note, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend(title="Algorithm", fontsize=8, title_fontsize=9, loc='best', framealpha=0.8)

plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.88)
plot_out = OUTPUT_DIR / "clustering_comparison.png"
plt.savefig(plot_out, dpi=300, bbox_inches='tight')
print(f"Plot saved to {plot_out}")
plt.show()

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"Outputs saved in: {OUTPUT_DIR}")
print("Files generated:")
for f in ["X_normalized.npy", "words.txt", "clustering_metrics.csv", "clustering_comparison.png"]:
    p = OUTPUT_DIR / f
    if p.exists():
        print(f"  - {f}")
print("  - *_clusters.csv (for each algorithm)")