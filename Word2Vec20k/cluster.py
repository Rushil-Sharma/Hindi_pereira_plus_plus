"""
GPU-Accelerated Word2Vec-Hindi Clustering Pipeline for Shabd 34k Dataset.
Model: AbhishekBiswas12/word2vec-hindi (Hugging Face)

GPU Acceleration:
  - Vector extraction & L2 normalization via PyTorch CUDA / CuPy
  - KMeans, MiniBatchKMeans, Gaussian Mixture via RAPIDS cuML on GPU
  - Automatic graceful fallback to scikit-learn on CPU if cuML/CUDA not present
  - 4 fast algorithms (Spectral Clustering excluded for fast runtime)
"""

from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

# PyTorch unpickling expects 'Word2Vec' class in __main__
class Word2Vec(nn.Module):
    pass

sys.modules["__main__"].Word2Vec = Word2Vec

# ---------------------------------------------------------------------------
# GPU Detection & Libraries
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> Computation Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

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
    MiniBatchKMeans, AgglomerativeClustering, KMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize as sk_normalize


# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
HINDI_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "Word2Vec" else SCRIPT_DIR

KAGGLE_WORKING = Path("/kaggle/working")
if KAGGLE_WORKING.exists():
    OUT_DIR = KAGGLE_WORKING
else:
    OUT_DIR = SCRIPT_DIR

# Max words to cluster (set to 20000 for high speed, or None for all available in vocab ~28k)
NUM_WORDS = 20000
RANDOM_STATE = 42
N_CLUSTERS = 200

# Roots to search for datasets and files
SEARCH_ROOTS = [
    OUT_DIR,
    SCRIPT_DIR,
    HINDI_DIR,
    HINDI_DIR / "Shabd",
    HINDI_DIR / "Shabd" / "fygme-osfstorage-archive",
    Path("/kaggle/input"),
    Path("/kaggle/input/datasets/rushilsharma13/34-hindi"),
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
        "*Shabd*34k*.csv",
        "*34k*.csv",
        "*Shabd*.csv",
    ]
    found = _find_file(patterns)
    if found:
        return found

    print("\n>>> Could not find the Shabd CSV. Searching available directories...")
    for root in SEARCH_ROOTS:
        if root.exists():
            print(f"--- {root} ---")
            for r, _, fs in os.walk(root):
                for f in fs:
                    if f.endswith(".csv"):
                        print("   ", os.path.join(r, f))
    raise FileNotFoundError("Shabd 34k CSV dataset not found.")


# ---------------------------------------------------------------------------
# 1. Load Word2Vec Hindi Model and Vocab from Hugging Face
# ---------------------------------------------------------------------------
REPO_ID = "AbhishekBiswas12/word2vec-hindi"

local_bin = _find_file(["Word2Vec_hindi.bin", "*word2vec*.bin"])
if local_bin and local_bin.exists():
    print(f"Found local model binary: {local_bin}")
    model_bin_path = str(local_bin)
else:
    print(f"Downloading Word2Vec_hindi.bin from Hugging Face ({REPO_ID})...")
    model_bin_path = hf_hub_download(repo_id=REPO_ID, filename="Word2Vec_hindi.bin")

local_vocab = _find_file(["vocab.txt", "*vocab*.txt"])
if local_vocab and local_vocab.exists():
    print(f"Found local vocabulary file: {local_vocab}")
    vocab_path = str(local_vocab)
else:
    print(f"Downloading vocab.txt from Hugging Face ({REPO_ID})...")
    vocab_path = hf_hub_download(repo_id=REPO_ID, filename="vocab.txt")

print(f"Loading Word2Vec model weights onto {DEVICE}...")
model = torch.load(model_bin_path, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()

print("Loading vocabulary index map...")
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
print(f"Loaded {len(vocab_to_idx)} vocabulary words.")


# ---------------------------------------------------------------------------
# 2. Load Shabd 34k Dataset and Filter Words
# ---------------------------------------------------------------------------
SHABD_PATH = _find_shabd_csv()
print(f"Using Shabd CSV: {SHABD_PATH}")
print(f"Output directory: {OUT_DIR}")

print("Loading Shabd dataset...")
shabd_data = pd.read_csv(SHABD_PATH, encoding="utf-8-sig", low_memory=False)
shabd_data.columns = shabd_data.columns.str.strip()

if "Frequency" in shabd_data.columns:
    shabd_data = shabd_data.sort_values(by="Frequency", ascending=False)

raw_words = shabd_data["Word"].dropna().astype(str).str.strip().unique().tolist()
print(f"Total unique words in Shabd 34k: {len(raw_words)}")

selected_words = []
selected_indices = []

for w in raw_words:
    if w in vocab_to_idx:
        selected_words.append(w)
        selected_indices.append(vocab_to_idx[w])
        if NUM_WORDS is not None and len(selected_words) >= NUM_WORDS:
            break

print(f"Selected {len(selected_words)} words matching Word2Vec vocab (NUM_WORDS cap: {NUM_WORDS}).")


# ---------------------------------------------------------------------------
# 3. GPU-Accelerated Extraction & L2 Normalization
# ---------------------------------------------------------------------------
print("Extracting and normalizing embeddings...")
words = selected_words

if DEVICE.type == "cuda":
    print("Running GPU tensor indexing and L2 normalization via PyTorch CUDA...")
    indices_t = torch.tensor(selected_indices, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        raw_t = model.input_embedding_layer.weight[indices_t]
        norm_t = torch.nn.functional.normalize(raw_t, p=2, dim=1)
        X_normalized = norm_t.cpu().numpy()
elif _HAS_CUPY:
    print("Running GPU L2 normalization via CuPy...")
    embeddings_matrix = model.input_embedding_layer.weight.detach().cpu().numpy()
    X = embeddings_matrix[selected_indices].astype(np.float32)
    X_gpu = cp.asarray(X)
    norms = cp.linalg.norm(X_gpu, axis=1, keepdims=True)
    norms = cp.where(norms == 0, cp.float32(1.0), norms)
    X_normalized = cp.asnumpy(X_gpu / norms)
else:
    print("Running CPU L2 normalization (sklearn)...")
    embeddings_matrix = model.input_embedding_layer.weight.detach().cpu().numpy()
    X = embeddings_matrix[selected_indices].astype(np.float32)
    X_normalized = sk_normalize(X, norm="l2")

print(f"Normalized embedding matrix: {X_normalized.shape}")

# Save normalized embeddings and word list
np.save(OUT_DIR / "X_word2vec_normalized.npy", X_normalized)
np.save(OUT_DIR / "X_normalized.npy", X_normalized)

with open(OUT_DIR / "words_word2vec.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))
with open(OUT_DIR / "words.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))

print(f"Saved normalized embeddings and word lists to {OUT_DIR}")


# ---------------------------------------------------------------------------
# 4. Clustering (GPU accelerated via cuML if available, else fast sklearn)
# ---------------------------------------------------------------------------
if _HAS_CUML:
    print("\n>>> RAPIDS cuML detected — executing KMeans, MiniBatch, GMM directly on GPU!")
    algorithms = {
        "MiniBatchKMeans": cuMiniBatchKMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
        ),
        "Voronoi_Partition": cuKMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
        ),
        "Gaussian_Mixture": cuGaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="diag",
            random_state=RANDOM_STATE,
        ),
        "Agglomerative_Cosine": AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="cosine", linkage="average"
        ),
    }
else:
    print("\n>>> cuML not detected — executing optimized scikit-learn on CPU.")
    algorithms = {
        "MiniBatchKMeans": MiniBatchKMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto"
        ),
        "Voronoi_Partition": KMeans(
            n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto"
        ),
        "Gaussian_Mixture": GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="diag",
            random_state=RANDOM_STATE,
        ),
        "Agglomerative_Cosine": AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="cosine", linkage="average"
        ),
    }

# ---------------------------------------------------------------------------
# 5. Fit & Save Cluster Assignments
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
    print(f"Saved results to {out_path.name}")

print("\nClustering complete for all 4 algorithms.")
