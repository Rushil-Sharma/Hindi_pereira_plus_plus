"""
GPU-accelerated RoBERTa-Hindi clustering pipeline.

What runs on GPU:
  - RoBERTa embedding extraction (batched inference) — HUGE win
  - L2 normalization (CuPy)
  - KMeans, MiniBatchKMeans, GaussianMixture (RAPIDS cuML)

What stays on CPU:
  - Agglomerative, Spectral clustering (no mature GPU impl.)

Falls back gracefully if cuML / CuPy are not installed.
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

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
SCRIPT_DIR = Path(__file__).resolve().parent
HINDI_DIR = SCRIPT_DIR.parent

SHABD_PATH = (
    HINDI_DIR / "Shabd" / "fygme-osfstorage-archive"
    / "Shabd Psycholinguistic database_34k Words List.csv"
)

RANDOM_STATE = 42
N_CLUSTERS = 200
BATCH_SIZE = 64  # tune based on GPU VRAM

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
# 2. Load RoBERTa-Hindi model and tokenizer
# ---------------------------------------------------------------------------
MODEL_NAME = "flax-community/roberta-hindi"

print(f"Loading tokenizer and model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}.")

# ---------------------------------------------------------------------------
# 3. Batched GPU embedding extraction  ← KEY OPTIMIZATION
# ---------------------------------------------------------------------------
def extract_embeddings(words, batch_size=BATCH_SIZE):
    """
    Extract mean-pooled RoBERTa embeddings for a list of words in batches.
    Returns a NumPy array of shape (len(words), hidden_dim).
    """
    all_embeddings = []
    n_batches = (len(words) + batch_size - 1) // batch_size

    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16,  # single words rarely exceed this
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling with attention mask (excludes padding)
        last_hidden = outputs.last_hidden_state          # (B, L, H)
        mask = inputs["attention_mask"].unsqueeze(-1)    # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)         # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        embeddings = (summed / counts).cpu().numpy()

        all_embeddings.append(embeddings)

        if (i // batch_size + 1) % 20 == 0:
            print(f"  Embedded batch {i//batch_size + 1}/{n_batches}")

    return np.vstack(all_embeddings).astype(np.float32)


print("Extracting RoBERTa embeddings (batched)...")
X = extract_embeddings(unique_words)   # (10000, 768)
words = unique_words
print(f"Raw embedding matrix: {X.shape}")

# ---------------------------------------------------------------------------
# 4. L2 normalization — GPU (CuPy) if available, else CPU
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

np.save(SCRIPT_DIR / "X_roberta_normalized.npy", X_normalized)
with open(SCRIPT_DIR / "words_roberta.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))

# ---------------------------------------------------------------------------
# 5. Clustering
# ---------------------------------------------------------------------------
if _HAS_CUML:
    print("\n>>> RAPIDS cuML detected — KMeans / MiniBatchKMeans / GMM on GPU.")
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
        # No GPU implementation — keep on CPU
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
    out_path = SCRIPT_DIR / f"roberta_{name}_clusters.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {name} results to {out_path.name}")

print("\nClustering complete for all 5 algorithms.")