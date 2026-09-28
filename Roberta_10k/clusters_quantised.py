"""
GPU-accelerated clustering with quantized RoBERTa-Hindi.
- Uses BitsAndBytesConfig for 8-bit quantization (requires CUDA + bitsandbytes).
- Falls back to FP16 if quantization is not available.
- Falls back to full precision on CPU if no GPU.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sklearn.cluster import (
    MiniBatchKMeans, AgglomerativeClustering, SpectralClustering, KMeans
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent   # .../Hindi/Roberta
HINDI_DIR = SCRIPT_DIR.parent                  # .../Hindi

SHABD_PATH = (
    HINDI_DIR / "Shabd" / "fygme-osfstorage-archive"
    / "Shabd Psycholinguistic database_34k Words List.csv"
)

# ---------------------------------------------------------------------------
# 1. Load the Shabd dataset (top 10,000 words)
# ---------------------------------------------------------------------------
print("Loading Shabd dataset...")
shabd_data = pd.read_csv(SHABD_PATH, encoding="utf-8-sig", low_memory=False)
shabd_data.columns = shabd_data.columns.str.strip()

shabd_data = shabd_data.sort_values(by="Frequency", ascending=False).head(10000)
unique_words = shabd_data["Word"].dropna().astype(str).str.strip().unique().tolist()
print(f"Total unique words selected for clustering: {len(unique_words)}")

# ---------------------------------------------------------------------------
# 2. Load RoBERTa-Hindi model (quantized if possible)
# ---------------------------------------------------------------------------
MODEL_NAME = "flax-community/roberta-hindi"
print(f"Loading tokenizer and model: {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Decide how to load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = None

if device.type == "cuda":
    try:
        # Try 8-bit quantization
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto"
        )
        print("Loaded model in 8-bit quantized mode.")
    except Exception as e:
        print(f"8-bit quantization failed ({e}). Trying FP16...")
        try:
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16
            ).to(device)
            print("Loaded model in FP16 mode.")
        except Exception as e2:
            print(f"FP16 failed ({e2}). Loading full precision on CPU.")
            model = AutoModel.from_pretrained(MODEL_NAME)
            device = torch.device("cpu")
            model.to(device)
else:
    # No GPU – load full precision on CPU
    print("No CUDA GPU detected. Loading full precision on CPU.")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)

model.eval()
print("Model ready.\n")

# ---------------------------------------------------------------------------
# 3. Extract embeddings (batched)
# ---------------------------------------------------------------------------
def extract_embeddings(words, batch_size=64):
    """Mean-pooled RoBERTa embeddings for a list of words."""
    all_emb = []
    n_batches = (len(words) + batch_size - 1) // batch_size

    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16
        )
        # Move inputs to the model's device
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling with attention mask
        last_hidden = outputs.last_hidden_state          # (B, L, H)
        mask = inputs["attention_mask"].unsqueeze(-1)    # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)         # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        emb = (summed / counts).cpu().numpy()
        all_emb.append(emb)

        if (i // batch_size + 1) % 20 == 0:
            print(f"  Batch {i//batch_size + 1}/{n_batches}")

    return np.vstack(all_emb).astype(np.float32)

print("Extracting embeddings...")
X = extract_embeddings(unique_words)
words = unique_words
print(f"Embedding matrix shape: {X.shape}")

# L2 normalize
X_normalized = normalize(X, norm='l2')
print(f"Normalized shape: {X_normalized.shape}")

# Save for later use
np.save(SCRIPT_DIR / "X_roberta_normalized.npy", X_normalized)
with open(SCRIPT_DIR / "words_roberta.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(words))

# ---------------------------------------------------------------------------
# 4. Clustering (k=200)
# ---------------------------------------------------------------------------
N_CLUSTERS = 200

algorithms = {
    "MiniBatchKMeans": MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10),
    "Agglomerative_Cosine": AgglomerativeClustering(
        n_clusters=N_CLUSTERS, metric='cosine', linkage='average'
    ),
    "Spectral": SpectralClustering(
        n_clusters=N_CLUSTERS,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans',
        random_state=42
    ),
    "Voronoi_Partition": KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10),
    "Gaussian_Mixture": GaussianMixture(
        n_components=N_CLUSTERS, covariance_type='diag', random_state=42
    )
}

for name, algo in algorithms.items():
    print(f"\nRunning {name}...")
    labels = algo.fit_predict(X_normalized)
    results_df = pd.DataFrame({"word": words, "cluster": labels})
    out_path = SCRIPT_DIR / f"roberta_{name}_clusters.csv"
    results_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved {name} results to {out_path.name}")

print("\nClustering complete for all 5 algorithms.")