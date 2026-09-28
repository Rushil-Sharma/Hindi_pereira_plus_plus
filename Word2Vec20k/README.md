# Word2Vec20k: Hindi Word2Vec Word Clustering & Evaluation Pipeline

This directory contains the pipeline for extracting, clustering, and evaluating **300-dimensional Word2Vec embeddings** for the top 20,000 most frequent Hindi words found in the **Shabd Psycholinguistic Database** and present in the Word2Vec vocabulary.

---

## 1. Model Overview & Training Characteristics

### What is the Model?
- **Model Source**: [`AbhishekBiswas12/word2vec-hindi`](https://huggingface.co/AbhishekBiswas12/word2vec-hindi) on Hugging Face Hub.
- **Artifact**: PyTorch `.bin` model file (`Word2Vec_hindi.bin`) plus a companion `vocab.txt` vocabulary index file.
- **Architecture**: A custom PyTorch `nn.Module` implementing the **Word2Vec** Skip-gram architecture. The core lookup table is exposed via the `model.input_embedding_layer.weight` tensor of shape `(vocab_size, 300)`.
- **Embedding Dimension**: $d = 300$.

### Language Scope: Monolingual or Multilingual?
- **Language**: **Monolingual Hindi**.
- The model is trained exclusively on Hindi-language corpora encoded in the Devanagari script. It captures Hindi-specific distributional semantics, morphological patterns, and syntactic co-occurrence statistics.

### How was it Trained?
- **Objective**: **Word2Vec Skip-gram with Negative Sampling (SGNS)**.
  - Given a center word $w_t$, the model is trained to maximize the log-probability of its surrounding context words within a sliding window and minimize the probability assigned to randomly sampled (negative) words:
    $$\mathcal{L} = \sum_{t=1}^T \sum_{-c \le j \le c,\, j \neq 0} \left[ \log \sigma(\mathbf{v}_{w_{t+j}}^\top \mathbf{v}_{w_t}) + \sum_{k=1}^K \mathbb{E}_{w_k \sim P_n(w)} \log \sigma(-\mathbf{v}_{w_k}^\top \mathbf{v}_{w_t}) \right]$$
  - Final word embeddings are the **input embedding vectors** from the lookup table (not the output/context vectors).

### Key Difference from FastText
| Property | Word2Vec (this folder) | FastText (IndicFT_10k) |
| :--- | :--- | :--- |
| Subword Information | ❌ None — whole-word tokens only | ✅ Character $n$-grams ($n \in [3,6]$) |
| OOV Handling | ❌ Cannot embed unseen words | ✅ Composed from subword $n$-grams |
| Morphological Sharing | ❌ Inflected forms have independent vectors | ✅ Shares parameters across morphological variants |
| Vocabulary Coverage | Limited to training vocabulary (~28k tokens) | Unbounded (any word can be embedded) |

Because Word2Vec has no subword mechanism, only the **intersection** of the Shabd wordlist and the model's vocabulary is embedded. The script filters Shabd words to those present in `vocab.txt`, yielding up to 20,000 matched words.

---

## 2. Underlying Assumptions & Theoretical Foundations

1. **Distributional Hypothesis**:
   - Words appearing in similar sentential contexts are assumed to have similar meanings (Harris, 1954). The Skip-gram objective operationalizes this by training embeddings to predict surrounding context from a center word.
2. **Whole-Word Tokenization**:
   - Each unique word surface form is treated as an atomic unit. Inflectional variants such as मेरा/मेरे/मेरी (my — masculine/oblique/feminine) receive completely independent embeddings, unlike FastText which shares subword parameters across them.
3. **Static Representation**:
   - Each word type receives exactly one fixed vector regardless of context. Polysemy and homonymy are collapsed into a single geometric locus, unlike RoBERTa whose 12-layer transformer produces dynamically context-sensitive representations.
4. **Vocabulary Intersection Constraint**:
   - **Critical assumption**: A word from the Shabd list is skipped entirely if it does not appear in the Word2Vec vocabulary (`vocab.txt`). The pipeline explicitly reports how many Shabd words were matched, with the cap set at `NUM_WORDS = 20000`.
5. **$L_2$ Normalization & Spherical Equivalence**:
   - Raw embedding vectors are normalized to unit length on the hypersphere $S^{299}$:
     $$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$
   - On unit-norm vectors, squared Euclidean distance equals twice the Cosine dissimilarity:
     $$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = 2 - 2\cos(\hat{\mathbf{u}}, \hat{\mathbf{v}})$$
     This makes centroid-based algorithms (K-Means, GMM) effectively optimize angular/semantic proximity.
6. **Silhouette Subsampling**:
   - For 20,000 words, computing the full $O(N^2)$ pairwise Silhouette matrix is computationally prohibitive. The evaluation script uses a random sample of 10,000 points to approximate the Silhouette score, yielding virtually identical statistical conclusions in a fraction of the time.

---

## 3. Pipeline Architecture & Code Implementation

```
Word2Vec20k/
├── clusters.py               # Step 1: Model loading, embedding extraction, normalization & k=200 clustering
├── evaluate_clusters.py      # Step 2: Multi-k evaluation across 4 cluster quality metrics
└── plot_metrics.py           # Step 3: Visualization of metric curves across k
```

### 1. Embedding Extraction & Clustering ([`clusters.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clusters.py))

**Data Ingestion**:
- Downloads `Word2Vec_hindi.bin` and `vocab.txt` from Hugging Face Hub (or uses local copies if found).
- Loads the PyTorch model using `torch.load(..., weights_only=False)`, mapping it to the appropriate device (CUDA or CPU).
- Reads `vocab.txt` to build a `word → index` dictionary mapping each Hindi word to its row in the embedding weight matrix.
- Reads the Shabd 34k CSV, sorts by `"Frequency"` descending, and iterates through words in frequency order, selecting only those present in the Word2Vec vocabulary up to `NUM_WORDS = 20000`.

**Embedding Extraction (GPU-accelerated)**:
- Looks up selected word indices in the embedding weight tensor `model.input_embedding_layer.weight`:
  - **CUDA path**: Index selection performed directly on GPU using `torch.Tensor.index_select`, then L2-normalized via `torch.nn.functional.normalize`, yielding $\mathbf{X}_\text{norm} \in \mathbb{R}^{N \times 300}$.
  - **CuPy path**: Weights transferred to CPU, filtered, then normalized on GPU via CuPy array operations.
  - **CPU fallback**: Pure NumPy + scikit-learn normalization.

**Clustering Algorithms ($k = 200$, 4 algorithms — Spectral is excluded for runtime)**:
1. **MiniBatchKMeans**: Stochastic mini-batch centroid updates; best speed on large datasets.
2. **Voronoi_Partition (Standard K-Means)**: Full batch centroid minimization partitioning the 300-d space.
3. **Gaussian_Mixture (GMM)**: EM-based soft probabilistic clustering with diagonal covariance.
4. **Agglomerative_Cosine**: Hierarchical bottom-up agglomeration using cosine distance and average linkage.

> [!NOTE]
> **Spectral Clustering is omitted** in Word2Vec20k (unlike the other two pipelines) because constructing the $k$-NN affinity graph for 20,000 points is prohibitively slow and memory-intensive. The 5-algorithm suite (IndicFT, RoBERTa) becomes a **4-algorithm suite** here.

### 2. Multi-$k$ Evaluation ([`evaluate_clusters.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/evaluate_clusters.py))
Sweeps $k \in \{50, 100, 150, 200, 250, 300\}$ across 4 algorithms, computing:
- **WCSS (Inertia)**:
  $$\text{WCSS} = \sum_{j=1}^k \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|_2^2$$
  For models with `inertia_` attribute (KMeans, MiniBatchKMeans), the attribute is read directly. For Agglomerative and GMM, WCSS is computed via a GPU-vectorized loop over cluster IDs using PyTorch CUDA tensors when available.
- **Silhouette Coefficient** (sampled at 10k for efficiency):
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
- **Davies-Bouldin Index**:
  $$\text{DB} = \frac{1}{k} \sum_{i=1}^k \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)}$$
- **Calinski-Harabasz Index (Variance Ratio)**:
  $$\text{CH} = \frac{\text{Tr}(\mathbf{B}_k)/(k-1)}{\text{Tr}(\mathbf{W}_k)/(N-k)}$$
- Results saved to both [`clustering_metrics_word2vec.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_metrics_word2vec.csv) and [`clustering_metrics.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_metrics.csv).

### 3. Metric Visualization ([`plot_metrics.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/plot_metrics.py))
Reads the metrics CSV and generates two identical copies of a $2 \times 2$ faceted line plot:
- [`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_comparison.png) and [`clustering_comparison_word2vec.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_comparison_word2vec.png) (identical content, different filenames for compatibility with both Kaggle and local paths).

---

## 4. Output Files Description

| File Name | Format | Description |
| :--- | :--- | :--- |
| [`words.txt`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/words.txt) / [`words_word2vec.txt`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/words_word2vec.txt) | Plain text (UTF-8) | Newline-delimited list of words selected (up to 20,000). Both files are identical; two names are kept for compatibility across environments. |
| [`X_normalized.npy`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/X_normalized.npy) / [`X_word2vec_normalized.npy`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/X_word2vec_normalized.npy) | NumPy binary | L2-normalized embedding matrix of shape `(N, 300)` (`float32`), where `N ≤ 20000`. Saved under two filenames for cross-environment compatibility. |
| [`MiniBatchKMeans_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/MiniBatchKMeans_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for MiniBatchKMeans ($k=200$). Columns: `word`, `cluster`. |
| [`Voronoi_Partition_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/Voronoi_Partition_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for Voronoi Partition / Standard K-Means ($k=200$). |
| [`Gaussian_Mixture_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/Gaussian_Mixture_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for Gaussian Mixture Model ($k=200$). |
| [`Agglomerative_Cosine_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/Agglomerative_Cosine_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for Agglomerative Cosine clustering ($k=200$). |
| [`clustering_metrics.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_metrics.csv) / [`clustering_metrics_word2vec.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_metrics_word2vec.csv) | CSV | Evaluation metrics across $k \in \{50, 100, 150, 200, 250, 300\}$ recording WCSS, Silhouette, Davies-Bouldin, and Calinski-Harabasz for all 4 algorithms. Saved under two filenames. |
| [`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_comparison.png) / [`clustering_comparison_word2vec.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_comparison_word2vec.png) | High-res PNG (300 DPI) | 4-panel diagnostic figure comparing all 4 algorithms across metrics and $k$ values. |

---

## 5. Interpretation of Evaluation Plots ([`clustering_comparison_word2vec.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Word2Vec20k/clustering_comparison_word2vec.png))

> [!IMPORTANT]
> **Cross-model comparison note**: Word2Vec has the largest word count (up to 20k vs 10k for IndicFT and RoBERTa), the same embedding dimension as FastText (300), but the fewest algorithms (4, no Spectral). Metric magnitudes are not directly comparable across the three models without normalizing for vocabulary size and dimensionality.

1. **Top-Left: Elbow Curve (WCSS vs. $k$)** (Lower is better):
   - Word2Vec WCSS values are dramatically higher ($\approx 15,500 – 18,500$) than both FastText ($\approx 6,800 – 8,300$) and RoBERTa ($\approx 2,900 – 4,300$) across all $k$.
   - This is largely a consequence of **twice the number of points** (20k vs 10k) — WCSS naturally scales with $N$. The higher magnitude does **not** imply worse clustering quality in isolation.
   - **Agglomerative Cosine** consistently shows the highest WCSS of all 4 algorithms, reinforcing that its linkage criterion (minimizing average cosine distance, not Euclidean squared deviation) is misaligned with the WCSS objective.
   - WCSS decreases only very slightly from $k=50$ to $k=300$ for all algorithms (unlike the steeper drops seen in FastText and RoBERTa), suggesting Word2Vec embeddings are more diffuse and harder to partition into tight Euclidean clusters.

2. **Top-Right: Silhouette Score vs. $k$** (Higher is better):
   - **Crucially, all 4 algorithms produce negative or near-zero Silhouette scores across all $k$**. Scores range from approximately $-0.08$ (MiniBatchKMeans at $k=300$) to $+0.006$ (Agglomerative Cosine at $k=50$).
   - **Agglomerative Cosine** is the only algorithm achieving marginally positive Silhouette values at low $k$, and slides to negative values at high $k$.
   - **MiniBatchKMeans** deteriorates most severely, with Silhouette scores becoming increasingly negative as $k$ grows.
   - This is a strong diagnostic signal: the Word2Vec embedding space is far more isotropic and less cluster-structured than either the FastText or RoBERTa spaces in this dataset. There may be no clear natural cluster partitioning for this vocabulary at the scales of $k$ tested.

3. **Bottom-Left: Davies-Bouldin Index vs. $k$** (Lower is better):
   - **MiniBatchKMeans** exhibits the sharpest DB improvement as $k$ increases, dropping from $5.83$ at $k=50$ to $2.41$ at $k=300$.
   - **Voronoi Partition** and **Gaussian Mixture** have consistently higher DB indices ($\approx 5.5 – 5.9$) that barely improve with $k$, suggesting centroids remain close relative to cluster spread.
   - DB indices overall are much higher than RoBERTa's (where $\approx 2.2 – 3.6$) and similar to FastText's worst performers, again confirming lower cluster separability.

4. **Bottom-Right: Calinski-Harabasz Index vs. $k$** (Higher is better):
   - Despite negative Silhouette scores, CH scores are substantial: **Voronoi Partition** leads at $k=50$ ($\approx 79.0$), declining to $\approx 17.4$ at $k=300$.
   - **Gaussian Mixture** closely tracks Voronoi Partition across all $k$.
   - **Agglomerative Cosine** produces the lowest CH scores (as low as $\approx 6.6$ at $k=300$), consistent with its non-Euclidean optimization objective.
   - The CH score pattern broadly mirrors RoBERTa (which also starts high at $k=50$ and declines), but Word2Vec's CH values are very similar in magnitude despite having twice as many words — a relative indicator of lower between-cluster separability per point.

---

## 6. How to Run the Pipeline

```bash
# 1. Download model, extract embeddings, normalize, and cluster at k=200
python clusters.py

# 2. Evaluate clustering metrics across k in [50, 100, 150, 200, 250, 300]
python evaluate_clusters.py

# 3. Generate 4-panel diagnostic plot
python plot_metrics.py
```

> [!TIP]
> The scripts are **Kaggle-compatible**: if a `/kaggle/working` directory is detected, all outputs (`.npy`, `.csv`, `.png`) are automatically redirected there. Locally, everything is saved to the `Word2Vec20k/` directory.
