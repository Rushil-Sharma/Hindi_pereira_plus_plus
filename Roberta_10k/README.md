# Roberta_10k: Hindi RoBERTa Word Clustering & Evaluation Pipeline

This directory contains the pipeline for extracting, clustering, and evaluating 768-dimensional contextualized embeddings from the **RoBERTa-Hindi** transformer model for the top 10,000 most frequent Hindi words selected from the **Shabd Psycholinguistic Database**.

---

## 1. Model Overview & Training Characteristics

### What is the Model?
- **Model Identifier**: [`flax-community/roberta-hindi`](https://huggingface.co/flax-community/roberta-hindi) (available on the Hugging Face Hub).
- **Architecture**: **RoBERTa-base** (*Robustly Optimized BERT Approach*, Liu et al., 2019).
  - 12 Transformer encoder layers
  - 768 hidden dimensions ($d_{\text{model}} = 768$)
  - 12 multi-head self-attention heads
  - 3,072 feed-forward intermediate dimensionality
  - Approximately 125 million parameters
- **Tokenizer**: Byte-Pair Encoding (BPE) subword tokenizer trained with a vocabulary size of 50,265 tokens specifically optimized for Hindi Devanagari script.

### Language Scope: Monolingual or Multilingual?
- **Language**: **Monolingual Hindi**.
- Unlike massive multilingual models (e.g., mBERT or XLM-RoBERTa), this model was trained **exclusively on Hindi text**. This eliminates cross-lingual interference and curse-of-multilinguality parameter dilution, allowing the entire 125M parameter capacity to model Devanagari syntax, morphology, and semantics.
- **Training Corpora**:
  - OSCAR Hindi corpus (filtered web crawl)
  - Hindi Wikipedia dumps
  - CC100 Hindi subset

### How was it Trained?
- **Pre-training Objective**: Masked Language Modeling (MLM).
  - Randomly masks 15% of tokens in a sequence; the model is trained to predict the original masked tokens conditioned on bidirectional contextual representations.
  - Utilizes dynamic masking (mask pattern generated on the fly per epoch).
  - Omits Next Sentence Prediction (NSP), following RoBERTa design principles.
- **Optimization**: Trained on Google Cloud TPUs during the Hugging Face Flax/JAX Community Sprint using AdamW optimizer with weight decay and linear learning rate warmup.

---

## 2. Underlying Assumptions & Theoretical Foundations

1. **Contextual Representations for Isolated Lexical Items**:
   - Transformers are inherently designed for sequence-level contextual embeddings. When fed isolated words ($L \le 16$), the model maps the word into its "default" contextual subspace.
2. **Subword Aggregation (Mean Pooling)**:
   - A single Hindi word may decompose into one or more BPE subword tokens (e.g., roots + vowel diacritics / inflectional suffixes).
   - To obtain a single fixed vector $\mathbf{e}_w \in \mathbb{R}^{768}$ per word, the token hidden states from the last transformer layer are averaged using an attention-mask weighted mean:
     $$\mathbf{e}_w = \frac{\sum_{i=1}^L m_i \mathbf{h}_i}{\sum_{i=1}^L m_i}$$
     where $m_i \in \{0, 1\}$ is the attention mask value ($m_i = 0$ for padding tokens).
3. **Hyperspherical $L_2$ Normalization**:
   - Transformer output states often suffer from anisotropy (the representation "cone" problem where cosine similarities are artificially high).
   - Projecting all vectors onto the unit sphere $S^{767}$ via $L_2$ normalization ($\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$) standardizes variance and aligns Euclidean distance with Cosine similarity:
     $$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = 2 - 2 \cos(\hat{\mathbf{u}}, \hat{\mathbf{v}})$$
4. **Vocabulary Selection**:
   - Uses the top 10,000 most frequent words from the [Shabd Psycholinguistic Database](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Shabd/fygme-osfstorage-archive/Shabd%20Psycholinguistic%20database_34k%20Words%20List.csv) to represent high-frequency core Hindi vocabulary.

---

## 3. Pipeline Architecture & Code Implementation

```
Roberta_10k/
├── clusters.py               # Step 1: Batched GPU inference, L2 norm & k=200 clustering
├── clusters_quantised.py     # Alternative: 8-bit quantized inference (BitsAndBytes)
├── evaluate_clusters.py      # Step 2: Multi-k evaluation across 4 cluster metrics
└── plot_metrics.py           # Step 3: Visualization of metric curves across k
```

### 1. Batched GPU Extraction & Clustering ([`clusters.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/clusters.py))
- **Batched Transformer Inference**: Words are tokenized in batches of size 64 with dynamic padding (`max_length=16`), transferred to CUDA, and passed through `AutoModel` under `torch.no_grad()`.
- **Mean Pooling**: Computes mask-aware token averaging to yield matrix $\mathbf{X} \in \mathbb{R}^{10000 \times 768}$.
- **$L_2$ Normalization**: Unit-normalizes embeddings using CuPy (GPU) or scikit-learn (CPU).
- **5 Clustering Algorithms ($k=200$)**:
  1. **MiniBatchKMeans**: GPU-accelerated via cuML or scikit-learn mini-batch optimization.
  2. **Voronoi_Partition (Standard K-Means)**: Partitions the 768-d space into Voronoi cells.
  3. **Gaussian_Mixture (GMM)**: Expectation-Maximization soft clustering with diagonal covariance.
  4. **Agglomerative_Cosine**: Hierarchical agglomeration with cosine distance and average linkage.
  5. **Spectral Clustering**: 10-nearest-neighbors affinity graph with normalized Laplacian.

### 2. Quantized Alternative ([`clusters_quantised.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/clusters_quantised.py))
- For low-VRAM environments, uses `BitsAndBytesConfig(load_in_8bit=True)` to quantize model weights to 8-bit integers, halving GPU memory consumption while preserving embedding quality.

### 3. Quantitative Evaluation ([`evaluate_clusters.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/evaluate_clusters.py))
Systematically sweeps $k \in \{50, 100, 150, 200, 250, 300\}$ across all 5 algorithms:
- **WCSS / Inertia**: Within-cluster sum of squared Euclidean deviations from cluster centroids.
- **Silhouette Coefficient**: Degree of intra-cluster cohesion vs. inter-cluster separation ($[-1, 1]$).
- **Davies-Bouldin Index**: Ratio of within-cluster dispersion to separation between cluster centers.
- **Calinski-Harabasz Index**: Between-cluster variance to within-cluster variance ratio.
- Outputs [`clustering_metrics_roberta.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/clustering_metrics_roberta.csv).

### 4. Metric Visualization ([`plot_metrics.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/plot_metrics.py))
Generates a $2 \times 2$ grid plot saved to [`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/clustering_comparison.png).

---

## 4. Output Files Description

| File Name | Format | Description |
| :--- | :--- | :--- |
| [`words_roberta.txt`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/words_roberta.txt) | Plain text (UTF-8) | Newline-delimited list of 10,000 words extracted from the Shabd database. |
| [`X_roberta_normalized.npy`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/X_roberta_normalized.npy) | NumPy binary | Normalized embedding array of shape `(10000, 768)` (`float32`). |
| [`roberta_MiniBatchKMeans_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/roberta_MiniBatchKMeans_clusters.csv) | CSV (`utf-8-sig`) | Cluster labels for MiniBatchKMeans ($k=200$). Columns: `word`, `cluster`. |
| [`roberta_Voronoi_Partition_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/roberta_Voronoi_Partition_clusters.csv) | CSV (`utf-8-sig`) | Cluster labels for Voronoi Partition / K-Means ($k=200$). |
| [`roberta_Gaussian_Mixture_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/roberta_Gaussian_Mixture_clusters.csv) | CSV (`utf-8-sig`) | Cluster labels for Gaussian Mixture Model ($k=200$). |
| [`roberta_Agglomerative_Cosine_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/roberta_Agglomerative_Cosine_clusters.csv) | CSV (`utf-8-sig`) | Cluster labels for Agglomerative Cosine ($k=200$). |
| [`roberta_Spectral_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/roberta_Spectral_clusters.csv) | CSV (`utf-8-sig`) | Cluster labels for Spectral Clustering ($k=200$). |
| [`clustering_metrics_roberta.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/clustering_metrics_roberta.csv) | CSV | Quantitative metrics (WCSS, Silhouette, DB, CH) across $k \in [50, 300]$. |
| [`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/clustering_comparison.png) | High-res PNG | 4-subplot comparison curves for all clustering algorithms. |

---

## 5. Interpretation of Evaluation Plots ([`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Roberta_10k/clustering_comparison.png))

1. **Top-Left: Elbow Curve (WCSS vs. $k$)** (Lower is better):
   - WCSS in RoBERTa space decreases from $\approx 3537 - 4295$ at $k=50$ down to $\approx 2865 - 3391$ at $k=300$.
   - **Crucial Comparison**: RoBERTa WCSS values are roughly **half** the magnitude of FastText WCSS values (~7000-8000), even though RoBERTa embeddings have higher dimensionality (768 vs 300). This demonstrates that RoBERTa embeddings form much tighter, denser clusters in latent space.
   - **Voronoi Partition (K-Means)** and **Gaussian Mixture** provide the sharpest variance minimization.
2. **Top-Right: Silhouette Score vs. $k$** (Higher is better):
   - **Voronoi Partition** achieves the highest Silhouette score of all models, rising from $0.0396$ ($k=50$) to **$0.0534$** ($k=300$).
   - **Gaussian Mixture** follows closely behind ($0.0455$ at $k=300$).
   - Unlike FastText where MiniBatchKMeans collapsed below zero, MiniBatchKMeans here maintains positive Silhouette scores throughout ($0.025 - 0.043$).
   - Agglomerative Cosine and Spectral clustering start low at $k=50$ but climb steadily to positive values as granularity increases.
3. **Bottom-Left: Davies-Bouldin Index vs. $k$** (Lower is better):
   - **Agglomerative Cosine** demonstrates outstanding separation in cosine space, achieving an exceptionally low DB index of **$2.26$** at $k=300$ (significantly superior to Voronoi's $2.84$).
   - All models show continuous improvement in DB index as $k$ increases from 50 to 300.
4. **Bottom-Right: Calinski-Harabasz Index vs. $k$** (Higher is better):
   - CH scores start high at $k=50$ ($\approx 80.08$ for Voronoi Partition) and taper down to $\approx 23.39$ at $k=300$.
   - Voronoi Partition and GMM consistently dominate across the entire spectrum of $k$.

---

## 6. How to Run the Pipeline

```bash
# 1. Extract RoBERTa embeddings and cluster at k=200
python clusters.py

# Alternatively, run 8-bit quantized extraction:
# python clusters_quantised.py

# 2. Evaluate clustering metrics across k in [50, 100, 150, 200, 250, 300]
python evaluate_clusters.py

# 3. Plot the diagnostic comparison figure
python plot_metrics.py
```
