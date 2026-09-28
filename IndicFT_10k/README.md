# IndicFT_10k: Hindi FastText Word Clustering & Evaluation Pipeline

This directory contains the pipeline for generating, clustering, and evaluating 300-dimensional **FastText** embeddings for the top 10,000 most frequent Hindi words selected from the **Shabd Psycholinguistic Database**.

---

## 1. Model Overview & Training Characteristics

### What is the Model?
- **Model Architecture**: **FastText** (Bojanowski et al., 2017), an extension of the Word2Vec Skip-gram model incorporating subword information.
- **Source / Artifact**:
  - Primary: Local IndicNLP FastText binary ([`indicnlp.ft.hi.300.bin`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/indicnlp.ft.hi.300.bin)) developed by the IndicNLP Suite (AI4Bharat / Kakwani et al., 2020).
  - Fallback: Hugging Face repository [`facebook/fasttext-hi-vectors`](https://huggingface.co/facebook/fasttext-hi-vectors) (`model.bin`).
- **Embedding Dimension**: $d = 300$.

### Language Scope: Monolingual or Multilingual?
- **Language**: **Monolingual Hindi**.
- The model is trained specifically on Hindi natural language corpora (Hindi Wikipedia dumps and Hindi Common Crawl corpus), capturing Devnagari script semantics and syntactic regularities.

### How was it Trained?
- **Objective**: Continuous Skip-gram with Negative Sampling (SGNS).
- **Subword Information (Character n-grams)**:
  - Unlike standard Word2Vec, which assigns a distinct vector to each unique token, FastText represents each word as a bag of character $n$-grams (typically $n \in [3, 6]$) bounded by special boundary symbols `<` and `>`, in addition to the word itself.
  - The vector representation of a word $w$ is the sum of the vector representations of its constituent character $n$-grams:
    $$\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$$
    where $\mathcal{G}_w \subset \{1, \dots, G\}$ is the set of $n$-grams appearing in $w$, and $\mathbf{z}_g$ is the learned embedding of $n$-gram $g$.
- **Relevance to Hindi**: Hindi is morphologically rich with complex inflectional morphology (e.g., verb conjugations, plural inflections, postpositional affixes). Subword representations allow the model to share morphological features across words sharing common roots (dhātus) and affixes (pratyayas), and gracefully generalize to out-of-vocabulary (OOV) words.

---

## 2. Underlying Assumptions & Theoretical Foundations

1. **Distributional Hypothesis**:
   - Assumes words that occur in similar linguistic distributions and linguistic contexts share similar semantic meanings (Harris, 1954; Firth, 1957).
2. **Static Representation Assumption**:
   - Each word type has a single, fixed vector representation in $\mathbb{R}^{300}$, collapsing polysemy and homonymy into one geometric locus.
3. **Lexical Representation of the Lexicon**:
   - The top 10,000 words by frequency from the [Shabd Psycholinguistic Database](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Shabd/fygme-osfstorage-archive/Shabd%20Psycholinguistic%20database_34k%20Words%20List.csv) are assumed to form the core lexical repository of adult Hindi speakers, capturing high-frequency functional and conceptual vocabulary.
4. **$L_2$ Normalization & Spherical Geometry**:
   - Raw FastText vectors have varying lengths reflecting word frequency and subword composition.
   - All vectors are projectively normalized to unit length on the unit hypersphere $S^{299}$:
     $$\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$
   - **Crucial Mathematical Consequence**: On unit-norm vectors, squared Euclidean distance is strictly proportional to Cosine distance:
     $$\|\hat{\mathbf{u}} - \hat{\mathbf{v}}\|_2^2 = \|\hat{\mathbf{u}}\|_2^2 + \|\hat{\mathbf{v}}\|_2^2 - 2(\hat{\mathbf{u}} \cdot \hat{\mathbf{v}}) = 2 - 2 \cos(\hat{\mathbf{u}}, \hat{\mathbf{v}})$$
     This ensures standard Euclidean centroid-based clustering algorithms (like K-Means and GMM) effectively optimize spherical cosine similarity.

---

## 3. Pipeline Architecture & Code Implementation

The pipeline in this directory consists of four modular scripts:

```
IndicFT_10k/
├── clusters.py               # Step 1: Feature extraction, normalization & k=200 clustering
├── evaluate_clusters.py      # Step 2: Multi-k evaluation across 4 cluster quality metrics
├── plot_metrics.py           # Step 3: Visualization of metric curves across k
└── word_net.py               # Step 4: External semantic coherence evaluation via IndoWordNet
```

### 1. Feature Extraction & Clustering ([`clusters.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/clusters.py))
- **Data Ingestion**: Reads [`Shabd Psycholinguistic database_34k Words List.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/Shabd/fygme-osfstorage-archive/Shabd%20Psycholinguistic%20database_34k%20Words%20List.csv), strips whitespace, sorts words by `"Frequency"` in descending order, and extracts the top 10,000 unique words.
- **Embedding Lookup**: Queries FastText using a multi-threaded pool (`ThreadPoolExecutor`), outputting raw matrix $\mathbf{X} \in \mathbb{R}^{10000 \times 300}$.
- **$L_2$ Normalization**: Normalizes $\mathbf{X}$ using CuPy (GPU) or scikit-learn (CPU).
- **Clustering Algorithms ($k = 200$)**:
  1. **MiniBatchKMeans**: Centroid-based K-Means using stochastic mini-batches for fast convergence.
  2. **Voronoi_Partition (Standard K-Means)**: Partitions the embedding space into convex Voronoi polyhedra by minimizing within-cluster sum of squares.
  3. **Gaussian_Mixture (GMM)**: Soft probabilistic clustering fitting 200 Gaussian components with diagonal covariance matrices (`covariance_type="diag"`).
  4. **Agglomerative_Cosine**: Hierarchical bottom-up agglomerative clustering using cosine distance metric and average linkage.
  5. **Spectral Clustering**: Constructs a 10-nearest-neighbors affinity graph, computes the normalized graph Laplacian, projects data into the low-dimensional spectral embedding space, and clusters using K-Means.
- **Hardware Acceleration**: Automatic fallback between RAPIDS cuML (NVIDIA GPU) and scikit-learn (CPU).

### 2. Multi-$k$ Quantitative Evaluation ([`evaluate_clusters.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/evaluate_clusters.py))
Evaluates the 5 clustering algorithms across $k \in \{50, 100, 150, 200, 250, 300\}$:
- **WCSS (Within-Cluster Sum of Squares / Inertia)**:
  $$\text{WCSS} = \sum_{j=1}^k \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol{\mu}_j\|_2^2$$
- **Silhouette Coefficient**:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  where $a(i)$ is the mean intra-cluster distance, and $b(i)$ is the mean nearest-cluster distance. Range: $[-1, 1]$.
- **Davies-Bouldin Index**:
  $$R_{ij} = \frac{s_i + s_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)}, \quad \text{DB} = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} R_{ij}$$
  Measures cluster similarity as the ratio of within-cluster spread to between-cluster separation.
- **Calinski-Harabasz Score (Variance Ratio Criterion)**:
  $$\text{CH} = \frac{\text{Tr}(\mathbf{B}_k) / (k - 1)}{\text{Tr}(\mathbf{W}_k) / (N - k)}$$
  Ratio of between-cluster dispersion to within-cluster dispersion.

### 3. Metric Plotting ([`plot_metrics.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/plot_metrics.py))
Reads [`clustering_metrics.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/clustering_metrics.csv) and generates a $2 \times 2$ faceted comparison figure saved as [`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/clustering_comparison.png).

### 4. IndoWordNet Semantic Coherence ([`word_net.py`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/word_net.py))
- Uses [`pyiwn`](https://github.com/cfilt/pyiwn) (IndoWordNet) to compute ontology-based semantic coherence.
- Calculates pairwise path similarity and hypernym-based Jaccard similarity across all pairs of words inside each cluster.

---

## 4. Output Files Description

| File Name | Format | Description |
| :--- | :--- | :--- |
| [`words.txt`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/words.txt) | Plain text (UTF-8) | Newline-separated list of the 10,000 vocabulary words selected from Shabd. |
| [`X_normalized.npy`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/X_normalized.npy) | NumPy binary | L2-normalized embedding matrix of shape `(10000, 300)` (`float32`). |
| [`word_vector_dict.npy`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/word_vector_dict.npy) | Serialized dict | Mapping of each Hindi word string to its raw 300-d vector. |
| [`MiniBatchKMeans_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/MiniBatchKMeans_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for MiniBatchKMeans ($k=200$). Columns: `word`, `cluster`. |
| [`Voronoi_Partition_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/Voronoi_Partition_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for Voronoi Partition / K-Means ($k=200$). |
| [`Gaussian_Mixture_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/Gaussian_Mixture_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for Gaussian Mixture ($k=200$). |
| [`Agglomerative_Cosine_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/Agglomerative_Cosine_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for Agglomerative Cosine ($k=200$). |
| [`Spectral_clusters.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/Spectral_clusters.csv) | CSV (`utf-8-sig`) | Word-to-cluster assignments for Spectral Clustering ($k=200$). |
| [`clustering_metrics.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/clustering_metrics.csv) | CSV | Evaluation table across $k \in \{50, 100, 150, 200, 250, 300\}$ recording WCSS, Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics. |
| [`wordnet_coherence_results.csv`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/wordnet_coherence_results.csv) | CSV | Summary of mean and std IndoWordNet synset coherence per algorithm. |
| [`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/clustering_comparison.png) | High-res PNG | 4-panel diagnostic plot comparing clustering performance metrics across $k$. |

---

## 5. Interpretation of Evaluation Plots ([`clustering_comparison.png`](file:///c:/Users/Rushil/OneDrive/Desktop/ASSignments/sem3/CogSci/pereira_plus_plus/Hindi/IndicFT_10k/clustering_comparison.png))

The generated plot contains 4 subplots illustrating the structural behavior of the FastText embedding space:

1. **Top-Left: Elbow Method (WCSS vs. $k$)** (Lower is better):
   - WCSS decreases smoothly from $k=50$ ($\approx 7744 - 8293$) to $k=300$ ($\approx 6813 - 7261$).
   - **Voronoi Partition (K-Means)** and **Gaussian Mixture** achieve the lowest WCSS at all $k$, indicating tightest cluster compactness.
   - **Agglomerative Cosine** shows higher WCSS because its linkage criterion prioritizes tree-based average distance rather than minimizing squared centroid Euclidean variance.
2. **Top-Right: Mean Silhouette Score vs. $k$** (Higher is better):
   - **Voronoi Partition** and **Gaussian Mixture** produce consistently positive Silhouette scores increasing from $\approx 0.018$ at $k=50$ to $\approx 0.024$ at $k=300$.
   - **MiniBatchKMeans** degrades severely into negative Silhouette territory as $k$ increases (dropping to $-0.039$ at $k=300$), because mini-batch sampling introduces high variance in high-dimensional space with many centroids.
   - Agglomerative and Spectral clustering improve steadily with higher $k$, achieving positive scores above $k=150$.
3. **Bottom-Left: Davies-Bouldin Index vs. $k$** (Lower is better):
   - DB index monotonically declines for all algorithms as $k$ increases, indicating clusters become more separated relative to their radii.
   - **MiniBatchKMeans** exhibits the lowest DB index ($2.89$ at $k=300$), while Voronoi Partition stabilizes around $3.68$.
4. **Bottom-Right: Calinski-Harabasz Index vs. $k$** (Higher is better):
   - CH score is highest at $k=50$ ($\approx 30.6$ for Voronoi Partition) and decreases as $k$ increases.
   - This occurs because CH penalizes the increase in degrees of freedom ($k-1$) more heavily than the reduction in within-cluster dispersion. Voronoi Partition and GMM consistently dominate across all $k$.

---

## 6. How to Run the Pipeline

```bash
# 1. Generate embeddings and cluster at k=200
python clusters.py

# 2. Compute evaluation metrics across k in [50, 100, 150, 200, 250, 300]
python evaluate_clusters.py

# 3. Generate metric comparison plots
python plot_metrics.py

# 4. (Optional) Run IndoWordNet semantic coherence evaluation
python word_net.py
```
