"""
Minimalist White & Gold Presentation Generator (Complete 3-Model Benchmark)
Models: IndicFastText (10k) vs RoBERTa-Hindi (10k) vs Word2Vec-Hindi (20k) on Shabd
Style: Ultra-clean, modern, minimalist luxury (Matte White, Champagne Gold, Charcoal Gray, Sharp Rectangles)
Each model features: 2 Code Slides, 1 Dedicated Full-Page Picture Slide, 1 Analysis Slide
"""

from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
HINDI_DIR = SCRIPT_DIR.parent
OUTPUT_PPT = SCRIPT_DIR / "Hindi_Embedding_Clustering_Analysis.pptx"

INDICFT_PNG = HINDI_DIR / "IndicFT_10k" / "clustering_comparison.png"
ROBERTA_PNG = HINDI_DIR / "Roberta_10k" / "clustering_comparison.png"
WORD2VEC_PNG = HINDI_DIR / "Word2Vec" / "clustering_comparison.png"

# ---------------------------------------------------------------------------
# Presentation Setup (16:9 Widescreen)
# ---------------------------------------------------------------------------
prs = pptx.Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank_layout = prs.slide_layouts[6]

# ---------------------------------------------------------------------------
# Color Palette: Minimalist White & Champagne Gold
# ---------------------------------------------------------------------------
BG_WHITE = RGBColor(255, 255, 255)         # Crisp Matte White
CARD_WHITE = RGBColor(255, 255, 255)       # Card Fill
GOLD_CHAMPAGNE = RGBColor(212, 175, 55)    # #D4AF37 Rich Champagne Gold
GOLD_LINE = RGBColor(220, 195, 125)        # Subtle Gold for Dividers/Borders
TEXT_CHARCOAL = RGBColor(34, 34, 34)       # #222222 Deep Charcoal Gray
TEXT_MUTED = RGBColor(105, 105, 105)       # Soft Muted Gray
CARD_BORDER = RGBColor(228, 222, 210)      # Delicate sharp gold-tinted border
TABLE_ALT_ROW = RGBColor(250, 249, 246)    # Warm soft off-white

FONT_SERIF = "Georgia"
FONT_SANS = "Segoe UI"


def set_bg(slide):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = BG_WHITE


def add_gold_divider(slide, left, top, width):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = GOLD_CHAMPAGNE
    line.line.fill.background()
    return line


def add_minimalist_header(slide, category_text, title_text, subtitle_text=""):
    cat_box = slide.shapes.add_textbox(Inches(0.9), Inches(0.45), Inches(11.5), Inches(0.3))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.name = FONT_SANS
    p_c.font.size = Pt(9.5)
    p_c.font.bold = True
    p_c.font.color.rgb = GOLD_CHAMPAGNE

    title_box = slide.shapes.add_textbox(Inches(0.9), Inches(0.75), Inches(11.5), Inches(0.55))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    tf_t.margin_left = tf_t.margin_right = tf_t.margin_top = tf_t.margin_bottom = 0
    p_t = tf_t.paragraphs[0]
    p_t.text = title_text
    p_t.font.name = FONT_SERIF
    p_t.font.size = Pt(22)
    p_t.font.bold = True
    p_t.font.color.rgb = TEXT_CHARCOAL

    if subtitle_text:
        sub_box = slide.shapes.add_textbox(Inches(0.9), Inches(1.32), Inches(11.5), Inches(0.35))
        tf_s = sub_box.text_frame
        tf_s.word_wrap = True
        tf_s.margin_left = tf_s.margin_right = tf_s.margin_top = tf_s.margin_bottom = 0
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle_text
        p_s.font.name = FONT_SANS
        p_s.font.size = Pt(11)
        p_s.font.color.rgb = TEXT_MUTED

    add_gold_divider(slide, Inches(0.9), Inches(1.72), Inches(11.533))


def add_card(slide, left, top, width, height):
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_WHITE
    card.line.color.rgb = CARD_BORDER
    card.line.width = Pt(1)
    return card


# ===========================================================================
# SLIDE 1: TITLE SLIDE
# ===========================================================================
slide1 = prs.slides.add_slide(blank_layout)
set_bg(slide1)

add_gold_divider(slide1, Inches(5.66), Inches(1.4), Inches(2.0))

title_box = slide1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(10.933), Inches(3.2))
tf1 = title_box.text_frame
tf1.word_wrap = True

p_sub_top = tf1.paragraphs[0]
p_sub_top.text = "COGNITIVE SCIENCE & COMPUTATIONAL LINGUISTICS"
p_sub_top.font.name = FONT_SANS
p_sub_top.font.size = Pt(10.5)
p_sub_top.font.bold = True
p_sub_top.font.color.rgb = GOLD_CHAMPAGNE
p_sub_top.alignment = PP_ALIGN.CENTER
p_sub_top.space_after = Pt(16)

p_main = tf1.add_paragraph()
p_main.text = "Hindi Semantic Embedding Spaces"
p_main.font.name = FONT_SERIF
p_main.font.size = Pt(36)
p_main.font.bold = True
p_main.font.color.rgb = TEXT_CHARCOAL
p_main.alignment = PP_ALIGN.CENTER
p_main.space_after = Pt(12)

p_desc = tf1.add_paragraph()
p_desc.text = "A Comparative Clustering & Geometric Evaluation: IndicFastText, RoBERTa-Hindi, and Word2Vec on Shabd"
p_desc.font.name = FONT_SANS
p_desc.font.size = Pt(14)
p_desc.font.color.rgb = TEXT_MUTED
p_desc.alignment = PP_ALIGN.CENTER

stat_w = Inches(3.4)
stat_gap = Inches(0.4)
stat_left = Inches(1.166)
stat_top = Inches(5.3)

stats_data = [
    ("3 EMBEDDINGS", "IndicFastText · RoBERTa · Word2Vec"),
    ("SHABD DATASET", "10,000 to 20,000 Lexical Entries"),
    ("k = 50 → 300", "Multi-Granularity Geometric Benchmarks")
]

for idx, (val, label) in enumerate(stats_data):
    s_left = stat_left + idx * (stat_w + stat_gap)
    card_s = add_card(slide1, s_left, stat_top, stat_w, Inches(1.2))
    add_gold_divider(slide1, s_left + Inches(0.3), stat_top, stat_w - Inches(0.6))
    
    s_box = slide1.shapes.add_textbox(s_left, stat_top + Inches(0.2), stat_w, Inches(0.8))
    tf_s = s_box.text_frame
    tf_s.word_wrap = True
    
    p1 = tf_s.paragraphs[0]
    p1.text = val
    p1.font.name = FONT_SERIF
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = GOLD_CHAMPAGNE
    p1.alignment = PP_ALIGN.CENTER
    
    p2 = tf_s.add_paragraph()
    p2.text = label
    p2.font.name = FONT_SANS
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_MUTED
    p2.alignment = PP_ALIGN.CENTER


# ===========================================================================
# SLIDE 2: INDICFT_10K — CODE (1/2)
# ===========================================================================
slide2 = prs.slides.add_slide(blank_layout)
set_bg(slide2)
add_minimalist_header(
    slide2,
    "IndicFT_10k Pipeline (1/2)",
    "Lexical Ingestion & FastText Extraction",
    "Processing top 10,000 Hindi entries from Shabd and extracting 300D static vectors"
)

cw = Inches(5.6)
ch = Inches(5.0)

c1 = add_card(slide2, Inches(0.9), Inches(1.95), cw, ch)
add_gold_divider(slide2, Inches(1.2), Inches(1.95), cw - Inches(0.6))
b1 = slide2.shapes.add_textbox(Inches(1.2), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Dataset Ingestion & Selection"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s2_1 = [
    ("Source Corpus:", "Loaded the Shabd Psycholinguistic Database containing 33,676 entries."),
    ("Frequency Ranking:", "Sorted words by frequency to isolate the top 10,000 core lexical items."),
    ("Lexical Sanitization:", "Cleaned extraneous whitespace and removed missing/corrupt tokens."),
    ("Cognitive Focus:", "Targets high-frequency vocabulary reflecting human lexical accessibility.")
]

for title, body in bullets_s2_1:
    p = tf1.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide2, Inches(6.833), Inches(1.95), cw, ch)
add_gold_divider(slide2, Inches(7.133), Inches(1.95), cw - Inches(0.6))
b2 = slide2.shapes.add_textbox(Inches(7.133), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Vector Extraction & Scaling"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s2_2 = [
    ("Model Architecture:", "Employed 300D FastText vectors (indicnlp.ft.hi.300.bin) from AI4Bharat."),
    ("Hugging Face Fallback:", "Integrated auto-download fallback to facebook/fasttext-hi-vectors."),
    ("Parallel Lookup:", "Extracted vectors across CPU threads using ThreadPoolExecutor."),
    ("Subword Construction:", "FastText builds token vectors by summing character n-gram vectors."),
    ("L2 Normalization:", "Unit-norm projected (||x||_2 = 1) onto a 300D hypersphere using CuPy / sklearn.")
]

for title, body in bullets_s2_2:
    p = tf2.add_paragraph()
    p.space_before = Pt(11)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 3: INDICFT_10K — CODE (2/2)
# ===========================================================================
slide3 = prs.slides.add_slide(blank_layout)
set_bg(slide3)
add_minimalist_header(
    slide3,
    "IndicFT_10k Pipeline (2/2)",
    "Clustering Paradigms & Evaluation Suite",
    "Running 5 distinct geometric clustering algorithms across k = 50 to 300"
)

c1 = add_card(slide3, Inches(0.9), Inches(1.95), cw, ch)
add_gold_divider(slide3, Inches(1.2), Inches(1.95), cw - Inches(0.6))
b1 = slide3.shapes.add_textbox(Inches(1.2), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Five Clustering Paradigms"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

algos_s3 = [
    ("Voronoi Partition:", "Standard K-Means with 10 restarts, partitioning the 300D space."),
    ("MiniBatchKMeans:", "Stochastic batch updates enabling fast large-scale clustering."),
    ("Gaussian Mixture (GMM):", "Probabilistic soft clustering with diagonal covariance matrices."),
    ("Agglomerative Cosine:", "Hierarchical bottom-up tree based on average cosine distance."),
    ("Spectral Clustering:", "Graph-based clustering over a 10-nearest-neighbor affinity matrix.")
]

for title, body in algos_s3:
    p = tf1.add_paragraph()
    p.space_before = Pt(10)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide3, Inches(6.833), Inches(1.95), cw, ch)
add_gold_divider(slide3, Inches(7.133), Inches(1.95), cw - Inches(0.6))
b2 = slide3.shapes.add_textbox(Inches(7.133), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Multi-Metric Benchmark Suite"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

eval_s3 = [
    ("Systematic Sweep:", "Evaluated across k in [50, 100, 150, 200, 250, 300] (30 conditions)."),
    ("WCSS / Inertia:", "Measures intra-cluster tightness (lower is better)."),
    ("Silhouette Score:", "Measures inter-cluster separation vs. cohesion (higher is better)."),
    ("Davies-Bouldin Index:", "Ratio of cluster scatter to cluster distance (lower is better)."),
    ("Calinski-Harabasz:", "Ratio of between-cluster to within-cluster variance (higher is better).")
]

for title, body in eval_s3:
    p = tf2.add_paragraph()
    p.space_before = Pt(10)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 4: INDICFT_10K — DEDICATED FULL-PAGE PICTURE
# ===========================================================================
slide4 = prs.slides.add_slide(blank_layout)
set_bg(slide4)

if INDICFT_PNG.exists():
    img_h = Inches(7.0)
    img_w = Inches(7.0 * (3976 / 3216))
    img_left = (prs.slide_width - img_w) / 2
    img_top = (prs.slide_height - img_h) / 2
    slide4.shapes.add_picture(str(INDICFT_PNG), img_left, img_top, width=img_w, height=img_h)


# ===========================================================================
# SLIDE 5: INDICFT_10K — METHODOLOGICAL ANALYSIS
# ===========================================================================
slide5 = prs.slides.add_slide(blank_layout)
set_bg(slide5)
add_minimalist_header(
    slide5,
    "IndicFT_10k Analysis",
    "Methodological & Geometric Mechanisms",
    "Explaining empirical behaviors through morphological composition and hypersphere geometry"
)

cw3 = Inches(3.64)
ch3 = Inches(5.0)

c1 = add_card(slide5, Inches(0.9), Inches(1.95), cw3, ch3)
add_gold_divider(slide5, Inches(1.15), Inches(1.95), cw3 - Inches(0.5))
b1 = slide5.shapes.add_textbox(Inches(1.15), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Morphological Bleed"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s5_1 = [
    ("n-Gram Blending:", "FastText sums subwords. Words sharing Hindi affixes (e.g. -ता, -कर) share subword vectors."),
    ("Orthographic Pull:", "Inflected forms group together regardless of core semantic divergence."),
    ("Low Silhouette:", "Explains the low silhouette peak (0.024); points form a dense continuous manifold.")
]
for title, body in pts_s5_1:
    p = tf1.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide5, Inches(4.84), Inches(1.95), cw3, ch3)
add_gold_divider(slide5, Inches(5.09), Inches(1.95), cw3 - Inches(0.5))
b2 = slide5.shapes.add_textbox(Inches(5.09), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Hypersphere Geometry"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s5_2 = [
    ("Spherical Equivalence:", "On unit-normalized vectors, Euclidean distance strictly mirrors cosine distance."),
    ("Voronoi Advantage:", "K-Means minimizes Euclidean variance, naturally optimizing cosine alignment on unit spheres."),
    ("CH Decay Mechanics:", "The (k-1) normalization factor in Calinski-Harabasz outpaces between-cluster variance growth.")
]
for title, body in pts_s5_2:
    p = tf2.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c3 = add_card(slide5, Inches(8.78), Inches(1.95), cw3, ch3)
add_gold_divider(slide5, Inches(9.03), Inches(1.95), cw3 - Inches(0.5))
b3 = slide5.shapes.add_textbox(Inches(9.03), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf3 = b3.text_frame
tf3.word_wrap = True

p = tf3.paragraphs[0]
p.text = "3. Algorithmic Dynamics"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s5_3 = [
    ("Agglomerative Cosine:", "Avoids spherical centroid assumptions, capturing natural branching to lead Davies-Bouldin."),
    ("Spectral Graph Density:", "Dense 10-NN graphs in 300D lead to noisy graph cuts and negative silhouette at k=50."),
    ("MiniBatch Centroid Drift:", "Stochastic subsampling accumulates noise, degrading silhouette at higher k.")
]
for title, body in pts_s5_3:
    p = tf3.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 6: ROBERTA_10K — CODE (1/2)
# ===========================================================================
slide6 = prs.slides.add_slide(blank_layout)
set_bg(slide6)
add_minimalist_header(
    slide6,
    "Roberta_10k Pipeline (1/2)",
    "Transformer Inference & Masked Pooling",
    "Extracting 768-dimensional contextual representations using RoBERTa-Hindi"
)

c1 = add_card(slide6, Inches(0.9), Inches(1.95), cw, ch)
add_gold_divider(slide6, Inches(1.2), Inches(1.95), cw - Inches(0.6))
b1 = slide6.shapes.add_textbox(Inches(1.2), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Model & Batched Inference"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s6_1 = [
    ("Model Architecture:", "Loaded flax-community/roberta-hindi (12-layer transformer, 768D)."),
    ("Subword Tokenization:", "Byte-level BPE with dynamic padding (max_length=16)."),
    ("GPU Batching:", "Processed tokens in batches of BATCH_SIZE = 64 using torch.no_grad()."),
    ("Device Allocation:", "Dynamically routed tensors to CUDA GPU for rapid forward passes.")
]

for title, body in bullets_s6_1:
    p = tf1.add_paragraph()
    p.space_before = Pt(13)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide6, Inches(6.833), Inches(1.95), cw, ch)
add_gold_divider(slide6, Inches(7.133), Inches(1.95), cw - Inches(0.6))
b2 = slide6.shapes.add_textbox(Inches(7.133), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Masked Mean-Pooling & Scaling"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s6_2 = [
    ("Masked Mean-Pooling:", "Multiplied last hidden states by attention masks to zero out padding."),
    ("Normalization Formula:", "Computed: sum(hidden * mask) / mask.sum(), isolating pure lexical semantics."),
    ("Dense Output:", "Yielded a clean embedding matrix X of shape (10000, 768)."),
    ("Hypersphere Normalization:", "Scaled vectors to unit norm (||x||_2 = 1) for spherical clustering."),
    ("Artifact Storage:", "Persisted X_roberta_normalized.npy and words_roberta.txt.")
]

for title, body in bullets_s6_2:
    p = tf2.add_paragraph()
    p.space_before = Pt(11)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 7: ROBERTA_10K — CODE (2/2)
# ===========================================================================
slide7 = prs.slides.add_slide(blank_layout)
set_bg(slide7)
add_minimalist_header(
    slide7,
    "Roberta_10k Pipeline (2/2)",
    "Clustering Suite, Quantization & Benchmarking",
    "Running 5 clustering algorithms on 768D space with GPU speedups"
)

c1 = add_card(slide7, Inches(0.9), Inches(1.95), cw, ch)
add_gold_divider(slide7, Inches(1.2), Inches(1.95), cw - Inches(0.6))
b1 = slide7.shapes.add_textbox(Inches(1.2), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. High-Dimensional Clustering"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

algos_s7 = [
    ("Strict Consistency:", "Executed the exact same 5 algorithms to guarantee benchmark validity."),
    ("cuML GPU Acceleration:", "Leveraged cuKMeans and cuGMM to accelerate 768D clustering."),
    ("Quantization Pipeline:", "Tested FP16/INT8 representation efficiency in clusters_quantised.py."),
    ("Cluster Output Integrity:", "Saved CSV assignments ensuring all k unique partitions were populated.")
]

for title, body in algos_s7:
    p = tf1.add_paragraph()
    p.space_before = Pt(13)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide7, Inches(6.833), Inches(1.95), cw, ch)
add_gold_divider(slide7, Inches(7.133), Inches(1.95), cw - Inches(0.6))
b2 = slide7.shapes.add_textbox(Inches(7.133), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Evaluation Suite & Logging"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

eval_s7 = [
    ("Parameter Sweep:", "Evaluated across k in [50, 100, 150, 200, 250, 300] directly on 768D."),
    ("Metric Pipeline:", "Automated calculation of WCSS, Silhouette, DB Index, and CH Score."),
    ("Exact WCSS Computation:", "Employed exact centroid deviation for models without native inertia."),
    ("Automated Visual Reporting:", "Generated clustering_comparison.png via plot_metrics.py.")
]

for title, body in eval_s7:
    p = tf2.add_paragraph()
    p.space_before = Pt(13)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 8: ROBERTA_10K — DEDICATED FULL-PAGE PICTURE
# ===========================================================================
slide8 = prs.slides.add_slide(blank_layout)
set_bg(slide8)

if ROBERTA_PNG.exists():
    img_h = Inches(7.0)
    img_w = Inches(7.0 * (3977 / 3216))
    img_left = (prs.slide_width - img_w) / 2
    img_top = (prs.slide_height - img_h) / 2
    slide8.shapes.add_picture(str(ROBERTA_PNG), img_left, img_top, width=img_w, height=img_h)


# ===========================================================================
# SLIDE 9: ROBERTA_10K — METHODOLOGICAL ANALYSIS
# ===========================================================================
slide9 = prs.slides.add_slide(blank_layout)
set_bg(slide9)
add_minimalist_header(
    slide9,
    "Roberta_10k Analysis",
    "Methodological & Geometric Mechanisms",
    "Why deep contextual pretraining dramatically outperforms static subword embeddings"
)

c1 = add_card(slide9, Inches(0.9), Inches(1.95), cw3, ch3)
add_gold_divider(slide9, Inches(1.15), Inches(1.95), cw3 - Inches(0.5))
b1 = slide9.shapes.add_textbox(Inches(1.15), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Contextual Semantics"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s9_1 = [
    ("Deep Context:", "Pretrained with Masked Language Modeling across millions of sentences."),
    ("Morphological Immunity:", "Words with identical grammatical endings are not artificially merged."),
    ("Impact on WCSS & CH:", "Genuine semantic cohesion produces dense, well-separated cluster cores.")
]
for title, body in pts_s9_1:
    p = tf1.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide9, Inches(4.84), Inches(1.95), cw3, ch3)
add_gold_divider(slide9, Inches(5.09), Inches(1.95), cw3 - Inches(0.5))
b2 = slide9.shapes.add_textbox(Inches(5.09), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Representation Anisotropy"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s9_2 = [
    ("Anisotropic Cones:", "Transformer vectors naturally populate narrow directional cones in 768D space."),
    ("Angular Dispersal:", "L2 normalization separates semantic classes into distinct sub-cones."),
    ("768D Expressivity:", "High dimensionality provides ample space to isolate fine-grained lexical categories.")
]
for title, body in pts_s9_2:
    p = tf2.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c3 = add_card(slide9, Inches(8.78), Inches(1.95), cw3, ch3)
add_gold_divider(slide9, Inches(9.03), Inches(1.95), cw3 - Inches(0.5))
b3 = slide9.shapes.add_textbox(Inches(9.03), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf3 = b3.text_frame
tf3.word_wrap = True

p = tf3.paragraphs[0]
p.text = "3. Algorithmic Synergy"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s9_3 = [
    ("Voronoi Precision:", "K-Means spherical Voronoi cells fit transformer directional cones with high fidelity."),
    ("Hierarchical Capture:", "Agglomerative Cosine captures taxonomic hierarchies (hyponyms) cleanly."),
    ("Stable Spectral Cuts:", "Contextual coherence cleans the 10-NN graph, avoiding negative silhouette at low k.")
]
for title, body in pts_s9_3:
    p = tf3.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 10: WORD2VEC — CODE (1/2: MODEL ARCHITECTURE & INGESTION)
# ===========================================================================
slide10 = prs.slides.add_slide(blank_layout)
set_bg(slide10)
add_minimalist_header(
    slide10,
    "Word2Vec-Hindi Pipeline (1/2)",
    "Skip-Gram Model & Lexical Alignment",
    "AbhishekBiswas12/word2vec-hindi architecture and alignment with 20,000 Shabd entries"
)

c1 = add_card(slide10, Inches(0.9), Inches(1.95), cw, ch)
add_gold_divider(slide10, Inches(1.2), Inches(1.95), cw - Inches(0.6))
b1 = slide10.shapes.add_textbox(Inches(1.2), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Model Architecture & Training Corpus"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s10_1 = [
    ("Model Origin:", "Employs AbhishekBiswas12/word2vec-hindi, a PyTorch Word2Vec Skip-gram model."),
    ("Training Scale:", "Trained on 82M+ tokens across 5 Hindi corpora (Bible, IITB Parallel, Wikipedia)."),
    ("Vocabulary Capacity:", "Spans 508,205 unique lexical entries with 300-dimensional embeddings."),
    ("PyTorch Unpickling:", "Registers custom Word2Vec class into __main__ for clean state deserialization."),
    ("Objective:", "Trained with negative sampling (BCEWithLogitsLoss) capturing word-level co-occurrences.")
]

for title, body in bullets_s10_1:
    p = tf1.add_paragraph()
    p.space_before = Pt(11)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide10, Inches(6.833), Inches(1.95), cw, ch)
add_gold_divider(slide10, Inches(7.133), Inches(1.95), cw - Inches(0.6))
b2 = slide10.shapes.add_textbox(Inches(7.133), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Shabd Alignment & GPU Slicing"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s10_2 = [
    ("Shabd 34k Filtering:", "Scanned 33,676 Shabd words; identified 28,121 matches in the Word2Vec vocabulary."),
    ("Frequency Subsampling:", "Selected the top 20,000 frequent matching entries (NUM_WORDS=20000) for fast execution."),
    ("Zero-Copy GPU Slicing:", "Directly indexes input_embedding_layer.weight on GPU via PyTorch CUDA tensors."),
    ("GPU Normalization:", "Applies torch.nn.functional.normalize(X, p=2, dim=1) in milliseconds on CUDA."),
    ("Artifact Storage:", "Persists X_word2vec_normalized.npy (20000, 300) and words_word2vec.txt.")
]

for title, body in bullets_s10_2:
    p = tf2.add_paragraph()
    p.space_before = Pt(11)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 11: WORD2VEC — CODE (2/2: CLUSTERING & RUNTIME OPTIMIZATION)
# ===========================================================================
slide11 = prs.slides.add_slide(blank_layout)
set_bg(slide11)
add_minimalist_header(
    slide11,
    "Word2Vec-Hindi Pipeline (2/2)",
    "Clustering Suite & Runtime Optimization",
    "Exclusion of Spectral Clustering for high speed and GPU-accelerated evaluation across k = 50 to 300"
)

c1 = add_card(slide11, Inches(0.9), Inches(1.95), cw, ch)
add_gold_divider(slide11, Inches(1.2), Inches(1.95), cw - Inches(0.6))
b1 = slide11.shapes.add_textbox(Inches(1.2), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Active 4-Algorithm Suite"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s11_1 = [
    ("MiniBatchKMeans:", "Stochastic fast K-Means approximation completing in ~3.2 seconds."),
    ("Voronoi Partition:", "Exact K-Means with k-means++ auto-initialization completing in ~6.3 seconds."),
    ("Gaussian Mixture (GMM):", "Probabilistic EM clustering with diagonal covariance matrix (~20.5 seconds)."),
    ("Agglomerative Cosine:", "Hierarchical average-linkage tree built directly on cosine distance (~83.6 seconds)."),
    ("Spectral Removal:", "Omitted Spectral Clustering, eliminating 20,000x20,000 graph and eigen-decomposition bottleneck.")
]

for title, body in bullets_s11_1:
    p = tf1.add_paragraph()
    p.space_before = Pt(11)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide11, Inches(6.833), Inches(1.95), cw, ch)
add_gold_divider(slide11, Inches(7.133), Inches(1.95), cw - Inches(0.6))
b2 = slide11.shapes.add_textbox(Inches(7.133), Inches(2.15), cw - Inches(0.6), ch - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. GPU Acceleration & Fast Sweep"
p.font.name = FONT_SERIF
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

bullets_s11_2 = [
    ("RAPIDS cuML:", "Accelerates cuKMeans, cuMiniBatch, and cuGMM on CUDA GPU, dropping runtimes by 10x-50x."),
    ("Vectorized GPU WCSS:", "PyTorch CUDA tensors compute non-native inertia for Agglomerative in milliseconds."),
    ("Sampled Silhouette:", "Computes silhouette on a 10,000 random subset in ~2.6s, eliminating O(N^2) memory lockup."),
    ("Runtime Profile:", "Full cluster.py completes in ~1.5 min on CPU (~15s on GPU); evaluate_cluster.py takes ~10 min on CPU (~2 min on GPU)."),
    ("Reporting & Plotting:", "Outputs clustering_metrics_word2vec.csv and 2x2 comparison grid in plot_metric.py.")
]

for title, body in bullets_s11_2:
    p = tf2.add_paragraph()
    p.space_before = Pt(11)
    p.font.name = FONT_SANS
    p.font.size = Pt(11)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 12: WORD2VEC — DEDICATED FULL-PAGE PICTURE
# ===========================================================================
slide12 = prs.slides.add_slide(blank_layout)
set_bg(slide12)

if WORD2VEC_PNG.exists():
    img_h = Inches(7.0)
    img_w = Inches(7.0 * (4003 / 3216))
    img_left = (prs.slide_width - img_w) / 2
    img_top = (prs.slide_height - img_h) / 2
    slide12.shapes.add_picture(str(WORD2VEC_PNG), img_left, img_top, width=img_w, height=img_h)


# ===========================================================================
# SLIDE 13: WORD2VEC — METHODOLOGICAL ANALYSIS & EMPIRICAL FINDINGS
# ===========================================================================
slide13 = prs.slides.add_slide(blank_layout)
set_bg(slide13)
add_minimalist_header(
    slide13,
    "Word2Vec-Hindi Analysis",
    "Methodological & Geometric Mechanisms",
    "How word-level skip-gram training shapes semantic cluster geometry compared to subwords and transformers"
)

c1 = add_card(slide13, Inches(0.9), Inches(1.95), cw3, ch3)
add_gold_divider(slide13, Inches(1.15), Inches(1.95), cw3 - Inches(0.5))
b1 = slide13.shapes.add_textbox(Inches(1.15), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf1 = b1.text_frame
tf1.word_wrap = True

p = tf1.paragraphs[0]
p.text = "1. Whole-Word vs. Subwords"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s13_1 = [
    ("No Morphological Bleed:", "Word2Vec treats tokens as atomic IDs. Words sharing suffixes (e.g. -कर) are not artificially bound."),
    ("Distributional Semantics:", "Embeddings reflect co-occurrence in a 5-token window, grouping topical and associative cohorts."),
    ("Clean Lexical Boundaries:", "Produces sharper categorical separation than FastText's continuous n-gram manifold.")
]
for title, body in pts_s13_1:
    p = tf1.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c2 = add_card(slide13, Inches(4.84), Inches(1.95), cw3, ch3)
add_gold_divider(slide13, Inches(5.09), Inches(1.95), cw3 - Inches(0.5))
b2 = slide13.shapes.add_textbox(Inches(5.09), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf2 = b2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "2. Static vs. Contextual Space"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s13_2 = [
    ("Higher CH than FastText:", "Calinski-Harabasz peaks at 78.99 (k=50) and reaches 24.47 (k=200), nearly 2x FastText (12.87)."),
    ("Isotropic Distribution:", "Word2Vec vectors spread more uniformly across 300D, avoiding RoBERTa's ultra-narrow anisotropic cone."),
    ("WCSS & Separation:", "MiniBatch achieves excellent DB separation (2.85 at k=200), while Voronoi leads cluster density.")
]
for title, body in pts_s13_2:
    p = tf2.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

c3 = add_card(slide13, Inches(8.78), Inches(1.95), cw3, ch3)
add_gold_divider(slide13, Inches(9.03), Inches(1.95), cw3 - Inches(0.5))
b3 = slide13.shapes.add_textbox(Inches(9.03), Inches(2.15), cw3 - Inches(0.5), ch3 - Inches(0.4))
tf3 = b3.text_frame
tf3.word_wrap = True

p = tf3.paragraphs[0]
p.text = "3. Algorithmic Fit & Efficiency"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

pts_s13_3 = [
    ("Voronoi Partition:", "Directly optimizes spherical Euclidean scatter; ideal baseline for semantic categorization."),
    ("Agglomerative Cosine:", "Hierarchical cosine trees excel at isolating fine-grained sub-taxonomies (synonyms/antonyms)."),
    ("High Speed:", "Eliminating Spectral enables complete 4-algorithm multi-k benchmarking in ~10 minutes on CPU.")
]
for title, body in pts_s13_3:
    p = tf3.add_paragraph()
    p.space_before = Pt(12)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL


# ===========================================================================
# SLIDE 14: TRIPARTITE SYNTHESIS & COGNITIVE TAKEAWAYS (3-WAY MATRIX)
# ===========================================================================
slide14 = prs.slides.add_slide(blank_layout)
set_bg(slide14)
add_minimalist_header(
    slide14,
    "Synthesis & Conclusions",
    "Tripartite Synthesis & Cognitive Science Insights",
    "Comparative architecture matrix across IndicFastText, Word2Vec, and RoBERTa on Hindi semantics"
)

card_tbl = add_card(slide14, Inches(0.9), Inches(1.95), Inches(11.533), Inches(2.5))
add_gold_divider(slide14, Inches(1.2), Inches(1.95), Inches(10.933))

table_shape = slide14.shapes.add_table(6, 6, Inches(1.1), Inches(2.1), Inches(11.133), Inches(2.15))
table = table_shape.table

table.columns[0].width = Inches(2.2)
table.columns[1].width = Inches(1.8)
table.columns[2].width = Inches(1.8)
table.columns[3].width = Inches(1.8)
table.columns[4].width = Inches(1.8)
table.columns[5].width = Inches(1.733)

headers = ["Dimension / Paradigm", "IndicFastText", "Word2Vec-Hindi", "RoBERTa-Hindi", "Best Algorithm", "Cognitive Role"]
for j, h in enumerate(headers):
    cell = table.cell(0, j)
    cell.fill.solid()
    cell.fill.fore_color.rgb = GOLD_CHAMPAGNE
    p = cell.text_frame.paragraphs[0]
    p.text = h
    p.font.name = FONT_SANS
    p.font.bold = True
    p.font.size = Pt(9.5)
    p.font.color.rgb = BG_WHITE
    p.alignment = PP_ALIGN.CENTER

rows_data = [
    ("Representation Type", "Subword n-gram", "Word Skip-gram", "Contextual Attention", "-", "Morphological vs. Contextual"),
    ("Vector Dimensionality", "300 D", "300 D", "768 D", "-", "Hierarchical Capacity"),
    ("Morphological Bleed", "High (Affix bias)", "None (Word tokens)", "Context-filtered", "-", "Inflectional robustness"),
    ("Cluster Tightness (WCSS)", "7,065.12", "14,706.77 (20k pts)", "3,038.46 (10k pts)", "Voronoi (K-Means)", "Compact mental categories"),
    ("Separation (Davies-Bouldin)", "3.50 - 4.28", "2.85 - 5.84", "2.38 - 3.00", "Agglomerative / MiniBatch", "Clean hierarchical taxonomy")
]

for i, row in enumerate(rows_data):
    for j, val in enumerate(row):
        cell = table.cell(i + 1, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = TABLE_ALT_ROW if i % 2 == 0 else CARD_WHITE
        p = cell.text_frame.paragraphs[0]
        p.text = val
        p.font.name = FONT_SANS
        p.font.size = Pt(9.5)
        p.font.color.rgb = TEXT_CHARCOAL
        if j == 0:
            p.font.bold = True
        elif j == 4:
            p.font.bold = True
            p.font.color.rgb = GOLD_CHAMPAGNE
            p.alignment = PP_ALIGN.CENTER
        else:
            p.alignment = PP_ALIGN.CENTER

card_cog = add_card(slide14, Inches(0.9), Inches(4.7), Inches(11.533), Inches(2.3))
add_gold_divider(slide14, Inches(1.2), Inches(4.7), Inches(10.933))
b_c = slide14.shapes.add_textbox(Inches(1.2), Inches(4.85), Inches(10.933), Inches(2.0))
tf_c = b_c.text_frame
tf_c.word_wrap = True

p = tf_c.paragraphs[0]
p.text = "Cognitive Science Takeaways"
p.font.name = FONT_SERIF
p.font.size = Pt(15)
p.font.bold = True
p.font.color.rgb = TEXT_CHARCOAL

cog_points = [
    ("Mental Lexicon Architecture:", "Human semantic memory is multi-tiered: static models like Word2Vec reflect baseline associative semantic networks, while RoBERTa mirrors active contextual disambiguation in real-time language processing."),
    ("The Inflectional Hazard in Hindi:", "Subword n-gram models (FastText) suffer from morphological confounding, whereas Word2Vec and RoBERTa isolate conceptual meaning independent of grammatical inflections."),
    ("Optimal Algorithmic Recommendation:", "For downstream cognitive tasks (e.g. semantic priming, neural decoding), Voronoi K-Means and Agglomerative Cosine at k ≈ 150-200 provide the most robust balance of intra-cluster coherence and inter-cluster separation.")
]

for title, body in cog_points:
    p = tf_c.add_paragraph()
    p.space_before = Pt(5)
    p.font.name = FONT_SANS
    p.font.size = Pt(10.5)
    r1 = p.add_run()
    r1.text = title + " "
    r1.font.bold = True
    r1.font.color.rgb = GOLD_CHAMPAGNE
    r2 = p.add_run()
    r2.text = body
    r2.font.color.rgb = TEXT_CHARCOAL

prs.save(str(OUTPUT_PPT))
print(f"Minimalist White & Gold presentation updated with 14 slides (including Word2Vec dedicated image): {OUTPUT_PPT}")
