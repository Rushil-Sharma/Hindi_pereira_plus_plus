"""
WordNet Coherence Evaluation for Hindi Word Clusters
- Cached synset lookups & pairwise similarities
- Parallelised across CPU cores with multiprocessing
- Produces CSV + bar plot
"""

from pathlib import Path
from itertools import combinations
from functools import lru_cache
import multiprocessing as mp
import os, sys, traceback

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyiwn

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTERS_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Global WordNet handle (initialised once per worker process)
# ---------------------------------------------------------------------------
_iwn = None

def _init_worker():
    """Called once in each worker process to load WordNet."""
    global _iwn
    try:
        lang = pyiwn.Language.HINDI
    except AttributeError:
        lang = 'hindi'
    _iwn = pyiwn.IndoWordNet(lang)


@lru_cache(maxsize=None)
def _synsets(word):
    """Cached synset lookup. Returns a tuple (hashable, cacheable)."""
    try:
        return tuple(_iwn.synsets(word))
    except Exception:
        return ()


def _hypernym_jaccard(s1, s2):
    try:
        h1 = set(s1.hypernymy())
        h2 = set(s2.hypernymy())
    except Exception:
        return 0.0
    if not h1 or not h2:
        return 0.0
    inter = len(h1 & h2)
    union = len(h1 | h2)
    return inter / union if union > 0 else 0.0


@lru_cache(maxsize=None)
def _pair_similarity(w1, w2):
    """Cached pairwise similarity. Call with sorted args to maximise hits."""
    synsets1 = _synsets(w1)
    synsets2 = _synsets(w2)
    if not synsets1 or not synsets2:
        return None

    max_sim = 0.0
    found = False
    for s1 in synsets1:
        for s2 in synsets2:
            sim = None
            if hasattr(s1, 'path_similarity'):
                try:
                    sim = s1.path_similarity(s2)
                except Exception:
                    sim = None
            if sim is None:
                sim = _hypernym_jaccard(s1, s2)
            if sim is not None and sim > max_sim:
                max_sim = sim
                found = True
    return max_sim if found else None


def cluster_coherence(words):
    if len(words) < 2:
        return None
    sims = []
    for w1, w2 in combinations(words, 2):
        a, b = (w1, w2) if w1 <= w2 else (w2, w1)  # canonical order → cache hits
        sim = _pair_similarity(a, b)
        if sim is not None:
            sims.append(sim)
    return float(np.mean(sims)) if sims else None


# ---------------------------------------------------------------------------
# Worker entry point — takes a whole cluster
# ---------------------------------------------------------------------------
def _score_cluster(words):
    return cluster_coherence(words)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Initializing Hindi WordNet...", flush=True)
    _init_worker()          # also initialises in the parent, for caching sanity
    print("Hindi WordNet loaded.", flush=True)

    cluster_files = sorted(CLUSTERS_DIR.glob("*_clusters.csv"))
    if not cluster_files:
        print(f"\nNo cluster files found in {CLUSTERS_DIR}.", flush=True)
        print("Run your clustering script first to produce *_clusters.csv.", flush=True)
        return

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"\nFound {len(cluster_files)} cluster files. "
          f"Using {n_workers} worker processes.", flush=True)

    results = []

    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for filepath in cluster_files:
            algorithm_name = filepath.stem.replace("_clusters", "")
            print(f"\n{'='*60}", flush=True)
            print(f"Evaluating: {algorithm_name}", flush=True)
            print(f"{'='*60}", flush=True)

            try:
                df = pd.read_csv(filepath, encoding="utf-8-sig")
            except Exception as e:
                print(f"  Could not read {filepath.name}: {e}", flush=True)
                continue

            df.columns = df.columns.str.lower()
            if "word" not in df.columns or "cluster" not in df.columns:
                print(f"  Skipping {filepath.name}: missing columns.", flush=True)
                continue

            # Build list of clusters (each a list of words) with >= 2 words
            clusters = []
            skipped = 0
            for cid, group in df.groupby("cluster"):
                words = group["word"].dropna().astype(str).tolist()
                if len(words) < 2:
                    skipped += 1
                else:
                    clusters.append(words)

            if not clusters:
                print("  No evaluable clusters.", flush=True)
                results.append({
                    "Algorithm": algorithm_name,
                    "Mean_WordNet_Coherence": None,
                    "Std_WordNet_Coherence": None,
                    "Clusters_Evaluated": 0,
                    "Clusters_Skipped": skipped,
                })
                continue

            # Parallel scoring of clusters
            scores = pool.map(_score_cluster, clusters, chunksize=4)
            valid = [s for s in scores if s is not None]
            skipped += len(scores) - len(valid)

            if valid:
                mean_c = float(np.mean(valid))
                std_c = float(np.std(valid))
                print(f"  Clusters evaluated: {len(valid)}", flush=True)
                print(f"  Clusters skipped:   {skipped}", flush=True)
                print(f"  Mean coherence:     {mean_c:.4f}", flush=True)
                print(f"  Std dev:            {std_c:.4f}", flush=True)
                results.append({
                    "Algorithm": algorithm_name,
                    "Mean_WordNet_Coherence": round(mean_c, 4),
                    "Std_WordNet_Coherence": round(std_c, 4),
                    "Clusters_Evaluated": len(valid),
                    "Clusters_Skipped": skipped,
                })
            else:
                print("  No valid cluster scores.", flush=True)
                results.append({
                    "Algorithm": algorithm_name,
                    "Mean_WordNet_Coherence": None,
                    "Std_WordNet_Coherence": None,
                    "Clusters_Evaluated": 0,
                    "Clusters_Skipped": skipped,
                })

    # -----------------------------------------------------------------------
    # CSV output
    # -----------------------------------------------------------------------
    if not results:
        print("\nNo results produced.", flush=True)
        return

    results_df = pd.DataFrame(results)
    out_csv = SCRIPT_DIR / "wordnet_coherence_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n{'='*60}", flush=True)
    print(f"Results saved to: {out_csv}", flush=True)
    print(results_df.to_string(index=False), flush=True)

    # -----------------------------------------------------------------------
    # Bar plot
    # -----------------------------------------------------------------------
    plot_df = results_df.dropna(subset=["Mean_WordNet_Coherence"]).copy()
    if plot_df.empty:
        print("\nNothing to plot.", flush=True)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(plot_df))
    ax.bar(
        x,
        plot_df["Mean_WordNet_Coherence"],
        yerr=plot_df["Std_WordNet_Coherence"],
        capsize=5,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["Algorithm"], rotation=30, ha="right")
    ax.set_ylabel("Mean WordNet Coherence")
    ax.set_title("Semantic Coherence of Clustering Algorithms (Hindi WordNet)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_png = SCRIPT_DIR / "wordnet_coherence_plot.png"
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"\nPlot saved to: {out_png}", flush=True)


if __name__ == "__main__":
    # Windows needs 'spawn'; multiprocessing handles that automatically,
    # but the __main__ guard is mandatory.
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)