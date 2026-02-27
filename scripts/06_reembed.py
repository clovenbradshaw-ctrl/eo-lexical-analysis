"""
Step 6: Re-embed verbs using LLM type-signature reasoning.

Three embedding modes:
  A) Type-signature only: embed the LLM's input→output reasoning
  B) Group centroids: embed original definitions, test whether LLM-defined groups separate
  C) Combined: verb + definition + type-signature + operator label

Then run clustering analysis to see if k=9 emerges when embeddings
carry transformation-type information rather than just lexical semantics.

Run: python scripts/06_reembed.py [--backend openai|sentence-transformers]

Requires: OPENAI_API_KEY (for openai backend)
Requires: data/llm_classifications.json (from step 4)

Outputs:
  data/reembed_*.npz
  output/reembed_report.json
  output/visualizations/reembed_*.png
"""

import argparse, json, os, sys, time
import numpy as np
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
SCRIPTS_DIR = os.path.dirname(__file__)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

sys.path.insert(0, SCRIPTS_DIR)
from operator_definitions import OPERATORS, HELIX_ORDER, TRIADS, ROLES

def log(msg):
    print(msg, flush=True)


# ─── Embedding backends ──────────────────────────────────────────

def embed_openai(texts, model="text-embedding-3-large"):
    from openai import OpenAI
    client = OpenAI()
    batch_size = 2048
    all_embs = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        bn = i // batch_size + 1
        total_b = (len(texts) - 1) // batch_size + 1
        log(f"    Batch {bn}/{total_b}: {len(batch)} texts...")
        response = client.embeddings.create(input=batch, model=model)
        all_embs.extend([item.embedding for item in response.data])
        log(f"      ✓ {len(all_embs)}/{len(texts)} done [{time.time()-t0:.1f}s]")
    return np.array(all_embs)

def embed_st(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    log(f"    Loading {model_name}...")
    model = SentenceTransformer(model_name)
    log(f"    Embedding {len(texts)} texts...")
    embs = model.encode(texts, show_progress_bar=True, batch_size=256)
    return np.array(embs)

BACKENDS = {"openai": embed_openai, "sentence-transformers": embed_st}


def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_n @ b_n.T


# ─── Load data ────────────────────────────────────────────────────

def load_data():
    log("Loading LLM classifications...")
    cls_path = os.path.join(DATA_DIR, "llm_classifications.json")
    if not os.path.exists(cls_path):
        log(f"ERROR: {cls_path} not found. Run 04_llm_classify.py first.")
        sys.exit(1)
    with open(cls_path) as f:
        cls_data = json.load(f)
    cls = cls_data["classifications"]
    log(f"  {len(cls)} classifications loaded")

    # Also load original corpus for definitions
    log("Loading original corpus...")
    with open(os.path.join(DATA_DIR, "combined_corpus.json")) as f:
        corpus = json.load(f)
    corpus_lookup = {}
    for entry in corpus:
        v = entry["verb"]
        wn = entry["sources"].get("wordnet", {})
        defs = wn.get("definitions", [])
        definition = ""
        if defs:
            good = [d for d in defs if len(d) > 10]
            definition = min(good, key=len) if good else defs[0]
        fn = entry["sources"].get("framenet", {})
        frames = list(dict.fromkeys(fn.get("frames", [])))[:2]
        corpus_lookup[v] = {"definition": definition, "frames": frames}
    log(f"  {len(corpus_lookup)} verb definitions loaded")

    return cls, corpus_lookup


# ─── Build embedding texts ───────────────────────────────────────

def build_texts(cls, corpus_lookup):
    log("\nBuilding three embedding text sets...")

    verbs = []
    texts_sig = []      # A: type-signature only
    texts_def = []      # B: original definition only
    texts_combined = []  # C: everything together
    labels = []
    valid_ops = set(HELIX_ORDER)

    for c in cls:
        op = c.get("operator", "")
        if op not in valid_ops:
            continue

        verb = c["verb"]
        reason = c.get("reason", "")
        scale = c.get("scale", "")
        confidence = c.get("confidence", "")

        lookup = corpus_lookup.get(verb, {})
        definition = lookup.get("definition", "")
        frames = lookup.get("frames", [])

        # A: Type-signature reasoning from LLM
        sig_text = f"{verb}: {reason}"
        if scale:
            sig_text += f" Scale: {scale}."

        # B: Original definition (same as step 2)
        def_text = verb
        if definition:
            def_text += f". {definition}"
        if frames:
            def_text += f". Frames: {', '.join(frames)}"

        # C: Combined — everything
        combined_parts = [verb]
        combined_parts.append(f"Operator: {op}")
        if definition:
            combined_parts.append(f"Definition: {definition}")
        if reason:
            combined_parts.append(f"Transformation: {reason}")
        if scale:
            combined_parts.append(f"Scale: {scale}")
        if frames:
            combined_parts.append(f"Frames: {', '.join(frames)}")
        combined_text = ". ".join(combined_parts)

        verbs.append(verb)
        texts_sig.append(sig_text)
        texts_def.append(def_text)
        texts_combined.append(combined_text)
        labels.append(HELIX_ORDER.index(op))

    labels = np.array(labels)
    log(f"  Valid verbs: {len(verbs)}")
    log(f"  Label distribution: {dict(Counter(HELIX_ORDER[l] for l in labels))}")

    log(f"\n  Sample texts:")
    for i in range(min(3, len(verbs))):
        log(f"    [{verbs[i]}]")
        log(f"      Sig:  {texts_sig[i][:80]}...")
        log(f"      Def:  {texts_def[i][:80]}...")
        log(f"      Comb: {texts_combined[i][:80]}...")

    return verbs, texts_sig, texts_def, texts_combined, labels


# ─── Clustering analysis ─────────────────────────────────────────

def analyze_embeddings(embs, labels, verbs, mode_name):
    log(f"\n  {'─'*50}")
    log(f"  ANALYSIS: {mode_name}")
    log(f"  {'─'*50}")
    log(f"  Shape: {embs.shape}")

    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from scipy.optimize import linear_sum_assignment

    n_sample = min(5000, len(embs))

    # K-means at k=9
    log(f"  K-means k=9...")
    t0 = time.time()
    km = KMeans(n_clusters=9, n_init=20, random_state=42)
    km_labels = km.fit_predict(embs)
    log(f"    ✓ [{time.time()-t0:.1f}s]")

    ari = adjusted_rand_score(labels, km_labels)
    nmi = normalized_mutual_info_score(labels, km_labels)
    sil = silhouette_score(embs, km_labels, metric="cosine", sample_size=n_sample)

    # Hungarian matching
    conf = np.zeros((9, 9), dtype=int)
    for sl, kl in zip(labels, km_labels):
        conf[sl, kl] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    matched_acc = conf[row_ind, col_ind].sum() / len(labels)

    log(f"    ARI:              {ari:.4f}")
    log(f"    NMI:              {nmi:.4f}")
    log(f"    Silhouette:       {sil:.4f}")
    log(f"    Matched accuracy: {matched_acc:.4f}")

    log(f"    Operator ↔ Cluster:")
    for r, c in zip(row_ind, col_ind):
        overlap = conf[r, c]
        total = (labels == r).sum()
        pct = overlap / total if total > 0 else 0
        log(f"      {HELIX_ORDER[r]:3s} → cluster {c}: {overlap}/{total} ({pct:.1%})")

    # K sweep
    log(f"  K sweep (3-20)...")
    sweep = []
    for k in range(3, 21):
        t0 = time.time()
        km_k = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km_k.fit_predict(embs)
        sil_k = silhouette_score(embs, labels_k, metric="cosine", sample_size=n_sample)
        marker = " ◄── EO" if k == 9 else ""
        log(f"    k={k:2d}: silhouette={sil_k:.4f}  inertia={km_k.inertia_:12.1f}  [{time.time()-t0:.1f}s]{marker}")
        sweep.append({"k": k, "silhouette": float(sil_k), "inertia": float(km_k.inertia_)})

    best_k = max(sweep, key=lambda x: x["silhouette"])["k"]
    log(f"    Best k by silhouette: {best_k}")

    # Group centroid separation
    log(f"  Computing group centroid separation...")
    centroids = np.zeros((9, embs.shape[1]))
    for i in range(9):
        mask = labels == i
        if mask.sum() > 0:
            centroids[i] = embs[mask].mean(axis=0)

    centroid_sim = cosine_sim(centroids, centroids)
    off_diag = centroid_sim[np.triu_indices(9, k=1)]
    log(f"    Mean inter-centroid similarity: {np.mean(off_diag):.4f}")
    log(f"    Min inter-centroid similarity:  {np.min(off_diag):.4f}")
    log(f"    Max inter-centroid similarity:  {np.max(off_diag):.4f}")

    # Compare to random grouping baseline
    log(f"  Computing random baseline (100 permutations)...")
    random_seps = []
    for _ in range(100):
        perm = np.random.permutation(labels)
        rand_centroids = np.zeros((9, embs.shape[1]))
        for i in range(9):
            mask = perm == i
            if mask.sum() > 0:
                rand_centroids[i] = embs[mask].mean(axis=0)
        rand_sim = cosine_sim(rand_centroids, rand_centroids)
        rand_off = rand_sim[np.triu_indices(9, k=1)]
        random_seps.append(np.mean(rand_off))

    actual_sep = float(np.mean(off_diag))
    random_mean = float(np.mean(random_seps))
    random_std = float(np.std(random_seps))
    z_score = (actual_sep - random_mean) / random_std if random_std > 0 else 0

    log(f"    Actual mean inter-centroid sim:  {actual_sep:.4f}")
    log(f"    Random baseline mean:            {random_mean:.4f} ± {random_std:.4f}")
    log(f"    Z-score (negative = more separated): {z_score:.2f}")

    if actual_sep < random_mean:
        log(f"    ✓ LLM-defined groups are MORE separated than random ({random_mean - actual_sep:.4f} lower)")
    else:
        log(f"    ✗ LLM-defined groups are NOT more separated than random")

    # Triadic structure
    log(f"  Triadic centroid structure...")
    triad_map = {"existence": [0,1,2], "structure": [3,4,5], "interpretation": [6,7,8]}
    all_intra = []
    all_inter = []
    for i in range(9):
        for j in range(i+1, 9):
            same = any(i in idx and j in idx for idx in triad_map.values())
            if same:
                all_intra.append(float(centroid_sim[i, j]))
            else:
                all_inter.append(float(centroid_sim[i, j]))
    if all_intra and all_inter:
        log(f"    Intra-triad centroid sim: {np.mean(all_intra):.4f}")
        log(f"    Inter-triad centroid sim: {np.mean(all_inter):.4f}")
        log(f"    Ratio: {np.mean(all_intra)/np.mean(all_inter):.4f}")

    return {
        "mode": mode_name,
        "shape": list(embs.shape),
        "kmeans_k9": {"ari": float(ari), "nmi": float(nmi), "silhouette": float(sil), "matched_accuracy": float(matched_acc)},
        "sweep": sweep,
        "best_k": best_k,
        "centroid_separation": {
            "actual_mean_sim": actual_sep,
            "random_mean_sim": random_mean,
            "random_std": random_std,
            "z_score": z_score,
        },
        "centroid_matrix": centroid_sim.tolist(),
    }


# ─── Visualizations ──────────────────────────────────────────────

def visualize_comparison(results):
    log("\n  Generating comparison visualizations...")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not available."); return

    modes = [r["mode"] for r in results]
    aris = [r["kmeans_k9"]["ari"] for r in results]
    nmis = [r["kmeans_k9"]["nmi"] for r in results]
    sils = [r["kmeans_k9"]["silhouette"] for r in results]
    accs = [r["kmeans_k9"]["matched_accuracy"] for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, vals, title in zip(axes, [aris, nmis, sils, accs], ["ARI", "NMI", "Silhouette", "Matched Accuracy"]):
        bars = ax.bar(modes, vals, color=["#e74c3c", "#3498db", "#2ecc71"])
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
        ax.tick_params(axis='x', rotation=15)
    plt.suptitle("Re-embedding Analysis: Does type-signature info improve clustering?", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "reembed_comparison.png"), dpi=150)
    plt.close()
    log("  ✓ reembed_comparison.png")

    # K-sweep comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for r, color in zip(results, colors):
        ks = [s["k"] for s in r["sweep"]]
        sils = [s["silhouette"] for s in r["sweep"]]
        ax.plot(ks, sils, "o-", color=color, label=r["mode"], linewidth=2)
    ax.axvline(x=9, color="black", linestyle="--", alpha=0.5, label="k=9 (EO)")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-sweep comparison across embedding modes")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "reembed_ksweep.png"), dpi=150)
    plt.close()
    log("  ✓ reembed_ksweep.png")

    # Centroid separation comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    actual = [r["centroid_separation"]["actual_mean_sim"] for r in results]
    random = [r["centroid_separation"]["random_mean_sim"] for r in results]
    x = np.arange(len(modes))
    w = 0.35
    ax.bar(x - w/2, actual, w, label="Actual groups", color="#3498db")
    ax.bar(x + w/2, random, w, label="Random groups", color="#95a5a6")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Mean inter-centroid cosine similarity")
    ax.set_title("Are LLM-defined groups more separated than random?\n(Lower = more separated)")
    ax.legend()
    for i, (a, r_val) in enumerate(zip(actual, random)):
        z = results[i]["centroid_separation"]["z_score"]
        ax.text(i, max(a, r_val) + 0.005, f"z={z:.1f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "reembed_separation.png"), dpi=150)
    plt.close()
    log("  ✓ reembed_separation.png")


# ─── Main ─────────────────────────────────────────────────────────

def run(backend_name):
    t_start = time.time()
    log("=" * 60)
    log(f"STEP 6: RE-EMBED WITH TYPE SIGNATURES (backend: {backend_name})")
    log("=" * 60)

    embed_fn = BACKENDS[backend_name]
    cls, corpus_lookup = load_data()
    verbs, texts_sig, texts_def, texts_combined, labels = build_texts(cls, corpus_lookup)

    results = []

    # A: Type-signature only
    log(f"\n{'='*50}")
    log(f"EMBEDDING MODE A: Type-signature reasoning only")
    log(f"{'='*50}")
    embs_sig = embed_fn(texts_sig)
    np.savez_compressed(os.path.join(DATA_DIR, "reembed_signature.npz"), embeddings=embs_sig, labels=labels)
    log(f"  ✓ Saved reembed_signature.npz")
    results.append(analyze_embeddings(embs_sig, labels, verbs, "Type-signature"))

    # B: Original definition only
    log(f"\n{'='*50}")
    log(f"EMBEDDING MODE B: Original definition only")
    log(f"{'='*50}")
    embs_def = embed_fn(texts_def)
    np.savez_compressed(os.path.join(DATA_DIR, "reembed_definition.npz"), embeddings=embs_def, labels=labels)
    log(f"  ✓ Saved reembed_definition.npz")
    results.append(analyze_embeddings(embs_def, labels, verbs, "Definition"))

    # C: Combined
    log(f"\n{'='*50}")
    log(f"EMBEDDING MODE C: Combined (verb + definition + type-signature + operator)")
    log(f"{'='*50}")
    embs_comb = embed_fn(texts_combined)
    np.savez_compressed(os.path.join(DATA_DIR, "reembed_combined.npz"), embeddings=embs_comb, labels=labels)
    log(f"  ✓ Saved reembed_combined.npz")
    results.append(analyze_embeddings(embs_comb, labels, verbs, "Combined"))

    # Compare
    log(f"\n{'='*60}")
    log(f"COMPARISON SUMMARY")
    log(f"{'='*60}")
    log(f"\n  {'Mode':<20s} {'ARI':>8s} {'NMI':>8s} {'Silh':>8s} {'Match%':>8s} {'BestK':>6s} {'Z-sep':>8s}")
    log(f"  {'─'*68}")
    for r in results:
        k9 = r["kmeans_k9"]
        z = r["centroid_separation"]["z_score"]
        log(f"  {r['mode']:<20s} {k9['ari']:8.4f} {k9['nmi']:8.4f} {k9['silhouette']:8.4f} {k9['matched_accuracy']:8.4f} {r['best_k']:6d} {z:8.2f}")

    visualize_comparison(results)

    # Save full report
    report = {"results": results, "verbs": verbs, "labels": labels.tolist()}
    with open(os.path.join(OUTPUT_DIR, "reembed_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"\n  ✓ Saved reembed_report.json")

    log(f"\n{'='*60}")
    log(f"STEP 6 COMPLETE in {time.time()-t_start:.1f}s")
    log(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), default="openai")
    args = parser.parse_args()
    run(args.backend)
