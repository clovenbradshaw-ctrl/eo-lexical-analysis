"""
Step 3: Analyze embedding space.

Tests: Completeness, Minimality, Orthogonality, Clustering, Boundaries.

Run: python scripts/03_analyze.py [--embed_level full_spec|short_def|seed_verbs]
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

def load_data():
    log("Loading embeddings...")
    verb_data = np.load(os.path.join(DATA_DIR, "verb_embeddings.npz"))
    log(f"  ✓ verb_embeddings.npz — keys: {list(verb_data.keys())}")
    op_data = np.load(os.path.join(DATA_DIR, "operator_embeddings.npz"))
    log(f"  ✓ operator_embeddings.npz — keys: {list(op_data.keys())}")
    with open(os.path.join(DATA_DIR, "embedding_metadata.json")) as f:
        meta = json.load(f)
    log(f"  ✓ metadata — backend: {meta['backend']}, dim: {meta['embedding_dim']}, verbs: {meta['num_verbs']}")
    return verb_data, op_data, meta

def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_n @ b_n.T


# ═══════════════════════════════════════════════════════════════════
# TEST 1: COMPLETENESS
# ═══════════════════════════════════════════════════════════════════

def test_completeness(verb_embs, op_embs, verb_names, top_n=20):
    log("\n" + "="*60)
    log("TEST 1: COMPLETENESS")
    log("Does every verb map to at least one operator?")
    log("="*60)
    
    log("\n  Computing similarity matrix...")
    t0 = time.time()
    sim = cosine_sim(verb_embs, op_embs)
    log(f"  ✓ Similarity matrix: {sim.shape} [{time.time()-t0:.1f}s]")
    
    nearest_idx = np.argmax(sim, axis=1)
    nearest_sim = np.max(sim, axis=1)
    
    log(f"\n  ── Distribution of verbs per operator ──")
    op_counts = Counter()
    op_verbs = defaultdict(list)
    for i, (oi, s) in enumerate(zip(nearest_idx, nearest_sim)):
        op = HELIX_ORDER[oi]
        op_counts[op] += 1
        op_verbs[op].append((verb_names[i], float(s)))
    
    for op in HELIX_ORDER:
        op_verbs[op].sort(key=lambda x: -x[1])
    
    max_count = max(op_counts.values())
    for op in HELIX_ORDER:
        c = op_counts.get(op, 0)
        pct = c / len(verb_names) * 100
        bar = "█" * int(c / max_count * 30)
        log(f"    {op:3s}: {c:5d} ({pct:5.1f}%) {bar}")
    
    log(f"\n  ── Statistics ──")
    log(f"    Mean max similarity:   {np.mean(nearest_sim):.4f}")
    log(f"    Median max similarity: {np.median(nearest_sim):.4f}")
    log(f"    Std:                   {np.std(nearest_sim):.4f}")
    log(f"    Min (worst fit):       {np.min(nearest_sim):.4f}")
    log(f"    Max (best fit):        {np.max(nearest_sim):.4f}")
    
    log(f"\n  ── Top {top_n} verbs per operator ──")
    for op in HELIX_ORDER:
        top = op_verbs[op][:top_n]
        verbs_str = ", ".join(f"{v}({s:.3f})" for v, s in top[:8])
        log(f"    {op}: {verbs_str}...")
    
    orphan_idx = np.argsort(nearest_sim)[:50]
    log(f"\n  ── Top 15 orphan candidates (far from ALL operators) ──")
    orphans = []
    for i in orphan_idx[:15]:
        all_sims = {HELIX_ORDER[j]: float(sim[i,j]) for j in range(9)}
        top2 = sorted(all_sims.items(), key=lambda x: -x[1])[:2]
        log(f"    {verb_names[i]:25s} max={nearest_sim[i]:.4f}  nearest={HELIX_ORDER[nearest_idx[i]]:3s}  2nd={top2[1][0]}({top2[1][1]:.4f})")
        orphans.append({"verb": verb_names[i], "max_sim": float(nearest_sim[i]), "nearest": HELIX_ORDER[nearest_idx[i]], "all_sims": all_sims})
    
    report = {
        "stats": {"mean": float(np.mean(nearest_sim)), "median": float(np.median(nearest_sim)), "std": float(np.std(nearest_sim)), "min": float(np.min(nearest_sim))},
        "distribution": {op: op_counts.get(op,0) for op in HELIX_ORDER},
        "top_verbs": {op: op_verbs[op][:top_n] for op in HELIX_ORDER},
        "orphans": orphans,
    }
    with open(os.path.join(OUTPUT_DIR, "completeness_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved completeness_report.json")
    return sim, nearest_idx, nearest_sim


# ═══════════════════════════════════════════════════════════════════
# TEST 2: MINIMALITY
# ═══════════════════════════════════════════════════════════════════

def test_minimality(verb_embs, op_embs, verb_names, sim, nearest_idx):
    log("\n" + "="*60)
    log("TEST 2: MINIMALITY")
    log("What happens when we remove each operator?")
    log("="*60)
    
    report = {}
    for ri, rop in enumerate(HELIX_ORDER):
        owned = nearest_idx == ri
        n_owned = int(owned.sum())
        
        if n_owned == 0:
            log(f"\n  Remove {rop}: 0 verbs → POTENTIALLY REDUNDANT (no verbs map here)")
            report[rop] = {"owned": 0, "verdict": "POTENTIALLY REDUNDANT"}
            continue
        
        remaining = [i for i in range(9) if i != ri]
        remaining_ops = [HELIX_ORDER[i] for i in remaining]
        reduced = sim[:, remaining]
        
        new_nearest = np.argmax(reduced[owned], axis=1)
        new_ops = [remaining_ops[i] for i in new_nearest]
        new_sims = np.max(reduced[owned], axis=1)
        old_sims = sim[owned, ri]
        drops = old_sims - new_sims
        
        scatter = Counter(new_ops)
        top_target, top_count = scatter.most_common(1)[0]
        concentration = top_count / n_owned
        mean_drop = float(np.mean(drops))
        
        if concentration > 0.8 and mean_drop < 0.05:
            verdict = f"⚠ POSSIBLY REDUNDANT → {top_target} absorbs {concentration:.0%}, drop={mean_drop:.4f}"
        elif mean_drop > 0.15:
            verdict = f"✓ NECESSARY — large gap (drop={mean_drop:.4f})"
        else:
            verdict = f"✓ NEEDED — scatters to {len(scatter)} ops, drop={mean_drop:.4f}"
        
        log(f"\n  Remove {rop}: {n_owned} verbs")
        log(f"    {verdict}")
        log(f"    Scatter: {dict(scatter.most_common(4))}")
        log(f"    Mean/Max drop: {mean_drop:.4f} / {float(np.max(drops)):.4f}")
        
        # Show worst orphans
        worst = np.argsort(-drops)[:5]
        for w in worst:
            idx = np.where(owned)[0][w]
            log(f"    Worst: {verb_names[idx]:20s} drop={drops[w]:.4f} → {new_ops[w]}")
        
        report[rop] = {
            "owned": n_owned, "scatter": dict(scatter.most_common()),
            "concentration": float(concentration), "primary_absorber": top_target,
            "mean_drop": mean_drop, "max_drop": float(np.max(drops)),
            "verdict": verdict,
        }
    
    with open(os.path.join(OUTPUT_DIR, "minimality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved minimality_report.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# TEST 3: ORTHOGONALITY
# ═══════════════════════════════════════════════════════════════════

def test_orthogonality(op_embs):
    log("\n" + "="*60)
    log("TEST 3: ORTHOGONALITY")
    log("Are the nine operators semantically distinct?")
    log("="*60)
    
    sim = cosine_sim(op_embs, op_embs)
    
    log(f"\n  ── Full 9×9 Similarity Matrix ──")
    log(f"       {'  '.join(f'{op:5s}' for op in HELIX_ORDER)}")
    for i, op in enumerate(HELIX_ORDER):
        row = "  ".join(f"{sim[i,j]:5.3f}" for j in range(9))
        log(f"  {op:3s}  {row}")
    
    pairs = []
    for i in range(9):
        for j in range(i+1, 9):
            pairs.append((HELIX_ORDER[i], HELIX_ORDER[j], float(sim[i,j])))
    pairs.sort(key=lambda x: -x[2])
    
    off_diag = sim[np.triu_indices(9, k=1)]
    log(f"\n  ── Off-diagonal stats ──")
    log(f"    Mean: {np.mean(off_diag):.4f}")
    log(f"    Max:  {np.max(off_diag):.4f}")
    log(f"    Min:  {np.min(off_diag):.4f}")
    
    log(f"\n  ── Most similar pairs (potential conflation) ──")
    for a, b, s in pairs[:5]:
        same_triad = "SAME TRIAD" if any(a in t["operators"] and b in t["operators"] for t in TRIADS.values()) else ""
        same_role = "SAME ROLE" if any(a in ops and b in ops for ops in ROLES.values()) else ""
        log(f"    {a:3s} ↔ {b:3s}: {s:.4f}  {same_triad} {same_role}")
    
    log(f"\n  ── Most distant pairs ──")
    for a, b, s in pairs[-5:]:
        log(f"    {a:3s} ↔ {b:3s}: {s:.4f}")
    
    log(f"\n  ── Triadic coherence (should be higher within triads) ──")
    for tname, tinfo in TRIADS.items():
        ops = tinfo["operators"]
        idx = [HELIX_ORDER.index(op) for op in ops]
        intra = [float(sim[idx[a], idx[b]]) for a in range(3) for b in range(a+1, 3)]
        log(f"    {tname:15s} ({','.join(ops)}): mean intra={np.mean(intra):.4f}  values={[f'{v:.3f}' for v in intra]}")
    
    log(f"\n  ── Role coherence (self-similarity test) ──")
    for rname, rops in ROLES.items():
        idx = [HELIX_ORDER.index(op) for op in rops]
        intra = [float(sim[idx[a], idx[b]]) for a in range(3) for b in range(a+1, 3)]
        log(f"    {rname:8s} ({','.join(rops)}): mean={np.mean(intra):.4f}  values={[f'{v:.3f}' for v in intra]}")
    
    # Compare: mean intra-triad vs mean inter-triad
    all_intra = []
    all_inter = []
    for i in range(9):
        for j in range(i+1, 9):
            same = any(HELIX_ORDER[i] in t["operators"] and HELIX_ORDER[j] in t["operators"] for t in TRIADS.values())
            if same: all_intra.append(float(sim[i,j]))
            else: all_inter.append(float(sim[i,j]))
    log(f"\n  ── Triad structure test ──")
    log(f"    Mean INTRA-triad similarity: {np.mean(all_intra):.4f} (n={len(all_intra)})")
    log(f"    Mean INTER-triad similarity: {np.mean(all_inter):.4f} (n={len(all_inter)})")
    log(f"    Ratio (higher = more structure): {np.mean(all_intra)/np.mean(all_inter):.4f}")
    
    report = {
        "matrix": sim.tolist(), "operator_order": HELIX_ORDER,
        "pairs_ranked": [{"a": a, "b": b, "sim": s} for a,b,s in pairs],
        "stats": {"mean": float(np.mean(off_diag)), "max": float(np.max(off_diag)), "min": float(np.min(off_diag))},
        "intra_triad_mean": float(np.mean(all_intra)),
        "inter_triad_mean": float(np.mean(all_inter)),
    }
    with open(os.path.join(OUTPUT_DIR, "orthogonality_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved orthogonality_report.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# TEST 4: UNSUPERVISED CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def test_clustering(verb_embs, op_embs, verb_names):
    log("\n" + "="*60)
    log("TEST 4: UNSUPERVISED CLUSTERING")
    log("Does k=9 emerge naturally?")
    log("="*60)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from scipy.optimize import linear_sum_assignment
    
    sim = cosine_sim(verb_embs, op_embs)
    supervised = np.argmax(sim, axis=1)
    
    # K-means at k=9
    log(f"\n  Running K-means (k=9, 20 inits)...")
    t0 = time.time()
    km = KMeans(n_clusters=9, n_init=20, random_state=42)
    km_labels = km.fit_predict(verb_embs)
    log(f"  ✓ K-means done [{time.time()-t0:.1f}s]")
    
    log(f"  Computing metrics...")
    ari = adjusted_rand_score(supervised, km_labels)
    nmi = normalized_mutual_info_score(supervised, km_labels)
    
    log(f"  Computing silhouette (may take a moment on large corpus)...")
    t0 = time.time()
    n_sample = min(5000, len(verb_embs))
    sil = silhouette_score(verb_embs, km_labels, metric="cosine", sample_size=n_sample)
    log(f"  ✓ Silhouette done [{time.time()-t0:.1f}s]")
    
    # Hungarian matching
    conf = np.zeros((9,9), dtype=int)
    for sl, kl in zip(supervised, km_labels):
        conf[sl, kl] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    matched_acc = conf[row_ind, col_ind].sum() / len(verb_names)
    
    log(f"\n  ── K-means k=9 Results ──")
    log(f"    Adjusted Rand Index:     {ari:.4f}")
    log(f"    Normalized Mutual Info:  {nmi:.4f}")
    log(f"    Silhouette (cosine):     {sil:.4f}")
    log(f"    Hungarian matched acc:   {matched_acc:.4f}")
    
    log(f"\n  ── Operator ↔ Cluster matching ──")
    for r, c in zip(row_ind, col_ind):
        overlap = conf[r,c]
        total = (supervised == r).sum()
        log(f"    {HELIX_ORDER[r]:3s} → cluster {c}: {overlap}/{total} ({overlap/total:.1%})")
    
    # K sweep
    log(f"\n  ── K sweep (3-20) ──")
    sweep = []
    for k in range(3, 21):
        t0 = time.time()
        km_k = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km_k.fit_predict(verb_embs)
        sil_k = silhouette_score(verb_embs, labels_k, metric="cosine", sample_size=n_sample)
        elapsed = time.time() - t0
        marker = " ◄── EO" if k == 9 else ""
        log(f"    k={k:2d}: silhouette={sil_k:.4f}  inertia={km_k.inertia_:12.1f}  [{elapsed:.1f}s]{marker}")
        sweep.append({"k": k, "silhouette": float(sil_k), "inertia": float(km_k.inertia_)})
    
    best_k = max(sweep, key=lambda x: x["silhouette"])["k"]
    log(f"\n    Best k by silhouette: {best_k}")
    
    # HDBSCAN
    try:
        from sklearn.cluster import HDBSCAN
        log(f"\n  Running HDBSCAN (min_cluster_size=50)...")
        t0 = time.time()
        hdb = HDBSCAN(min_cluster_size=50, metric="cosine")
        hdb_labels = hdb.fit_predict(verb_embs)
        n_c = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = int((hdb_labels == -1).sum())
        log(f"  ✓ HDBSCAN: {n_c} clusters, {n_noise} noise [{time.time()-t0:.1f}s]")
        hdbscan_result = {"n_clusters": n_c, "n_noise": n_noise, "sizes": dict(Counter(hdb_labels.tolist()))}
    except Exception as e:
        log(f"  HDBSCAN skipped: {e}")
        hdbscan_result = None
    
    report = {
        "kmeans_k9": {"ari": float(ari), "nmi": float(nmi), "silhouette": float(sil), "matched_accuracy": float(matched_acc)},
        "sweep": sweep, "best_k": best_k,
        "hdbscan": hdbscan_result,
    }
    with open(os.path.join(OUTPUT_DIR, "clustering_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved clustering_report.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# TEST 5: BOUNDARY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def test_boundaries(sim, verb_names):
    log("\n" + "="*60)
    log("TEST 5: BOUNDARY ANALYSIS")
    log("Which verbs sit between two operators?")
    log("="*60)
    
    top2_idx = np.argsort(-sim, axis=1)[:, :2]
    top2_sim = np.array([[sim[i, top2_idx[i,0]], sim[i, top2_idx[i,1]]] for i in range(len(verb_names))])
    ambiguity = top2_sim[:, 0] - top2_sim[:, 1]
    
    log(f"\n  ── Ambiguity stats ──")
    log(f"    Mean gap (top1 - top2): {np.mean(ambiguity):.4f}")
    log(f"    Median gap:             {np.median(ambiguity):.4f}")
    log(f"    Min gap (most ambig):   {np.min(ambiguity):.4f}")
    
    most_ambig = np.argsort(ambiguity)
    
    log(f"\n  ── 20 most ambiguous verbs ──")
    boundary_verbs = []
    for i in most_ambig[:20]:
        op1 = HELIX_ORDER[top2_idx[i,0]]
        op2 = HELIX_ORDER[top2_idx[i,1]]
        log(f"    {verb_names[i]:25s} {op1}({top2_sim[i,0]:.4f}) vs {op2}({top2_sim[i,1]:.4f})  gap={ambiguity[i]:.4f}")
        boundary_verbs.append({"verb": verb_names[i], "op1": op1, "sim1": float(top2_sim[i,0]), "op2": op2, "sim2": float(top2_sim[i,1]), "gap": float(ambiguity[i])})
    
    # Which boundaries are busiest?
    boundaries = Counter()
    for i in most_ambig[:200]:
        a, b = sorted([HELIX_ORDER[top2_idx[i,0]], HELIX_ORDER[top2_idx[i,1]]])
        boundaries[f"{a}-{b}"] += 1
    
    log(f"\n  ── Busiest operator boundaries (from top 200 ambiguous) ──")
    for bnd, cnt in boundaries.most_common(10):
        bar = "█" * cnt
        log(f"    {bnd:10s}: {cnt:3d} {bar}")
    
    report = {
        "stats": {"mean_gap": float(np.mean(ambiguity)), "median_gap": float(np.median(ambiguity))},
        "most_ambiguous": boundary_verbs,
        "busiest_boundaries": dict(boundaries.most_common(15)),
    }
    with open(os.path.join(OUTPUT_DIR, "boundary_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved boundary_report.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

def visualize(verb_embs, op_embs, verb_names, nearest_idx):
    log("\n" + "="*60)
    log("VISUALIZATIONS")
    log("="*60)
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        log("  matplotlib not available, skipping."); return
    
    colors = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45"]
    
    # PCA
    log("  Computing PCA...")
    t0 = time.time()
    pca = PCA(n_components=2)
    all_embs = np.vstack([verb_embs, op_embs])
    all_2d = pca.fit_transform(all_embs)
    v2d = all_2d[:len(verb_embs)]
    o2d = all_2d[len(verb_embs):]
    log(f"  ✓ PCA done [{time.time()-t0:.1f}s]")
    log(f"    Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    for i, op in enumerate(HELIX_ORDER):
        mask = nearest_idx == i
        ax.scatter(v2d[mask,0], v2d[mask,1], c=colors[i], alpha=0.3, s=8, label=op)
    for i, op in enumerate(HELIX_ORDER):
        ax.scatter(o2d[i,0], o2d[i,1], c=colors[i], s=200, marker="*", edgecolors="black", linewidths=1.5)
        ax.annotate(op, (o2d[i,0], o2d[i,1]), fontsize=12, fontweight="bold", xytext=(5,5), textcoords="offset points")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("EO Operator Space: PCA Projection", fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "pca_operator_space.png"), dpi=150)
    plt.close()
    log("  ✓ pca_operator_space.png")
    
    # Heatmap
    log("  Generating heatmap...")
    op_sim = cosine_sim(op_embs, op_embs)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(op_sim, cmap="RdYlBu_r", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(HELIX_ORDER, fontsize=11); ax.set_yticklabels(HELIX_ORDER, fontsize=11)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{op_sim[i,j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if op_sim[i,j] > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Inter-Operator Similarity Matrix", fontsize=14)
    for start, name in [(0,"Existence"),(3,"Structure"),(6,"Interpretation")]:
        rect = plt.Rectangle((start-0.5, start-0.5), 3, 3, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "operator_similarity_heatmap.png"), dpi=150)
    plt.close()
    log("  ✓ operator_similarity_heatmap.png")
    
    # K-sweep plot
    try:
        with open(os.path.join(OUTPUT_DIR, "clustering_report.json")) as f:
            cr = json.load(f)
        sweep = cr.get("sweep", [])
        if sweep:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ks = [s["k"] for s in sweep]
            ax1.plot(ks, [s["silhouette"] for s in sweep], "bo-")
            ax1.axvline(x=9, color="red", linestyle="--", alpha=0.7, label="k=9 (EO)")
            ax1.set_xlabel("k"); ax1.set_ylabel("Silhouette"); ax1.set_title("Silhouette vs k"); ax1.legend()
            ax2.plot(ks, [s["inertia"] for s in sweep], "go-")
            ax2.axvline(x=9, color="red", linestyle="--", alpha=0.7, label="k=9 (EO)")
            ax2.set_xlabel("k"); ax2.set_ylabel("Inertia"); ax2.set_title("Elbow Plot"); ax2.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(VIZ_DIR, "k_sweep.png"), dpi=150)
            plt.close()
            log("  ✓ k_sweep.png")
    except: pass
    
    # UMAP
    try:
        from umap import UMAP
        log("  Computing UMAP (this takes a minute)...")
        t0 = time.time()
        reducer = UMAP(n_components=2, metric="cosine", n_neighbors=30, min_dist=0.1, random_state=42)
        all_umap = reducer.fit_transform(all_embs)
        vu = all_umap[:len(verb_embs)]; ou = all_umap[len(verb_embs):]
        log(f"  ✓ UMAP done [{time.time()-t0:.1f}s]")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        for i, op in enumerate(HELIX_ORDER):
            mask = nearest_idx == i
            ax.scatter(vu[mask,0], vu[mask,1], c=colors[i], alpha=0.3, s=8, label=op)
        for i, op in enumerate(HELIX_ORDER):
            ax.scatter(ou[i,0], ou[i,1], c=colors[i], s=200, marker="*", edgecolors="black", linewidths=1.5)
            ax.annotate(op, (ou[i,0], ou[i,1]), fontsize=12, fontweight="bold", xytext=(5,5), textcoords="offset points")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_title("EO Operator Space: UMAP Projection", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, "umap_operator_space.png"), dpi=150)
        plt.close()
        log("  ✓ umap_operator_space.png")
    except ImportError:
        log("  umap-learn not installed, skipping UMAP. pip install umap-learn")
    
    log(f"\n  All visualizations in {VIZ_DIR}/")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_level", choices=["short_def","full_spec","seed_verbs"], default="full_spec")
    parser.add_argument("--top_n", type=int, default=20)
    args = parser.parse_args()
    
    t_start = time.time()
    log("="*60)
    log("STEP 3: ANALYSIS")
    log("="*60)
    
    verb_data, op_data, meta = load_data()
    verb_embs = verb_data["enriched"]
    op_embs = op_data[args.embed_level]
    verb_names = meta["verb_names"]
    
    log(f"\nUsing: {args.embed_level} operator embeddings")
    log(f"Verb embeddings:     {verb_embs.shape}")
    log(f"Operator embeddings: {op_embs.shape}")
    
    sim, nearest_idx, nearest_sim = test_completeness(verb_embs, op_embs, verb_names, args.top_n)
    test_minimality(verb_embs, op_embs, verb_names, sim, nearest_idx)
    test_orthogonality(op_embs)
    test_clustering(verb_embs, op_embs, verb_names)
    test_boundaries(sim, verb_names)
    visualize(verb_embs, op_embs, verb_names, nearest_idx)
    
    log(f"\n{'='*60}")
    log(f"ALL ANALYSIS COMPLETE in {time.time()-t_start:.1f}s")
    log(f"Reports: {os.path.abspath(OUTPUT_DIR)}/")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
