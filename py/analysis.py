"""
Consolidated analysis functions for the EO Notebook Experiment App.
Extracted from scripts 03, 08, 11, 12 — adapted to run in Pyodide (no file I/O).
All functions take numpy arrays / dicts and return result dicts.
"""

import numpy as np
from collections import Counter, defaultdict

HELIX_ORDER = ["NUL", "DES", "INS", "SEG", "CON", "SYN", "ALT", "SUP", "REC"]

TRIADS = {
    "existence":      {"operators": ["NUL", "DES", "INS"]},
    "structure":      {"operators": ["SEG", "CON", "SYN"]},
    "interpretation": {"operators": ["ALT", "SUP", "REC"]},
}

ROLES = {
    "ground":  ["NUL", "SEG", "ALT"],
    "figure":  ["DES", "CON", "SUP"],
    "pattern": ["INS", "SYN", "REC"],
}


# ── helpers ──────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two matrices (row-wise)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_n @ b_n.T


def cosine_sim_vec(a, B):
    """Cosine similarity between a vector and a matrix."""
    a_n = a / (np.linalg.norm(a) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return B_n @ a_n


# ═══════════════════════════════════════════════════════════════
# TEST 1: COMPLETENESS
# ═══════════════════════════════════════════════════════════════

def test_completeness(verb_embs, op_embs, verb_names, top_n=20):
    """Does every verb map to at least one operator?"""
    sim = cosine_sim(verb_embs, op_embs)
    nearest_idx = np.argmax(sim, axis=1)
    nearest_sim = np.max(sim, axis=1)

    op_counts = Counter()
    op_verbs = defaultdict(list)
    for i, (oi, s) in enumerate(zip(nearest_idx, nearest_sim)):
        op = HELIX_ORDER[oi]
        op_counts[op] += 1
        op_verbs[op].append((verb_names[i], float(s)))

    for op in HELIX_ORDER:
        op_verbs[op].sort(key=lambda x: -x[1])

    orphan_idx = np.argsort(nearest_sim)[:50]
    orphans = []
    for i in orphan_idx[:15]:
        all_sims = {HELIX_ORDER[j]: float(sim[i, j]) for j in range(9)}
        top2 = sorted(all_sims.items(), key=lambda x: -x[1])[:2]
        orphans.append({
            "verb": verb_names[i],
            "max_sim": float(nearest_sim[i]),
            "nearest": HELIX_ORDER[nearest_idx[i]],
            "all_sims": all_sims,
        })

    return {
        "sim_matrix": sim,
        "nearest_idx": nearest_idx,
        "nearest_sim": nearest_sim,
        "stats": {
            "mean": float(np.mean(nearest_sim)),
            "median": float(np.median(nearest_sim)),
            "std": float(np.std(nearest_sim)),
            "min": float(np.min(nearest_sim)),
            "max": float(np.max(nearest_sim)),
        },
        "distribution": {op: op_counts.get(op, 0) for op in HELIX_ORDER},
        "top_verbs": {op: op_verbs[op][:top_n] for op in HELIX_ORDER},
        "orphans": orphans,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 2: MINIMALITY
# ═══════════════════════════════════════════════════════════════

def test_minimality(verb_embs, op_embs, verb_names, sim, nearest_idx):
    """What happens when we remove each operator?"""
    report = {}
    for ri, rop in enumerate(HELIX_ORDER):
        owned = nearest_idx == ri
        n_owned = int(owned.sum())

        if n_owned == 0:
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
            verdict = f"POSSIBLY REDUNDANT -> {top_target} absorbs {concentration:.0%}, drop={mean_drop:.4f}"
        elif mean_drop > 0.15:
            verdict = f"NECESSARY — large gap (drop={mean_drop:.4f})"
        else:
            verdict = f"NEEDED — scatters to {len(scatter)} ops, drop={mean_drop:.4f}"

        report[rop] = {
            "owned": n_owned,
            "scatter": dict(scatter.most_common()),
            "concentration": float(concentration),
            "primary_absorber": top_target,
            "mean_drop": mean_drop,
            "max_drop": float(np.max(drops)),
            "verdict": verdict,
        }

    return report


# ═══════════════════════════════════════════════════════════════
# TEST 3: ORTHOGONALITY
# ═══════════════════════════════════════════════════════════════

def test_orthogonality(op_embs):
    """Are the nine operators semantically distinct?"""
    sim = cosine_sim(op_embs, op_embs)

    pairs = []
    for i in range(9):
        for j in range(i + 1, 9):
            pairs.append({"a": HELIX_ORDER[i], "b": HELIX_ORDER[j], "sim": float(sim[i, j])})
    pairs.sort(key=lambda x: -x["sim"])

    off_diag = sim[np.triu_indices(9, k=1)]

    all_intra, all_inter = [], []
    for i in range(9):
        for j in range(i + 1, 9):
            same = any(
                HELIX_ORDER[i] in t["operators"] and HELIX_ORDER[j] in t["operators"]
                for t in TRIADS.values()
            )
            (all_intra if same else all_inter).append(float(sim[i, j]))

    return {
        "matrix": sim.tolist(),
        "operator_order": HELIX_ORDER,
        "pairs_ranked": pairs,
        "stats": {
            "mean": float(np.mean(off_diag)),
            "max": float(np.max(off_diag)),
            "min": float(np.min(off_diag)),
        },
        "intra_triad_mean": float(np.mean(all_intra)),
        "inter_triad_mean": float(np.mean(all_inter)),
    }


# ═══════════════════════════════════════════════════════════════
# TEST 4: UNSUPERVISED CLUSTERING
# ═══════════════════════════════════════════════════════════════

def test_clustering(verb_embs, op_embs, verb_names, k_range=(3, 21)):
    """Does k=9 emerge naturally?"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from scipy.optimize import linear_sum_assignment

    sim = cosine_sim(verb_embs, op_embs)
    supervised = np.argmax(sim, axis=1)

    # K-means at k=9
    km = KMeans(n_clusters=9, n_init=20, random_state=42)
    km_labels = km.fit_predict(verb_embs)

    ari = adjusted_rand_score(supervised, km_labels)
    nmi = normalized_mutual_info_score(supervised, km_labels)
    n_sample = min(5000, len(verb_embs))
    sil = silhouette_score(verb_embs, km_labels, metric="cosine", sample_size=n_sample)

    # Hungarian matching
    conf = np.zeros((9, 9), dtype=int)
    for sl, kl in zip(supervised, km_labels):
        conf[sl, kl] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    matched_acc = conf[row_ind, col_ind].sum() / len(verb_names)

    matching = []
    for r, c in zip(row_ind, col_ind):
        overlap = conf[r, c]
        total = int((supervised == r).sum())
        matching.append({
            "operator": HELIX_ORDER[r],
            "cluster": int(c),
            "overlap": int(overlap),
            "total": total,
            "pct": float(overlap / total) if total > 0 else 0,
        })

    # K sweep
    sweep = []
    for k in range(k_range[0], k_range[1]):
        km_k = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels_k = km_k.fit_predict(verb_embs)
        sil_k = silhouette_score(verb_embs, labels_k, metric="cosine", sample_size=n_sample)
        sweep.append({"k": k, "silhouette": float(sil_k), "inertia": float(km_k.inertia_)})

    best_k = max(sweep, key=lambda x: x["silhouette"])["k"]

    return {
        "kmeans_k9": {
            "ari": float(ari),
            "nmi": float(nmi),
            "silhouette": float(sil),
            "matched_accuracy": float(matched_acc),
            "matching": matching,
        },
        "sweep": sweep,
        "best_k": best_k,
    }


# ═══════════════════════════════════════════════════════════════
# TEST 5: BOUNDARY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def test_boundaries(sim, verb_names):
    """Which verbs sit between two operators?"""
    top2_idx = np.argsort(-sim, axis=1)[:, :2]
    top2_sim = np.array([[sim[i, top2_idx[i, 0]], sim[i, top2_idx[i, 1]]] for i in range(len(verb_names))])
    ambiguity = top2_sim[:, 0] - top2_sim[:, 1]

    most_ambig = np.argsort(ambiguity)
    boundary_verbs = []
    for i in most_ambig[:30]:
        op1 = HELIX_ORDER[top2_idx[i, 0]]
        op2 = HELIX_ORDER[top2_idx[i, 1]]
        boundary_verbs.append({
            "verb": verb_names[i],
            "op1": op1, "sim1": float(top2_sim[i, 0]),
            "op2": op2, "sim2": float(top2_sim[i, 1]),
            "gap": float(ambiguity[i]),
        })

    boundaries = Counter()
    for i in most_ambig[:200]:
        a, b = sorted([HELIX_ORDER[top2_idx[i, 0]], HELIX_ORDER[top2_idx[i, 1]]])
        boundaries[f"{a}-{b}"] += 1

    return {
        "stats": {"mean_gap": float(np.mean(ambiguity)), "median_gap": float(np.median(ambiguity))},
        "most_ambiguous": boundary_verbs,
        "busiest_boundaries": dict(boundaries.most_common(15)),
    }


# ═══════════════════════════════════════════════════════════════
# TOPOLOGY: CENTROIDS, TERRITORY, EVENT HORIZONS
# ═══════════════════════════════════════════════════════════════

def compute_topology(records, embs):
    """Compute operator centroids, territories, and event horizons.

    Args:
        records: list of dicts with 'operator', 'verb', 'confidence', 'reason', etc.
        embs: numpy array of embeddings (one per record)

    Returns:
        dict with operator topology results
    """
    # Compute centroids
    centroids = np.zeros((9, embs.shape[1]))
    for i, op in enumerate(HELIX_ORDER):
        mask = np.array([r["operator"] == op for r in records])
        if mask.sum() > 0:
            centroids[i] = embs[mask].mean(axis=0)

    # Centroid similarity
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    centroid_sims = c_norm @ c_norm.T

    results = []
    for op_name in HELIX_ORDER:
        op_idx = HELIX_ORDER.index(op_name)
        op_mask = np.array([r["operator"] == op_name for r in records])
        op_indices = np.where(op_mask)[0]

        if op_mask.sum() < 3:
            results.append(None)
            continue

        op_embs = embs[op_indices]
        op_records = [records[i] for i in op_indices]
        centroid = centroids[op_idx]

        self_sims = cosine_sim_vec(centroid, op_embs)

        other_centroids = np.delete(centroids, op_idx, axis=0)
        other_names = [h for i, h in enumerate(HELIX_ORDER) if i != op_idx]

        max_other_sims = np.zeros(len(op_embs))
        nearest_other = []
        for vi in range(len(op_embs)):
            sims_to_others = cosine_sim_vec(op_embs[vi], other_centroids)
            best_idx = np.argmax(sims_to_others)
            max_other_sims[vi] = sims_to_others[best_idx]
            nearest_other.append(other_names[best_idx])

        margin = self_sims - max_other_sims

        # Prototypes
        centroid_order = np.argsort(-self_sims)
        prototypes = []
        for idx in centroid_order[:10]:
            r = op_records[idx]
            prototypes.append({
                "verb": r["verb"],
                "margin": float(margin[idx]),
                "reason": r.get("reason", ""),
            })

        # Event horizon
        contested_order = np.argsort(np.abs(margin))
        horizon = []
        for idx in contested_order[:15]:
            r = op_records[idx]
            horizon.append({
                "verb": r["verb"],
                "neighbor": nearest_other[idx],
                "margin": float(margin[idx]),
            })

        # Territory
        core = int((margin > 0.10).sum())
        settled = int(((margin > 0.02) & (margin <= 0.10)).sum())
        contested = int(((margin >= -0.02) & (margin <= 0.02)).sum())
        foreign = int((margin < -0.02).sum())

        # Neighbor sims
        neighbors = []
        for i, other_op in enumerate(HELIX_ORDER):
            if other_op == op_name:
                continue
            neighbors.append({
                "op": other_op,
                "sim": float(centroid_sims[op_idx, HELIX_ORDER.index(other_op)]),
            })
        neighbors.sort(key=lambda x: -x["sim"])

        # Scale distribution
        scales = dict(Counter(r.get("scale", "?") for r in op_records).most_common())

        results.append({
            "operator": op_name,
            "total": len(op_records),
            "territory": {"core": core, "settled": settled, "contested": contested, "foreign": foreign},
            "margin_mean": float(np.mean(margin)),
            "prototypes": prototypes[:5],
            "horizon": horizon[:8],
            "neighbors": neighbors[:4],
            "scales": scales,
        })

    return {"operators": results, "centroid_sims": centroid_sims.tolist()}


# ═══════════════════════════════════════════════════════════════
# FALSIFICATION: RANDOM BASELINE + K-TESTING
# ═══════════════════════════════════════════════════════════════

def compute_z_score(embeddings, labels, n_permutations=100):
    """Compute z-score of mean inter-centroid cosine distance vs random."""
    from sklearn.metrics.pairwise import cosine_similarity

    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)

    centroids = []
    for lab in unique_labels:
        mask = labels == lab
        centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)

    sim_matrix = cosine_similarity(centroids)
    n_ops = len(unique_labels)
    real_mean_sim = 0
    count = 0
    for i in range(n_ops):
        for j in range(i + 1, n_ops):
            real_mean_sim += sim_matrix[i, j]
            count += 1
    real_mean_sim /= count

    rng = np.random.RandomState(42)
    random_sims = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        perm_centroids = []
        for lab in unique_labels:
            mask = perm_labels == lab
            perm_centroids.append(embeddings[mask].mean(axis=0))
        perm_centroids = np.array(perm_centroids)
        perm_sim = cosine_similarity(perm_centroids)
        s = sum(perm_sim[i, j] for i in range(n_ops) for j in range(i + 1, n_ops)) / count
        random_sims.append(s)

    random_mean = np.mean(random_sims)
    random_std = np.std(random_sims)
    z = (real_mean_sim - random_mean) / random_std if random_std > 0 else 0

    return {
        "real_mean_sim": float(real_mean_sim),
        "random_mean": float(random_mean),
        "random_std": float(random_std),
        "z_score": float(z),
    }


def test_falsification(embeddings, labels, n_random_taxonomies=20):
    """Run falsification tests: random baseline + k-configuration.

    Args:
        embeddings: numpy array (n_verbs, dim)
        labels: numpy array of operator indices (0-8)

    Returns:
        dict with test results
    """
    # Test A: EO z-score
    eo_z = compute_z_score(embeddings, labels, n_permutations=100)

    # Random taxonomy z-scores
    rng = np.random.RandomState(123)
    random_z_scores = []
    for _ in range(n_random_taxonomies):
        shuffled = rng.permutation(labels)
        z_result = compute_z_score(embeddings, shuffled, n_permutations=30)
        random_z_scores.append(z_result["z_score"])

    eo_vs_random = abs(eo_z["z_score"]) / max(abs(np.mean(random_z_scores)), 0.1)

    return {
        "eo_z_score": eo_z["z_score"],
        "eo_details": eo_z,
        "random_z_mean": float(np.mean(random_z_scores)),
        "random_z_std": float(np.std(random_z_scores)),
        "random_z_range": [float(min(random_z_scores)), float(max(random_z_scores))],
        "random_z_scores": random_z_scores,
        "eo_vs_random_ratio": float(eo_vs_random),
        "verdict": "EO captures real structure" if eo_vs_random > 5 else "WARNING: may be artifactual",
    }


# ═══════════════════════════════════════════════════════════════
# PCA PROJECTION (for visualization data)
# ═══════════════════════════════════════════════════════════════

def compute_pca(verb_embs, op_embs, nearest_idx):
    """Compute PCA projection for Plotly visualization."""
    from sklearn.decomposition import PCA

    all_embs = np.vstack([verb_embs, op_embs])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embs)
    v2d = all_2d[: len(verb_embs)]
    o2d = all_2d[len(verb_embs) :]

    return {
        "verb_coords": v2d.tolist(),
        "op_coords": o2d.tolist(),
        "variance_explained": [float(v) for v in pca.explained_variance_ratio_],
        "nearest_idx": nearest_idx.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# CROSS-LINGUISTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_crossling(all_langs_data):
    """Analyze cross-linguistic operator distributions.

    Args:
        all_langs_data: dict of {lang_name: {classifications: [...], family, era, region, morph_type}}

    Returns:
        dict with distribution analysis
    """
    results = {}
    for lang, data in all_langs_data.items():
        cls = data.get("classifications", [])
        op_counts = Counter()
        for c in cls:
            op = c.get("operator", "").upper().strip()
            if op in HELIX_ORDER:
                op_counts[op] += 1

        total = sum(op_counts.values())
        if total == 0:
            continue

        op_pcts = {op: op_counts[op] / total * 100 for op in HELIX_ORDER}

        triad_pcts = {}
        for tname, tinfo in TRIADS.items():
            triad_pcts[tname] = sum(op_pcts.get(op, 0) for op in tinfo["operators"])

        results[lang] = {
            "family": data.get("family", ""),
            "era": data.get("era", ""),
            "region": data.get("region", ""),
            "morph_type": data.get("morph_type", ""),
            "n_classified": total,
            "n_verbs": len(data.get("verbs", {})),
            "op_counts": dict(op_counts),
            "op_pcts": op_pcts,
            "triad_pcts": triad_pcts,
        }

    return {"n_languages": len(results), "languages": results}
