"""
Step 7: Deep Dive Explorer — Map operator clusters in embedding space.

For each operator, visualize its verbs in 2D, identify sub-clusters,
show nearest neighbors, and export rich content for analysis.

Run: python scripts/07_explore.py [--operator REC] [--all]
     python scripts/07_explore.py --operator REC
     python scripts/07_explore.py --all

Requires: data/llm_classifications.json, data/reembed_*.npz (from step 6)
Falls back to generating embeddings if needed.
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


def load_all():
    log("Loading data...")

    # Classifications
    with open(os.path.join(DATA_DIR, "llm_classifications.json")) as f:
        cls_data = json.load(f)
    cls = cls_data["classifications"]
    log(f"  {len(cls)} classifications")

    # Try to load definition embeddings from step 6
    emb_path = os.path.join(DATA_DIR, "reembed_definition.npz")
    sig_path = os.path.join(DATA_DIR, "reembed_signature.npz")
    comb_path = os.path.join(DATA_DIR, "reembed_combined.npz")

    embs = {}
    for name, path in [("definition", emb_path), ("signature", sig_path), ("combined", comb_path)]:
        if os.path.exists(path):
            data = np.load(path)
            embs[name] = data["embeddings"]
            log(f"  ✓ {name} embeddings: {embs[name].shape}")

    if not embs:
        log("  No embeddings found. Run step 6 first.")
        sys.exit(1)

    # Build aligned arrays
    valid_ops = set(HELIX_ORDER)
    records = []
    for c in cls:
        if c.get("operator") in valid_ops:
            records.append(c)

    log(f"  {len(records)} valid records")
    return records, embs


def cosine_sim_vec(a, B):
    """Similarity of one vector against a matrix."""
    a_n = a / (np.linalg.norm(a) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return B_n @ a_n


def explore_operator(op_name, records, embs, n_subclusters=5, n_neighbors=10):
    log(f"\n{'='*70}")
    log(f"  DEEP DIVE: {op_name} ({OPERATORS[op_name]['symbol']}) — {OPERATORS[op_name]['short_def'][:60]}")
    log(f"{'='*70}")

    # Filter to this operator
    op_indices = [i for i, r in enumerate(records) if r["operator"] == op_name]
    op_records = [records[i] for i in op_indices]
    log(f"\n  Total verbs: {len(op_records)}")

    if len(op_records) < 5:
        log(f"  Too few verbs for analysis.")
        return None

    # Use best available embedding
    emb_key = "combined" if "combined" in embs else "definition" if "definition" in embs else list(embs.keys())[0]
    all_embs = embs[emb_key]
    op_embs = all_embs[op_indices]
    log(f"  Using {emb_key} embeddings: {op_embs.shape}")

    # ── Scale breakdown ──
    scales = Counter(r.get("scale", "?") for r in op_records)
    log(f"\n  ── Scale breakdown ──")
    for s, c in scales.most_common():
        pct = c / len(op_records) * 100
        log(f"    {s:15s}: {c:4d} ({pct:.1f}%)")

    # ── Confidence breakdown ──
    confs = Counter(r.get("confidence", "?") for r in op_records)
    log(f"\n  ── Confidence ──")
    for c, n in confs.most_common():
        log(f"    {c:8s}: {n:4d}")

    # ── Alternative operators (where does it border?) ──
    alts = Counter()
    for r in op_records:
        alt = r.get("alternative")
        if alt:
            alts[alt] += 1
    if alts:
        log(f"\n  ── Borders with (alternative operators) ──")
        for alt, c in alts.most_common():
            log(f"    {op_name} ↔ {alt:3s}: {c:4d} verbs")

    # ── Sub-clustering ──
    from sklearn.cluster import KMeans

    k = min(n_subclusters, len(op_records) // 5, 8)
    if k < 2:
        k = 2

    log(f"\n  ── Sub-clustering (k={k}) ──")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    sub_labels = km.fit_predict(op_embs)

    subclusters = []
    for si in range(k):
        mask = sub_labels == si
        sub_records = [op_records[i] for i, m in enumerate(mask) if m]
        sub_embs = op_embs[mask]

        # Find verbs closest to sub-centroid
        centroid = km.cluster_centers_[si]
        sims = cosine_sim_vec(centroid, sub_embs)
        top_idx = np.argsort(-sims)[:15]

        sub_scales = Counter(r.get("scale", "?") for r in sub_records)
        dominant_scale = sub_scales.most_common(1)[0][0]

        # Get the high-confidence core
        core = [r for r in sub_records if r.get("confidence") == "high"]
        core_verbs = [r["verb"] for r in core[:10]]

        # Sample reasons
        reasons = [r.get("reason", "")[:80] for r in sub_records[:3]]

        log(f"\n    Sub-cluster {si+1} ({mask.sum()} verbs, dominant: {dominant_scale})")
        log(f"    Core: {', '.join(core_verbs[:8])}")
        log(f"    Scales: {dict(sub_scales.most_common())}")
        log(f"    Nearest to centroid:")
        for idx in top_idx[:8]:
            r = sub_records[idx]
            log(f"      {r['verb']:25s} [{r.get('scale','?')}] {r.get('reason','')[:55]}...")
        log(f"    Sample reasoning:")
        for reason in reasons:
            log(f"      → {reason}...")

        subclusters.append({
            "id": si,
            "size": int(mask.sum()),
            "dominant_scale": dominant_scale,
            "scales": dict(sub_scales.most_common()),
            "core_verbs": core_verbs,
            "nearest_to_centroid": [
                {"verb": sub_records[idx]["verb"],
                 "scale": sub_records[idx].get("scale", ""),
                 "reason": sub_records[idx].get("reason", ""),
                 "confidence": sub_records[idx].get("confidence", "")}
                for idx in top_idx[:15]
            ],
        })

    # ── Nearest neighbors from OTHER operators ──
    log(f"\n  ── Nearest neighbors from other operators ──")
    op_centroid = np.mean(op_embs, axis=0)
    other_indices = [i for i, r in enumerate(records) if r["operator"] != op_name]
    other_embs = all_embs[other_indices]
    other_records = [records[i] for i in other_indices]

    sims = cosine_sim_vec(op_centroid, other_embs)
    nearest = np.argsort(-sims)[:20]

    log(f"  Verbs from other operators closest to {op_name}'s center:")
    boundary_verbs = []
    for idx in nearest:
        r = other_records[idx]
        s = sims[idx]
        log(f"    {r['verb']:25s} → {r['operator']:3s} (sim={s:.4f}) [{r.get('scale','?')}] {r.get('reason','')[:50]}...")
        boundary_verbs.append({
            "verb": r["verb"], "operator": r["operator"],
            "similarity": float(s), "scale": r.get("scale", ""),
            "reason": r.get("reason", ""),
        })

    # ── Farthest verbs within the operator ──
    dists = cosine_sim_vec(op_centroid, op_embs)
    farthest = np.argsort(dists)[:10]
    log(f"\n  ── Most peripheral {op_name} verbs (farthest from centroid) ──")
    periphery = []
    for idx in farthest:
        r = op_records[idx]
        s = dists[idx]
        alt = r.get("alternative") or ""
        log(f"    {r['verb']:25s} (sim={s:.4f}) [{r.get('scale','?')}] alt:{alt:3s} {r.get('reason','')[:50]}...")
        periphery.append({
            "verb": r["verb"], "similarity": float(s),
            "alternative": alt, "scale": r.get("scale", ""),
            "reason": r.get("reason", ""),
        })

    # ── Full verb list ──
    all_verbs = []
    for r in sorted(op_records, key=lambda x: x["verb"]):
        all_verbs.append({
            "verb": r["verb"],
            "confidence": r.get("confidence", ""),
            "scale": r.get("scale", ""),
            "reason": r.get("reason", ""),
            "alternative": r.get("alternative"),
        })

    report = {
        "operator": op_name,
        "symbol": OPERATORS[op_name]["symbol"],
        "total_verbs": len(op_records),
        "scales": dict(scales.most_common()),
        "confidence": dict(confs.most_common()),
        "borders": dict(alts.most_common()),
        "subclusters": subclusters,
        "boundary_verbs_from_other_ops": boundary_verbs,
        "peripheral_verbs": periphery,
        "all_verbs": all_verbs,
    }

    outpath = os.path.join(OUTPUT_DIR, f"explore_{op_name}.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved {outpath}")

    return report


# ─── Visualization ────────────────────────────────────────────────

def visualize_operator(op_name, records, embs, report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        log("  matplotlib not available."); return

    emb_key = "combined" if "combined" in embs else "definition" if "definition" in embs else list(embs.keys())[0]
    all_embs = embs[emb_key]

    # PCA on all verbs, highlight this operator
    log(f"  Generating {op_name} visualization...")
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embs)

    op_mask = np.array([r["operator"] == op_name for r in records])
    colors_map = {op: f"C{i}" for i, op in enumerate(HELIX_ORDER)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: all operators, this one highlighted
    ax = axes[0]
    for i, op in enumerate(HELIX_ORDER):
        mask = np.array([r["operator"] == op for r in records])
        alpha = 0.6 if op == op_name else 0.05
        size = 15 if op == op_name else 3
        ax.scatter(all_2d[mask, 0], all_2d[mask, 1], s=size, alpha=alpha, label=op, c=f"C{i}")
    ax.legend(fontsize=9)
    ax.set_title(f"{op_name} in context (PCA)", fontsize=13)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # Right: just this operator, colored by sub-cluster
    ax = axes[1]
    op_2d = all_2d[op_mask]
    op_records_list = [r for r in records if r["operator"] == op_name]

    # Re-run sub-clustering for coloring
    from sklearn.cluster import KMeans
    k = min(5, len(op_records_list) // 5, 8)
    if k < 2: k = 2
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    op_embs_only = all_embs[op_mask]
    sub_labels = km.fit_predict(op_embs_only)

    for si in range(k):
        mask = sub_labels == si
        # Dominant scale for label
        sub_recs = [op_records_list[j] for j, m in enumerate(mask) if m]
        dom_scale = Counter(r.get("scale", "?") for r in sub_recs).most_common(1)[0][0]
        ax.scatter(op_2d[mask, 0], op_2d[mask, 1], s=20, alpha=0.6,
                   label=f"Sub-{si+1} ({mask.sum()}, {dom_scale})")

    # Label some verbs
    for i in range(0, len(op_2d), max(1, len(op_2d) // 25)):
        ax.annotate(op_records_list[i]["verb"], (op_2d[i, 0], op_2d[i, 1]),
                    fontsize=6, alpha=0.7)

    ax.legend(fontsize=9)
    ax.set_title(f"{op_name} sub-clusters", fontsize=13)

    plt.suptitle(f"Deep Dive: {op_name} ({OPERATORS[op_name]['symbol']}) — {OPERATORS[op_name]['short_def'][:50]}",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"explore_{op_name}.png"), dpi=150)
    plt.close()
    log(f"  ✓ explore_{op_name}.png")


# ─── Full map visualization ──────────────────────────────────────

def visualize_full_map(records, embs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        return

    log(f"\n  Generating full operator map...")
    emb_key = "combined" if "combined" in embs else "definition" if "definition" in embs else list(embs.keys())[0]
    all_embs = embs[emb_key]

    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embs)

    colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
              "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

    fig, ax = plt.subplots(figsize=(18, 14))
    for i, op in enumerate(HELIX_ORDER):
        mask = np.array([r["operator"] == op for r in records])
        ax.scatter(all_2d[mask, 0], all_2d[mask, 1], s=8, alpha=0.3, c=colors[i], label=f"{op} ({mask.sum()})")

        # Centroid
        centroid = all_2d[mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], s=200, c=colors[i], marker="*",
                   edgecolors="black", linewidths=1.5)
        ax.annotate(op, centroid, fontsize=14, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points")

    ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"Full EO Operator Map (PCA, {emb_key} embeddings)", fontsize=14)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "explore_full_map.png"), dpi=150)
    plt.close()
    log(f"  ✓ explore_full_map.png")

    # Scale-colored version
    fig, ax = plt.subplots(figsize=(18, 14))
    scale_colors = {"physical": "#e74c3c", "psychological": "#9b59b6",
                    "social": "#3498db", "informational": "#2ecc71"}
    for scale, color in scale_colors.items():
        mask = np.array([r.get("scale") == scale for r in records])
        ax.scatter(all_2d[mask, 0], all_2d[mask, 1], s=8, alpha=0.3, c=color, label=f"{scale} ({mask.sum()})")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title("Verb Space Colored by Scale", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "explore_scale_map.png"), dpi=150)
    plt.close()
    log(f"  ✓ explore_scale_map.png")


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", type=str, default=None, help="Explore one operator (e.g. REC)")
    parser.add_argument("--all", action="store_true", help="Explore all nine operators")
    args = parser.parse_args()

    if not args.operator and not args.all:
        args.all = True

    records, embs = load_all()

    operators_to_explore = HELIX_ORDER if args.all else [args.operator.upper()]

    for op in operators_to_explore:
        if op not in HELIX_ORDER:
            log(f"  Unknown operator: {op}")
            continue
        report = explore_operator(op, records, embs)
        if report:
            visualize_operator(op, records, embs, report)

    visualize_full_map(records, embs)

    log(f"\n{'='*60}")
    log(f"EXPLORATION COMPLETE")
    log(f"Reports: {OUTPUT_DIR}/explore_*.json")
    log(f"Visuals: {VIZ_DIR}/explore_*.png")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
