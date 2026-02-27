"""
Step 8: Empirical Definitions, Centroids, and Event Horizons.

For each operator, extracts:
  - CENTROID: the geometric center and its nearest verbs (prototypes)
  - EVENT HORIZON: the boundary where this operator's pull equals a neighbor's
  - EMPIRICAL DEFINITION: common patterns in high-confidence reasoning texts
  - TERRITORY MAP: core (deep inside), settled (clearly this operator), 
    contested (near boundary), and foreign (closer to another centroid)

Run: python scripts/08_topology.py
Requires: data/llm_classifications.json, data/reembed_combined.npz (or definition.npz)
Outputs: output/topology_report.json, output/topology_summary.txt
"""

import json, os, sys, re
import numpy as np
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
SCRIPTS_DIR = os.path.dirname(__file__)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

sys.path.insert(0, SCRIPTS_DIR)
from operator_definitions import OPERATORS, HELIX_ORDER

def log(msg):
    print(msg, flush=True)


def load_data():
    log("Loading data...")
    with open(os.path.join(DATA_DIR, "llm_classifications.json")) as f:
        cls_data = json.load(f)
    cls = cls_data["classifications"]

    # Try combined first, then definition, then signature
    embs = None
    emb_name = None
    for name in ["combined", "definition", "signature"]:
        path = os.path.join(DATA_DIR, f"reembed_{name}.npz")
        if os.path.exists(path):
            data = np.load(path)
            embs = data["embeddings"]
            emb_name = name
            break

    if embs is None:
        log("ERROR: No embeddings found. Run step 6 first.")
        sys.exit(1)

    # Filter to valid operators
    valid = set(HELIX_ORDER)
    records = [c for c in cls if c.get("operator") in valid]

    if len(records) != embs.shape[0]:
        log(f"  WARNING: {len(records)} records but {embs.shape[0]} embeddings. Truncating to min.")
        n = min(len(records), embs.shape[0])
        records = records[:n]
        embs = embs[:n]

    log(f"  {len(records)} verbs, {embs.shape[1]}d embeddings ({emb_name})")
    return records, embs


def cosine_sim_vec(a, B):
    a_n = a / (np.linalg.norm(a) + 1e-10)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return B_n @ a_n


def extract_common_patterns(reasons):
    """Find recurring structural phrases in reasoning texts."""
    # Extract input→output patterns
    input_patterns = []
    output_patterns = []
    change_words = Counter()

    for r in reasons:
        r_lower = r.lower()
        # Extract "Input: X" patterns
        inp = re.search(r'input:\s*(.+?)(?:\.|output:|;|$)', r_lower)
        if inp:
            input_patterns.append(inp.group(1).strip())
        out = re.search(r'output:\s*(.+?)(?:\.|;|change:|$)', r_lower)
        if out:
            output_patterns.append(out.group(1).strip())

        # Count transformation-describing words
        for word in re.findall(r'\b\w+\b', r_lower):
            if word in ('the', 'a', 'an', 'is', 'of', 'to', 'in', 'with', 'and',
                        'or', 'that', 'this', 'it', 'its', 'for', 'as', 'from',
                        'input', 'output', 'state', 'verb', 'new', 'same'):
                continue
            change_words[word] += 1

    return input_patterns, output_patterns, change_words


def analyze_operator(op_name, records, embs, all_centroids, centroid_sims):
    op_idx = HELIX_ORDER.index(op_name)
    op_mask = np.array([r["operator"] == op_name for r in records])
    op_indices = np.where(op_mask)[0]
    op_embs = embs[op_indices]
    op_records = [records[i] for i in op_indices]

    if len(op_records) < 3:
        return None

    centroid = all_centroids[op_idx]

    # ── Similarity of each verb to its own centroid and all others ──
    self_sims = cosine_sim_vec(centroid, op_embs)

    other_centroids = np.delete(all_centroids, op_idx, axis=0)
    other_names = [h for i, h in enumerate(HELIX_ORDER) if i != op_idx]

    # For each verb: max similarity to any OTHER centroid
    max_other_sims = np.zeros(len(op_embs))
    nearest_other = []
    for vi in range(len(op_embs)):
        sims_to_others = cosine_sim_vec(op_embs[vi], other_centroids)
        best_other_idx = np.argmax(sims_to_others)
        max_other_sims[vi] = sims_to_others[best_other_idx]
        nearest_other.append(other_names[best_other_idx])

    # ── Territory zones ──
    margin = self_sims - max_other_sims
    # Core: margin > 0.1 (deep inside)
    # Settled: margin > 0.02 (clearly this operator)
    # Contested: margin between -0.02 and 0.02 (on the boundary)
    # Foreign: margin < -0.02 (closer to another centroid)
    core_mask = margin > 0.10
    settled_mask = (margin > 0.02) & ~core_mask
    contested_mask = (margin >= -0.02) & (margin <= 0.02)
    foreign_mask = margin < -0.02

    # ── Prototypes (nearest to centroid) ──
    centroid_order = np.argsort(-self_sims)
    prototypes = []
    for idx in centroid_order[:10]:
        r = op_records[idx]
        prototypes.append({
            "verb": r["verb"],
            "self_sim": float(self_sims[idx]),
            "margin": float(margin[idx]),
            "scale": r.get("scale", ""),
            "reason": r.get("reason", ""),
            "confidence": r.get("confidence", ""),
        })

    # ── Event horizon verbs (contested zone) ──
    contested_order = np.argsort(np.abs(margin))
    horizon_verbs = []
    for idx in contested_order[:20]:
        r = op_records[idx]
        horizon_verbs.append({
            "verb": r["verb"],
            "self_sim": float(self_sims[idx]),
            "nearest_other_op": nearest_other[idx],
            "other_sim": float(max_other_sims[idx]),
            "margin": float(margin[idx]),
            "scale": r.get("scale", ""),
            "reason": r.get("reason", ""),
            "alternative": r.get("alternative"),
        })

    # ── Foreign verbs (misclassified or compositional) ──
    foreign_order = np.argsort(margin)
    foreign_verbs = []
    for idx in foreign_order[:10]:
        if margin[idx] >= 0:
            break
        r = op_records[idx]
        foreign_verbs.append({
            "verb": r["verb"],
            "self_sim": float(self_sims[idx]),
            "nearest_other_op": nearest_other[idx],
            "other_sim": float(max_other_sims[idx]),
            "margin": float(margin[idx]),
            "scale": r.get("scale", ""),
            "reason": r.get("reason", ""),
        })

    # ── Empirical definition extraction ──
    high_conf = [r for r in op_records if r.get("confidence") == "high"]
    reasons = [r.get("reason", "") for r in high_conf]
    input_patterns, output_patterns, change_words = extract_common_patterns(reasons)

    # Top transformation words (excluding common ones)
    top_words = change_words.most_common(20)

    # ── Nearest neighboring operators ──
    neighbor_sims = []
    for i, other_op in enumerate(HELIX_ORDER):
        if other_op == op_name:
            continue
        sim = float(centroid_sims[op_idx, HELIX_ORDER.index(other_op)])
        neighbor_sims.append({"operator": other_op, "centroid_similarity": sim})
    neighbor_sims.sort(key=lambda x: -x["centroid_similarity"])

    # ── Summary stats ──
    result = {
        "operator": op_name,
        "symbol": OPERATORS[op_name]["symbol"],
        "total_verbs": len(op_records),
        "territory": {
            "core": int(core_mask.sum()),
            "settled": int(settled_mask.sum()),
            "contested": int(contested_mask.sum()),
            "foreign": int(foreign_mask.sum()),
        },
        "margin_stats": {
            "mean": float(np.mean(margin)),
            "median": float(np.median(margin)),
            "std": float(np.std(margin)),
            "min": float(np.min(margin)),
            "max": float(np.max(margin)),
            "pct_positive": float((margin > 0).sum() / len(margin) * 100),
        },
        "prototypes": prototypes,
        "horizon_verbs": horizon_verbs,
        "foreign_verbs": foreign_verbs,
        "nearest_neighbors": neighbor_sims[:4],
        "empirical_definition": {
            "top_transformation_words": top_words,
            "sample_input_patterns": input_patterns[:8],
            "sample_output_patterns": output_patterns[:8],
        },
        "scales": dict(Counter(r.get("scale", "?") for r in op_records).most_common()),
    }

    return result


def generate_summary(results):
    """Generate human-readable topology summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("EO OPERATOR TOPOLOGY: CENTROIDS, HORIZONS, AND EMPIRICAL DEFINITIONS")
    lines.append("=" * 70)

    for res in results:
        if res is None:
            continue
        op = res["operator"]
        sym = res["symbol"]
        terr = res["territory"]
        margin = res["margin_stats"]

        lines.append(f"\n{'─'*70}")
        lines.append(f"  {op} ({sym}) — {res['total_verbs']} verbs")
        lines.append(f"{'─'*70}")

        # Territory
        total = res["total_verbs"]
        lines.append(f"\n  TERRITORY:")
        lines.append(f"    Core (margin > 0.10):     {terr['core']:4d} ({terr['core']/total*100:.1f}%)")
        lines.append(f"    Settled (0.02 < m < 0.10): {terr['settled']:4d} ({terr['settled']/total*100:.1f}%)")
        lines.append(f"    Contested (-0.02 < m < 0.02): {terr['contested']:4d} ({terr['contested']/total*100:.1f}%)")
        lines.append(f"    Foreign (margin < -0.02):  {terr['foreign']:4d} ({terr['foreign']/total*100:.1f}%)")
        lines.append(f"    Mean margin: {margin['mean']:.4f}  |  {margin['pct_positive']:.1f}% closer to own centroid")

        # Prototypes
        lines.append(f"\n  PROTOTYPES (centroid-nearest, the 'purest' {op} verbs):")
        for p in res["prototypes"][:5]:
            lines.append(f"    {p['verb']:25s} [margin={p['margin']:.3f}, {p['scale']}]")
            lines.append(f"      → {p['reason'][:80]}")

        # Empirical definition
        emp = res["empirical_definition"]
        words = [w for w, _ in emp["top_transformation_words"][:10]]
        lines.append(f"\n  EMPIRICAL DEFINITION (from high-confidence reasoning):")
        lines.append(f"    Key words: {', '.join(words)}")
        if emp["sample_input_patterns"]:
            lines.append(f"    Input pattern:  \"{emp['sample_input_patterns'][0][:70]}\"")
        if emp["sample_output_patterns"]:
            lines.append(f"    Output pattern: \"{emp['sample_output_patterns'][0][:70]}\"")

        # Event horizon
        lines.append(f"\n  EVENT HORIZON (boundary verbs, margin ≈ 0):")
        for h in res["horizon_verbs"][:5]:
            alt = h.get("alternative") or ""
            lines.append(f"    {h['verb']:25s} ↔ {h['nearest_other_op']:3s} [margin={h['margin']:+.3f}] alt:{alt}")
            lines.append(f"      → {h['reason'][:70]}")

        # Nearest operators
        lines.append(f"\n  NEAREST OPERATORS:")
        for n in res["nearest_neighbors"][:3]:
            lines.append(f"    {op} ↔ {n['operator']:3s}: centroid sim = {n['centroid_similarity']:.4f}")

        # Foreign verbs
        if res["foreign_verbs"]:
            lines.append(f"\n  FOREIGN TERRITORY ({len(res['foreign_verbs'])} verbs closer to another centroid):")
            for f in res["foreign_verbs"][:3]:
                lines.append(f"    {f['verb']:25s} → actually near {f['nearest_other_op']:3s} [margin={f['margin']:+.3f}]")

    return "\n".join(lines)


def visualize_topology(results, records, embs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not available."); return

    # Territory composition bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    ops = [r["operator"] for r in results if r]
    core = [r["territory"]["core"] for r in results if r]
    settled = [r["territory"]["settled"] for r in results if r]
    contested = [r["territory"]["contested"] for r in results if r]
    foreign = [r["territory"]["foreign"] for r in results if r]

    x = np.arange(len(ops))
    w = 0.6
    ax.bar(x, core, w, label="Core", color="#2ecc71")
    ax.bar(x, settled, w, bottom=core, label="Settled", color="#3498db")
    ax.bar(x, contested, w, bottom=np.array(core)+np.array(settled), label="Contested", color="#f39c12")
    ax.bar(x, foreign, w, bottom=np.array(core)+np.array(settled)+np.array(contested), label="Foreign", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(ops, fontsize=12)
    ax.set_ylabel("Number of verbs")
    ax.set_title("Operator Territory Composition\n(green = deep core, red = closer to another operator's centroid)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "topology_territory.png"), dpi=150)
    plt.close()
    log("  ✓ topology_territory.png")

    # Margin distribution per operator
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for i, op in enumerate(HELIX_ORDER):
        ax = axes[i // 3][i % 3]
        op_mask = np.array([r["operator"] == op for r in records])
        if op_mask.sum() == 0:
            continue

        res = results[i]
        if res is None:
            continue

        # Recompute margins for histogram
        op_idx = HELIX_ORDER.index(op)
        op_embs_local = embs[op_mask]
        self_sims = cosine_sim_vec(results[i]["prototypes"][0]["self_sim"] if results[i]["prototypes"] else 0, op_embs_local) if len(op_embs_local) > 0 else []

        margins = []
        for r in results[i]["prototypes"] + results[i]["horizon_verbs"] + results[i]["foreign_verbs"]:
            margins.append(r["margin"])

        if len(margins) > 5:
            ax.hist(margins, bins=20, color="#3498db", alpha=0.7, edgecolor="black")
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Event horizon")
        ax.set_title(f"{op} ({res['total_verbs']} verbs)", fontsize=11)
        ax.set_xlabel("Margin (self - nearest other)")

    plt.suptitle("Margin Distributions per Operator\n(right of red = own territory, left = foreign)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "topology_margins.png"), dpi=150)
    plt.close()
    log("  ✓ topology_margins.png")

    # Centroid similarity heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sims = np.zeros((9, 9))
    for i, ri in enumerate(results):
        if ri is None: continue
        for n in ri["nearest_neighbors"]:
            j = HELIX_ORDER.index(n["operator"])
            sims[i, j] = n["centroid_similarity"]
            sims[j, i] = n["centroid_similarity"]
    np.fill_diagonal(sims, 1.0)

    im = ax.imshow(sims, cmap="RdYlGn_r", vmin=0.3, vmax=0.8)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(HELIX_ORDER, fontsize=11)
    ax.set_yticklabels(HELIX_ORDER, fontsize=11)
    for i in range(9):
        for j in range(9):
            if sims[i, j] > 0:
                ax.text(j, i, f"{sims[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Centroid cosine similarity")
    ax.set_title("Operator Centroid Similarity Matrix")

    # Draw triad boxes
    for start in [0, 3, 6]:
        rect = plt.Rectangle((start-0.5, start-0.5), 3, 3, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "topology_centroid_heatmap.png"), dpi=150)
    plt.close()
    log("  ✓ topology_centroid_heatmap.png")


def run():
    log("=" * 60)
    log("STEP 8: OPERATOR TOPOLOGY")
    log("=" * 60)

    records, embs = load_data()

    # Compute all centroids
    log("\nComputing centroids...")
    centroids = np.zeros((9, embs.shape[1]))
    for i, op in enumerate(HELIX_ORDER):
        mask = np.array([r["operator"] == op for r in records])
        if mask.sum() > 0:
            centroids[i] = embs[mask].mean(axis=0)
            log(f"  {op}: {mask.sum()} verbs")

    # Centroid similarity matrix
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    centroid_sims = c_norm @ c_norm.T

    # Analyze each operator
    results = []
    for op in HELIX_ORDER:
        log(f"\n{'='*60}")
        log(f"  Analyzing {op}...")
        res = analyze_operator(op, records, embs, centroids, centroid_sims)
        results.append(res)

        if res:
            terr = res["territory"]
            log(f"  Territory: {terr['core']} core, {terr['settled']} settled, "
                f"{terr['contested']} contested, {terr['foreign']} foreign")
            log(f"  Mean margin: {res['margin_stats']['mean']:.4f} "
                f"({res['margin_stats']['pct_positive']:.1f}% in own territory)")
            log(f"  Prototypes: {', '.join(p['verb'] for p in res['prototypes'][:5])}")
            log(f"  Nearest neighbor: {res['nearest_neighbors'][0]['operator']} "
                f"(sim={res['nearest_neighbors'][0]['centroid_similarity']:.4f})")

    # Generate summary
    summary = generate_summary(results)
    summary_path = os.path.join(OUTPUT_DIR, "topology_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    log(f"\n  ✓ Saved {summary_path}")
    print(summary)

    # Save full report
    report_path = os.path.join(OUTPUT_DIR, "topology_report.json")
    with open(report_path, "w") as f:
        json.dump({"operators": results}, f, indent=2, default=str)
    log(f"  ✓ Saved {report_path}")

    # Visualize
    visualize_topology(results, records, embs)

    log(f"\n{'='*60}")
    log(f"STEP 8 COMPLETE")
    log(f"{'='*60}")


if __name__ == "__main__":
    run()
