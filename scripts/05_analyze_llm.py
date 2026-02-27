"""
Step 5: Analyze LLM classifications.

Same tests as embedding analysis but on LLM-reasoned classifications:
  1. Completeness (distribution)
  2. Minimality (confidence-weighted)  
  3. Orthogonality (confusion matrix between operators)
  4. Boundary analysis (low-confidence and alternative-operator patterns)
  5. Comparison with embedding results

Run: python scripts/05_analyze_llm.py

Outputs:
  output/llm_completeness.json
  output/llm_confusion.json
  output/llm_boundaries.json
  output/llm_comparison.json
  output/visualizations/llm_*.png
"""

import json, os, sys, time
import numpy as np
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
VIZ_DIR = os.path.join(OUTPUT_DIR, "visualizations")
SCRIPTS_DIR = os.path.dirname(__file__)
os.makedirs(VIZ_DIR, exist_ok=True)

sys.path.insert(0, SCRIPTS_DIR)
from operator_definitions import OPERATORS, HELIX_ORDER, TRIADS, ROLES

def log(msg):
    print(msg, flush=True)


def load_classifications():
    path = os.path.join(DATA_DIR, "llm_classifications.json")
    if not os.path.exists(path):
        log(f"ERROR: {path} not found. Run 04_llm_classify.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    cls = data["classifications"]
    log(f"Loaded {len(cls)} classifications (backend: {data['metadata']['backend']})")
    return data, cls


# ═══════════════════════════════════════════════════════════════════
# TEST 1: COMPLETENESS & DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════

def test_completeness(cls):
    log("\n" + "="*60)
    log("TEST 1: COMPLETENESS & DISTRIBUTION")
    log("="*60)
    
    # Distribution by operator
    op_counts = Counter(c.get("operator", "UNKNOWN") for c in cls)
    total = len(cls)
    
    log(f"\n  ── Verb distribution across operators ──")
    max_c = max(op_counts.get(op, 0) for op in HELIX_ORDER) or 1
    for op in HELIX_ORDER:
        c = op_counts.get(op, 0)
        pct = c / total * 100
        bar = "█" * int(c / max_c * 30)
        log(f"    {op:3s}: {c:5d} ({pct:5.1f}%) {bar}")
    
    # Confidence by operator
    log(f"\n  ── Confidence by operator ──")
    for op in HELIX_ORDER:
        op_cls = [c for c in cls if c.get("operator") == op]
        if not op_cls:
            continue
        confs = Counter(c.get("confidence", "?").lower() for c in op_cls)
        high_pct = confs.get("high", 0) / len(op_cls) * 100
        med_pct = confs.get("medium", 0) / len(op_cls) * 100
        low_pct = confs.get("low", 0) / len(op_cls) * 100
        log(f"    {op:3s}: high={high_pct:5.1f}%  med={med_pct:5.1f}%  low={low_pct:5.1f}%  (n={len(op_cls)})")
    
    # Top verbs per operator (high confidence)
    log(f"\n  ── Top high-confidence verbs per operator ──")
    for op in HELIX_ORDER:
        high = [c for c in cls if c.get("operator") == op and c.get("confidence","").lower() == "high"]
        verbs = [c["verb"] for c in high[:12]]
        log(f"    {op:3s}: {', '.join(verbs)}")
    
    # Low confidence verbs (the interesting ones)
    low_conf = [c for c in cls if c.get("confidence","").lower() == "low"]
    log(f"\n  ── Low-confidence classifications ({len(low_conf)} total) ──")
    for c in low_conf[:15]:
        alt = c.get("alternative", "")
        reason = c.get("reason", "")[:70]
        log(f"    {c['verb']:25s} → {c['operator']:3s} (alt: {alt:3s})  {reason}")
    
    report = {
        "distribution": {op: op_counts.get(op, 0) for op in HELIX_ORDER},
        "total": total,
        "confidence_summary": {
            "high": sum(1 for c in cls if c.get("confidence","").lower() == "high"),
            "medium": sum(1 for c in cls if c.get("confidence","").lower() == "medium"),
            "low": sum(1 for c in cls if c.get("confidence","").lower() == "low"),
        },
    }
    
    with open(os.path.join(OUTPUT_DIR, "llm_completeness.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved llm_completeness.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# TEST 2: CONFUSION / ORTHOGONALITY
# ═══════════════════════════════════════════════════════════════════

def test_confusion(cls):
    log("\n" + "="*60)
    log("TEST 2: OPERATOR CONFUSION MATRIX")
    log("(from alternative operator assignments)")
    log("="*60)
    
    # Build confusion from alternatives
    confusion = defaultdict(lambda: defaultdict(int))
    for c in cls:
        op = c.get("operator", "")
        alt = c.get("alternative")
        if op in HELIX_ORDER and alt and alt.upper() in HELIX_ORDER:
            confusion[op][alt.upper()] += 1
    
    # Print matrix
    log(f"\n  ── Confusion matrix (primary → most common alternative) ──")
    log(f"  Primary  {'  '.join(f'{op:>5s}' for op in HELIX_ORDER)}  | Most confused with")
    for op in HELIX_ORDER:
        row = confusion.get(op, {})
        vals = [row.get(op2, 0) for op2 in HELIX_ORDER]
        row_str = "  ".join(f"{v:5d}" for v in vals)
        total_alts = sum(vals)
        if total_alts > 0:
            top_alt = max(row.items(), key=lambda x: x[1])
            confused = f"{top_alt[0]} ({top_alt[1]}/{total_alts} = {top_alt[1]/total_alts:.0%})"
        else:
            confused = "(no alternatives)"
        log(f"  {op:3s}      {row_str}  | {confused}")
    
    # Symmetric confusion (merge both directions)
    sym_confusion = Counter()
    for op in HELIX_ORDER:
        for op2, count in confusion.get(op, {}).items():
            pair = tuple(sorted([op, op2]))
            sym_confusion[pair] += count
    
    log(f"\n  ── Most confused operator pairs (bidirectional) ──")
    for (a, b), count in sym_confusion.most_common(15):
        same_triad = "SAME TRIAD" if any(a in t["operators"] and b in t["operators"] for t in TRIADS.values()) else ""
        log(f"    {a:3s} ↔ {b:3s}: {count:4d} verbs confused  {same_triad}")
    
    # Triadic confusion patterns
    intra_triad = 0
    inter_triad = 0
    for (a, b), count in sym_confusion.items():
        same = any(a in t["operators"] and b in t["operators"] for t in TRIADS.values())
        if same:
            intra_triad += count
        else:
            inter_triad += count
    
    total_confused = intra_triad + inter_triad
    if total_confused > 0:
        log(f"\n  ── Confusion by triad structure ──")
        log(f"    Intra-triad confusion: {intra_triad:5d} ({intra_triad/total_confused:.1%})")
        log(f"    Inter-triad confusion: {inter_triad:5d} ({inter_triad/total_confused:.1%})")
        log(f"    (Higher intra = operators within triads are harder to distinguish)")
    
    report = {
        "confusion": {op: dict(confusion.get(op, {})) for op in HELIX_ORDER},
        "symmetric_pairs": {f"{a}-{b}": c for (a,b), c in sym_confusion.most_common()},
        "intra_triad_confusion": intra_triad,
        "inter_triad_confusion": inter_triad,
    }
    
    with open(os.path.join(OUTPUT_DIR, "llm_confusion.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved llm_confusion.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# TEST 3: BOUNDARY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def test_boundaries(cls):
    log("\n" + "="*60)
    log("TEST 3: BOUNDARY ANALYSIS")
    log("Where does the LLM hesitate?")
    log("="*60)
    
    # Group by operator pair boundary
    boundaries = defaultdict(list)
    for c in cls:
        op = c.get("operator", "")
        alt = c.get("alternative")
        conf = c.get("confidence", "").lower()
        if alt and alt.upper() in HELIX_ORDER and conf in ("low", "medium"):
            pair = tuple(sorted([op, alt.upper()]))
            boundaries[pair].append(c)
    
    log(f"\n  ── Boundary regions (medium/low confidence with alternatives) ──")
    for (a, b), verbs in sorted(boundaries.items(), key=lambda x: -len(x[1]))[:12]:
        log(f"\n    {a} ↔ {b}: {len(verbs)} contested verbs")
        for v in verbs[:5]:
            reason = v.get("reason", "")[:65]
            log(f"      {v['verb']:22s} → {v['operator']:3s}  {reason}")
    
    # Specific critical distinctions from EO
    critical_pairs = [
        ("DES", "SEG", "types vs sets"),
        ("CON", "SYN", "link vs merge"),
        ("ALT", "REC", "switch vs rebuild"),
        ("ALT", "SUP", "one-at-a-time vs simultaneous"),
        ("NUL", "SEG", "mark absence vs exclude"),
        ("DES", "INS", "type vs instance"),
        ("SEG", "SYN", "split vs combine"),
    ]
    
    log(f"\n  ── Critical EO distinctions: does the LLM honor them? ──")
    for a, b, desc in critical_pairs:
        pair = tuple(sorted([a, b]))
        contested = boundaries.get(pair, [])
        total_a = sum(1 for c in cls if c.get("operator") == a)
        total_b = sum(1 for c in cls if c.get("operator") == b)
        log(f"    {a:3s} vs {b:3s} ({desc}): {len(contested)} contested | {a}={total_a}, {b}={total_b}")
        if contested:
            examples = [c["verb"] for c in contested[:5]]
            log(f"      Examples: {', '.join(examples)}")
    
    report = {
        "boundaries": {f"{a}-{b}": [{"verb": c["verb"], "operator": c["operator"], "reason": c.get("reason","")} for c in verbs[:20]]
                       for (a,b), verbs in sorted(boundaries.items(), key=lambda x: -len(x[1]))},
    }
    
    with open(os.path.join(OUTPUT_DIR, "llm_boundaries.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved llm_boundaries.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# TEST 4: COMPARISON WITH EMBEDDING RESULTS
# ═══════════════════════════════════════════════════════════════════

def test_comparison(cls):
    log("\n" + "="*60)
    log("TEST 4: COMPARISON WITH EMBEDDING RESULTS")
    log("="*60)
    
    # Load embedding metadata
    meta_path = os.path.join(DATA_DIR, "embedding_metadata.json")
    if not os.path.exists(meta_path):
        log("  No embedding results to compare. Skipping.")
        return
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Load embedding similarity matrix
    try:
        verb_data = np.load(os.path.join(DATA_DIR, "verb_embeddings.npz"))
        op_data = np.load(os.path.join(DATA_DIR, "operator_embeddings.npz"))
        verb_embs = verb_data["enriched"]
        op_embs = op_data["full_spec"]
        
        # Cosine similarity
        v_norm = verb_embs / (np.linalg.norm(verb_embs, axis=1, keepdims=True) + 1e-10)
        o_norm = op_embs / (np.linalg.norm(op_embs, axis=1, keepdims=True) + 1e-10)
        sim = v_norm @ o_norm.T
        
        emb_labels = np.argmax(sim, axis=1)
        emb_names = meta["verb_names"]
    except Exception as e:
        log(f"  Could not load embeddings: {e}")
        return
    
    # Build LLM label lookup
    llm_lookup = {c["verb"]: c.get("operator", "UNKNOWN") for c in cls}
    
    # Compare
    agree = 0
    disagree = 0
    disagreements = []
    
    for i, vname in enumerate(emb_names):
        if vname in llm_lookup:
            emb_op = HELIX_ORDER[emb_labels[i]]
            llm_op = llm_lookup[vname]
            if llm_op in HELIX_ORDER:
                if emb_op == llm_op:
                    agree += 1
                else:
                    disagree += 1
                    if len(disagreements) < 200:
                        disagreements.append({
                            "verb": vname,
                            "embedding": emb_op,
                            "llm": llm_op,
                            "emb_sim": float(sim[i, HELIX_ORDER.index(emb_op)]),
                            "llm_emb_sim": float(sim[i, HELIX_ORDER.index(llm_op)]),
                        })
    
    total_compared = agree + disagree
    agreement_rate = agree / total_compared if total_compared > 0 else 0
    
    log(f"\n  Verbs compared:  {total_compared}")
    log(f"  Agreement:       {agree} ({agreement_rate:.1%})")
    log(f"  Disagreement:    {disagree} ({1-agreement_rate:.1%})")
    
    # Where do they disagree?
    log(f"\n  ── Disagreement patterns ──")
    dis_pairs = Counter((d["embedding"], d["llm"]) for d in disagreements)
    for (emb, llm), count in dis_pairs.most_common(15):
        log(f"    Embedding says {emb:3s}, LLM says {llm:3s}: {count} verbs")
    
    # Most interesting disagreements (where embedding similarity was high for wrong operator)
    disagreements.sort(key=lambda d: d["emb_sim"] - d["llm_emb_sim"], reverse=True)
    log(f"\n  ── Biggest disagreements (embedding was confident, LLM disagreed) ──")
    for d in disagreements[:10]:
        delta = d["emb_sim"] - d["llm_emb_sim"]
        log(f"    {d['verb']:22s} emb={d['embedding']:3s}({d['emb_sim']:.3f}) vs llm={d['llm']:3s}({d['llm_emb_sim']:.3f})  Δ={delta:+.3f}")
    
    # Per-operator agreement
    log(f"\n  ── Agreement by operator ──")
    for op in HELIX_ORDER:
        op_verbs = [(d["embedding"], d["llm"]) for d in disagreements if d["embedding"] == op or d["llm"] == op]
        emb_count = sum(1 for e, l in op_verbs if e == op)
        llm_count = sum(1 for e, l in op_verbs if l == op)
        log(f"    {op:3s}: embedding assigned {emb_count} disputed verbs here, LLM assigned {llm_count}")
    
    report = {
        "total_compared": total_compared,
        "agreement": agree,
        "agreement_rate": agreement_rate,
        "disagreement_patterns": {f"{e}-{l}": c for (e,l), c in dis_pairs.most_common()},
        "sample_disagreements": disagreements[:50],
    }
    
    with open(os.path.join(OUTPUT_DIR, "llm_comparison.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"\n  ✓ Saved llm_comparison.json")
    return report


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════

def visualize(cls):
    log("\n" + "="*60)
    log("VISUALIZATIONS")
    log("="*60)
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  matplotlib not available."); return
    
    colors = ["#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45"]
    
    # Distribution bar chart
    op_counts = Counter(c.get("operator","?") for c in cls)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = [op_counts.get(op, 0) for op in HELIX_ORDER]
    bars = ax.bar(HELIX_ORDER, counts, color=colors)
    
    # Color by triad
    for i, bar in enumerate(bars):
        if i < 3: bar.set_edgecolor("red"); bar.set_linewidth(2)
        elif i < 6: bar.set_edgecolor("blue"); bar.set_linewidth(2)
        else: bar.set_edgecolor("green"); bar.set_linewidth(2)
    
    for i, (op, c) in enumerate(zip(HELIX_ORDER, counts)):
        ax.text(i, c + 20, str(c), ha="center", fontsize=10)
    
    ax.set_title("LLM Classification: Verb Distribution by Operator", fontsize=14)
    ax.set_ylabel("Number of verbs")
    
    # Add triad labels
    ax.text(1, max(counts)*1.1, "Existence", ha="center", fontsize=11, color="red", fontstyle="italic")
    ax.text(4, max(counts)*1.1, "Structure", ha="center", fontsize=11, color="blue", fontstyle="italic")
    ax.text(7, max(counts)*1.1, "Interpretation", ha="center", fontsize=11, color="green", fontstyle="italic")
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "llm_distribution.png"), dpi=150)
    plt.close()
    log("  ✓ llm_distribution.png")
    
    # Confidence by operator
    fig, ax = plt.subplots(figsize=(12, 6))
    high_counts = []
    med_counts = []
    low_counts = []
    for op in HELIX_ORDER:
        op_cls = [c for c in cls if c.get("operator") == op]
        confs = Counter(c.get("confidence","?").lower() for c in op_cls)
        high_counts.append(confs.get("high", 0))
        med_counts.append(confs.get("medium", 0))
        low_counts.append(confs.get("low", 0))
    
    x = np.arange(9)
    w = 0.6
    ax.bar(x, high_counts, w, label="High", color="#2ecc71")
    ax.bar(x, med_counts, w, bottom=high_counts, label="Medium", color="#f39c12")
    ax.bar(x, low_counts, w, bottom=[h+m for h,m in zip(high_counts, med_counts)], label="Low", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(HELIX_ORDER)
    ax.legend()
    ax.set_title("Classification Confidence by Operator", fontsize=14)
    ax.set_ylabel("Number of verbs")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "llm_confidence.png"), dpi=150)
    plt.close()
    log("  ✓ llm_confidence.png")
    
    # Confusion heatmap from alternatives
    confusion = np.zeros((9, 9), dtype=int)
    for c in cls:
        op = c.get("operator", "")
        alt = c.get("alternative")
        if op in HELIX_ORDER and alt and alt.upper() in HELIX_ORDER:
            i = HELIX_ORDER.index(op)
            j = HELIX_ORDER.index(alt.upper())
            confusion[i, j] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion, cmap="YlOrRd")
    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(HELIX_ORDER, fontsize=11)
    ax.set_yticklabels(HELIX_ORDER, fontsize=11)
    ax.set_xlabel("Alternative operator")
    ax.set_ylabel("Primary operator")
    for i in range(9):
        for j in range(9):
            if confusion[i,j] > 0:
                ax.text(j, i, str(confusion[i,j]), ha="center", va="center", fontsize=9,
                        color="white" if confusion[i,j] > confusion.max()*0.5 else "black")
    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title("Operator Confusion Matrix (Primary → Alternative)", fontsize=14)
    for start in [0, 3, 6]:
        rect = plt.Rectangle((start-0.5, start-0.5), 3, 3, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "llm_confusion_heatmap.png"), dpi=150)
    plt.close()
    log("  ✓ llm_confusion_heatmap.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log("="*60)
    log("STEP 5: ANALYZE LLM CLASSIFICATIONS")
    log("="*60)
    
    data, cls = load_classifications()
    
    test_completeness(cls)
    test_confusion(cls)
    test_boundaries(cls)
    test_comparison(cls)
    visualize(cls)
    
    log(f"\n{'='*60}")
    log(f"ANALYSIS COMPLETE in {time.time()-t0:.1f}s")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
