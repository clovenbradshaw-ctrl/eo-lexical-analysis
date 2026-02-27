#!/usr/bin/env python3
"""
Step 09f: Third Axis — Figure/Pattern/Ground of the Referent
=============================================================

Tests the hypothesis that verb semantics has a third independent axis:
  Axis 1 (EO positions): figure/pattern/ground of the OPERATION
  Axis 2 (EO triads):    figure/pattern/ground of the ONTOLOGICAL DOMAIN
  Axis 3 (new):          figure/pattern/ground of the REFERENT

  Figure referent:  verb is about particulars, bounded things, discrete entities
  Pattern referent: verb is about relationships, regularities, structures between things
  Ground referent:  verb is about contexts, substrates, conditions, backgrounds

If this axis is real and independent, we get 3x3x3 = 27 cells.

Uses English verbs (already classified into EO operators, already embedded).
Phase 1: Classify verbs on the referent axis via Claude
Phase 2: Test geometric separation
Phase 3: Test independence from EO axes

Requires:
  pip install anthropic openai numpy scikit-learn
  export ANTHROPIC_API_KEY=sk-...

Usage:
  python scripts/09f_third_axis.py              # all phases
  python scripts/09f_third_axis.py --classify   # classify only
  python scripts/09f_third_axis.py --analyze    # analyze only
"""

import json, os, sys, time, re, argparse
import numpy as np
from collections import Counter
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
CROSSLING = os.path.join(DATA, "crossling")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']

TRIADS = {
    'NUL': 'Existence', 'DES': 'Existence', 'INS': 'Existence',
    'SEG': 'Structure', 'CON': 'Structure', 'SYN': 'Structure',
    'ALT': 'Interpretation', 'SUP': 'Interpretation', 'REC': 'Interpretation',
}

POSITIONS = {
    'NUL': 'Differentiate', 'SEG': 'Differentiate', 'ALT': 'Differentiate',
    'DES': 'Relate', 'CON': 'Relate', 'SUP': 'Relate',
    'INS': 'Generate', 'SYN': 'Generate', 'REC': 'Generate',
}


# ══════════════════════════════════════════════════════════════
#  PHASE 1: CLASSIFY VERBS ON REFERENT AXIS
# ══════════════════════════════════════════════════════════════

def phase1_classify():
    """Classify English verbs as figure/pattern/ground referent."""
    try:
        import anthropic
    except ImportError:
        print("  ✗ pip install anthropic")
        return

    client = anthropic.Anthropic()

    # Load English classified verbs
    eng_dir = os.path.join(CROSSLING, "English")
    classified_file = os.path.join(eng_dir, "classified.json")
    output_file = os.path.join(eng_dir, "referent_axis.json")

    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
        print(f"  Already classified: {len(existing)} verbs")
        return

    if not os.path.exists(classified_file):
        print("  ✗ No English classified.json found")
        return

    with open(classified_file) as f:
        data = json.load(f)

    verbs = []
    for c in data['classifications']:
        op = c.get('operator', '').upper().strip()
        if op not in HELIX:
            continue
        verb = c.get('verb', '').strip()
        gloss = c.get('gloss', '').strip()
        if verb:
            verbs.append({'verb': verb, 'operator': op, 'gloss': gloss})

    print(f"  Classifying {len(verbs)} English verbs on referent axis...")

    system_prompt = """You are a cognitive linguist classifying verbs along a single dimension:
what kind of REFERENT does the verb point to?

Three categories:

FIGURE: The verb is about particulars, bounded entities, discrete things, specific objects
or events. The referent is something you could point to, count, or isolate.
Examples: "kick" (specific action on specific thing), "build" (creates discrete object),
"arrive" (bounded event), "name" (picks out particular), "kill" (targets specific entity)

PATTERN: The verb is about relationships, regularities, structures, or connections between
things. The referent is not a thing but a relation, rule, similarity, or recurring form.
Examples: "resemble" (relationship), "correlate" (regularity), "govern" (structural relation),
"alternate" (pattern of change), "rank" (ordering relation)

GROUND: The verb is about contexts, substrates, conditions, backgrounds, states, or the
medium in which things happen. The referent is not a thing or relation but a condition,
atmosphere, or enabling context.
Examples: "exist" (condition), "rain" (environmental state), "persist" (ongoing condition),
"underlie" (substrate), "permeate" (diffuse presence), "afford" (enabling condition)

Classify each verb. When uncertain, ask: would you describe the verb's referent as
"a thing" (FIGURE), "a relationship" (PATTERN), or "a situation/condition" (GROUND)?

Respond ONLY as a JSON array: [{"verb": "...", "referent": "FIGURE|PATTERN|GROUND"}]
No explanations."""

    all_results = []
    batch_size = 80

    for batch_start in range(0, len(verbs), batch_size):
        batch = verbs[batch_start:batch_start + batch_size]

        verb_list = "\n".join(f"  {v['verb']}" + (f" ({v['gloss']})" if v['gloss'] else "")
                              for v in batch)

        prompt = f"""Classify these verbs as FIGURE, PATTERN, or GROUND:

{verb_list}

Return JSON array: [{{"verb": "...", "referent": "FIGURE|PATTERN|GROUND"}}]"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text.strip()
            if '```' in text:
                match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
                if match:
                    text = match.group(1).strip()

            batch_results = json.loads(text)

            # Merge with operator info
            verb_op_map = {v['verb']: v['operator'] for v in batch}
            verb_gloss_map = {v['verb']: v['gloss'] for v in batch}
            for r in batch_results:
                r['operator'] = verb_op_map.get(r.get('verb', ''), '')
                r['gloss'] = verb_gloss_map.get(r.get('verb', ''), '')
                r['referent'] = r.get('referent', '').upper().strip()

            all_results.extend(batch_results)

            n_done = min(batch_start + batch_size, len(verbs))
            print(f"    {n_done}/{len(verbs)}", end="\r", flush=True)

        except Exception as e:
            print(f"\n    ✗ Batch {batch_start}: {e}")
            time.sleep(5)
            continue

        time.sleep(1)

    # Save
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n    ✓ {len(all_results)} verbs classified")

    # Quick distribution
    ref_counts = Counter(r.get('referent', '') for r in all_results)
    total = sum(ref_counts.values())
    for ref in ['FIGURE', 'PATTERN', 'GROUND']:
        n = ref_counts.get(ref, 0)
        print(f"      {ref}: {n} ({100*n/total:.1f}%)")


# ══════════════════════════════════════════════════════════════
#  PHASE 2: ANALYZE
# ══════════════════════════════════════════════════════════════

def compute_z_score(embeddings, labels, n_perm=500):
    """Z-score of inter-centroid separation vs random permutations."""
    from sklearn.metrics.pairwise import cosine_similarity

    unique = sorted(set(labels))
    n_groups = len(unique)
    if n_groups < 2:
        return None

    label_map = {u: i for i, u in enumerate(unique)}
    mapped = np.array([label_map[l] for l in labels])

    centroids = []
    for i in range(n_groups):
        mask = mapped == i
        if mask.sum() == 0:
            centroids.append(np.zeros(embeddings.shape[1]))
        else:
            centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)

    sim = cosine_similarity(centroids)
    pairs = [sim[i,j] for i in range(n_groups) for j in range(i+1, n_groups)]
    real_mean = np.mean(pairs)

    rng = np.random.RandomState(42)
    random_means = []
    for _ in range(n_perm):
        perm = rng.permutation(mapped)
        perm_centroids = []
        for i in range(n_groups):
            mask = perm == i
            if mask.sum() == 0:
                perm_centroids.append(np.zeros(embeddings.shape[1]))
            else:
                perm_centroids.append(embeddings[mask].mean(axis=0))
        perm_centroids = np.array(perm_centroids)

        perm_sim = cosine_similarity(perm_centroids)
        perm_pairs = [perm_sim[i,j] for i in range(n_groups) for j in range(i+1, n_groups)]
        random_means.append(np.mean(perm_pairs))

    rand_mean = np.mean(random_means)
    rand_std = np.std(random_means)
    z = (real_mean - rand_mean) / rand_std if rand_std > 0 else 0

    return {
        'z_score': float(z),
        'real_sim': float(real_mean),
        'random_mean': float(rand_mean),
        'random_std': float(rand_std),
        'n_groups': n_groups,
        'n_items': len(labels),
    }


def compute_ari_nmi(labels_a, labels_b):
    """Compute ARI and NMI between two classification systems."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(labels_a, labels_b)
    nmi = normalized_mutual_info_score(labels_a, labels_b)
    return ari, nmi


def phase2_analyze():
    """Test third axis for separation and independence."""
    from sklearn.metrics.pairwise import cosine_similarity

    eng_dir = os.path.join(CROSSLING, "English")
    referent_file = os.path.join(eng_dir, "referent_axis.json")
    embeddings_file = os.path.join(eng_dir, "definition_embeddings.npz")
    definitions_file = os.path.join(eng_dir, "native_definitions.json")

    if not os.path.exists(referent_file):
        print("  ✗ No referent_axis.json — run --classify first")
        return

    if not os.path.exists(embeddings_file):
        print("  ✗ No definition_embeddings.npz — run 09d first")
        return

    # Load referent classifications
    with open(referent_file) as f:
        referents = json.load(f)
    ref_map = {}
    for r in referents:
        v = r.get('verb', '').strip()
        ref = r.get('referent', '').upper().strip()
        op = r.get('operator', '').upper().strip()
        if v and ref in ['FIGURE', 'PATTERN', 'GROUND'] and op in HELIX:
            ref_map[v] = {'referent': ref, 'operator': op}

    # Load embeddings
    emb_data = np.load(embeddings_file, allow_pickle=True)
    emb_verbs = emb_data['verbs'].tolist()
    emb_operators = emb_data['operators'].tolist()
    embeddings = emb_data['embeddings']

    # Match: only verbs that have both embeddings and referent classification
    matched_idx = []
    matched_refs = []
    matched_ops = []
    matched_triads = []
    matched_positions = []
    matched_verbs = []

    for i, (verb, op) in enumerate(zip(emb_verbs, emb_operators)):
        if verb in ref_map and op in HELIX:
            matched_idx.append(i)
            matched_refs.append(ref_map[verb]['referent'])
            matched_ops.append(op)
            matched_triads.append(TRIADS[op])
            matched_positions.append(POSITIONS[op])
            matched_verbs.append(verb)

    matched_emb = embeddings[matched_idx]

    print(f"\n{'='*70}")
    print(f"  THIRD AXIS ANALYSIS: {len(matched_verbs)} matched English verbs")
    print(f"{'='*70}")

    # Distribution
    ref_counts = Counter(matched_refs)
    total = len(matched_refs)
    print(f"\n  Referent distribution:")
    for ref in ['FIGURE', 'PATTERN', 'GROUND']:
        n = ref_counts.get(ref, 0)
        print(f"    {ref:10s}: {n:5d} ({100*n/total:.1f}%)")

    # Cross-tab: operator × referent
    print(f"\n  CROSS-TABULATION: Operator × Referent")
    print(f"  {'Op':>5s} {'FIGURE':>8s} {'PATTERN':>8s} {'GROUND':>8s} {'Total':>8s}")
    print("  " + "-"*42)
    for op in HELIX:
        fig = sum(1 for o, r in zip(matched_ops, matched_refs) if o == op and r == 'FIGURE')
        pat = sum(1 for o, r in zip(matched_ops, matched_refs) if o == op and r == 'PATTERN')
        grd = sum(1 for o, r in zip(matched_ops, matched_refs) if o == op and r == 'GROUND')
        tot = fig + pat + grd
        if tot > 0:
            print(f"  {op:>5s} {fig:>8d} {pat:>8d} {grd:>8d} {tot:>8d}  [{TRIADS[op]}/{POSITIONS[op]}]")

    # ══════════════════════════════════════════════════════════
    #  Z-SCORES FOR ALL AXES
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  GEOMETRIC SEPARATION: ALL THREE AXES")
    print(f"{'='*70}")

    results = {}

    # Axis 1: EO operators (9 groups)
    z_ops = compute_z_score(matched_emb, matched_ops, n_perm=500)
    results['operators'] = z_ops
    print(f"\n  EO operators (9 groups):    z = {z_ops['z_score']:+.1f}")

    # Axis 1a: Positions (3 groups)
    z_pos = compute_z_score(matched_emb, matched_positions, n_perm=500)
    results['positions'] = z_pos
    print(f"  Positions (3 groups):       z = {z_pos['z_score']:+.1f}")

    # Axis 1b: Triads (3 groups)
    z_tri = compute_z_score(matched_emb, matched_triads, n_perm=500)
    results['triads'] = z_tri
    print(f"  Triads (3 groups):          z = {z_tri['z_score']:+.1f}")

    # Axis 2: Referent (3 groups)
    z_ref = compute_z_score(matched_emb, matched_refs, n_perm=500)
    results['referent'] = z_ref
    print(f"  Referent FPG (3 groups):    z = {z_ref['z_score']:+.1f}")

    # Random baseline
    rng = np.random.RandomState(99)
    rand_z = []
    for _ in range(20):
        shuffled = list(matched_refs)
        rng.shuffle(shuffled)
        rz = compute_z_score(matched_emb, shuffled, n_perm=100)
        if rz:
            rand_z.append(rz['z_score'])
    results['random'] = float(np.mean(rand_z))
    print(f"  Random 3-way (baseline):    z = {np.mean(rand_z):+.1f}")

    # ══════════════════════════════════════════════════════════
    #  INDEPENDENCE TESTS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  INDEPENDENCE: Is the referent axis orthogonal to EO?")
    print(f"{'='*70}")

    # ARI / NMI
    ari_ref_op, nmi_ref_op = compute_ari_nmi(matched_refs, matched_ops)
    ari_ref_tri, nmi_ref_tri = compute_ari_nmi(matched_refs, matched_triads)
    ari_ref_pos, nmi_ref_pos = compute_ari_nmi(matched_refs, matched_positions)
    ari_tri_pos, nmi_tri_pos = compute_ari_nmi(matched_triads, matched_positions)

    print(f"\n  {'Comparison':>35s} {'ARI':>8s} {'NMI':>8s}")
    print("  " + "-"*55)
    print(f"  {'Referent vs Operators':>35s} {ari_ref_op:+8.4f} {nmi_ref_op:8.4f}")
    print(f"  {'Referent vs Triads':>35s} {ari_ref_tri:+8.4f} {nmi_ref_tri:8.4f}")
    print(f"  {'Referent vs Positions':>35s} {ari_ref_pos:+8.4f} {nmi_ref_pos:8.4f}")
    print(f"  {'Triads vs Positions (control)':>35s} {ari_tri_pos:+8.4f} {nmi_tri_pos:8.4f}")

    print(f"\n  ARI = 0 means independent. ARI = 1 means identical.")
    print(f"  NMI = 0 means no mutual information. NMI = 1 means identical.")

    # ══════════════════════════════════════════════════════════
    #  COMBINED: 27-CELL ANALYSIS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  27-CELL ANALYSIS: Position × Triad × Referent")
    print(f"{'='*70}")

    # Create 27-cell labels
    cell_labels = []
    for pos, tri, ref in zip(matched_positions, matched_triads, matched_refs):
        cell_labels.append(f"{pos}_{tri}_{ref}")

    cell_counts = Counter(cell_labels)
    n_cells = len(cell_counts)
    populated = sum(1 for c in cell_counts.values() if c >= 5)

    print(f"\n  Total cells with any verbs: {n_cells}/27")
    print(f"  Cells with ≥5 verbs: {populated}/27")

    # Show populated cells
    print(f"\n  {'Cell':>40s} {'n':>5s}")
    print("  " + "-"*50)
    for cell, count in sorted(cell_counts.items(), key=lambda x: -x[1]):
        if count >= 3:
            print(f"  {cell:>40s} {count:>5d}")

    # Z-score for 27 cells
    if populated >= 10:
        z_27 = compute_z_score(matched_emb, cell_labels, n_perm=500)
        results['27_cells'] = z_27
        print(f"\n  Z-score for 27-cell grouping: z = {z_27['z_score']:+.1f}")
    else:
        print(f"\n  Too few populated cells for 27-cell z-score")

    # ══════════════════════════════════════════════════════════
    #  CONDITIONAL TESTS: Does referent add info beyond EO?
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  CONDITIONAL: Does referent separate WITHIN EO groups?")
    print(f"{'='*70}")

    # Within each operator: does referent still separate?
    print(f"\n  Within each EO operator:")
    for op in HELIX:
        mask = [o == op for o in matched_ops]
        if sum(mask) < 30:
            continue
        op_emb = matched_emb[mask]
        op_refs = [r for r, m in zip(matched_refs, mask) if m]
        if len(set(op_refs)) < 2:
            continue
        z = compute_z_score(op_emb, op_refs, n_perm=200)
        if z:
            print(f"    {op:>5s} (n={sum(mask):4d}): referent z = {z['z_score']:+.1f}")

    # Within each triad: does referent still separate?
    print(f"\n  Within each triad:")
    for tri in ['Existence', 'Structure', 'Interpretation']:
        mask = [t == tri for t in matched_triads]
        if sum(mask) < 30:
            continue
        tri_emb = matched_emb[mask]
        tri_refs = [r for r, m in zip(matched_refs, mask) if m]
        z = compute_z_score(tri_emb, tri_refs, n_perm=300)
        if z:
            print(f"    {tri:>20s} (n={sum(mask):4d}): referent z = {z['z_score']:+.1f}")

    # Within each position: does referent still separate?
    print(f"\n  Within each position:")
    for pos in ['Differentiate', 'Relate', 'Generate']:
        mask = [p == pos for p in matched_positions]
        if sum(mask) < 30:
            continue
        pos_emb = matched_emb[mask]
        pos_refs = [r for r, m in zip(matched_refs, mask) if m]
        z = compute_z_score(pos_emb, pos_refs, n_perm=300)
        if z:
            print(f"    {pos:>20s} (n={sum(mask):4d}): referent z = {z['z_score']:+.1f}")

    # ══════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Three axes of verb semantics:")
    print(f"    Axis 1 — Positions (operation type):   z = {z_pos['z_score']:+.1f}")
    print(f"    Axis 2 — Triads (ontological domain):  z = {z_tri['z_score']:+.1f}")
    print(f"    Axis 3 — Referent (figure/pattern/ground): z = {z_ref['z_score']:+.1f}")
    print(f"    Random baseline:                       z = {np.mean(rand_z):+.1f}")

    print(f"\n  Independence (ARI ≈ 0 = orthogonal):")
    print(f"    Referent vs Operators: ARI = {ari_ref_op:+.4f}")
    print(f"    Referent vs Triads:    ARI = {ari_ref_tri:+.4f}")
    print(f"    Referent vs Positions: ARI = {ari_ref_pos:+.4f}")

    is_real = z_ref['z_score'] < -3
    is_independent = abs(ari_ref_op) < 0.1 and abs(ari_ref_tri) < 0.1 and abs(ari_ref_pos) < 0.1

    print(f"\n  VERDICT:")
    print(f"    Geometrically real?  {'YES' if is_real else 'NO'} (z = {z_ref['z_score']:+.1f})")
    print(f"    Independent of EO?   {'YES' if is_independent else 'PARTIAL'}")
    if is_real and is_independent:
        print(f"    → Third axis CONFIRMED: 3×3×3 = 27 phase space is empirically supported")
    elif is_real:
        print(f"    → Third axis is real but partially correlated with EO axes")
    else:
        print(f"    → Third axis not confirmed")

    # Save
    report = {
        'n_verbs': len(matched_verbs),
        'distribution': dict(ref_counts),
        'z_positions': z_pos['z_score'],
        'z_triads': z_tri['z_score'],
        'z_referent': z_ref['z_score'],
        'z_random': float(np.mean(rand_z)),
        'ari_referent_operators': ari_ref_op,
        'ari_referent_triads': ari_ref_tri,
        'ari_referent_positions': ari_ref_pos,
        'nmi_referent_operators': nmi_ref_op,
        'nmi_referent_triads': nmi_ref_tri,
        'nmi_referent_positions': nmi_ref_pos,
    }
    if 'z_27' in dir():
        report['z_27_cells'] = z_27['z_score']

    with open(os.path.join(OUT, "third_axis.json"), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Saved third_axis.json")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()

    run_all = not (args.classify or args.analyze)

    if args.classify or run_all:
        print(f"{'='*70}")
        print(f"  PHASE 1: Classify English verbs on referent axis")
        print(f"{'='*70}")
        phase1_classify()

    if args.analyze or run_all:
        phase2_analyze()


if __name__ == '__main__':
    main()
