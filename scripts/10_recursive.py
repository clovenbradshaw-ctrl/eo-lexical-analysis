#!/usr/bin/env python3
"""
Step 10: Recursive Operator Structure
======================================
Tests the hypothesis that sub-clusters within each operator correspond
to OTHER operators acting as modifiers, not to scales or arbitrary k.

Approach:
  1. For each operator, find the natural number of sub-clusters (sweep k=2..12)
  2. Characterize each sub-cluster by its nearest OTHER operator centroid
  3. Test: do sub-clusters align better with operator-crossings or with scales?
  4. Build the operator × operator matrix

If the helix is self-similar, then NUL's internal landscape should be
shaped by SEG, CON, INS, ALT, REC etc. acting as modifiers on absence.
"""

import json, os, sys
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────
print("Loading data...")
with open(os.path.join(DATA, "llm_classifications.json")) as f:
    cls = json.load(f)["classifications"]

embs = np.load(os.path.join(DATA, "reembed_combined.npz"))["embeddings"]
print(f"  {len(cls)} verbs, {embs.shape[1]} dimensions")

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']
SCALES = ['physical','social','psychological','informational']

# Build indices and centroids
op_indices = {op: [] for op in HELIX}
for i, c in enumerate(cls):
    if c['operator'] in HELIX:
        op_indices[c['operator']].append(i)

centroids = {}
for op in HELIX:
    idx = op_indices[op]
    centroids[op] = embs[idx].mean(axis=0)

# ══════════════════════════════════════════════════════════════════
#  1. FIND NATURAL SUB-CLUSTER COUNT PER OPERATOR
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  1. NATURAL SUB-CLUSTER COUNTS (silhouette sweep k=2..12)")
print("="*70)

natural_k = {}

for op in HELIX:
    idx = op_indices[op]
    op_embs = embs[idx]
    n = len(idx)

    if n < 20:
        print(f"\n  {op}: only {n} verbs, testing k=2..{min(5,n-1)}")
        k_range = range(2, min(6, n))
    else:
        k_range = range(2, 13)

    scores = []
    for k in k_range:
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(op_embs)
        # Check all clusters have at least 2 members
        counts = Counter(labels)
        if min(counts.values()) < 2:
            scores.append((k, -1))
            continue
        sil = silhouette_score(op_embs, labels, metric='cosine', sample_size=min(5000, n))
        scores.append((k, sil))

    if not scores:
        natural_k[op] = 2
        continue

    best_k, best_sil = max(scores, key=lambda x: x[1])

    print(f"\n  {op} ({n} verbs):")
    for k, sil in scores:
        marker = " ★" if k == best_k else ""
        bar = "█" * int(max(0, sil) * 100) if sil >= 0 else ""
        print(f"    k={k:2d}: sil={sil:+.4f} {bar}{marker}")

    natural_k[op] = best_k

print(f"\n  Natural k values: {natural_k}")

# ══════════════════════════════════════════════════════════════════
#  2. CLUSTER AT NATURAL K AND CHARACTERIZE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  2. SUB-CLUSTER CHARACTERIZATION")
print("="*70)

all_subclusters = {}

for op in HELIX:
    idx = op_indices[op]
    op_embs = embs[idx]
    k = natural_k[op]
    n = len(idx)

    if n < 4:
        print(f"\n  {op}: too few verbs for sub-clustering")
        continue

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(op_embs)

    print(f"\n{'='*60}")
    print(f"  {op} ({n} verbs) → {k} natural sub-clusters")
    print(f"{'='*60}")

    subclusters = []

    for ci in range(k):
        mask = labels == ci
        cluster_idx = [idx[j] for j in range(n) if mask[j]]
        cluster_embs = op_embs[mask]
        cluster_centroid = cluster_embs.mean(axis=0)

        # ── Nearest OTHER operator centroid ────────────────────────
        other_sims = {}
        for other_op in HELIX:
            if other_op == op:
                continue
            sim = cosine_similarity([cluster_centroid], [centroids[other_op]])[0, 0]
            other_sims[other_op] = sim

        # Also compute similarity to own centroid
        self_sim = cosine_similarity([cluster_centroid], [centroids[op]])[0, 0]

        nearest_ops = sorted(other_sims.items(), key=lambda x: -x[1])

        # ── Scale distribution ────────────────────────────────────
        scale_counts = Counter(cls[j]['scale'] for j in cluster_idx)
        total_in_cluster = len(cluster_idx)
        scale_pcts = {s: scale_counts.get(s, 0) / total_in_cluster * 100 for s in SCALES}
        dominant_scale = max(scale_pcts.items(), key=lambda x: x[1])

        # ── Exemplar verbs (closest to sub-cluster centroid) ──────
        verb_sims = cosine_similarity([cluster_centroid], cluster_embs)[0]
        top_verb_idx = np.argsort(-verb_sims)[:8]
        exemplars = [cls[cluster_idx[j]]['verb'] for j in top_verb_idx]

        # ── Alternative operator distribution ─────────────────────
        alt_counts = Counter()
        for j in cluster_idx:
            alt = cls[j].get('alternative', '')
            if alt and alt in HELIX:
                alt_counts[alt] += 1

        # ── Confidence distribution ───────────────────────────────
        conf_counts = Counter(cls[j].get('confidence', 'high') for j in cluster_idx)

        subcluster_data = {
            'cluster_id': ci,
            'size': total_in_cluster,
            'nearest_operator': nearest_ops[0][0],
            'nearest_op_sim': nearest_ops[0][1],
            'second_nearest': nearest_ops[1][0],
            'second_nearest_sim': nearest_ops[1][1],
            'self_sim': self_sim,
            'pull_ratio': nearest_ops[0][1] / self_sim if self_sim > 0 else 0,
            'dominant_scale': dominant_scale[0],
            'scale_pcts': scale_pcts,
            'exemplars': exemplars,
            'alternatives': dict(alt_counts.most_common(3)),
            'confidence': dict(conf_counts),
        }
        subclusters.append(subcluster_data)

        # Print
        pull = "→" if subcluster_data['pull_ratio'] > 0.98 else "↗" if subcluster_data['pull_ratio'] > 0.95 else "↑"
        print(f"\n  Sub-{ci} ({total_in_cluster} verbs) {pull} {nearest_ops[0][0]} (sim={nearest_ops[0][1]:.3f}, self={self_sim:.3f})")
        print(f"    Interpretation: {op} × {nearest_ops[0][0]} (pull_ratio={subcluster_data['pull_ratio']:.3f})")
        print(f"    Scale: {dominant_scale[0]} ({dominant_scale[1]:.0f}%)", end="")
        for s in SCALES:
            if s != dominant_scale[0] and scale_pcts[s] > 15:
                print(f", {s} ({scale_pcts[s]:.0f}%)", end="")
        print()
        print(f"    Exemplars: {', '.join(exemplars)}")
        if alt_counts:
            alt_str = ", ".join(f"{a}:{c}" for a, c in alt_counts.most_common(3))
            print(f"    Alternatives: {alt_str}")

    all_subclusters[op] = subclusters

# ══════════════════════════════════════════════════════════════════
#  3. ALIGNMENT TEST: OPERATOR-CROSSING VS SCALE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  3. WHAT EXPLAINS SUB-CLUSTERS BETTER: OPERATOR-CROSSING OR SCALE?")
print("="*70)

for op in HELIX:
    if op not in all_subclusters:
        continue
    subs = all_subclusters[op]
    if len(subs) < 3:
        continue

    # For each pair of sub-clusters, compute:
    # (a) Do they differ more by nearest-operator or by dominant-scale?
    nearest_ops = [s['nearest_operator'] for s in subs]
    dom_scales = [s['dominant_scale'] for s in subs]

    unique_ops = len(set(nearest_ops))
    unique_scales = len(set(dom_scales))

    # Information: how many distinct values does each variable take?
    # If all sub-clusters point to different operators → operator explains structure
    # If all sub-clusters have different scales → scale explains structure
    op_diversity = unique_ops / len(subs)
    scale_diversity = unique_scales / len(subs)

    # Stronger test: compute how much variance in sub-cluster centroids
    # is explained by grouping by nearest-op vs grouping by scale
    print(f"\n  {op} ({len(subs)} sub-clusters):")
    print(f"    Unique nearest operators: {unique_ops}/{len(subs)} ({', '.join(nearest_ops)})")
    print(f"    Unique dominant scales:   {unique_scales}/{len(subs)} ({', '.join(dom_scales)})")

    if op_diversity > scale_diversity:
        print(f"    → OPERATOR-CROSSING explains more ({op_diversity:.1%} vs {scale_diversity:.1%} diversity)")
    elif scale_diversity > op_diversity:
        print(f"    → SCALE explains more ({scale_diversity:.1%} vs {op_diversity:.1%} diversity)")
    else:
        print(f"    → TIE ({op_diversity:.1%} diversity each)")

# ══════════════════════════════════════════════════════════════════
#  4. BUILD THE OPERATOR × OPERATOR MATRIX
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  4. OPERATOR × OPERATOR MATRIX")
print("="*70)
print("  Each cell: how many sub-clusters of row-operator point toward column-operator\n")

# Build matrix
cross_matrix = defaultdict(lambda: defaultdict(int))
cross_sizes = defaultdict(lambda: defaultdict(int))

for op in HELIX:
    if op not in all_subclusters:
        continue
    for sub in all_subclusters[op]:
        nearest = sub['nearest_operator']
        cross_matrix[op][nearest] += 1
        cross_sizes[op][nearest] += sub['size']

# Print matrix
print(f"  {'':8s}", end="")
for op in HELIX:
    print(f"{op:>6s}", end="")
print(f"  {'total':>6s}")
print("  " + "-" * 70)

for op in HELIX:
    print(f"  {op:8s}", end="")
    total = 0
    for target in HELIX:
        if target == op:
            print(f"{'·':>6s}", end="")
        else:
            c = cross_matrix[op][target]
            total += c
            if c > 0:
                print(f"{c:6d}", end="")
            else:
                print(f"{'':>6s}", end="")
    print(f"{total:6d}")

# Size-weighted version
print(f"\n  Size-weighted (verb counts):")
print(f"  {'':8s}", end="")
for op in HELIX:
    print(f"{op:>7s}", end="")
print()
print("  " + "-" * 75)

for op in HELIX:
    print(f"  {op:8s}", end="")
    for target in HELIX:
        if target == op:
            print(f"{'·':>7s}", end="")
        else:
            c = cross_sizes[op][target]
            if c > 0:
                print(f"{c:7d}", end="")
            else:
                print(f"{'':>7s}", end="")
    print()

# ══════════════════════════════════════════════════════════════════
#  5. WHICH OPERATORS ARE MOST "MODIFYING"?
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  5. OPERATOR INFLUENCE (how often each operator modifies others)")
print(f"{'='*70}")

# Column sums of cross_matrix
influence = defaultdict(int)
influence_sizes = defaultdict(int)
for op in HELIX:
    for target in HELIX:
        if target != op:
            influence[target] += cross_matrix[op][target]
            influence_sizes[target] += cross_sizes[op][target]

for op in sorted(HELIX, key=lambda x: -influence[x]):
    print(f"  {op}: appears as modifier in {influence[op]} sub-clusters ({influence_sizes[op]} verbs)")

# ══════════════════════════════════════════════════════════════════
#  6. RECIPROCITY TEST
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  6. RECIPROCITY (does A modify B when B modifies A?)")
print(f"{'='*70}")

pairs = []
for i, op1 in enumerate(HELIX):
    for op2 in HELIX[i+1:]:
        a_to_b = cross_matrix[op1][op2]
        b_to_a = cross_matrix[op2][op1]
        if a_to_b > 0 or b_to_a > 0:
            pairs.append((op1, op2, a_to_b, b_to_a))

pairs.sort(key=lambda x: -(x[2] + x[3]))
for op1, op2, a2b, b2a in pairs:
    direction = "↔" if a2b > 0 and b2a > 0 else "→" if a2b > 0 else "←"
    print(f"  {op1} {direction} {op2}: {op1}×{op2}={a2b}, {op2}×{op1}={b2a}")

# ══════════════════════════════════════════════════════════════════
#  7. SELF-SIMILARITY TEST
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  7. SELF-SIMILARITY: Do sub-clusters recapitulate helix structure?")
print(f"{'='*70}")

# For each operator with enough sub-clusters, check if the
# sub-cluster-to-nearest-operator mapping follows triad structure
TRIAD_OF = {}
for tname, tops in [('Existence',['NUL','DES','INS']),
                     ('Structure',['SEG','CON','SYN']),
                     ('Interpretation',['ALT','SUP','REC'])]:
    for op in tops:
        TRIAD_OF[op] = tname

for op in HELIX:
    if op not in all_subclusters or len(all_subclusters[op]) < 3:
        continue

    subs = all_subclusters[op]
    own_triad = TRIAD_OF[op]

    same_triad = 0
    diff_triad = 0
    for sub in subs:
        nearest = sub['nearest_operator']
        if TRIAD_OF[nearest] == own_triad:
            same_triad += 1
        else:
            diff_triad += 1

    print(f"  {op} ({own_triad}): {same_triad} sub-clusters → same triad, {diff_triad} → different triads")

# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.decomposition import PCA

    # Fig 1: Operator × Operator heatmap
    matrix = np.zeros((9, 9))
    for i, op1 in enumerate(HELIX):
        for j, op2 in enumerate(HELIX):
            if op1 != op2:
                matrix[i, j] = cross_sizes[op1][op2]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal')

    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(HELIX, fontsize=11, fontweight='bold')
    ax.set_yticklabels(HELIX, fontsize=11, fontweight='bold')

    for i in range(9):
        for j in range(9):
            if i == j:
                ax.text(j, i, "·", ha='center', va='center', fontsize=14)
            elif matrix[i, j] > 0:
                color = 'white' if matrix[i, j] > matrix.max() * 0.6 else 'black'
                ax.text(j, i, f"{int(matrix[i,j])}", ha='center', va='center',
                       fontsize=8, color=color)

    # Triad boxes
    for start in [0, 3, 6]:
        rect = plt.Rectangle((start-0.5, start-0.5), 3, 3, fill=False,
                            edgecolor='black', linewidth=2.5)
        ax.add_patch(rect)

    plt.colorbar(im, label='Verbs in sub-cluster')
    ax.set_xlabel("Modifier operator (nearest centroid)", fontsize=12)
    ax.set_ylabel("Base operator", fontsize=12)
    ax.set_title("Recursive Operator Structure\n(row operator's sub-clusters → nearest column operator)",
                fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "recursive_operator_matrix.png"), dpi=150)
    plt.close()
    print("  ✓ recursive_operator_matrix.png")

    # Fig 2: PCA of all sub-cluster centroids, colored by base operator,
    # with arrows to nearest other operator
    fig, ax = plt.subplots(figsize=(14, 10))

    op_colors = {
        'NUL': '#e74c3c', 'DES': '#e67e22', 'INS': '#f1c40f',
        'SEG': '#3498db', 'CON': '#2ecc71', 'SYN': '#9b59b6',
        'ALT': '#1abc9c', 'SUP': '#e91e63', 'REC': '#8bc34a',
    }

    # Collect all sub-cluster centroids
    sub_centroids = []
    sub_labels = []
    sub_nearest = []
    sub_sizes = []

    for op in HELIX:
        idx = op_indices[op]
        op_embs = embs[idx]
        k = natural_k.get(op, 2)
        if len(idx) < 4:
            continue

        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(op_embs)

        for ci in range(k):
            mask = labels == ci
            if mask.sum() < 2:
                continue
            sc = op_embs[mask].mean(axis=0)
            sub_centroids.append(sc)
            sub_labels.append(op)

            # Nearest other operator
            best_other = None
            best_sim = -1
            for other in HELIX:
                if other == op:
                    continue
                sim = cosine_similarity([sc], [centroids[other]])[0, 0]
                if sim > best_sim:
                    best_sim = sim
                    best_other = other
            sub_nearest.append(best_other)
            sub_sizes.append(int(mask.sum()))

    if sub_centroids:
        all_points = np.array(sub_centroids + [centroids[op] for op in HELIX])
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_points)

        n_sub = len(sub_centroids)
        sub_2d = all_2d[:n_sub]
        centroid_2d = all_2d[n_sub:]

        # Plot sub-cluster centroids
        for i in range(n_sub):
            op = sub_labels[i]
            nearest = sub_nearest[i]
            size = max(30, min(300, sub_sizes[i] / 3))
            ax.scatter(sub_2d[i, 0], sub_2d[i, 1], c=op_colors[op], s=size,
                      alpha=0.6, edgecolors=op_colors[nearest], linewidth=2, zorder=3)

        # Plot main centroids as stars
        for i, op in enumerate(HELIX):
            ax.scatter(centroid_2d[i, 0], centroid_2d[i, 1], c=op_colors[op],
                      s=500, marker='*', edgecolors='black', linewidth=2, zorder=5)
            ax.annotate(op, (centroid_2d[i, 0], centroid_2d[i, 1]),
                       fontsize=14, fontweight='bold', ha='center', va='bottom',
                       xytext=(0, 12), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.9, edgecolor='black'))

        ax.set_title("Sub-cluster Centroids\n(fill = base operator, border = nearest other operator, size ∝ verb count)",
                    fontsize=13)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

        # Legend
        patches = [mpatches.Patch(color=op_colors[op], label=op) for op in HELIX]
        ax.legend(handles=patches, loc='upper right', fontsize=9, ncol=3)

        plt.tight_layout()
        plt.savefig(os.path.join(VIZ, "recursive_subcluster_map.png"), dpi=150)
        plt.close()
        print("  ✓ recursive_subcluster_map.png")

except ImportError:
    print("  (matplotlib not available)")

# ── SAVE ──────────────────────────────────────────────────────────
report = {
    'natural_k': natural_k,
    'subclusters': {op: subs for op, subs in all_subclusters.items()},
    'cross_matrix': {op: dict(targets) for op, targets in cross_matrix.items()},
    'cross_sizes': {op: dict(targets) for op, targets in cross_sizes.items()},
}
with open(os.path.join(OUT, "recursive_report.json"), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*70}")
print("  RECURSIVE STRUCTURE ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"  Report: {OUT}/recursive_report.json")
print(f"  Visuals: {VIZ}/recursive_*.png")
