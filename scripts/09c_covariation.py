#!/usr/bin/env python3
"""
Step 09c: Triad Co-Variation Analysis
=======================================

The real test of whether {NUL,DES,INS}, {SEG,CON,SYN}, {ALT,SUP,REC}
form natural groupings is whether operators within a triad co-vary
across languages.

If Structure is a real category, then languages with high SEG should
also tend toward high CON and SYN. If the triads are arbitrary,
within-triad correlations should be no stronger than between-triad.

Tests:
  1. 9×9 correlation matrix of operators across 28 languages
  2. Mean within-triad vs between-triad correlation
  3. Factor analysis: do the operators naturally cluster into 3 groups?
  4. Compare EO triads against data-driven groupings
  5. Position co-variation: do Position-1 ops co-vary? Position-2? Position-3?
"""

import json, os
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']
TRIADS = {
    'Existence': ['NUL','DES','INS'],
    'Structure': ['SEG','CON','SYN'],
    'Interpretation': ['ALT','SUP','REC'],
}
POSITIONS = {
    'Position 1 (differentiate)': ['NUL','SEG','ALT'],
    'Position 2 (relate)': ['DES','CON','SUP'],
    'Position 3 (generate)': ['INS','SYN','REC'],
}

# ══════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════
print("Loading cross-linguistic data...")

all_langs = {}
for lang_dir in sorted(Path(DATA).iterdir()):
    if not lang_dir.is_dir():
        continue
    classified_file = lang_dir / "classified.json"
    if not classified_file.exists():
        continue
    with open(classified_file) as f:
        data = json.load(f)
    
    lang = data['language']
    cls = data['classifications']
    op_counts = Counter()
    for c in cls:
        op = c.get('operator', '').upper().strip()
        if op in HELIX:
            op_counts[op] += 1
    total = sum(op_counts.values())
    if total == 0:
        continue
    
    op_pcts = {op: op_counts[op] / total * 100 for op in HELIX}
    all_langs[lang] = {
        'family': data['family'],
        'era': data['era'],
        'n_classified': total,
        'op_pcts': op_pcts,
    }

langs = sorted(all_langs.keys())
N = len(langs)
print(f"  {N} languages")

# Build matrix: rows=languages, cols=operators
M = np.array([[all_langs[l]['op_pcts'][op] for op in HELIX] for l in langs])

# ══════════════════════════════════════════════════════════════
#  TEST 1: RAW CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 1: OPERATOR CORRELATION MATRIX (across 28 languages)")
print(f"{'='*70}")

corr = np.corrcoef(M.T)  # 9×9

print(f"\n         ", end="")
for op in HELIX:
    print(f" {op:>6s}", end="")
print()
print("  " + "-"*65)

for i, op1 in enumerate(HELIX):
    print(f"  {op1:>6s} ", end="")
    for j, op2 in enumerate(HELIX):
        r = corr[i,j]
        if i == j:
            print(f"   --- ", end="")
        elif abs(r) >= 0.5:
            print(f" {r:+.3f}*", end="")
        elif abs(r) >= 0.3:
            print(f" {r:+.3f} ", end="")
        else:
            print(f"  {r:+.2f} ", end="")
    print()

print("\n  * = |r| ≥ 0.5")


# ══════════════════════════════════════════════════════════════
#  TEST 2: WITHIN-TRIAD vs BETWEEN-TRIAD CORRELATION
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 2: WITHIN-TRIAD vs BETWEEN-TRIAD CORRELATION")
print(f"{'='*70}")

within_triad = []
between_triad = []

triad_of = {}
for tname, ops in TRIADS.items():
    for op in ops:
        triad_of[op] = tname

for i, op1 in enumerate(HELIX):
    for j, op2 in enumerate(HELIX):
        if i >= j:
            continue
        r = corr[i,j]
        if triad_of[op1] == triad_of[op2]:
            within_triad.append((op1, op2, r, triad_of[op1]))
        else:
            between_triad.append((op1, op2, r))

print("\n  WITHIN-TRIAD correlations:")
for op1, op2, r, tname in sorted(within_triad, key=lambda x: -x[2]):
    marker = "***" if abs(r) >= 0.5 else "**" if abs(r) >= 0.3 else ""
    print(f"    {op1:>4s} ↔ {op2:<4s}  r = {r:+.3f}  [{tname}] {marker}")

print(f"\n  Mean within-triad:  r = {np.mean([r for _,_,r,_ in within_triad]):+.3f}")
print(f"  Mean between-triad: r = {np.mean([r for _,_,r in between_triad]):+.3f}")

# Permutation test: is within-triad mean significantly higher?
rng = np.random.RandomState(42)
observed_diff = np.mean([r for _,_,r,_ in within_triad]) - np.mean([r for _,_,r in between_triad])

all_pairs = [(i,j,corr[i,j]) for i in range(9) for j in range(i+1,9)]
n_within = len(within_triad)  # 9 within-triad pairs
n_perms = 10000
perm_diffs = []

for _ in range(n_perms):
    perm = rng.permutation(len(all_pairs))
    fake_within = [all_pairs[p][2] for p in perm[:n_within]]
    fake_between = [all_pairs[p][2] for p in perm[n_within:]]
    perm_diffs.append(np.mean(fake_within) - np.mean(fake_between))

p_value = np.mean([d >= observed_diff for d in perm_diffs])
print(f"\n  Observed difference: {observed_diff:+.3f}")
print(f"  Permutation p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  ✓ Within-triad correlations are significantly stronger (p={p_value:.4f})")
else:
    print(f"  ✗ No significant difference (p={p_value:.4f})")


# ══════════════════════════════════════════════════════════════
#  TEST 3: WITHIN-POSITION vs BETWEEN-POSITION CORRELATION  
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 3: WITHIN-POSITION vs BETWEEN-POSITION CORRELATION")
print(f"{'='*70}")

pos_of = {}
for pname, ops in POSITIONS.items():
    for op in ops:
        pos_of[op] = pname

within_pos = []
between_pos = []

for i, op1 in enumerate(HELIX):
    for j, op2 in enumerate(HELIX):
        if i >= j:
            continue
        r = corr[i,j]
        if pos_of[op1] == pos_of[op2]:
            within_pos.append((op1, op2, r, pos_of[op1]))
        else:
            between_pos.append((op1, op2, r))

print("\n  WITHIN-POSITION correlations:")
for op1, op2, r, pname in sorted(within_pos, key=lambda x: -x[2]):
    marker = "***" if abs(r) >= 0.5 else "**" if abs(r) >= 0.3 else ""
    print(f"    {op1:>4s} ↔ {op2:<4s}  r = {r:+.3f}  [{pname}] {marker}")

print(f"\n  Mean within-position:  r = {np.mean([r for _,_,r,_ in within_pos]):+.3f}")
print(f"  Mean between-position: r = {np.mean([r for _,_,r in between_pos]):+.3f}")

observed_diff_pos = np.mean([r for _,_,r,_ in within_pos]) - np.mean([r for _,_,r in between_pos])
perm_diffs_pos = []
for _ in range(n_perms):
    perm = rng.permutation(len(all_pairs))
    fake_within = [all_pairs[p][2] for p in perm[:n_within]]
    fake_between = [all_pairs[p][2] for p in perm[n_within:]]
    perm_diffs_pos.append(np.mean(fake_within) - np.mean(fake_between))

p_value_pos = np.mean([d >= observed_diff_pos for d in perm_diffs_pos])
print(f"\n  Observed difference: {observed_diff_pos:+.3f}")
print(f"  Permutation p-value: {p_value_pos:.4f}")


# ══════════════════════════════════════════════════════════════
#  TEST 4: DATA-DRIVEN CLUSTERING — What groups does PCA find?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 4: PCA — What structure does the data naturally show?")
print(f"{'='*70}")

# Standardize columns
M_std = (M - M.mean(axis=0)) / M.std(axis=0)

# PCA via SVD
U, S, Vt = np.linalg.svd(M_std, full_matrices=False)
explained = (S**2) / (S**2).sum()

print("\n  Principal components (variance explained):")
for i in range(min(6, len(S))):
    print(f"    PC{i+1}: {explained[i]*100:.1f}%")
print(f"    Cumulative: {explained[:3].sum()*100:.1f}% in 3 PCs")

# Loadings: which operators load on which PCs?
loadings = Vt[:3].T  # 9 operators × 3 PCs

print(f"\n  Operator loadings on first 3 PCs:")
print(f"  {'Op':>6s} {'PC1':>8s} {'PC2':>8s} {'PC3':>8s}  {'Triad':>15s}")
print("  " + "-"*50)
for i, op in enumerate(HELIX):
    tname = triad_of[op]
    markers = []
    for j in range(3):
        if abs(loadings[i,j]) > 0.4:
            markers.append(f"PC{j+1}")
    marker_str = ", ".join(markers) if markers else ""
    print(f"  {op:>6s} {loadings[i,0]:+8.3f} {loadings[i,1]:+8.3f} {loadings[i,2]:+8.3f}  [{tname}] {marker_str}")

# Do the PCs separate triads?
print(f"\n  If triads are real, operators in the same triad should load similarly.")
print(f"  Visual check: do Existence operators cluster? Structure? Interpretation?")


# ══════════════════════════════════════════════════════════════
#  TEST 5: HIERARCHICAL CLUSTERING of operators
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 5: HIERARCHICAL CLUSTERING of operators by co-variation")
print(f"{'='*70}")

# Use correlation distance: d = 1 - r
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

try:
    dist = 1 - corr  # correlation distance
    # Force perfect symmetry and zero diagonal
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist)
    
    Z = linkage(condensed, method='average')
    
    print("\n  Dendrogram (average linkage on correlation distance):")
    print("  Merge order (earlier = more similar):")
    
    labels = list(HELIX)
    clusters = [[op] for op in HELIX]
    
    for i, (c1, c2, dist_val, n) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        g1 = clusters[c1] if c1 < len(HELIX) else clusters[c1]
        g2 = clusters[c2] if c2 < len(HELIX) else clusters[c2]
        merged = g1 + g2
        clusters.append(merged)
        
        # Check if this merge is within-triad
        triads_involved = set(triad_of[op] for op in merged)
        triad_info = ", ".join(sorted(triads_involved))
        
        g1_str = "+".join(g1)
        g2_str = "+".join(g2)
        print(f"    Step {i+1}: ({g1_str}) + ({g2_str}) = d={dist_val:.3f}  [{triad_info}]")
    
    # Cut at k=3 and compare to EO triads
    cluster_labels = fcluster(Z, t=3, criterion='maxclust')
    
    print(f"\n  Cut at k=3:")
    for k in range(1, 4):
        members = [HELIX[i] for i in range(9) if cluster_labels[i] == k]
        triads = [triad_of[op] for op in members]
        print(f"    Cluster {k}: {', '.join(members)}  (triads: {', '.join(set(triads))})")
    
    # Compare to EO triads
    from sklearn.metrics import adjusted_rand_score
    eo_labels = [0,0,0, 1,1,1, 2,2,2]  # EO triads
    ari = adjusted_rand_score(eo_labels, cluster_labels)
    print(f"\n  ARI(data-driven k=3, EO triads): {ari:.3f}")
    if ari > 0.3:
        print(f"  → Data-driven clusters partially match EO triads")
    elif ari > 0:
        print(f"  → Weak match")
    else:
        print(f"  → No match — data-driven groupings differ from EO triads")

except ImportError:
    print("  scipy not available")


# ══════════════════════════════════════════════════════════════
#  TEST 6: REMOVE HIGH-INS BIAS — Partial correlations
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 6: PARTIAL CORRELATIONS (controlling for INS)")
print(f"{'='*70}")
print("  INS dominates. Remove its effect and see what structure remains.\n")

# Partial correlation: correlate residuals after regressing out INS
ins_col = M[:, HELIX.index('INS')]

partial_corr = np.zeros((9,9))
for i in range(9):
    for j in range(9):
        if i == j:
            partial_corr[i,j] = 1.0
            continue
        if HELIX[i] == 'INS' or HELIX[j] == 'INS':
            partial_corr[i,j] = 0  # Skip INS itself
            continue
        
        # Residualize both variables against INS
        x = M[:, i]
        y = M[:, j]
        
        # Regress out INS
        slope_x = np.polyfit(ins_col, x, 1)
        slope_y = np.polyfit(ins_col, y, 1)
        resid_x = x - np.polyval(slope_x, ins_col)
        resid_y = y - np.polyval(slope_y, ins_col)
        
        if np.std(resid_x) > 0 and np.std(resid_y) > 0:
            partial_corr[i,j] = np.corrcoef(resid_x, resid_y)[0,1]

# Print non-INS partial correlations
non_ins = [op for op in HELIX if op != 'INS']
non_ins_idx = [HELIX.index(op) for op in non_ins]

print(f"         ", end="")
for op in non_ins:
    print(f" {op:>6s}", end="")
print()
print("  " + "-"*55)

for i, op1 in zip(non_ins_idx, non_ins):
    print(f"  {op1:>6s} ", end="")
    for j, op2 in zip(non_ins_idx, non_ins):
        r = partial_corr[i,j]
        if i == j:
            print(f"   --- ", end="")
        elif abs(r) >= 0.5:
            print(f" {r:+.3f}*", end="")
        elif abs(r) >= 0.3:
            print(f" {r:+.3f} ", end="")
        else:
            print(f"  {r:+.2f} ", end="")
    print()

# Within vs between triad (excluding INS pairs)
within_partial = []
between_partial = []

for i, op1 in zip(non_ins_idx, non_ins):
    for j, op2 in zip(non_ins_idx, non_ins):
        if i >= j:
            continue
        r = partial_corr[i,j]
        if triad_of[op1] == triad_of[op2]:
            within_partial.append((op1, op2, r, triad_of[op1]))
        else:
            between_partial.append((op1, op2, r))

print(f"\n  After controlling for INS:")
print(f"  Mean within-triad:  r = {np.mean([r for _,_,r,_ in within_partial]):+.3f}")
print(f"  Mean between-triad: r = {np.mean([r for _,_,r in between_partial]):+.3f}")

print(f"\n  Within-triad pairs (controlling for INS):")
for op1, op2, r, tname in sorted(within_partial, key=lambda x: -x[2]):
    marker = "***" if abs(r) >= 0.5 else "**" if abs(r) >= 0.3 else ""
    print(f"    {op1:>4s} ↔ {op2:<4s}  r = {r:+.3f}  [{tname}] {marker}")


# ══════════════════════════════════════════════════════════════
#  TEST 7: COMPOSITIONAL ARTIFACT CHECK
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 7: COMPOSITIONAL ARTIFACT CHECK")
print(f"{'='*70}")
print("  Percentages sum to 100, so negative correlations are expected.")
print("  Use log-ratio transform to remove compositional constraint.\n")

# Centered log-ratio transform
M_clr = np.zeros_like(M)
for i in range(N):
    # Replace zeros with small value for log
    row = M[i].copy()
    row[row == 0] = 0.01
    log_row = np.log(row)
    M_clr[i] = log_row - log_row.mean()

corr_clr = np.corrcoef(M_clr.T)

print(f"  CLR-transformed correlation matrix:")
print(f"         ", end="")
for op in HELIX:
    print(f" {op:>6s}", end="")
print()
print("  " + "-"*65)

for i, op1 in enumerate(HELIX):
    print(f"  {op1:>6s} ", end="")
    for j, op2 in enumerate(HELIX):
        r = corr_clr[i,j]
        if i == j:
            print(f"   --- ", end="")
        elif abs(r) >= 0.5:
            print(f" {r:+.3f}*", end="")
        elif abs(r) >= 0.3:
            print(f" {r:+.3f} ", end="")
        else:
            print(f"  {r:+.02f} ", end="")
    print()

# Within vs between triad on CLR
within_clr = []
between_clr = []
for i, op1 in enumerate(HELIX):
    for j, op2 in enumerate(HELIX):
        if i >= j:
            continue
        r = corr_clr[i,j]
        if triad_of[op1] == triad_of[op2]:
            within_clr.append((op1, op2, r, triad_of[op1]))
        else:
            between_clr.append((op1, op2, r))

print(f"\n  CLR-transformed:")
print(f"  Mean within-triad:  r = {np.mean([r for _,_,r,_ in within_clr]):+.3f}")
print(f"  Mean between-triad: r = {np.mean([r for _,_,r in between_clr]):+.3f}")

# Permutation test on CLR
observed_diff_clr = np.mean([r for _,_,r,_ in within_clr]) - np.mean([r for _,_,r in between_clr])
all_pairs_clr = [(i,j,corr_clr[i,j]) for i in range(9) for j in range(i+1,9)]
perm_diffs_clr = []
for _ in range(n_perms):
    perm = rng.permutation(len(all_pairs_clr))
    fake_within = [all_pairs_clr[p][2] for p in perm[:n_within]]
    fake_between = [all_pairs_clr[p][2] for p in perm[n_within:]]
    perm_diffs_clr.append(np.mean(fake_within) - np.mean(fake_between))

p_value_clr = np.mean([d >= observed_diff_clr for d in perm_diffs_clr])
print(f"\n  Observed difference (CLR): {observed_diff_clr:+.3f}")
print(f"  Permutation p-value: {p_value_clr:.4f}")


# ══════════════════════════════════════════════════════════════
#  TEST 8: EXHAUSTIVE TRIAD COMPARISON
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST 8: EXHAUSTIVE TRIAD COMPARISON")
print(f"{'='*70}")
print("  Of all 280 ways to partition 9 operators into 3×3,")
print("  which partition maximizes within-group correlation?\n")

from itertools import combinations

def all_3x3_partitions_idx():
    """Generate all ways to partition indices 0-8 into 3 groups of 3."""
    seen = set()
    for g1 in combinations(range(9), 3):
        remaining = [i for i in range(9) if i not in g1]
        for g2 in combinations(remaining, 3):
            g3 = tuple(i for i in remaining if i not in g2)
            canon = tuple(sorted([tuple(sorted(g1)), tuple(sorted(g2)), tuple(sorted(g3))]))
            if canon not in seen:
                seen.add(canon)
                yield canon

# Score each partition by mean within-group correlation
partition_scores = []

# Use CLR correlations (composition-corrected)
for partition in all_3x3_partitions_idx():
    within_rs = []
    for group in partition:
        for i in range(3):
            for j in range(i+1, 3):
                within_rs.append(corr_clr[group[i], group[j]])
    
    mean_within = np.mean(within_rs)
    
    # Is this EO?
    eo_partition = tuple(sorted([
        tuple(sorted([HELIX.index(op) for op in ['NUL','DES','INS']])),
        tuple(sorted([HELIX.index(op) for op in ['SEG','CON','SYN']])),
        tuple(sorted([HELIX.index(op) for op in ['ALT','SUP','REC']])),
    ]))
    is_eo = (partition == eo_partition)
    
    groups_named = [[HELIX[i] for i in g] for g in partition]
    partition_scores.append((mean_within, is_eo, groups_named))

partition_scores.sort(key=lambda x: -x[0])

print(f"  Top 10 partitions by mean within-group CLR correlation:")
print(f"  {'Rank':>4s} {'Mean r':>8s} {'Groups':>55s} {'EO?':>5s}")
print("  " + "-"*80)

eo_rank = None
for i, (score, is_eo, groups) in enumerate(partition_scores[:15]):
    groups_str = " | ".join(",".join(g) for g in groups)
    marker = " ← EO" if is_eo else ""
    print(f"  {i+1:4d} {score:+8.3f}  {groups_str:55s}{marker}")
    if is_eo:
        eo_rank = i + 1

if eo_rank is None:
    for i, (score, is_eo, groups) in enumerate(partition_scores):
        if is_eo:
            eo_rank = i + 1
            print(f"\n  EO partition rank: #{eo_rank} out of {len(partition_scores)}")
            print(f"  EO mean within-group CLR correlation: {score:+.3f}")
            break

print(f"\n  Distribution of scores:")
all_scores = [s[0] for s in partition_scores]
print(f"    Best:  {max(all_scores):+.3f}")
print(f"    Worst: {min(all_scores):+.3f}")
print(f"    Mean:  {np.mean(all_scores):+.3f}")
if eo_rank:
    eo_score = partition_scores[eo_rank-1][0]
    percentile = (1 - eo_rank / len(partition_scores)) * 100
    print(f"    EO:    {eo_score:+.3f} (rank #{eo_rank}, top {percentile:.0f}%)")


# ══════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Correlation heatmap with triad borders
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, corr_mat, title in [
        (axes[0], corr, "Raw Correlation"),
        (axes[1], corr_clr, "CLR-Transformed Correlation"),
    ]:
        im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax.set_xticks(range(9))
        ax.set_xticklabels(HELIX, fontsize=10, fontweight='bold')
        ax.set_yticks(range(9))
        ax.set_yticklabels(HELIX, fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=13)
        
        # Add values
        for i in range(9):
            for j in range(9):
                color = 'white' if abs(corr_mat[i,j]) > 0.5 else 'black'
                ax.text(j, i, f"{corr_mat[i,j]:.2f}", ha='center', va='center',
                       fontsize=7, color=color)
        
        # Triad borders
        for pos in [2.5, 5.5]:
            ax.axhline(pos, color='black', linewidth=2)
            ax.axvline(pos, color='black', linewidth=2)
    
    plt.colorbar(im, ax=axes, shrink=0.8, label='Correlation')
    plt.suptitle("Operator Co-Variation Across 28 Languages\n(triad borders shown)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "crossling_covariation.png"), dpi=150)
    plt.close()
    print(f"\n  ✓ crossling_covariation.png")

except ImportError:
    print("  (matplotlib not available)")


# ══════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  SUMMARY: ARE THE TRIADS REAL?")
print(f"{'='*70}")

print(f"""
  Raw correlation:
    Within-triad mean:  {np.mean([r for _,_,r,_ in within_triad]):+.3f}
    Between-triad mean: {np.mean([r for _,_,r in between_triad]):+.3f}
    p-value: {p_value:.4f}

  CLR-corrected (compositional artifacts removed):
    Within-triad mean:  {np.mean([r for _,_,r,_ in within_clr]):+.3f}
    Between-triad mean: {np.mean([r for _,_,r in between_clr]):+.3f}
    p-value: {p_value_clr:.4f}

  Hierarchical clustering vs EO triads: ARI = {ari:.3f}
  
  EO partition rank (by within-group CLR correlation): #{eo_rank}/{len(partition_scores)}
""")

# Save
report = {
    'raw_within_triad_mean': float(np.mean([r for _,_,r,_ in within_triad])),
    'raw_between_triad_mean': float(np.mean([r for _,_,r in between_triad])),
    'raw_p_value': float(p_value),
    'clr_within_triad_mean': float(np.mean([r for _,_,r,_ in within_clr])),
    'clr_between_triad_mean': float(np.mean([r for _,_,r in between_clr])),
    'clr_p_value': float(p_value_clr),
    'hierarchical_ari': float(ari),
    'eo_partition_rank': eo_rank,
    'total_partitions': len(partition_scores),
}
with open(os.path.join(OUT, "crossling_covariation.json"), 'w') as f:
    json.dump(report, f, indent=2)
print(f"  ✓ Saved crossling_covariation.json")
