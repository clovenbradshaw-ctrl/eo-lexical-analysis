#!/usr/bin/env python3
"""
Step 10c: Operator Influence Fields
====================================
Instead of sub-clustering (which finds no natural boundaries),
map every verb as a point in 9D "operator influence space" —
its cosine similarity to each of the 9 operator centroids.

Then ask:
  1. Within each operator, what direction does the gradient run?
  2. Does the pull direction predict the classifier's alternative?
  3. Does it predict scale?
  4. Does it predict confidence?
  5. What does the full 9D influence profile look like per operator?
  6. Which operator pairs exert mutual pull on each other's verbs?
"""

import json, os, numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

print("Loading data...")
with open(os.path.join(DATA, "llm_classifications.json")) as f:
    cls = json.load(f)["classifications"]
embs = np.load(os.path.join(DATA, "reembed_combined.npz"))["embeddings"]

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']
SCALES = ['physical','social','psychological','informational']
TRIADS = {'Existence':['NUL','DES','INS'],'Structure':['SEG','CON','SYN'],'Interpretation':['ALT','SUP','REC']}
TRIAD_OF = {op: t for t, ops in TRIADS.items() for op in ops}

op_indices = {op: [] for op in HELIX}
for i, c in enumerate(cls):
    if c['operator'] in HELIX:
        op_indices[c['operator']].append(i)

# ── Compute centroids ─────────────────────────────────────────────
centroids = {}
for op in HELIX:
    centroids[op] = embs[op_indices[op]].mean(axis=0)

centroid_matrix = np.array([centroids[op] for op in HELIX])  # 9 x 3072

# ── Compute influence profiles for ALL verbs ──────────────────────
print("Computing influence profiles for all verbs...")
influence = cosine_similarity(embs, centroid_matrix)  # N x 9
print(f"  Influence matrix: {influence.shape}")

# ══════════════════════════════════════════════════════════════════
#  1. INFLUENCE PROFILES PER OPERATOR
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  1. MEAN INFLUENCE PROFILES")
print("="*70)
print(f"\n  Each row: mean similarity of that operator's verbs to ALL 9 centroids")
print(f"  {'':8s}", end="")
for op in HELIX:
    print(f"  →{op:>4s}", end="")
print()
print("  " + "-"*70)

for op in HELIX:
    idx = op_indices[op]
    mean_profile = influence[idx].mean(axis=0)
    print(f"  {op:8s}", end="")
    for j, target in enumerate(HELIX):
        val = mean_profile[j]
        # Bold the self-similarity and the strongest non-self
        print(f"  {val:.3f}", end="")
    # Strongest non-self pull
    non_self = [(HELIX[j], mean_profile[j]) for j in range(9) if HELIX[j] != op]
    strongest = max(non_self, key=lambda x: x[1])
    print(f"   pull→{strongest[0]}({strongest[1]:.3f})")

# ══════════════════════════════════════════════════════════════════
#  2. DOES PULL DIRECTION PREDICT CLASSIFIER'S ALTERNATIVE?
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  2. PULL DIRECTION vs CLASSIFIER ALTERNATIVE")
print("="*70)
print("  For verbs with an alternative, does the geometrically nearest")
print("  other centroid match what the classifier chose?\n")

match_count = 0
total_with_alt = 0
per_op_match = defaultdict(lambda: [0, 0])

for i, c in enumerate(cls):
    op = c['operator']
    alt = c.get('alternative', '')
    if op not in HELIX or alt not in HELIX:
        continue
    
    total_with_alt += 1
    
    # Geometric nearest (excluding self)
    profile = influence[i]
    non_self = [(HELIX[j], profile[j]) for j in range(9) if HELIX[j] != op]
    geo_nearest = max(non_self, key=lambda x: x[1])[0]
    
    if geo_nearest == alt:
        match_count += 1
        per_op_match[op][0] += 1
    per_op_match[op][1] += 1

print(f"  Overall: {match_count}/{total_with_alt} ({match_count/total_with_alt*100:.1f}%) geometric nearest = classifier alternative")
print()
for op in HELIX:
    m, t = per_op_match[op]
    if t > 0:
        print(f"  {op}: {m}/{t} ({m/t*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
#  3. DOES PULL DIRECTION PREDICT SCALE?
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  3. PULL DIRECTION vs SCALE")
print("="*70)
print("  Within each operator, do verbs of different scales")
print("  get pulled toward different other operators?\n")

for op in HELIX:
    idx = op_indices[op]
    if len(idx) < 50:
        continue
    
    print(f"  {op} ({len(idx)} verbs):")
    
    # For each scale, compute mean pull profile
    scale_profiles = {}
    for scale in SCALES:
        scale_idx = [j for j in idx if cls[j].get('scale') == scale]
        if len(scale_idx) < 10:
            continue
        mean_pull = influence[scale_idx].mean(axis=0)
        # Strongest non-self
        non_self = [(HELIX[k], mean_pull[k]) for k in range(9) if HELIX[k] != op]
        strongest = max(non_self, key=lambda x: x[1])
        scale_profiles[scale] = (strongest[0], strongest[1], len(scale_idx))
    
    for scale in SCALES:
        if scale in scale_profiles:
            target, sim, n = scale_profiles[scale]
            print(f"    {scale:15s} ({n:4d} verbs) → pulled toward {target} ({sim:.3f})")

# ══════════════════════════════════════════════════════════════════
#  4. DOES PULL STRENGTH PREDICT CONFIDENCE?
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  4. PULL STRENGTH vs CONFIDENCE")
print("="*70)
print("  Do high-confidence verbs sit deeper inside their operator")
print("  (further from any other centroid)?\n")

for conf_level in ['high', 'medium', 'low']:
    margins = []
    for i, c in enumerate(cls):
        if c.get('confidence') != conf_level or c['operator'] not in HELIX:
            continue
        op = c['operator']
        self_sim = influence[i, HELIX.index(op)]
        max_other = max(influence[i, j] for j in range(9) if HELIX[j] != op)
        margins.append(self_sim - max_other)
    
    if margins:
        print(f"  {conf_level:8s}: n={len(margins):5d}  mean_margin={np.mean(margins):.4f}  median={np.median(margins):.4f}  std={np.std(margins):.4f}")

# ══════════════════════════════════════════════════════════════════
#  5. THE PULL FIELD: WHICH PAIRS EXERT MUTUAL ATTRACTION?
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  5. MUTUAL PULL MATRIX")
print("="*70)
print("  Each cell: mean similarity of row-operator's verbs to column centroid,")
print("  MINUS the global mean similarity to that centroid (excess pull).\n")

# Global mean similarity to each centroid
global_mean = influence.mean(axis=0)  # 9 values

print(f"  Global mean pull: ", end="")
for j, op in enumerate(HELIX):
    print(f"{op}={global_mean[j]:.3f} ", end="")
print()

# Excess pull matrix
print(f"\n  Excess pull (above global mean):")
print(f"  {'':8s}", end="")
for op in HELIX:
    print(f"  →{op:>4s}", end="")
print()
print("  " + "-"*70)

excess_matrix = np.zeros((9, 9))
for i, op in enumerate(HELIX):
    idx = op_indices[op]
    mean_profile = influence[idx].mean(axis=0)
    excess = mean_profile - global_mean
    excess_matrix[i] = excess
    
    print(f"  {op:8s}", end="")
    for j in range(9):
        val = excess[j]
        if i == j:
            print(f"   [{val:+.3f}]", end="")
        elif abs(val) > 0.01:
            print(f"  {val:+.4f}", end="")
        else:
            print(f"  {'·':>6s}", end="")
    print()

# ══════════════════════════════════════════════════════════════════
#  6. TOP PULLED VERBS PER OPERATOR PAIR
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  6. VERBS MOST PULLED BETWEEN OPERATOR PAIRS")
print("="*70)
print("  For each operator, the 5 verbs with strongest pull toward")
print("  each neighboring operator.\n")

for op in HELIX:
    idx = op_indices[op]
    if len(idx) < 20:
        continue
    
    op_i = HELIX.index(op)
    
    # Find the 3 strongest non-self pulls for this operator overall
    mean_profile = influence[idx].mean(axis=0)
    non_self = [(HELIX[j], mean_profile[j]) for j in range(9) if j != op_i]
    top_pulls = sorted(non_self, key=lambda x: -x[1])[:3]
    
    print(f"\n  {op}:")
    for target_op, _ in top_pulls:
        target_j = HELIX.index(target_op)
        
        # For each verb in op, compute relative pull toward target
        # (similarity to target minus similarity to own centroid)
        relative_pulls = []
        for k, glob_idx in enumerate(idx):
            pull = influence[glob_idx, target_j] - influence[glob_idx, op_i]
            relative_pulls.append((glob_idx, pull))
        
        # Most pulled toward target (smallest margin = closest to defecting)
        relative_pulls.sort(key=lambda x: -x[1])
        
        print(f"    → {target_op} (top 5 most pulled):")
        for glob_idx, pull in relative_pulls[:5]:
            c = cls[glob_idx]
            own_sim = influence[glob_idx, op_i]
            target_sim = influence[glob_idx, target_j]
            alt = c.get('alternative', '')
            alt_str = f" [alt={alt}]" if alt else ""
            print(f"      {c['verb']:25s} self={own_sim:.3f} {target_op}={target_sim:.3f} gap={own_sim-target_sim:+.3f}{alt_str}")

# ══════════════════════════════════════════════════════════════════
#  7. OPERATOR INFLUENCE ENTROPY
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  7. INFLUENCE ENTROPY (how focused vs diffuse is each verb?)")
print("="*70)
print("  Low entropy = strongly belonging to one operator")
print("  High entropy = pulled equally by many operators\n")

from scipy.stats import entropy as scipy_entropy

for op in HELIX:
    idx = op_indices[op]
    if len(idx) < 10:
        continue
    
    entropies = []
    for j in idx:
        # Normalize influence profile to probability distribution
        profile = influence[j]
        # Shift to positive and normalize
        p = profile - profile.min() + 1e-10
        p = p / p.sum()
        ent = scipy_entropy(p)
        entropies.append(ent)
    
    # Also find the most and least entropic verbs
    ent_with_idx = list(zip(entropies, idx))
    ent_with_idx.sort(key=lambda x: x[0])
    
    print(f"  {op:3s} ({len(idx):5d} verbs): entropy mean={np.mean(entropies):.4f} std={np.std(entropies):.4f}")
    
    # Most focused (lowest entropy)
    print(f"    Most focused: ", end="")
    for ent, j in ent_with_idx[:3]:
        print(f"{cls[j]['verb']}({ent:.3f}) ", end="")
    print()
    
    # Most diffuse (highest entropy)
    print(f"    Most diffuse: ", end="")
    for ent, j in ent_with_idx[-3:]:
        print(f"{cls[j]['verb']}({ent:.3f}) ", end="")
    print()

# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    op_colors = {
        'NUL': '#e74c3c', 'DES': '#e67e22', 'INS': '#f1c40f',
        'SEG': '#3498db', 'CON': '#2ecc71', 'SYN': '#9b59b6',
        'ALT': '#1abc9c', 'SUP': '#e91e63', 'REC': '#8bc34a',
    }

    # Fig 1: Excess pull heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mask diagonal for clearer view
    display = excess_matrix.copy()
    
    vmax = max(abs(display.min()), abs(display.max()))
    im = ax.imshow(display, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels([f"→{op}" for op in HELIX], fontsize=10, fontweight='bold')
    ax.set_yticklabels(HELIX, fontsize=11, fontweight='bold')
    
    for i in range(9):
        for j in range(9):
            val = display[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            fmt = f"{val:+.3f}" if i != j else f"[{val:+.3f}]"
            ax.text(j, i, fmt, ha='center', va='center', fontsize=7, color=color)
    
    for start in [0, 3, 6]:
        rect = plt.Rectangle((start-0.5, start-0.5), 3, 3, fill=False,
                            edgecolor='black', linewidth=2.5)
        ax.add_patch(rect)
    
    plt.colorbar(im, label='Excess pull (above global mean)')
    ax.set_title("Operator Influence Field\n(red = attracted, blue = repelled, relative to global baseline)", fontsize=13)
    ax.set_xlabel("Pulled toward centroid of...", fontsize=11)
    ax.set_ylabel("Verbs belonging to...", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "influence_field.png"), dpi=150)
    plt.close()
    print("  ✓ influence_field.png")

    # Fig 2: Confidence vs margin scatterplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for conf_level, color, marker in [('high', '#2ecc71', 'o'), ('medium', '#f1c40f', 's'), ('low', '#e74c3c', '^')]:
        margins = []
        entropies_plot = []
        for i, c in enumerate(cls):
            if c.get('confidence') != conf_level or c['operator'] not in HELIX:
                continue
            op_i = HELIX.index(c['operator'])
            self_sim = influence[i, op_i]
            max_other = max(influence[i, j] for j in range(9) if j != op_i)
            margins.append(self_sim - max_other)
        
        if margins:
            ax.hist(margins, bins=50, alpha=0.5, color=color, label=f"{conf_level} (n={len(margins)})")
    
    ax.set_xlabel("Margin (self_sim - nearest_other_sim)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Classification Confidence vs Geometric Margin", fontsize=13)
    ax.legend()
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "influence_confidence.png"), dpi=150)
    plt.close()
    print("  ✓ influence_confidence.png")

    # Fig 3: Per-operator pull profiles (radar chart style, but as bar groups)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    for ax_i, op in enumerate(HELIX):
        ax = axes[ax_i // 3][ax_i % 3]
        idx = op_indices[op]
        
        if len(idx) < 10:
            ax.set_title(f"{op} (n={len(idx)})")
            ax.text(0.5, 0.5, "too few", ha='center', va='center', transform=ax.transAxes)
            continue
        
        mean_profile = influence[idx].mean(axis=0)
        
        # Sort by pull strength, excluding self
        op_j = HELIX.index(op)
        others = [(HELIX[j], mean_profile[j], j) for j in range(9) if j != op_j]
        others.sort(key=lambda x: -x[1])
        
        labels = [o[0] for o in others]
        values = [o[1] for o in others]
        colors_bar = [op_colors[o[0]] for o in others]
        
        bars = ax.barh(range(len(others)), values, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(others)))
        ax.set_yticklabels(labels, fontsize=9, fontweight='bold')
        ax.set_title(f"{op} (n={len(idx)}) — self={mean_profile[op_j]:.3f}", fontsize=11, fontweight='bold',
                    color=op_colors[op])
        ax.set_xlim(mean_profile.min() - 0.01, mean_profile.max() + 0.01)
        ax.invert_yaxis()
        
        # Add global mean reference line
        ax.axvline(global_mean.mean(), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.suptitle("Operator Influence Profiles\n(each operator's verbs: mean similarity to all other centroids)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "influence_profiles.png"), dpi=150)
    plt.close()
    print("  ✓ influence_profiles.png")

except ImportError:
    print("  (matplotlib not available)")

# ── SAVE ──────────────────────────────────────────────────────────
report = {
    'global_mean_pull': {HELIX[j]: float(global_mean[j]) for j in range(9)},
    'excess_matrix': {
        HELIX[i]: {HELIX[j]: float(excess_matrix[i,j]) for j in range(9)}
        for i in range(9)
    },
    'pull_predicts_alternative': {
        'overall': f"{match_count}/{total_with_alt} ({match_count/total_with_alt*100:.1f}%)",
        'per_operator': {op: f"{per_op_match[op][0]}/{per_op_match[op][1]}" for op in HELIX if per_op_match[op][1] > 0},
    },
}

report_path = os.path.join(OUT, "influence_report.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*70}")
print("  INFLUENCE FIELD ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"  Report: {report_path}")
print(f"  Visuals: {VIZ}/influence_*.png")
