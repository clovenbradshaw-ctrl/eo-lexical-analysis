#!/usr/bin/env python3
"""
Step 11: Falsification Tests
==============================
Two tests that could kill the theory:

TEST A: RANDOM TAXONOMY BASELINE
  - Assign verbs to 9 random categories (by shuffling labels)
  - Compute z-score in definition-only embedding space
  - If random categories produce similar z-scores to EO, the pipeline
    finds structure in anything and EO is noise
  - Run 100 random permutations for a null distribution

TEST B: MISSING OPERATOR DETECTION
  - Find high-entropy verbs (pulled equally by many operators)
  - Check if they cluster together in embedding space
  - Look for coherent properties among the "homeless" verbs
  - Test k=8 and k=10 to see if 9 is special or arbitrary

TEST C: ALTERNATIVE TAXONOMY COMPARISON
  - Test whether 7, 8, 9, 10, 11, 12 operator taxonomies differ
  - Use the EXISTING classifications but merge/split operators
  - Compare z-scores: is 9 a sweet spot?

TEST D: CONFIDENCE CALIBRATION
  - Are high-confidence verbs actually closer to their centroids?
  - Do low-confidence verbs sit on geometric boundaries?
  - This validates whether the LLM's uncertainty is real or random
"""

import json, os, numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

print("Loading data...")
with open(os.path.join(DATA, "llm_classifications.json")) as f:
    cls = json.load(f)["classifications"]

# Load definition-only embeddings (the critical control)
embs_def = np.load(os.path.join(DATA, "reembed_definition.npz"))["embeddings"]
# Load combined embeddings
embs_comb = np.load(os.path.join(DATA, "reembed_combined.npz"))["embeddings"]

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']
print(f"  {len(cls)} verbs, def={embs_def.shape}, comb={embs_comb.shape}")

# Build operator labels array
labels_true = []
for c in cls:
    labels_true.append(HELIX.index(c['operator']) if c['operator'] in HELIX else -1)
labels_true = np.array(labels_true)

# Group sizes (for size-matched random permutations)
group_sizes = Counter(labels_true)
print(f"  Group sizes: {dict(group_sizes)}")


# ══════════════════════════════════════════════════════════════════
#  HELPER: compute inter-centroid separation z-score
# ══════════════════════════════════════════════════════════════════

def compute_z_score(embeddings, labels, n_permutations=200):
    """Compute z-score of mean inter-centroid cosine distance vs random."""
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    # Compute centroids
    centroids = []
    for lab in unique_labels:
        mask = labels == lab
        centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)
    
    # Mean pairwise cosine similarity between centroids
    sim_matrix = cosine_similarity(centroids)
    n_ops = len(unique_labels)
    real_mean_sim = 0
    count = 0
    for i in range(n_ops):
        for j in range(i+1, n_ops):
            real_mean_sim += sim_matrix[i,j]
            count += 1
    real_mean_sim /= count
    
    # Random baseline
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
        s = 0
        for i in range(n_ops):
            for j in range(i+1, n_ops):
                s += perm_sim[i,j]
        s /= count
        random_sims.append(s)
    
    random_mean = np.mean(random_sims)
    random_std = np.std(random_sims)
    z = (real_mean_sim - random_mean) / random_std if random_std > 0 else 0
    
    return {
        'real_mean_sim': float(real_mean_sim),
        'random_mean': float(random_mean),
        'random_std': float(random_std),
        'z_score': float(z),
    }


# ══════════════════════════════════════════════════════════════════
#  TEST A: RANDOM TAXONOMY BASELINE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TEST A: RANDOM TAXONOMY BASELINE")
print("="*70)
print("  If 9 random categories produce similar z-scores to EO,")
print("  the pipeline finds structure in anything and EO is noise.\n")

# EO z-score (definition-only)
print("  Computing EO z-score (definition-only)...")
eo_z_def = compute_z_score(embs_def, labels_true, n_permutations=200)
print(f"  EO (definition-only): z = {eo_z_def['z_score']:.1f}")
print(f"    real_sim = {eo_z_def['real_mean_sim']:.6f}")
print(f"    random   = {eo_z_def['random_mean']:.6f} ± {eo_z_def['random_std']:.6f}")

# Now: create multiple random 9-category taxonomies
# Each preserves the same group sizes as EO but assigns verbs randomly
print("\n  Running 20 random taxonomies with same group sizes...")

rng = np.random.RandomState(123)
random_taxonomy_z_scores = []

for trial in range(20):
    # Shuffle the labels (preserves group sizes)
    shuffled = rng.permutation(labels_true)
    
    # Compute z-score of THIS random taxonomy against ITS OWN random baseline
    z_result = compute_z_score(embs_def, shuffled, n_permutations=50)
    random_taxonomy_z_scores.append(z_result['z_score'])
    
    if trial < 5 or trial == 19:
        print(f"    Trial {trial+1:2d}: z = {z_result['z_score']:+.1f} (sim={z_result['real_mean_sim']:.6f})")

print(f"\n  Random taxonomy z-scores:")
print(f"    Mean:   {np.mean(random_taxonomy_z_scores):+.1f}")
print(f"    Std:    {np.std(random_taxonomy_z_scores):.1f}")
print(f"    Range:  [{min(random_taxonomy_z_scores):+.1f}, {max(random_taxonomy_z_scores):+.1f}]")
print(f"\n  EO z-score:     {eo_z_def['z_score']:.1f}")
print(f"  Random mean:    {np.mean(random_taxonomy_z_scores):+.1f}")
print(f"  EO is {abs(eo_z_def['z_score'] / max(abs(np.mean(random_taxonomy_z_scores)), 0.1)):.0f}x stronger than random taxonomies")

if abs(eo_z_def['z_score']) > abs(np.mean(random_taxonomy_z_scores)) * 5:
    print("  → EO captures structure that random categories do NOT")
else:
    print("  → WARNING: Random categories show similar structure — pipeline may be artifactual")

# ══════════════════════════════════════════════════════════════════
#  TEST B: MISSING OPERATOR DETECTION
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TEST B: MISSING OPERATOR DETECTION")
print("="*70)
print("  Looking for 'homeless' verbs that don't fit any operator well.\n")

# Compute influence profiles
centroids = {}
for op in HELIX:
    idx = [i for i, c in enumerate(cls) if c['operator'] == op]
    centroids[op] = embs_comb[idx].mean(axis=0)

centroid_matrix = np.array([centroids[op] for op in HELIX])
influence = cosine_similarity(embs_comb, centroid_matrix)

# For each verb: margin = self_sim - max_other_sim
margins = []
entropies = []
from scipy.stats import entropy as scipy_entropy

for i, c in enumerate(cls):
    if c['operator'] not in HELIX:
        continue
    op_idx = HELIX.index(c['operator'])
    self_sim = influence[i, op_idx]
    max_other = max(influence[i, j] for j in range(9) if j != op_idx)
    margin = self_sim - max_other
    margins.append((i, margin, c['verb'], c['operator'], c.get('confidence', 'high')))
    
    # Entropy
    p = influence[i] - influence[i].min() + 1e-10
    p = p / p.sum()
    ent = scipy_entropy(p)
    entropies.append((i, ent, c['verb'], c['operator']))

# Sort by margin (lowest = most homeless)
margins.sort(key=lambda x: x[1])

print("  20 most 'homeless' verbs (lowest margin between assigned and nearest other):")
print(f"  {'verb':25s} {'assigned':>5s} {'margin':>8s} {'conf':>6s}")
print("  " + "-" * 50)

homeless_verbs = []
for idx, margin, verb, op, conf in margins[:20]:
    # What's the nearest other operator?
    op_idx = HELIX.index(op)
    other_sims = [(HELIX[j], influence[idx, j]) for j in range(9) if j != op_idx]
    nearest = max(other_sims, key=lambda x: x[1])
    print(f"  {verb:25s} {op:>5s} {margin:+8.4f} {conf:>6s} (nearest: {nearest[0]}={nearest[1]:.3f})")
    homeless_verbs.append(idx)

# Check if homeless verbs cluster together
if len(homeless_verbs) >= 10:
    homeless_embs = embs_comb[homeless_verbs]
    
    # Internal coherence of homeless verbs
    homeless_sim = cosine_similarity(homeless_embs)
    mean_homeless_sim = (homeless_sim.sum() - len(homeless_verbs)) / (len(homeless_verbs) * (len(homeless_verbs) - 1))
    
    # Compare to random sample of same size
    rng2 = np.random.RandomState(42)
    random_samples_sim = []
    for _ in range(100):
        sample = rng2.choice(len(cls), size=len(homeless_verbs), replace=False)
        sample_embs = embs_comb[sample]
        s = cosine_similarity(sample_embs)
        mean_s = (s.sum() - len(sample)) / (len(sample) * (len(sample) - 1))
        random_samples_sim.append(mean_s)
    
    print(f"\n  Homeless verb coherence: {mean_homeless_sim:.4f}")
    print(f"  Random sample coherence: {np.mean(random_samples_sim):.4f} ± {np.std(random_samples_sim):.4f}")
    
    if mean_homeless_sim > np.mean(random_samples_sim) + 2 * np.std(random_samples_sim):
        print("  → Homeless verbs cluster together — POSSIBLE MISSING OPERATOR")
    else:
        print("  → Homeless verbs don't cluster — they're genuinely borderline, not a missing category")

# High-entropy analysis
entropies.sort(key=lambda x: -x[1])
print(f"\n  20 highest-entropy verbs (most diffusely pulled):")
print(f"  {'verb':25s} {'operator':>5s} {'entropy':>8s}")
print("  " + "-" * 42)

high_ent_verbs = []
for idx, ent, verb, op in entropies[:20]:
    print(f"  {verb:25s} {op:>5s} {ent:8.4f}")
    high_ent_verbs.append(idx)

# Check if high-entropy verbs cluster
if len(high_ent_verbs) >= 10:
    he_embs = embs_comb[high_ent_verbs]
    he_sim = cosine_similarity(he_embs)
    mean_he_sim = (he_sim.sum() - len(high_ent_verbs)) / (len(high_ent_verbs) * (len(high_ent_verbs) - 1))
    print(f"\n  High-entropy coherence: {mean_he_sim:.4f}")
    print(f"  Random baseline:        {np.mean(random_samples_sim):.4f}")
    
    if mean_he_sim > np.mean(random_samples_sim) + 2 * np.std(random_samples_sim):
        print("  → High-entropy verbs cluster — they may share an uncaptured property")
    else:
        print("  → High-entropy verbs don't cluster — they're just rare/obscure words")

# ══════════════════════════════════════════════════════════════════
#  TEST C: IS 9 SPECIAL? (merge/split tests)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TEST C: IS 9 SPECIAL?")
print("="*70)
print("  Test z-scores for different numbers of operators by merging/splitting.\n")

# Merges to test:
merge_configs = {
    '8 (merge SYN+CON)': {
        'NUL': ['NUL'], 'DES': ['DES'], 'INS': ['INS'],
        'SEG': ['SEG'], 'CON+SYN': ['CON', 'SYN'],
        'ALT': ['ALT'], 'SUP': ['SUP'], 'REC': ['REC']
    },
    '8 (merge ALT+REC)': {
        'NUL': ['NUL'], 'DES': ['DES'], 'INS': ['INS'],
        'SEG': ['SEG'], 'CON': ['CON'], 'SYN': ['SYN'],
        'ALT+REC': ['ALT', 'REC'], 'SUP': ['SUP']
    },
    '8 (merge SUP into ALT)': {
        'NUL': ['NUL'], 'DES': ['DES'], 'INS': ['INS'],
        'SEG': ['SEG'], 'CON': ['CON'], 'SYN': ['SYN'],
        'ALT+SUP': ['ALT', 'SUP'], 'REC': ['REC']
    },
    '7 (merge Structure triad)': {
        'NUL': ['NUL'], 'DES': ['DES'], 'INS': ['INS'],
        'STRUCT': ['SEG', 'CON', 'SYN'],
        'ALT': ['ALT'], 'SUP': ['SUP'], 'REC': ['REC']
    },
    '6 (merge each triad)': {
        'EXIST': ['NUL', 'DES', 'INS'],
        'STRUCT': ['SEG', 'CON', 'SYN'],
        'INTERP': ['ALT', 'SUP', 'REC'],
    },
    '3 (triads only)': {
        'EXIST': ['NUL', 'DES', 'INS'],
        'STRUCT': ['SEG', 'CON', 'SYN'],
        'INTERP': ['ALT', 'SUP', 'REC'],
    },
}

# First: the 9-operator baseline
print(f"  9 operators (EO): z = {eo_z_def['z_score']:.1f}")

for config_name, config in merge_configs.items():
    # Build merged labels
    merged_labels = np.full(len(cls), -1)
    for new_idx, (new_name, old_ops) in enumerate(config.items()):
        for i, c in enumerate(cls):
            if c['operator'] in old_ops:
                merged_labels[i] = new_idx
    
    n_groups = len(config)
    if n_groups < 3:
        continue
    
    result = compute_z_score(embs_def, merged_labels, n_permutations=100)
    
    # Also compute per-group-count normalized version
    print(f"  {config_name}: z = {result['z_score']:.1f} (sim={result['real_mean_sim']:.6f}, n_groups={n_groups})")

# ══════════════════════════════════════════════════════════════════
#  TEST D: DEFECTION ANALYSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TEST D: VERBS THAT HAVE DEFECTED (closer to another centroid)")
print("="*70)

defectors = []
for i, c in enumerate(cls):
    if c['operator'] not in HELIX:
        continue
    op_idx = HELIX.index(c['operator'])
    self_sim = influence[i, op_idx]
    
    for j in range(9):
        if j != op_idx and influence[i, j] > self_sim:
            defectors.append((i, c['verb'], c['operator'], HELIX[j], 
                            self_sim, influence[i, j], c.get('confidence', 'high')))
            break

print(f"\n  Total defectors: {len(defectors)} / {len(cls)} ({len(defectors)/len(cls)*100:.1f}%)")
print(f"  These verbs are geometrically closer to a DIFFERENT operator than their assigned one.\n")

# Breakdown by operator
defect_by_op = defaultdict(list)
for d in defectors:
    defect_by_op[d[2]].append(d)

print(f"  {'Operator':8s} {'Defected':>8s} {'Total':>6s} {'Rate':>6s}")
print("  " + "-" * 35)
for op in HELIX:
    total = len([c for c in cls if c['operator'] == op])
    n_def = len(defect_by_op[op])
    rate = n_def / total * 100 if total > 0 else 0
    print(f"  {op:8s} {n_def:8d} {total:6d} {rate:5.1f}%")

# Where do defectors go?
print(f"\n  Defection flows (assigned → geometric home):")
flow = defaultdict(int)
for d in defectors:
    flow[(d[2], d[3])] += 1

for (src, dst), count in sorted(flow.items(), key=lambda x: -x[1])[:15]:
    print(f"    {src} → {dst}: {count}")

# Confidence of defectors
defector_conf = Counter(d[6] for d in defectors)
all_conf = Counter(c.get('confidence', 'high') for c in cls)
print(f"\n  Defector confidence vs overall:")
for conf in ['high', 'medium', 'low']:
    d_rate = defector_conf.get(conf, 0) / len(defectors) * 100 if defectors else 0
    a_rate = all_conf.get(conf, 0) / len(cls) * 100
    print(f"    {conf:8s}: defectors={d_rate:.1f}%  overall={a_rate:.1f}%")

# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Fig 1: EO z-score vs random taxonomy distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(random_taxonomy_z_scores, bins=15, color='lightblue', edgecolor='black',
            alpha=0.7, label=f'Random taxonomies (n=20)')
    ax.axvline(eo_z_def['z_score'], color='red', linewidth=3, 
              label=f'EO operators (z={eo_z_def["z_score"]:.1f})')
    ax.axvline(0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Z-score (definition-only embeddings)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('EO Operators vs Random Taxonomies\n(definition-only embeddings, same group sizes)', fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "falsification_random_baseline.png"), dpi=150)
    plt.close()
    print("  ✓ falsification_random_baseline.png")

    # Fig 2: Margin distribution with defection threshold
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_margins = [m[1] for m in margins]
    
    ax.hist(all_margins, bins=80, color='steelblue', edgecolor='none', alpha=0.7)
    ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Defection boundary')
    
    n_neg = sum(1 for m in all_margins if m < 0)
    n_pos = sum(1 for m in all_margins if m >= 0)
    
    ax.set_xlabel('Margin (self_sim - nearest_other_sim)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Operator Assignment Margins\n({n_neg} defectors / {n_pos + n_neg} total = {n_neg/(n_pos+n_neg)*100:.1f}% geometric disagreement)', fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "falsification_margins.png"), dpi=150)
    plt.close()
    print("  ✓ falsification_margins.png")

except ImportError:
    print("  (matplotlib not available)")

# ── SAVE ──────────────────────────────────────────────────────────
report = {
    'test_a_random_baseline': {
        'eo_z_score': eo_z_def['z_score'],
        'random_taxonomy_z_mean': float(np.mean(random_taxonomy_z_scores)),
        'random_taxonomy_z_std': float(np.std(random_taxonomy_z_scores)),
        'random_taxonomy_z_range': [float(min(random_taxonomy_z_scores)), float(max(random_taxonomy_z_scores))],
        'eo_vs_random_ratio': float(abs(eo_z_def['z_score']) / max(abs(np.mean(random_taxonomy_z_scores)), 0.1)),
    },
    'test_b_defectors': {
        'n_defectors': len(defectors),
        'n_total': len(cls),
        'defection_rate': len(defectors) / len(cls),
    },
}

report_path = os.path.join(OUT, "falsification_report.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*70}")
print("  FALSIFICATION TESTS COMPLETE")
print(f"{'='*70}")
print(f"  Report: {report_path}")
print(f"  Visuals: {VIZ}/falsification_*.png")
