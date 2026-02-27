#!/usr/bin/env python3
"""
Step 12: Taxonomy Comparison
==============================
Is EO finding real ontological structure, or just rediscovering
WordNet's filing system?

Compare z-scores in definition-only embedding space:
  1. EO operators (9 categories)
  2. WordNet lexicographer categories (~15 categories)
  3. Morphological groupings (prefix-based)
  4. Random taxonomies (baseline, from step 11)

If EO ≤ WordNet, the operators are downstream of lexicography.
If EO > WordNet, EO is finding structure the lexicographers didn't encode.
"""

import json, os, re, numpy as np
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

embs_def = np.load(os.path.join(DATA, "reembed_definition.npz"))["embeddings"]
embs_comb = np.load(os.path.join(DATA, "reembed_combined.npz"))["embeddings"]

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']
N = len(cls)
print(f"  {N} verbs, {embs_def.shape[1]} dimensions")


# ══════════════════════════════════════════════════════════════════
#  Z-SCORE COMPUTATION (reusable)
# ══════════════════════════════════════════════════════════════════

def compute_z_score(embeddings, labels, n_perm=200, label_name="taxonomy"):
    """Compute z-score of inter-centroid separation vs random permutations."""
    unique = sorted(set(labels))
    # Remove -1 (unassigned)
    unique = [u for u in unique if u >= 0]
    n_groups = len(unique)
    
    if n_groups < 2:
        return None
    
    # Map labels to 0..n_groups-1
    label_map = {u: i for i, u in enumerate(unique)}
    mapped = np.array([label_map.get(l, -1) for l in labels])
    
    # Group sizes
    sizes = Counter(mapped)
    sizes.pop(-1, None)
    
    # Compute centroids
    centroids = []
    for i in range(n_groups):
        mask = mapped == i
        if mask.sum() == 0:
            centroids.append(np.zeros(embeddings.shape[1]))
        else:
            centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)
    
    # Mean pairwise cosine similarity
    sim = cosine_similarity(centroids)
    pairs = []
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            pairs.append(sim[i,j])
    real_mean = np.mean(pairs)
    
    # Random baseline
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
        perm_pairs = []
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                perm_pairs.append(perm_sim[i,j])
        random_means.append(np.mean(perm_pairs))
    
    rand_mean = np.mean(random_means)
    rand_std = np.std(random_means)
    z = (real_mean - rand_mean) / rand_std if rand_std > 0 else 0
    
    return {
        'name': label_name,
        'n_groups': n_groups,
        'group_sizes': dict(Counter(mapped[mapped >= 0])),
        'real_mean_sim': float(real_mean),
        'random_mean': float(rand_mean),
        'random_std': float(rand_std),
        'z_score': float(z),
    }


# ══════════════════════════════════════════════════════════════════
#  TAXONOMY 1: EO OPERATORS (baseline)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TAXONOMY 1: EO OPERATORS")
print("="*70)

eo_labels = np.array([HELIX.index(c['operator']) if c['operator'] in HELIX else -1 for c in cls])
eo_result = compute_z_score(embs_def, eo_labels, n_perm=200, label_name="EO operators")
print(f"  {eo_result['n_groups']} groups, z = {eo_result['z_score']:.1f}")
print(f"  real_sim = {eo_result['real_mean_sim']:.6f}, random = {eo_result['random_mean']:.6f} ± {eo_result['random_std']:.6f}")


# ══════════════════════════════════════════════════════════════════
#  TAXONOMY 2: WORDNET LEXICOGRAPHER CATEGORIES
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TAXONOMY 2: WORDNET LEXICOGRAPHER CATEGORIES")
print("="*70)

# WordNet verb lexicographer files:
# verb.body, verb.change, verb.cognition, verb.communication,
# verb.competition, verb.consumption, verb.contact, verb.creation,
# verb.emotion, verb.motion, verb.perception, verb.possession,
# verb.social, verb.stative, verb.weather

# Extract from WordNet directly
try:
    from nltk.corpus import wordnet as wn
    import nltk
    
    # Map each verb in our dataset to its WordNet lexicographer category
    wn_categories = {
        29: 'body', 30: 'change', 31: 'cognition', 32: 'communication',
        33: 'competition', 34: 'consumption', 35: 'contact', 36: 'creation',
        37: 'emotion', 38: 'motion', 39: 'perception', 40: 'possession',
        41: 'social', 42: 'stative', 43: 'weather'
    }
    
    wn_labels = np.full(N, -1)
    wn_label_names = {}
    
    for i, c in enumerate(cls):
        verb = c['verb']
        synset_id = c.get('synset', '')
        
        # Try to get the synset directly
        synset = None
        if synset_id:
            try:
                synset = wn.synset(synset_id)
            except:
                pass
        
        if synset is None:
            # Try to look up by verb name
            # Clean the verb for WordNet lookup
            clean_verb = verb.replace(' ', '_').lower()
            synsets = wn.synsets(clean_verb, pos=wn.VERB)
            if synsets:
                synset = synsets[0]  # Take first sense
        
        if synset is not None:
            lex_id = synset.lexname()
            # Convert lexname to category number
            cat_name = lex_id.replace('verb.', '') if lex_id.startswith('verb.') else lex_id
            
            if cat_name not in wn_label_names:
                wn_label_names[cat_name] = len(wn_label_names)
            wn_labels[i] = wn_label_names[cat_name]
    
    n_mapped = (wn_labels >= 0).sum()
    print(f"  Mapped {n_mapped}/{N} verbs to WordNet categories")
    print(f"  Categories found: {len(wn_label_names)}")
    for cat, idx in sorted(wn_label_names.items(), key=lambda x: x[1]):
        count = (wn_labels == idx).sum()
        print(f"    {cat:20s}: {count:5d} verbs")
    
    wn_result = compute_z_score(embs_def, wn_labels, n_perm=200, label_name="WordNet lexicographer")
    print(f"\n  {wn_result['n_groups']} groups, z = {wn_result['z_score']:.1f}")
    print(f"  real_sim = {wn_result['real_mean_sim']:.6f}, random = {wn_result['random_mean']:.6f} ± {wn_result['random_std']:.6f}")

except ImportError:
    print("  ✗ NLTK not available. Install with: pip install nltk")
    print("    Then: python -c \"import nltk; nltk.download('wordnet')\"")
    wn_result = None


# ══════════════════════════════════════════════════════════════════
#  TAXONOMY 3: MORPHOLOGICAL (prefix-based)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TAXONOMY 3: MORPHOLOGICAL (prefix-based)")
print("="*70)

# Group verbs by common prefixes that carry semantic content
prefixes = {
    're-': re.compile(r'^re[a-z]'),      # redo, restructure
    'de-': re.compile(r'^de[a-z]'),      # destroy, devalue
    'un-': re.compile(r'^un[a-z]'),      # undo, unwind
    'dis-': re.compile(r'^dis[a-z]'),    # disconnect, disappear
    'over-': re.compile(r'^over[a-z]'),  # overflow, overpower
    'out-': re.compile(r'^out[a-z]'),    # outrun, outperform
    'mis-': re.compile(r'^mis[a-z]'),    # misunderstand, misplace
    'pre-': re.compile(r'^pre[a-z]'),    # predict, prepare
    'inter-': re.compile(r'^inter[a-z]'),# interact, interweave
    'trans-': re.compile(r'^trans[a-z]'),# transform, translate
    'co-/com-': re.compile(r'^co[mnr]?[a-z]'),  # combine, connect, correlate
    '-ize': re.compile(r'ize$'),         # modernize, ostracize
    '-ify': re.compile(r'ify$'),         # simplify, modify
    '-ate': re.compile(r'[^e]ate$'),     # create, dominate
    'other': re.compile(r'.*'),          # everything else
}

morph_labels = np.full(N, -1)
morph_names = {}
prefix_idx = 0

for prefix_name, pattern in prefixes.items():
    morph_names[prefix_name] = prefix_idx
    prefix_idx += 1

for i, c in enumerate(cls):
    verb = c['verb'].lower().strip()
    assigned = False
    for prefix_name, pattern in prefixes.items():
        if prefix_name == 'other':
            continue
        if pattern.search(verb):
            morph_labels[i] = morph_names[prefix_name]
            assigned = True
            break
    if not assigned:
        morph_labels[i] = morph_names['other']

print(f"  Morphological categories:")
for name, idx in sorted(morph_names.items(), key=lambda x: x[1]):
    count = (morph_labels == idx).sum()
    print(f"    {name:15s}: {count:5d} verbs")

morph_result = compute_z_score(embs_def, morph_labels, n_perm=200, label_name="Morphological")
print(f"\n  {morph_result['n_groups']} groups, z = {morph_result['z_score']:.1f}")
print(f"  real_sim = {morph_result['real_mean_sim']:.6f}, random = {morph_result['random_mean']:.6f} ± {morph_result['random_std']:.6f}")


# ══════════════════════════════════════════════════════════════════
#  TAXONOMY 4: VENDLER ASPECT CLASSES (via heuristic)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TAXONOMY 4: SCALE (from EO classification)")
print("="*70)

# Use the scale assignments from EO classification as an independent taxonomy
scale_map = {'physical': 0, 'social': 1, 'psychological': 2, 'informational': 3}
scale_labels = np.array([scale_map.get(c.get('scale', ''), -1) for c in cls])

n_with_scale = (scale_labels >= 0).sum()
print(f"  {n_with_scale} verbs with scale assignments")
for name, idx in scale_map.items():
    count = (scale_labels == idx).sum()
    print(f"    {name:15s}: {count:5d}")

scale_result = compute_z_score(embs_def, scale_labels, n_perm=200, label_name="Scale (4 categories)")
print(f"\n  {scale_result['n_groups']} groups, z = {scale_result['z_score']:.1f}")
print(f"  real_sim = {scale_result['real_mean_sim']:.6f}, random = {scale_result['random_mean']:.6f} ± {scale_result['random_std']:.6f}")


# ══════════════════════════════════════════════════════════════════
#  TAXONOMY 5: RANDOM (from falsification, for reference)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TAXONOMY 5: RANDOM BASELINE")
print("="*70)

rng = np.random.RandomState(99)
random_z_scores = []
for trial in range(20):
    shuffled = rng.permutation(eo_labels)
    r = compute_z_score(embs_def, shuffled, n_perm=50, label_name=f"random_{trial}")
    random_z_scores.append(r['z_score'])

print(f"  20 random taxonomies: z = {np.mean(random_z_scores):+.1f} ± {np.std(random_z_scores):.1f}")
print(f"  Range: [{min(random_z_scores):+.1f}, {max(random_z_scores):+.1f}]")


# ══════════════════════════════════════════════════════════════════
#  TAXONOMY 6: EO TRIADS (3 categories)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TAXONOMY 6: EO TRIADS (3 categories)")
print("="*70)

triad_map = {
    'NUL': 0, 'DES': 0, 'INS': 0,  # Existence
    'SEG': 1, 'CON': 1, 'SYN': 1,  # Structure
    'ALT': 2, 'SUP': 2, 'REC': 2,  # Interpretation
}
triad_labels = np.array([triad_map.get(c['operator'], -1) for c in cls])

triad_result = compute_z_score(embs_def, triad_labels, n_perm=200, label_name="EO triads")
print(f"  {triad_result['n_groups']} groups, z = {triad_result['z_score']:.1f}")


# ══════════════════════════════════════════════════════════════════
#  TAXONOMY 7: EO × SCALE CROSS (up to 36 categories)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  TAXONOMY 7: OPERATOR × SCALE (crossed)")
print("="*70)

cross_labels = np.full(N, -1)
cross_map = {}
for i, c in enumerate(cls):
    op = c['operator']
    scale = c.get('scale', '')
    if op in HELIX and scale in scale_map:
        key = f"{op}_{scale}"
        if key not in cross_map:
            cross_map[key] = len(cross_map)
        cross_labels[i] = cross_map[key]

n_cross = len(cross_map)
print(f"  {n_cross} operator×scale categories")
# Only include groups with >= 10 members
size_counts = Counter(cross_labels[cross_labels >= 0])
valid_groups = {k for k, v in size_counts.items() if v >= 10}
cross_labels_filtered = np.array([l if l in valid_groups else -1 for l in cross_labels])
n_valid = len(valid_groups)
print(f"  {n_valid} categories with >= 10 members")

cross_result = compute_z_score(embs_def, cross_labels_filtered, n_perm=100, label_name="Operator×Scale")
print(f"  z = {cross_result['z_score']:.1f}")


# ══════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  COMPARISON TABLE")
print("="*70)

results = [
    ("Random baseline (mean of 20)", 9, np.mean(random_z_scores), "—"),
    ("EO triads", 3, triad_result['z_score'], triad_result['real_mean_sim']),
    ("Scale (phys/soc/psych/info)", 4, scale_result['z_score'], scale_result['real_mean_sim']),
    ("EO operators", 9, eo_result['z_score'], eo_result['real_mean_sim']),
    ("Morphological (prefix)", morph_result['n_groups'], morph_result['z_score'], morph_result['real_mean_sim']),
]

if wn_result:
    results.append(("WordNet lexicographer", wn_result['n_groups'], wn_result['z_score'], wn_result['real_mean_sim']))

results.append((f"Operator×Scale", n_valid, cross_result['z_score'], cross_result['real_mean_sim']))

# Sort by z-score (most negative = most separated)
results.sort(key=lambda x: x[2])

print(f"\n  {'Taxonomy':35s} {'k':>4s} {'z-score':>10s} {'mean_sim':>10s} {'vs EO':>10s}")
print("  " + "-" * 75)

for name, k, z, sim in results:
    vs_eo = ""
    if isinstance(z, float) and name != "EO operators":
        if eo_result['z_score'] != 0:
            ratio = z / eo_result['z_score']
            vs_eo = f"{ratio:.2f}x"
    sim_str = f"{sim:.6f}" if isinstance(sim, float) else sim
    print(f"  {name:35s} {k:4d} {z:+10.1f} {sim_str:>10s} {vs_eo:>10s}")

# Key question
print(f"\n  KEY QUESTION: Does EO exceed WordNet's own taxonomy?")
if wn_result:
    if abs(eo_result['z_score']) > abs(wn_result['z_score']):
        print(f"  → YES. EO (z={eo_result['z_score']:.1f}) > WordNet (z={wn_result['z_score']:.1f})")
        print(f"  → EO is finding structure that WordNet's lexicographers did NOT encode")
    elif abs(eo_result['z_score']) > abs(wn_result['z_score']) * 0.8:
        print(f"  → COMPARABLE. EO (z={eo_result['z_score']:.1f}) ≈ WordNet (z={wn_result['z_score']:.1f})")
        print(f"  → EO captures similar-strength structure with fewer categories")
    else:
        print(f"  → NO. EO (z={eo_result['z_score']:.1f}) < WordNet (z={wn_result['z_score']:.1f})")
        print(f"  → EO may be partially rediscovering lexicographic categories")

# ══════════════════════════════════════════════════════════════════
#  CROSS-TABULATION: EO vs WORDNET
# ══════════════════════════════════════════════════════════════════
if wn_result:
    print(f"\n{'='*70}")
    print("  CROSS-TABULATION: EO × WORDNET")
    print(f"{'='*70}")
    print("  If EO = WordNet in disguise, each EO operator should map to 1-2 WN categories.\n")
    
    # Build cross-tab
    wn_name_list = sorted(wn_label_names.keys(), key=lambda x: wn_label_names[x])
    
    cross_tab = defaultdict(lambda: Counter())
    for i, c in enumerate(cls):
        if c['operator'] in HELIX and wn_labels[i] >= 0:
            wn_name = wn_name_list[wn_labels[i]]
            cross_tab[c['operator']][wn_name] += 1
    
    # Print: for each EO operator, top 3 WordNet categories
    print(f"  {'EO op':6s} {'n':>5s}  Top WordNet categories")
    print("  " + "-" * 60)
    for op in HELIX:
        if op not in cross_tab:
            continue
        total = sum(cross_tab[op].values())
        top3 = cross_tab[op].most_common(3)
        top3_str = ", ".join(f"{name}({count}/{total}={count/total*100:.0f}%)" for name, count in top3)
        # Concentration: does top-1 account for >50%?
        top1_pct = top3[0][1] / total * 100 if top3 else 0
        marker = " ← concentrated" if top1_pct > 60 else ""
        print(f"  {op:6s} {total:5d}  {top3_str}{marker}")
    
    # Reverse: for each WN category, top EO operators
    print(f"\n  {'WN cat':20s} {'n':>5s}  Top EO operators")
    print("  " + "-" * 60)
    
    wn_to_eo = defaultdict(lambda: Counter())
    for i, c in enumerate(cls):
        if c['operator'] in HELIX and wn_labels[i] >= 0:
            wn_name = wn_name_list[wn_labels[i]]
            wn_to_eo[wn_name][c['operator']] += 1
    
    for wn_name in sorted(wn_to_eo.keys()):
        total = sum(wn_to_eo[wn_name].values())
        top3 = wn_to_eo[wn_name].most_common(3)
        top3_str = ", ".join(f"{op}({count}/{total}={count/total*100:.0f}%)" for op, count in top3)
        print(f"  {wn_name:20s} {total:5d}  {top3_str}")


# ══════════════════════════════════════════════════════════════════
#  ORTHOGONALITY TEST
# ══════════════════════════════════════════════════════════════════
if wn_result:
    print(f"\n{'='*70}")
    print("  ORTHOGONALITY: Are EO and WordNet capturing the same or different structure?")
    print(f"{'='*70}")
    
    # Adjusted Rand Index between EO labels and WN labels
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # Only for verbs that have both labels
    both_mask = (eo_labels >= 0) & (wn_labels >= 0)
    eo_both = eo_labels[both_mask]
    wn_both = wn_labels[both_mask]
    
    ari = adjusted_rand_score(eo_both, wn_both)
    nmi = normalized_mutual_info_score(eo_both, wn_both)
    
    print(f"  Adjusted Rand Index: {ari:.4f}")
    print(f"  Normalized Mutual Info: {nmi:.4f}")
    print(f"  (ARI=0 means independent, ARI=1 means identical)")
    
    if ari < 0.05:
        print(f"  → EO and WordNet are nearly ORTHOGONAL — capturing different structure")
    elif ari < 0.15:
        print(f"  → WEAK overlap — mostly different structure with some shared signal")
    elif ari < 0.30:
        print(f"  → MODERATE overlap — partially related classifications")
    else:
        print(f"  → STRONG overlap — EO may be rediscovering WordNet categories")

    # Combined z-score: if they're orthogonal, combining should be even stronger
    print(f"\n  If orthogonal, combining EO + WordNet should produce even stronger separation:")
    print(f"  EO alone: z = {eo_result['z_score']:.1f} ({eo_result['n_groups']} groups)")
    print(f"  WN alone: z = {wn_result['z_score']:.1f} ({wn_result['n_groups']} groups)")
    print(f"  Operator×Scale: z = {cross_result['z_score']:.1f} ({n_valid} groups)")


# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Bar chart of z-scores
    fig, ax = plt.subplots(figsize=(12, 7))
    
    names = [r[0] for r in results]
    z_scores = [r[2] for r in results]
    
    colors = []
    for name in names:
        if 'Random' in name:
            colors.append('#95a5a6')
        elif 'EO op' in name:
            colors.append('#e74c3c')
        elif 'WordNet' in name:
            colors.append('#3498db')
        elif 'EO triad' in name:
            colors.append('#e67e22')
        elif 'Scale' in name and 'Operator' not in name:
            colors.append('#2ecc71')
        elif 'Morph' in name:
            colors.append('#9b59b6')
        else:
            colors.append('#f1c40f')
    
    bars = ax.barh(range(len(results)), z_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{n} (k={k})" for n, k, z, s in results], fontsize=10)
    ax.set_xlabel('Z-score (definition-only embeddings)', fontsize=12)
    ax.set_title('Taxonomy Comparison: Which classification best separates\nverb meanings in definition-only embedding space?', fontsize=14)
    ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.3)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (name, k, z, sim) in enumerate(results):
        ax.text(z - 1, i, f"z={z:+.1f}", va='center', ha='right', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "taxonomy_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ taxonomy_comparison.png")

except ImportError:
    print("  (matplotlib not available)")

# ── SAVE ──────────────────────────────────────────────────────────
report = {
    'results': {r[0]: {'k': r[1], 'z_score': float(r[2])} for r in results},
}
if wn_result:
    report['eo_vs_wordnet'] = {
        'ari': float(ari),
        'nmi': float(nmi),
    }

with open(os.path.join(OUT, "taxonomy_comparison.json"), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*70}")
print("  TAXONOMY COMPARISON COMPLETE")
print(f"{'='*70}")
print(f"  Report: {OUT}/taxonomy_comparison.json")
