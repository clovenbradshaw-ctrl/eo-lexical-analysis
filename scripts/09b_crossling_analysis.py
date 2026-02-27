#!/usr/bin/env python3
"""
Step 09b: Cross-Linguistic Falsification & Deep Analysis
=========================================================

Tests:
  A. CLASSIFIER BIAS: Does INS% correlate with language obscurity?
     If yes, the "universal gradient" is partly "universal ignorance."

  B. RANDOM FRAMEWORK: Shuffle verbs into 9 random categories per language.
     Does the triad gradient (largest > middle > smallest) persist?
     How often does random produce Existence > Structure > Interpretation?

  C. ALTERNATIVE FRAMEWORKS: What if we group by 3 equal-sized buckets
     instead of EO's triads? How stable are those across languages?

  D. CROSS-LINGUISTIC CONSISTENCY: Do typologically similar languages
     produce similar operator profiles? (Correlation matrix)

  E. INS-ALT TRADEOFF: Is it real or zero-sum classifier behavior?

  F. KOREAN DES INVESTIGATION: What's driving 27%?

  G. CONFIDENCE × LANGUAGE: Does classifier confidence track language
     familiarity?
"""

import json, os, sys
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

# ══════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading cross-linguistic data...")

all_langs = {}
all_verbs = {}  # language -> list of classification dicts

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
    triad_pcts = {}
    for tname, tops in TRIADS.items():
        triad_pcts[tname] = sum(op_pcts.get(op, 0) for op in tops)
    
    all_langs[lang] = {
        'family': data['family'],
        'era': data['era'],
        'region': data['region'],
        'morph_type': data['morph_type'],
        'n_classified': total,
        'op_counts': dict(op_counts),
        'op_pcts': op_pcts,
        'triad_pcts': triad_pcts,
    }
    all_verbs[lang] = cls

print(f"  {len(all_langs)} languages loaded")


# ══════════════════════════════════════════════════════════════
#  TEST A: CLASSIFIER BIAS — INS% vs Language Obscurity
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST A: CLASSIFIER BIAS — INS% vs Language Obscurity")
print(f"{'='*70}")

# Proxy for "how well does Claude know this language":
# 1. Sample size (smaller treebank = less well-known)
# 2. Manual obscurity ranking
# 3. Confidence distribution

obscurity_ranking = {
    # Well-known (Claude definitely knows these well)
    'English': 1, 'French': 1, 'German': 1, 'Russian': 1,
    'Japanese': 1, 'Korean': 2, 'Mandarin': 1, 'Arabic': 2,
    'Hindi': 2, 'Turkish': 2, 'Vietnamese': 2, 'Indonesian': 2,
    'Persian': 2, 'Finnish': 2, 'Basque': 3,
    # Ancient (Claude knows from training, but may struggle with verb meanings)
    'Ancient_Greek': 2, 'Latin': 2, 'Sanskrit': 3,
    'Classical_Chinese': 3, 'Old_Church_Slavonic': 4, 'Gothic': 4,
    'Old_French': 3, 'Old_East_Slavic': 4, 'Coptic': 5,
    # Less well-known
    'Tamil': 3, 'Tagalog': 3, 'Naija': 4,
    'Wolof': 5, 'Yoruba': 4, 'Uyghur': 4,
}

print("\n  INS% vs Obscurity (1=well-known, 5=obscure):")
print(f"  {'Language':25s} {'Obscurity':>9s} {'INS%':>6s} {'n':>6s}")
print("  " + "-"*50)

obs_ins_pairs = []
for lang in sorted(all_langs.keys(), key=lambda x: all_langs[x]['op_pcts']['INS']):
    d = all_langs[lang]
    obs = obscurity_ranking.get(lang, 3)
    ins_pct = d['op_pcts']['INS']
    obs_ins_pairs.append((obs, ins_pct, d['n_classified']))
    print(f"  {lang:25s} {obs:9d} {ins_pct:5.1f}% {d['n_classified']:6d}")

# Correlation
obs_arr = np.array([p[0] for p in obs_ins_pairs])
ins_arr = np.array([p[1] for p in obs_ins_pairs])
n_arr = np.array([p[2] for p in obs_ins_pairs])

corr_obs_ins = np.corrcoef(obs_arr, ins_arr)[0,1]
corr_n_ins = np.corrcoef(n_arr, ins_arr)[0,1]

print(f"\n  Correlation(obscurity, INS%): r = {corr_obs_ins:.3f}")
print(f"  Correlation(sample_size, INS%): r = {corr_n_ins:.3f}")

if abs(corr_obs_ins) > 0.4:
    print(f"  ⚠ MODERATE-STRONG correlation — INS inflation may track classifier ignorance")
elif abs(corr_obs_ins) > 0.2:
    print(f"  ⚠ Weak correlation — some classifier bias possible")
else:
    print(f"  ✓ No meaningful correlation — INS% not driven by language obscurity")


# ══════════════════════════════════════════════════════════════
#  TEST B: CONFIDENCE × LANGUAGE
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST B: CONFIDENCE × LANGUAGE")
print(f"{'='*70}")
print("  If Claude is guessing on obscure languages, confidence should drop.\n")

print(f"  {'Language':25s} {'High%':>6s} {'Med%':>6s} {'Low%':>6s} {'n':>6s} {'Obs':>4s}")
print("  " + "-"*55)

conf_ins_pairs = []
for lang in sorted(all_langs.keys(), key=lambda x: obscurity_ranking.get(x, 3)):
    verbs = all_verbs[lang]
    conf = Counter(v.get('confidence', 'unknown').lower() for v in verbs)
    total = sum(conf.values())
    if total == 0:
        continue
    high_pct = conf.get('high', 0) / total * 100
    med_pct = conf.get('medium', 0) / total * 100
    low_pct = conf.get('low', 0) / total * 100
    obs = obscurity_ranking.get(lang, 3)
    
    conf_ins_pairs.append((obs, high_pct, all_langs[lang]['op_pcts']['INS']))
    print(f"  {lang:25s} {high_pct:5.1f}% {med_pct:5.1f}% {low_pct:5.1f}% {total:6d} {obs:4d}")

obs_conf = np.array([p[0] for p in conf_ins_pairs])
high_conf = np.array([p[1] for p in conf_ins_pairs])
corr_obs_conf = np.corrcoef(obs_conf, high_conf)[0,1]
print(f"\n  Correlation(obscurity, high_confidence%): r = {corr_obs_conf:.3f}")

# Does high confidence correlate with INS%?
ins_conf = np.array([p[2] for p in conf_ins_pairs])
corr_conf_ins = np.corrcoef(high_conf, ins_conf)[0,1]
print(f"  Correlation(high_confidence%, INS%): r = {corr_conf_ins:.3f}")


# ══════════════════════════════════════════════════════════════
#  TEST C: RANDOM FRAMEWORK — Does any 9→3 grouping show gradient?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST C: RANDOM FRAMEWORK — Is the triad gradient trivial?")
print(f"{'='*70}")

# For each language, we have 9 operator percentages.
# EO groups them into triads: {NUL,DES,INS}, {SEG,CON,SYN}, {ALT,SUP,REC}
# The triad gradient is: Existence > Structure > Interpretation (always)
#
# Test: randomly assign 9 operators to 3 groups of 3. How often does
# the resulting "gradient" hold across all 28 languages simultaneously?

rng = np.random.RandomState(42)
n_trials = 10000
universal_gradient_count = 0
universal_ordering_counts = Counter()

for trial in range(n_trials):
    # Random partition of 9 operators into 3 groups of 3
    perm = rng.permutation(9)
    groups = [
        [HELIX[perm[0]], HELIX[perm[1]], HELIX[perm[2]]],
        [HELIX[perm[3]], HELIX[perm[4]], HELIX[perm[5]]],
        [HELIX[perm[6]], HELIX[perm[7]], HELIX[perm[8]]],
    ]
    
    # For each language, compute group sums
    all_ordered = True
    orderings = []
    
    for lang, d in all_langs.items():
        sums = []
        for g in groups:
            s = sum(d['op_pcts'].get(op, 0) for op in g)
            sums.append(s)
        
        # Sort to find ordering
        sorted_sums = sorted(sums, reverse=True)
        ordering = tuple(sorted_sums[i] > sorted_sums[i+1] for i in range(2))
        orderings.append(ordering)
    
    # Check: does the SAME group always come out largest across all languages?
    # (not just any ordering, but a consistent ordering)
    group_sums_per_lang = []
    for lang, d in all_langs.items():
        sums = tuple(sum(d['op_pcts'].get(op, 0) for op in g) for g in groups)
        group_sums_per_lang.append(sums)
    
    # Check if same group is always biggest, same always middle, same always smallest
    rankings = []
    for sums in group_sums_per_lang:
        rank = sorted(range(3), key=lambda i: -sums[i])
        rankings.append(tuple(rank))
    
    # Count how many languages share the most common ranking
    ranking_counts = Counter(rankings)
    most_common_ranking, mc_count = ranking_counts.most_common(1)[0]
    
    if mc_count == len(all_langs):  # ALL languages same ordering
        universal_gradient_count += 1

print(f"  {n_trials} random 9→3 partitions tested")
print(f"  Universal gradient (all 28 languages same ordering): {universal_gradient_count}/{n_trials}")
print(f"  = {universal_gradient_count/n_trials*100:.2f}%")

if universal_gradient_count == 0:
    print(f"  ✓ EO's universal triad ordering is non-trivial (p < {1/n_trials:.5f})")
else:
    print(f"  ⚠ {universal_gradient_count} random partitions also produce universal ordering")

# Also: how often does any partition produce >=25/28 same ordering?
almost_universal = 0
for trial in range(n_trials):
    perm = rng.permutation(9)
    groups = [
        [HELIX[perm[0]], HELIX[perm[1]], HELIX[perm[2]]],
        [HELIX[perm[3]], HELIX[perm[4]], HELIX[perm[5]]],
        [HELIX[perm[6]], HELIX[perm[7]], HELIX[perm[8]]],
    ]
    
    group_sums_per_lang = []
    for lang, d in all_langs.items():
        sums = tuple(sum(d['op_pcts'].get(op, 0) for op in g) for g in groups)
        group_sums_per_lang.append(sums)
    
    rankings = []
    for sums in group_sums_per_lang:
        rank = tuple(sorted(range(3), key=lambda i: -sums[i]))
        rankings.append(rank)
    
    ranking_counts = Counter(rankings)
    _, mc_count = ranking_counts.most_common(1)[0]
    if mc_count >= 25:
        almost_universal += 1

print(f"  ≥25/28 same ordering: {almost_universal}/{n_trials} ({almost_universal/n_trials*100:.2f}%)")


# ══════════════════════════════════════════════════════════════
#  TEST D: IS INS DOING ALL THE WORK?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST D: IS INS DOING ALL THE WORK?")
print(f"{'='*70}")
print("  INS is 35-62% everywhere. Any group containing INS will be largest.")
print("  Does the gradient hold if we remove INS from the picture?\n")

# Redistribute: remove INS, renormalize remaining 8 operators
# Then test: does {NUL,DES} > {SEG,CON,SYN} > {ALT,SUP,REC} hold?
# (NUL,DES are the non-INS Existence operators)

print(f"  {'Language':25s} {'NUL+DES':>8s} {'SEG+CON+SYN':>12s} {'ALT+SUP+REC':>12s} {'Gradient?':>10s}")
print("  " + "-"*70)

gradient_holds = 0
for lang in sorted(all_langs.keys()):
    d = all_langs[lang]
    # Renormalize without INS
    non_ins_total = 100 - d['op_pcts']['INS']
    if non_ins_total < 1:
        continue
    
    exist_no_ins = (d['op_pcts']['NUL'] + d['op_pcts']['DES']) / non_ins_total * 100
    structure = (d['op_pcts']['SEG'] + d['op_pcts']['CON'] + d['op_pcts']['SYN']) / non_ins_total * 100
    interp = (d['op_pcts']['ALT'] + d['op_pcts']['SUP'] + d['op_pcts']['REC']) / non_ins_total * 100
    
    holds = "✓" if structure > exist_no_ins and structure > interp else "✗"
    # Actually, without INS, Structure should be biggest since it has 3 ops vs 2
    # Better test: per-operator average
    exist_avg = exist_no_ins / 2
    struct_avg = structure / 3
    interp_avg = interp / 3
    
    print(f"  {lang:25s} {exist_no_ins:7.1f}% {structure:11.1f}% {interp:11.1f}%")
    
    # Does the ordering hold per-operator-average?
    # In EO: NUL+DES average should < SEG+CON+SYN average (Structure has more specific ops)
    # This is a different prediction

# Actually the right test is simpler: 
# Does the triad ordering hold when INS is removed and we compare
# Existence-remainder vs Structure vs Interpretation?
print(f"\n  With INS removed, Structure typically becomes largest (3 ops vs 2).")
print(f"  The meaningful test is whether the OPERATOR-LEVEL ordering is stable.")


# ══════════════════════════════════════════════════════════════
#  TEST E: INS-ALT TRADEOFF
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST E: INS-ALT TRADEOFF")
print(f"{'='*70}")
print("  Are INS and ALT inversely correlated across languages?")
print("  If yes: real typological variation, or classifier zero-sum?\n")

ins_pcts = [all_langs[l]['op_pcts']['INS'] for l in sorted(all_langs.keys())]
alt_pcts = [all_langs[l]['op_pcts']['ALT'] for l in sorted(all_langs.keys())]
con_pcts = [all_langs[l]['op_pcts']['CON'] for l in sorted(all_langs.keys())]
des_pcts = [all_langs[l]['op_pcts']['DES'] for l in sorted(all_langs.keys())]
nul_pcts = [all_langs[l]['op_pcts']['NUL'] for l in sorted(all_langs.keys())]
seg_pcts = [all_langs[l]['op_pcts']['SEG'] for l in sorted(all_langs.keys())]

pairs = [
    ('INS', 'ALT', ins_pcts, alt_pcts),
    ('INS', 'CON', ins_pcts, con_pcts),
    ('INS', 'DES', ins_pcts, des_pcts),
    ('INS', 'NUL', ins_pcts, nul_pcts),
    ('INS', 'SEG', ins_pcts, seg_pcts),
    ('ALT', 'CON', alt_pcts, con_pcts),
    ('ALT', 'DES', alt_pcts, des_pcts),
    ('NUL', 'SEG', nul_pcts, seg_pcts),
]

print(f"  {'Pair':>12s} {'r':>7s} {'Interpretation':>40s}")
print("  " + "-"*65)
for name1, name2, arr1, arr2 in pairs:
    r = np.corrcoef(arr1, arr2)[0,1]
    if abs(r) > 0.5:
        note = "STRONG"
    elif abs(r) > 0.3:
        note = "moderate"
    else:
        note = "weak"
    print(f"  {name1:>5s}↔{name2:<5s} {r:+.3f}   {note}")

# Key test: is INS-ALT tradeoff just compositional (they sum to ~constant)?
ins_plus_alt = [i + a for i, a in zip(ins_pcts, alt_pcts)]
print(f"\n  INS + ALT range: [{min(ins_plus_alt):.1f}%, {max(ins_plus_alt):.1f}%]")
print(f"  INS + ALT mean:  {np.mean(ins_plus_alt):.1f}% ± {np.std(ins_plus_alt):.1f}%")

# If INS+ALT is nearly constant, the tradeoff is compositional, not real
if np.std(ins_plus_alt) < 3:
    print(f"  ⚠ INS+ALT is nearly constant — tradeoff may be compositional")
else:
    print(f"  INS+ALT varies by {np.std(ins_plus_alt):.1f}pp — not purely compositional")


# ══════════════════════════════════════════════════════════════
#  TEST F: KOREAN DES INVESTIGATION
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST F: KOREAN DES = 27% — What's happening?")
print(f"{'='*70}")

if 'Korean' in all_verbs:
    korean_verbs = all_verbs['Korean']
    korean_des = [v for v in korean_verbs if v.get('operator', '').upper() == 'DES']
    
    print(f"  {len(korean_des)} Korean verbs classified as DES")
    print(f"  Sample (first 30):")
    for v in korean_des[:30]:
        gloss = v.get('gloss', '')
        conf = v.get('confidence', '')
        alt = v.get('alternative', '')
        verb = v.get('verb', '')
        alt_str = f" (alt: {alt})" if alt else ""
        print(f"    {verb:20s} → {gloss:30s} [{conf}]{alt_str}")
    
    # Compare confidence of DES verbs vs others
    des_conf = Counter(v.get('confidence', '').lower() for v in korean_des)
    other_conf = Counter(v.get('confidence', '').lower() for v in korean_verbs 
                        if v.get('operator', '').upper() != 'DES')
    
    des_high = des_conf.get('high', 0) / max(len(korean_des), 1) * 100
    other_high = other_conf.get('high', 0) / max(len(korean_verbs) - len(korean_des), 1) * 100
    
    print(f"\n  DES confidence: {des_high:.1f}% high")
    print(f"  Non-DES confidence: {other_high:.1f}% high")
    
    # What are the glosses? Are these really type-assignment verbs?
    des_glosses = [v.get('gloss', '') for v in korean_des if v.get('gloss')]
    print(f"\n  All DES glosses (unique):")
    for g in sorted(set(des_glosses)):
        count = des_glosses.count(g)
        marker = f" ×{count}" if count > 1 else ""
        print(f"    {g}{marker}")


# ══════════════════════════════════════════════════════════════
#  TEST G: OPERATOR PROFILE SIMILARITY (language clustering)
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST G: LANGUAGE CLUSTERING BY OPERATOR PROFILE")
print(f"{'='*70}")
print("  Do typologically similar languages have similar operator profiles?\n")

langs = sorted(all_langs.keys())
n_langs = len(langs)

# Build matrix of operator profiles
profiles = np.array([[all_langs[l]['op_pcts'][op] for op in HELIX] for l in langs])

# Compute cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(profiles)

# Find most and least similar pairs
pairs_list = []
for i in range(n_langs):
    for j in range(i+1, n_langs):
        pairs_list.append((langs[i], langs[j], sim_matrix[i,j],
                          all_langs[langs[i]]['family'], all_langs[langs[j]]['family']))

pairs_list.sort(key=lambda x: x[2])

print("  MOST DIFFERENT operator profiles:")
for l1, l2, sim, f1, f2 in pairs_list[:5]:
    same_fam = "SAME" if f1 == f2 else "diff"
    print(f"    {l1:20s} ↔ {l2:20s}  sim={sim:.4f}  [{f1} / {f2}] {same_fam}")

print("\n  MOST SIMILAR operator profiles:")
for l1, l2, sim, f1, f2 in pairs_list[-5:]:
    same_fam = "SAME" if f1 == f2 else "diff"
    print(f"    {l1:20s} ↔ {l2:20s}  sim={sim:.4f}  [{f1} / {f2}] {same_fam}")

# Do same-family languages cluster together?
same_fam_sims = [s for _, _, s, f1, f2 in pairs_list if f1 == f2]
diff_fam_sims = [s for _, _, s, f1, f2 in pairs_list if f1 != f2]

if same_fam_sims:
    print(f"\n  Same-family pairs: mean sim = {np.mean(same_fam_sims):.4f} (n={len(same_fam_sims)})")
    print(f"  Diff-family pairs: mean sim = {np.mean(diff_fam_sims):.4f} (n={len(diff_fam_sims)})")
    
    diff = np.mean(same_fam_sims) - np.mean(diff_fam_sims)
    if diff > 0.005:
        print(f"  → Same-family languages ARE more similar (+{diff:.4f})")
    else:
        print(f"  → No meaningful difference — operator profiles don't track family")


# ══════════════════════════════════════════════════════════════
#  TEST H: VARIANCE ANALYSIS — Which operators vary most?
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST H: OPERATOR VARIANCE ACROSS LANGUAGES")
print(f"{'='*70}")
print("  Which operators are stable (universal) vs variable (language-specific)?\n")

print(f"  {'Operator':>8s} {'Mean%':>7s} {'Std%':>7s} {'CV':>7s} {'Min%':>7s} {'Max%':>7s} {'Range':>7s}")
print("  " + "-"*55)

for op in HELIX:
    pcts = [all_langs[l]['op_pcts'][op] for l in all_langs]
    mean = np.mean(pcts)
    std = np.std(pcts)
    cv = std / mean if mean > 0 else 0
    print(f"  {op:>8s} {mean:6.1f}% {std:6.1f}% {cv:6.2f} {min(pcts):6.1f}% {max(pcts):6.1f}% {max(pcts)-min(pcts):6.1f}%")

print(f"\n  CV = coefficient of variation (std/mean). Higher = more variable.")
print(f"  Low CV = universal property. High CV = language-specific.")


# ══════════════════════════════════════════════════════════════
#  TEST I: ALTERNATIVE 9-CATEGORY FRAMEWORKS
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST I: HOW SPECIAL IS EO's SPECIFIC 9→3 GROUPING?")
print(f"{'='*70}")

# There are 9!/(3!3!3!) = 280 ways to partition 9 items into 3 groups of 3.
# EO claims {NUL,DES,INS},{SEG,CON,SYN},{ALT,SUP,REC} is the right one.
# For each partition, compute: how many languages show the SAME ordering?

from itertools import combinations

def all_3x3_partitions(items):
    """Generate all ways to partition 9 items into 3 groups of 3."""
    if len(items) != 9:
        return
    items = list(items)
    for g1 in combinations(range(9), 3):
        remaining1 = [i for i in range(9) if i not in g1]
        for g2 in combinations(remaining1, 3):
            g3 = tuple(i for i in remaining1 if i not in g2)
            # Canonicalize to avoid counting same partition multiple times
            groups = tuple(sorted([tuple(sorted(g1)), tuple(sorted(g2)), tuple(sorted(g3))]))
            yield groups

# Generate all unique partitions
all_partitions = set()
for p in all_3x3_partitions(HELIX):
    all_partitions.add(p)

print(f"  Total unique 9→3 partitions: {len(all_partitions)}")

# For each partition, compute consistency across languages
partition_scores = []

for partition in all_partitions:
    # Get operator names for each group
    groups = [[HELIX[i] for i in g] for g in partition]
    
    # Compute group sums per language and find ordering
    rankings = []
    for lang, d in all_langs.items():
        sums = [sum(d['op_pcts'].get(op, 0) for op in g) for g in groups]
        rank = tuple(sorted(range(3), key=lambda i: -sums[i]))
        rankings.append(rank)
    
    # Count most common ranking
    ranking_counts = Counter(rankings)
    most_common, mc_count = ranking_counts.most_common(1)[0]
    
    # Is this the EO partition?
    eo_groups = [
        tuple(sorted([HELIX.index(op) for op in ['NUL','DES','INS']])),
        tuple(sorted([HELIX.index(op) for op in ['SEG','CON','SYN']])),
        tuple(sorted([HELIX.index(op) for op in ['ALT','SUP','REC']])),
    ]
    eo_partition = tuple(sorted(eo_groups))
    is_eo = (partition == eo_partition)
    
    partition_scores.append((mc_count, is_eo, partition, groups))

partition_scores.sort(key=lambda x: -x[0])

print(f"\n  Top 10 most consistent partitions:")
print(f"  {'Rank':>4s} {'Consistency':>12s} {'Groups':>60s} {'EO?':>5s}")
print("  " + "-"*85)

eo_rank = None
for i, (count, is_eo, partition, groups) in enumerate(partition_scores[:10]):
    groups_str = " | ".join(",".join(g) for g in groups)
    marker = " ← EO" if is_eo else ""
    print(f"  {i+1:4d} {count:4d}/28 ({count/28*100:.0f}%)  {groups_str:55s}{marker}")
    if is_eo:
        eo_rank = i + 1

# Find EO's rank if not in top 10
if eo_rank is None:
    for i, (count, is_eo, _, _) in enumerate(partition_scores):
        if is_eo:
            eo_rank = i + 1
            eo_consistency = count
            break
    print(f"\n  EO partition rank: #{eo_rank} out of {len(all_partitions)}")
    print(f"  EO consistency: {eo_consistency}/28 ({eo_consistency/28*100:.0f}%)")
else:
    print(f"\n  EO partition rank: #{eo_rank} out of {len(all_partitions)}")

# Distribution of consistency scores
scores = [s[0] for s in partition_scores]
print(f"\n  Consistency distribution:")
print(f"    28/28 (perfect): {scores.count(28)} partitions")
print(f"    ≥25/28:          {sum(1 for s in scores if s >= 25)} partitions")
print(f"    ≥20/28:          {sum(1 for s in scores if s >= 20)} partitions")
print(f"    Mean:             {np.mean(scores):.1f}/28")


# ══════════════════════════════════════════════════════════════
#  TEST J: REC POVERTY
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  TEST J: REC POVERTY — Is restructuring also impoverished?")
print(f"{'='*70}")

rec_data = [(lang, d['op_pcts'].get('REC', 0), d['op_counts'].get('REC', 0))
            for lang, d in all_langs.items()]
rec_data.sort(key=lambda x: -x[1])

print(f"\n  {'Language':25s} {'REC%':>6s} {'Count':>6s}")
print("  " + "-"*40)
for lang, pct, count in rec_data:
    print(f"  {lang:25s} {pct:5.1f}% {count:6d}")

rec_pcts = [r[1] for r in rec_data]
print(f"\n  REC range: [{min(rec_pcts):.1f}%, {max(rec_pcts):.1f}%]")
print(f"  REC mean:  {np.mean(rec_pcts):.1f}% ± {np.std(rec_pcts):.1f}%")
print(f"  SUP mean:  {np.mean([all_langs[l]['op_pcts']['SUP'] for l in all_langs]):.1f}%")


# ══════════════════════════════════════════════════════════════
#  SAVE REPORT
# ══════════════════════════════════════════════════════════════

report = {
    'test_A_classifier_bias': {
        'corr_obscurity_ins': float(corr_obs_ins),
        'corr_samplesize_ins': float(corr_n_ins),
    },
    'test_C_random_framework': {
        'n_trials': n_trials,
        'universal_gradient_count': universal_gradient_count,
        'almost_universal_25': almost_universal,
    },
    'test_E_ins_alt_tradeoff': {
        'ins_alt_mean': float(np.mean(ins_plus_alt)),
        'ins_alt_std': float(np.std(ins_plus_alt)),
    },
    'test_H_variance': {
        op: {'mean': float(np.mean([all_langs[l]['op_pcts'][op] for l in all_langs])),
             'std': float(np.std([all_langs[l]['op_pcts'][op] for l in all_langs]))}
        for op in HELIX
    },
    'test_I_partition': {
        'eo_rank': eo_rank,
        'total_partitions': len(all_partitions),
        'eo_consistency': partition_scores[eo_rank-1][0] if eo_rank else None,
    },
}

with open(os.path.join(OUT, "crossling_falsification.json"), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*70}")
print("  CROSS-LINGUISTIC FALSIFICATION COMPLETE")
print(f"{'='*70}")
