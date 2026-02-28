#!/usr/bin/env python3
"""
Step 12b: Robust Taxonomy Comparison
======================================
Addresses statistical issues in the original 12_taxonomy.py:

1. k-MATCHED NULL BASELINES: For each real taxonomy (k groups), generate
   random taxonomies with the SAME k and SAME group-size distribution,
   then compare. This eliminates the confound where fewer groups → inflated |z|.

2. EFFECT SIZE (η²): Compute the proportion of total embedding variance
   explained by group membership (PERMANOVA-style). η² is comparable across k.

3. PERMUTATION p-VALUES with adequate count (1000 permutations).

4. MULTIPLE COMPARISON CORRECTION: Benjamini-Hochberg FDR across all
   taxonomies tested.

5. CONFIDENCE FILTERING: Optionally restrict to high-confidence LLM
   classifications only.

The key output is a comparison table where all taxonomies are ranked by
η² (effect size), which is directly comparable regardless of k.
"""

import json, os, re, sys, warnings
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore", category=FutureWarning)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

N_PERM = 1000          # permutations for real taxonomies
N_PERM_MATCHED = 100   # permutations per k-matched random trial
N_MATCHED_TRIALS = 30  # number of k-matched random taxonomies
SEED = 42
MIN_GROUP_SIZE = 5     # minimum members per group

# ══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading data...")
with open(os.path.join(DATA, "verbs.json")) as f:
    verbs = json.load(f)

N = len(verbs)
print(f"  {N} verbs loaded")

# Map abbreviated scales to full names
SCALE_MAP = {'phys': 'physical', 'soci': 'social', 'psyc': 'psychological', 'info': 'informational'}
CONF_MAP = {'H': 'high', 'M': 'medium', 'L': 'low'}
HELIX = ['NUL', 'DES', 'INS', 'SEG', 'CON', 'SYN', 'ALT', 'SUP', 'REC']
TRIAD_MAP = {
    'NUL': 'Existence', 'DES': 'Existence', 'INS': 'Existence',
    'SEG': 'Structure', 'CON': 'Structure', 'SYN': 'Structure',
    'ALT': 'Interpretation', 'SUP': 'Interpretation', 'REC': 'Interpretation',
}

# Print distribution
op_counts = Counter(v['op'] for v in verbs)
print("  Operator distribution:")
for op in HELIX:
    n = op_counts.get(op, 0)
    print(f"    {op}: {n:5d} ({n/N*100:.1f}%)")

conf_counts = Counter(CONF_MAP.get(v['c'], '?') for v in verbs)
print(f"  Confidence: {dict(conf_counts)}")


# ══════════════════════════════════════════════════════════════════
#  GENERATE EMBEDDINGS
# ══════════════════════════════════════════════════════════════════
EMB_CACHE = os.path.join(OUT, "taxonomy_embeddings.npz")

if os.path.exists(EMB_CACHE):
    print(f"\n  Loading cached embeddings from {EMB_CACHE}...")
    data = np.load(EMB_CACHE)
    embeddings = data['embeddings']
    print(f"  Shape: {embeddings.shape}")
else:
    print("\n  Generating definition embeddings...")
    from nltk.corpus import wordnet as wn

    # Get WordNet definitions for each verb
    definitions = []
    for v in verbs:
        verb_clean = v['v'].replace(' ', '_').replace("(can't) ", "").lower()
        synsets = wn.synsets(verb_clean, pos=wn.VERB)
        if synsets:
            definitions.append(synsets[0].definition())
        else:
            # Fall back to the LLM reason as definition proxy
            definitions.append(v.get('r', v['v']))

    n_wn = sum(1 for d, v in zip(definitions, verbs) if d != v.get('r', v['v']))
    print(f"  WordNet definitions: {n_wn}/{N} ({n_wn/N*100:.1f}%)")

    # Try sentence-transformers first; fall back to TF-IDF if unavailable
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(definitions, show_progress_bar=True, batch_size=256)
        embeddings = np.array(embeddings, dtype=np.float32)
        emb_method = 'all-MiniLM-L6-v2'
    except Exception as e:
        print(f"  sentence-transformers unavailable ({e}), using TF-IDF")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        # TF-IDF + LSA (dimensionality reduction) produces dense semantic vectors
        # suitable for cosine similarity comparisons
        tfidf = TfidfVectorizer(max_features=10000, stop_words='english',
                                sublinear_tf=True, ngram_range=(1, 2))
        X_sparse = tfidf.fit_transform(definitions)

        # Reduce to 384 dims via SVD (same order as MiniLM)
        n_components = min(384, X_sparse.shape[1] - 1, X_sparse.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        embeddings = svd.fit_transform(X_sparse).astype(np.float32)
        var_explained = svd.explained_variance_ratio_.sum()
        print(f"  TF-IDF + SVD: {X_sparse.shape[1]} features → {n_components} dims")
        print(f"  Variance explained: {var_explained:.1%}")
        emb_method = f'TF-IDF + SVD({n_components}d)'

    np.savez_compressed(EMB_CACHE, embeddings=embeddings)
    print(f"  Saved to {EMB_CACHE}: shape={embeddings.shape}")


# ══════════════════════════════════════════════════════════════════
#  STATISTICAL FUNCTIONS (vectorized for performance)
# ══════════════════════════════════════════════════════════════════

def _fast_eta_and_sim(X, y, n_groups, ss_total):
    """
    Fast combined computation of η² and mean pairwise cosine similarity.
    X: (n_valid, d) embedding matrix (already filtered to valid labels)
    y: (n_valid,) integer labels 0..n_groups-1
    ss_total: precomputed total sum of squares
    """
    d = X.shape[1]
    grand_mean = X.mean(axis=0)

    # Compute centroids via scatter-add
    centroids = np.zeros((n_groups, d), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    for g in range(n_groups):
        mask = y == g
        n_i = mask.sum()
        if n_i > 0:
            centroids[g] = X[mask].mean(axis=0)
            counts[g] = n_i

    # η² = SS_between / SS_total
    diffs = centroids - grand_mean
    ss_between = np.sum(counts[:, None] * diffs ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    # Mean pairwise cosine similarity between centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = centroids / norms
    sim_matrix = normed @ normed.T
    # Extract upper triangle
    triu_idx = np.triu_indices(n_groups, k=1)
    mean_sim = sim_matrix[triu_idx].mean() if len(triu_idx[0]) > 0 else 0.0

    return float(eta_sq), float(mean_sim)


def permutation_test(embeddings, labels, n_perm=1000, seed=42):
    """
    Permutation test with η² (effect size) and cosine similarity.
    Optimized: precomputes SS_total and filters to valid labels once.
    """
    unique = sorted(set(l for l in labels if l >= 0))
    n_groups = len(unique)
    if n_groups < 2:
        return None

    # Remap labels to 0..n_groups-1 and filter
    label_map = {u: i for i, u in enumerate(unique)}
    valid_mask = np.array([l >= 0 for l in labels])
    X = embeddings[valid_mask]
    y = np.array([label_map[l] for l in labels if l >= 0])

    # Precompute SS_total (invariant under permutation)
    grand_mean = X.mean(axis=0)
    ss_total = float(np.sum((X - grand_mean) ** 2))

    # Observed statistics
    real_eta, real_sim = _fast_eta_and_sim(X, y, n_groups, ss_total)

    # Null distribution
    rng = np.random.RandomState(seed)
    null_etas = np.empty(n_perm)
    null_sims = np.empty(n_perm)
    for p in range(n_perm):
        perm_y = rng.permutation(y)
        null_etas[p], null_sims[p] = _fast_eta_and_sim(X, perm_y, n_groups, ss_total)

    # z-scores
    eta_mean, eta_std = null_etas.mean(), null_etas.std()
    z_eta = (real_eta - eta_mean) / eta_std if eta_std > 0 else 0

    sim_mean, sim_std = null_sims.mean(), null_sims.std()
    z_sim = (real_sim - sim_mean) / sim_std if sim_std > 0 else 0

    # p-values (one-tailed)
    p_eta = float(max((null_etas >= real_eta).sum() / n_perm, 1 / n_perm))
    p_sim = float(max((null_sims <= real_sim).sum() / n_perm, 1 / n_perm))

    sizes = {int(g): int((y == i).sum()) for g, i in label_map.items()}

    return {
        'n_groups': n_groups,
        'group_sizes': sizes,
        'eta_sq': float(real_eta),
        'eta_sq_null_mean': float(eta_mean),
        'eta_sq_null_std': float(eta_std),
        'z_eta': float(z_eta),
        'p_eta': p_eta,
        'mean_sim': float(real_sim),
        'sim_null_mean': float(sim_mean),
        'sim_null_std': float(sim_std),
        'z_sim': float(z_sim),
        'p_sim': p_sim,
    }


def k_matched_null_baseline(embeddings, labels, n_trials=50, n_perm_per=200, seed=42):
    """
    Generate random taxonomies with the SAME k and SAME group-size distribution,
    then compute their η² and z_sim to determine what you'd expect by chance.

    Uses a lightweight approach: for each random taxonomy, compute only η²
    (not full permutation z) since the k-matched null IS the z-score calibration.
    """
    unique = sorted(set(l for l in labels if l >= 0))
    n_groups = len(unique)
    if n_groups < 2:
        return None

    label_map = {u: i for i, u in enumerate(unique)}
    valid_mask = np.array([l >= 0 for l in labels])
    X = embeddings[valid_mask]
    y_real = np.array([label_map[l] for l in labels if l >= 0])

    grand_mean = X.mean(axis=0)
    ss_total = float(np.sum((X - grand_mean) ** 2))
    sizes = [int((y_real == i).sum()) for i in range(n_groups)]

    rng = np.random.RandomState(seed)
    null_etas = np.empty(n_trials)
    null_z_sims = np.empty(n_trials)

    for trial in range(n_trials):
        # Random taxonomy: shuffle labels (preserving group sizes)
        y_rand = rng.permutation(y_real)
        rand_eta, rand_sim = _fast_eta_and_sim(X, y_rand, n_groups, ss_total)
        null_etas[trial] = rand_eta

        # Quick z_sim for this random taxonomy (fewer permutations)
        inner_sims = np.empty(n_perm_per)
        for p in range(n_perm_per):
            y_perm = rng.permutation(y_rand)
            _, s = _fast_eta_and_sim(X, y_perm, n_groups, ss_total)
            inner_sims[p] = s
        null_z_sims[trial] = (rand_sim - inner_sims.mean()) / inner_sims.std() if inner_sims.std() > 0 else 0

    return {
        'n_trials': n_trials,
        'null_z_sim_mean': float(null_z_sims.mean()),
        'null_z_sim_std': float(null_z_sims.std()),
        'null_eta_mean': float(null_etas.mean()),
        'null_eta_std': float(null_etas.std()),
        'k': n_groups,
        'sizes': sizes,
    }


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values and boolean array of rejections.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]

    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adjusted_p = sorted_p[i] * n / rank
        if i < n - 1:
            adjusted_p = min(adjusted_p, adjusted[sorted_idx[i + 1]])
        adjusted[sorted_idx[i]] = min(adjusted_p, 1.0)

    rejected = adjusted <= alpha
    return adjusted.tolist(), rejected.tolist()


# ══════════════════════════════════════════════════════════════════
#  BUILD LABEL ARRAYS
# ══════════════════════════════════════════════════════════════════
print("\nBuilding label arrays...")

# Confidence filter
high_conf_mask = np.array([v['c'] == 'H' for v in verbs])
med_conf_mask = np.array([v['c'] in ('H', 'M') for v in verbs])
print(f"  High confidence: {high_conf_mask.sum()}/{N} ({high_conf_mask.sum()/N*100:.1f}%)")
print(f"  High+Medium:     {med_conf_mask.sum()}/{N} ({med_conf_mask.sum()/N*100:.1f}%)")

# --- Operator labels (k=9) ---
eo_labels = np.array([HELIX.index(v['op']) if v['op'] in HELIX else -1 for v in verbs])

# --- Scale labels (k=4) ---
scale_names = ['physical', 'social', 'psychological', 'informational']
scale_label_map = {s: i for i, s in enumerate(scale_names)}
scale_labels = np.array([scale_label_map.get(SCALE_MAP.get(v.get('s', ''), ''), -1) for v in verbs])

# --- Triad labels (k=3) ---
triad_names = ['Existence', 'Structure', 'Interpretation']
triad_label_map = {t: i for i, t in enumerate(triad_names)}
triad_labels = np.array([triad_label_map.get(TRIAD_MAP.get(v['op'], ''), -1) for v in verbs])

# --- WordNet lexicographer categories ---
try:
    from nltk.corpus import wordnet as wn
    wn_labels = np.full(N, -1)
    wn_label_names = {}
    for i, v in enumerate(verbs):
        verb_clean = v['v'].replace(' ', '_').replace("(can't) ", "").lower()
        synsets = wn.synsets(verb_clean, pos=wn.VERB)
        if synsets:
            lex = synsets[0].lexname().replace('verb.', '')
            if lex not in wn_label_names:
                wn_label_names[lex] = len(wn_label_names)
            wn_labels[i] = wn_label_names[lex]
    n_wn = (wn_labels >= 0).sum()
    print(f"  WordNet: mapped {n_wn}/{N} verbs to {len(wn_label_names)} categories")
    has_wordnet = True
except ImportError:
    print("  WordNet: NLTK not available")
    wn_labels = np.full(N, -1)
    has_wordnet = False

# --- Morphological labels ---
prefixes = [
    ('re-', re.compile(r'^re[a-z]')),
    ('de-', re.compile(r'^de[a-z]')),
    ('un-', re.compile(r'^un[a-z]')),
    ('dis-', re.compile(r'^dis[a-z]')),
    ('over-', re.compile(r'^over[a-z]')),
    ('out-', re.compile(r'^out[a-z]')),
    ('mis-', re.compile(r'^mis[a-z]')),
    ('pre-', re.compile(r'^pre[a-z]')),
    ('inter-', re.compile(r'^inter[a-z]')),
    ('trans-', re.compile(r'^trans[a-z]')),
    ('co-/com-', re.compile(r'^co[mnr]?[a-z]')),
    ('-ize', re.compile(r'ize$')),
    ('-ify', re.compile(r'ify$')),
    ('-ate', re.compile(r'[^e]ate$')),
]
morph_labels = np.full(N, len(prefixes))  # default = "other" group
for i, v in enumerate(verbs):
    verb = v['v'].lower().strip()
    for j, (name, pattern) in enumerate(prefixes):
        if pattern.search(verb):
            morph_labels[i] = j
            break
n_morph_groups = len(set(morph_labels))
print(f"  Morphological: {n_morph_groups} groups (14 prefixes + other)")

# --- Operator × Scale (crossed) ---
cross_labels = np.full(N, -1)
cross_map = {}
for i, v in enumerate(verbs):
    op = v['op']
    scale = SCALE_MAP.get(v.get('s', ''), '')
    if op in HELIX and scale in scale_label_map:
        key = f"{op}_{scale}"
        if key not in cross_map:
            cross_map[key] = len(cross_map)
        cross_labels[i] = cross_map[key]

# Filter to groups with >= MIN_GROUP_SIZE members
size_counts = Counter(cross_labels[cross_labels >= 0])
valid_groups = {k for k, cnt in size_counts.items() if cnt >= MIN_GROUP_SIZE}
cross_labels_filtered = np.array([l if l in valid_groups else -1 for l in cross_labels])
n_cross_valid = len(valid_groups)
print(f"  Operator×Scale: {n_cross_valid} categories with >= {MIN_GROUP_SIZE} members")


# ══════════════════════════════════════════════════════════════════
#  RUN ALL TAXONOMY TESTS
# ══════════════════════════════════════════════════════════════════

taxonomies = [
    ("EO operators",           eo_labels,              9),
    ("EO triads",              triad_labels,           3),
    ("Scale",                  scale_labels,           4),
    ("Morphological",          morph_labels,           n_morph_groups),
    ("Operator×Scale",         cross_labels_filtered,  n_cross_valid),
]
if has_wordnet:
    taxonomies.append(("WordNet lexicographer", wn_labels, len(wn_label_names)))

print(f"\n{'='*80}")
print("  PERMUTATION TESTS ({} permutations each)".format(N_PERM))
print(f"{'='*80}")

results = {}
for name, labels, expected_k in taxonomies:
    print(f"\n  Testing: {name} (k={expected_k})...")
    r = permutation_test(embeddings, labels, n_perm=N_PERM, seed=SEED)
    if r:
        results[name] = r
        print(f"    η² = {r['eta_sq']:.6f} (null: {r['eta_sq_null_mean']:.6f} ± {r['eta_sq_null_std']:.6f})")
        print(f"    z(η²) = {r['z_eta']:+.1f}, p = {r['p_eta']:.4f}")
        print(f"    mean_sim = {r['mean_sim']:.6f}, z(sim) = {r['z_sim']:+.1f}")
    else:
        print(f"    SKIPPED (< 2 groups)")


# ══════════════════════════════════════════════════════════════════
#  k-MATCHED NULL BASELINES
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  k-MATCHED NULL BASELINES ({N_MATCHED_TRIALS} trials × {N_PERM_MATCHED} perms each)")
print(f"{'='*80}")
print("  For each taxonomy, generate random taxonomies with SAME k and group sizes")
print("  to determine what z-score and η² you'd expect by chance for that k.\n")

matched_nulls = {}
for name, labels, expected_k in taxonomies:
    if name not in results:
        continue
    print(f"  Computing null baseline for {name} (k={expected_k})...")
    mn = k_matched_null_baseline(embeddings, labels,
                                  n_trials=N_MATCHED_TRIALS,
                                  n_perm_per=N_PERM_MATCHED,
                                  seed=SEED + hash(name) % 10000)
    if mn:
        matched_nulls[name] = mn
        real_z = results[name]['z_sim']
        real_eta = results[name]['eta_sq']

        # "Corrected z": how many SDs above the k-matched null
        if mn['null_z_sim_std'] > 0:
            corrected_z = (real_z - mn['null_z_sim_mean']) / mn['null_z_sim_std']
        else:
            corrected_z = 0
        matched_nulls[name]['corrected_z'] = corrected_z

        # Excess η²: how much more than random with same k
        excess_eta = real_eta - mn['null_eta_mean']
        matched_nulls[name]['excess_eta'] = excess_eta

        print(f"    null z(sim) = {mn['null_z_sim_mean']:+.1f} ± {mn['null_z_sim_std']:.1f}")
        print(f"    real z(sim) = {real_z:+.1f}  →  corrected z = {corrected_z:+.1f}")
        print(f"    null η² = {mn['null_eta_mean']:.6f}, excess η² = {excess_eta:.6f}")


# ══════════════════════════════════════════════════════════════════
#  MULTIPLE COMPARISON CORRECTION (BH-FDR)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  MULTIPLE COMPARISON CORRECTION (Benjamini-Hochberg FDR)")
print(f"{'='*80}")

tax_names = list(results.keys())
p_values = [results[n]['p_eta'] for n in tax_names]
adjusted_p, rejected = benjamini_hochberg(p_values, alpha=0.05)

for i, name in enumerate(tax_names):
    results[name]['p_eta_adjusted'] = adjusted_p[i]
    results[name]['significant_bh'] = rejected[i]
    star = "***" if rejected[i] else "n.s."
    print(f"  {name:30s}  p_raw={p_values[i]:.4f}  p_adj={adjusted_p[i]:.4f}  {star}")


# ══════════════════════════════════════════════════════════════════
#  HIGH-CONFIDENCE SUBSET
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  HIGH-CONFIDENCE SUBSET (confidence = H only)")
print(f"{'='*80}")

hc_results = {}
hc_embeddings = embeddings[high_conf_mask]
for name, labels, expected_k in taxonomies:
    hc_labels = labels[high_conf_mask]
    unique_valid = set(l for l in hc_labels if l >= 0)
    # Remove groups with < MIN_GROUP_SIZE in HC subset
    sizes = Counter(l for l in hc_labels if l >= 0)
    small = {g for g, c in sizes.items() if c < MIN_GROUP_SIZE}
    if small:
        hc_labels = np.array([l if l not in small else -1 for l in hc_labels])
        unique_valid -= small

    n_valid = len(unique_valid)
    if n_valid < 2:
        print(f"  {name}: SKIPPED (only {n_valid} groups with >= {MIN_GROUP_SIZE} members)")
        continue

    print(f"  {name} (k={n_valid}, n={int((hc_labels >= 0).sum())})...")
    r = permutation_test(hc_embeddings, hc_labels, n_perm=N_PERM, seed=SEED)
    if r:
        hc_results[name] = r
        print(f"    η² = {r['eta_sq']:.6f}, z(η²) = {r['z_eta']:+.1f}")


# ══════════════════════════════════════════════════════════════════
#  ORTHOGONALITY (EO vs WordNet)
# ══════════════════════════════════════════════════════════════════
if has_wordnet:
    print(f"\n{'='*80}")
    print("  ORTHOGONALITY: EO vs WordNet")
    print(f"{'='*80}")

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    both_mask = (eo_labels >= 0) & (wn_labels >= 0)
    eo_both = eo_labels[both_mask]
    wn_both = wn_labels[both_mask]

    ari = adjusted_rand_score(eo_both, wn_both)
    nmi = normalized_mutual_info_score(eo_both, wn_both)

    print(f"  Adjusted Rand Index:        {ari:.4f}")
    print(f"  Normalized Mutual Info:      {nmi:.4f}")
    print(f"  (ARI=0 → independent, ARI=1 → identical)")

    if ari < 0.05:
        print(f"  → EO and WordNet are nearly ORTHOGONAL")
    elif ari < 0.15:
        print(f"  → WEAK overlap")
    else:
        print(f"  → MODERATE-STRONG overlap")


# ══════════════════════════════════════════════════════════════════
#  COMPARISON TABLE (corrected)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  CORRECTED COMPARISON TABLE")
print(f"{'='*80}")

# Build rows: sort by η² (descending — higher = more variance explained)
rows = []
for name in results:
    r = results[name]
    mn = matched_nulls.get(name, {})
    hc = hc_results.get(name)

    rows.append({
        'name': name,
        'k': r['n_groups'],
        'eta_sq': r['eta_sq'],
        'excess_eta': mn.get('excess_eta', r['eta_sq']),
        'z_sim': r['z_sim'],
        'corrected_z': mn.get('corrected_z', r['z_sim']),
        'p_adj': r.get('p_eta_adjusted', r['p_eta']),
        'significant': r.get('significant_bh', r['p_eta'] < 0.05),
        'hc_eta': hc['eta_sq'] if hc else None,
    })

rows.sort(key=lambda x: -x['eta_sq'])

header = f"  {'Taxonomy':30s} {'k':>3s} {'η²':>8s} {'excess η²':>10s} {'z(sim)':>8s} {'corrected_z':>12s} {'p_adj':>8s} {'sig':>5s} {'HC η²':>8s}"
print(header)
print("  " + "-" * (len(header) - 2))

for row in rows:
    sig = "***" if row['significant'] else "n.s."
    hc_str = f"{row['hc_eta']:.6f}" if row['hc_eta'] is not None else "—"
    print(f"  {row['name']:30s} {row['k']:3d} {row['eta_sq']:.6f} {row['excess_eta']:+10.6f} "
          f"{row['z_sim']:+8.1f} {row['corrected_z']:+12.1f} {row['p_adj']:8.4f} {sig:>5s} {hc_str:>8s}")


# ══════════════════════════════════════════════════════════════════
#  k-CONFOUND DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  k-CONFOUND DIAGNOSTIC")
print(f"{'='*80}")
print("  Correlation between k and z_sim (the original confound):")

ks = np.array([r['k'] for r in rows])
zs = np.array([r['z_sim'] for r in rows])
corrected_zs = np.array([r['corrected_z'] for r in rows])
etas = np.array([r['eta_sq'] for r in rows])

if len(ks) > 2:
    from scipy.stats import pearsonr
    r_kz, p_kz = pearsonr(ks, zs)
    r_kcz, p_kcz = pearsonr(ks, corrected_zs)
    r_ke, p_ke = pearsonr(ks, etas)
    print(f"  r(k, z_sim)       = {r_kz:+.3f}  (p={p_kz:.4f})  {'← CONFOUND' if abs(r_kz) > 0.5 else '← OK'}")
    print(f"  r(k, corrected_z) = {r_kcz:+.3f}  (p={p_kcz:.4f})  {'← CONFOUND' if abs(r_kcz) > 0.5 else '← OK'}")
    print(f"  r(k, η²)          = {r_ke:+.3f}  (p={p_ke:.4f})  {'← CONFOUND' if abs(r_ke) > 0.5 else '← OK'}")


# ══════════════════════════════════════════════════════════════════
#  KEY FINDINGS
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  KEY FINDINGS")
print(f"{'='*80}")

# Rank by excess η²
rows_by_excess = sorted(rows, key=lambda x: -x['excess_eta'])
print("\n  Ranking by excess η² (effect above k-matched random):")
for i, row in enumerate(rows_by_excess):
    sig = "***" if row['significant'] else "n.s."
    print(f"    {i+1}. {row['name']:30s}  excess η² = {row['excess_eta']:+.6f}  {sig}")

print(f"\n  Original z_sim ranking (CONFOUNDED by k):")
rows_by_z = sorted(rows, key=lambda x: x['z_sim'])
for i, row in enumerate(rows_by_z):
    print(f"    {i+1}. {row['name']:30s}  z = {row['z_sim']:+.1f}  (k={row['k']})")

# Does EO beat WordNet when properly compared?
if has_wordnet and 'EO operators' in results and 'WordNet lexicographer' in results:
    eo_excess = matched_nulls.get('EO operators', {}).get('excess_eta', 0)
    wn_excess = matched_nulls.get('WordNet lexicographer', {}).get('excess_eta', 0)
    print(f"\n  EO vs WordNet (k-corrected):")
    print(f"    EO operators:    excess η² = {eo_excess:+.6f}")
    print(f"    WordNet:         excess η² = {wn_excess:+.6f}")
    if eo_excess > wn_excess:
        print(f"    → EO explains MORE variance than WordNet after k correction")
    elif eo_excess > wn_excess * 0.8:
        print(f"    → EO and WordNet explain COMPARABLE variance after k correction")
    else:
        print(f"    → WordNet explains more variance; EO's original advantage was inflated by k")

# Orthogonality summary
if has_wordnet:
    print(f"\n  Orthogonality (strongest finding):")
    print(f"    ARI = {ari:.4f}, NMI = {nmi:.4f}")
    print(f"    → EO and WordNet capture DIFFERENT structure (this stands regardless of k)")

# Triads vs operators
if 'EO triads' in results and 'EO operators' in results:
    t_excess = matched_nulls.get('EO triads', {}).get('excess_eta', 0)
    o_excess = matched_nulls.get('EO operators', {}).get('excess_eta', 0)
    print(f"\n  EO triads vs operators:")
    print(f"    Triads (k=3):    excess η² = {t_excess:+.6f}")
    print(f"    Operators (k=9): excess η² = {o_excess:+.6f}")
    if t_excess > o_excess:
        print(f"    → Triads capture MORE structure per-group (coarser = stronger signal)")
    else:
        print(f"    → Operators capture MORE fine-grained structure")


# ══════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════
print("\n  Generating visualizations...")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: η² comparison
    ax = axes[0]
    names_sorted = [r['name'] for r in rows]
    etas_sorted = [r['eta_sq'] for r in rows]
    excess_sorted = [r['excess_eta'] for r in rows]

    colors = []
    for n in names_sorted:
        if 'EO op' in n:
            colors.append('#e74c3c')
        elif 'WordNet' in n:
            colors.append('#3498db')
        elif 'triad' in n:
            colors.append('#e67e22')
        elif 'Scale' in n and 'Operator' not in n:
            colors.append('#2ecc71')
        elif 'Morph' in n:
            colors.append('#9b59b6')
        else:
            colors.append('#f1c40f')

    y_pos = range(len(names_sorted))
    ax.barh(y_pos, etas_sorted, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} (k={r['k']})" for n, r in zip(names_sorted, rows)], fontsize=9)
    ax.set_xlabel('η² (variance explained)', fontsize=11)
    ax.set_title('Effect Size (η²)\nComparable across k', fontsize=12)
    ax.invert_yaxis()

    # Panel 2: Excess η² (above k-matched null)
    ax = axes[1]
    ax.barh(y_pos, excess_sorted, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} (k={r['k']})" for n, r in zip(names_sorted, rows)], fontsize=9)
    ax.set_xlabel('Excess η² (above k-matched null)', fontsize=11)
    ax.set_title('k-Corrected Effect\n(random with same k subtracted)', fontsize=12)
    ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.3)
    ax.invert_yaxis()

    # Panel 3: Original z_sim (showing the confound)
    ax = axes[2]
    z_sims = [r['z_sim'] for r in rows]
    ax.barh(y_pos, z_sims, color=colors, edgecolor='black', linewidth=0.5, alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} (k={r['k']})" for n, r in zip(names_sorted, rows)], fontsize=9)
    ax.set_xlabel('z-score (cosine sim)', fontsize=11)
    ax.set_title('Original z-score\n(NOT comparable across k)', fontsize=12)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ, "taxonomy_robust_comparison.png"), dpi=150)
    plt.close()
    print("  ✓ taxonomy_robust_comparison.png")

except ImportError:
    print("  (matplotlib not available)")


# ══════════════════════════════════════════════════════════════════
#  SAVE REPORT
# ══════════════════════════════════════════════════════════════════
report = {
    'method': {
        'n_permutations': N_PERM,
        'n_matched_trials': N_MATCHED_TRIALS,
        'n_perm_per_matched': N_PERM_MATCHED,
        'min_group_size': MIN_GROUP_SIZE,
        'embedding_model': emb_method if 'emb_method' in dir() else 'cached',
        'embedding_dim': int(embeddings.shape[1]),
        'n_verbs': N,
        'correction': 'Benjamini-Hochberg FDR',
    },
    'results': {},
    'high_confidence_results': {},
    'k_matched_nulls': {},
    'comparison_ranking_eta_sq': [r['name'] for r in rows],
    'comparison_ranking_excess_eta': [r['name'] for r in rows_by_excess],
}

for name in results:
    report['results'][name] = results[name]
    if name in matched_nulls:
        report['k_matched_nulls'][name] = matched_nulls[name]
    if name in hc_results:
        report['high_confidence_results'][name] = hc_results[name]

if has_wordnet:
    report['orthogonality'] = {
        'ari': float(ari),
        'nmi': float(nmi),
    }

with open(os.path.join(OUT, "taxonomy_robust.json"), 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*80}")
print("  ROBUST TAXONOMY COMPARISON COMPLETE")
print(f"{'='*80}")
print(f"  Report: {OUT}/taxonomy_robust.json")
print(f"  Visualization: {VIZ}/taxonomy_robust_comparison.png")
