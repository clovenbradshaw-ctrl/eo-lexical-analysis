#!/usr/bin/env python3
"""
Step 09e: Position (Column) Z-Score Analysis
==============================================

The covariation analysis (09c) showed:
  - Triads (rows) don't co-vary across languages (p=0.81)
  - Positions (columns) trend toward co-variation (p=0.28)
  - Data-driven clustering pulls one operator from each triad

The z-score analysis (09d) showed:
  - Triads are geometrically real (27/27 languages, z < -3.7)

This script tests: are POSITIONS also geometrically real?
  Position 1 (differentiate): NUL, SEG, ALT
  Position 2 (relate):        DES, CON, SUP
  Position 3 (generate):      INS, SYN, REC

If both axes produce geometric separation, the helix has genuine
two-dimensional structure: rows = semantic domain, columns = transformation type.

Also tests the data-driven grouping from 09c:
  Cluster A: NUL, SYN, ALT
  Cluster B: DES, SEG, REC
  Cluster C: INS, CON, SUP

Uses the same native-language definition embeddings from 09d.
"""

import json, os, sys
import numpy as np
from collections import Counter
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']

# Grouping schemes to test
GROUPINGS = {
    'triads': {
        'Existence':      ['NUL','DES','INS'],
        'Structure':      ['SEG','CON','SYN'],
        'Interpretation': ['ALT','SUP','REC'],
    },
    'positions': {
        'Differentiate': ['NUL','SEG','ALT'],
        'Relate':        ['DES','CON','SUP'],
        'Generate':      ['INS','SYN','REC'],
    },
    'diag_main': {
        'Diag_A': ['NUL','CON','REC'],  # main diagonal: null-state, bonding, rebuilding
        'Diag_B': ['DES','SYN','ALT'],  # typing, combining, reframing
        'Diag_C': ['INS','SEG','SUP'],  # appearing, cutting, holding contradiction
    },
    'diag_anti': {
        'Anti_A': ['INS','CON','ALT'],  # anti-diagonal: appearing, bonding, reframing
        'Anti_B': ['NUL','SYN','SUP'],  # null-state, combining, holding contradiction
        'Anti_C': ['DES','SEG','REC'],  # typing, cutting, rebuilding
    },
    'data_driven': {
        'Cluster_A': ['NUL','SYN','ALT'],
        'Cluster_B': ['DES','SEG','REC'],
        'Cluster_C': ['INS','CON','SUP'],
    },
}


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
        'n_verbs': len(labels),
    }


def assign_group(operator, grouping):
    """Map an operator to its group label under a given grouping scheme."""
    for group_name, ops in grouping.items():
        if operator in ops:
            return group_name
    return None


def main():
    print(f"{'='*70}")
    print("  POSITION & GROUPING Z-SCORE ANALYSIS")
    print(f"{'='*70}")
    
    # Collect results per language per grouping
    results = {}
    
    for lang_dir in sorted(Path(DATA).iterdir()):
        if not lang_dir.is_dir():
            continue
        
        embeddings_file = lang_dir / "definition_embeddings.npz"
        classified_file = lang_dir / "classified.json"
        
        if not embeddings_file.exists() or not classified_file.exists():
            continue
        
        emb_data = np.load(embeddings_file, allow_pickle=True)
        embeddings = emb_data['embeddings']
        operators = emb_data['operators'].tolist()
        
        with open(classified_file) as f:
            meta = json.load(f)
        
        lang = meta['language']
        
        valid_mask = [op in HELIX for op in operators]
        embeddings = embeddings[valid_mask]
        operators = [op for op, v in zip(operators, valid_mask) if v]
        
        if len(operators) < 30:
            continue
        
        lang_results = {
            'family': meta['family'],
            'era': meta['era'],
            'n_verbs': len(operators),
        }
        
        # Test each grouping scheme
        for scheme_name, grouping in GROUPINGS.items():
            labels = [assign_group(op, grouping) for op in operators]
            # Filter out None
            valid = [(e, l) for e, l in zip(range(len(labels)), labels) if l is not None]
            if len(valid) < 30:
                lang_results[scheme_name] = None
                continue
            
            valid_idx = [v[0] for v in valid]
            valid_labels = [v[1] for v in valid]
            valid_emb = embeddings[valid_idx]
            
            z = compute_z_score(valid_emb, valid_labels, n_perm=500)
            lang_results[scheme_name] = z
        
        # Also test 9 individual operators
        z_ops = compute_z_score(embeddings, operators, n_perm=500)
        lang_results['operators'] = z_ops
        
        results[lang] = lang_results
        
        t_z = lang_results['triads']['z_score'] if lang_results.get('triads') else float('nan')
        p_z = lang_results['positions']['z_score'] if lang_results.get('positions') else float('nan')
        dm_z = lang_results['diag_main']['z_score'] if lang_results.get('diag_main') else float('nan')
        da_z = lang_results['diag_anti']['z_score'] if lang_results.get('diag_anti') else float('nan')
        d_z = lang_results['data_driven']['z_score'] if lang_results.get('data_driven') else float('nan')
        o_z = lang_results['operators']['z_score'] if lang_results.get('operators') else float('nan')
        
        print(f"  {lang:25s}  ops={o_z:+7.1f}  tri={t_z:+7.1f}  pos={p_z:+7.1f}  diagM={dm_z:+7.1f}  diagA={da_z:+7.1f}  data={d_z:+7.1f}")
    
    # ══════════════════════════════════════════════════════════
    #  COMPARISON TABLE
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  COMPARISON: WHICH GROUPING PRODUCES STRONGEST SEPARATION?")
    print(f"{'='*70}")
    
    sorted_langs = sorted(results.keys(), key=lambda x: results[x].get('operators', {}).get('z_score', 0) if results[x].get('operators') else 0)
    
    print(f"\n  {'Language':25s} {'9 ops':>8s} {'Triads':>8s} {'Posit':>8s} {'DiagM':>8s} {'DiagA':>8s} {'DataDr':>8s} {'Best':>12s}")
    print("  " + "-"*95)
    
    for lang in sorted_langs:
        r = results[lang]
        scores = {}
        for key in ['operators', 'triads', 'positions', 'diag_main', 'diag_anti', 'data_driven']:
            if r.get(key) and r[key].get('z_score') is not None:
                scores[key] = r[key]['z_score']
            else:
                scores[key] = float('nan')
        
        # Which grouping is strongest (most negative z)?
        valid_scores = {k: v for k, v in scores.items() if not np.isnan(v) and k != 'operators'}
        best = min(valid_scores, key=valid_scores.get) if valid_scores else 'n/a'
        
        print(f"  {lang:25s} {scores['operators']:+8.1f} {scores['triads']:+8.1f} {scores['positions']:+8.1f} {scores.get('diag_main', float('nan')):+8.1f} {scores.get('diag_anti', float('nan')):+8.1f} {scores['data_driven']:+8.1f}  {best}")
    
    # Summary
    print(f"\n  SUMMARY ACROSS ALL LANGUAGES:")
    print(f"  {'Grouping':>15s} {'Mean z':>8s} {'Min z':>8s} {'Max z':>8s} {'All<-2':>7s} {'All<-3':>7s}")
    print("  " + "-"*60)
    
    for key, label in [('operators', '9 operators'), ('triads', 'Triads (rows)'), ('positions', 'Positions (cols)'), ('diag_main', 'Main diagonal'), ('diag_anti', 'Anti-diagonal'), ('data_driven', 'Data-driven')]:
        z_list = [results[l][key]['z_score'] for l in results if results[l].get(key) and results[l][key] is not None]
        if not z_list:
            continue
        all_lt2 = all(z < -1.96 for z in z_list)
        all_lt3 = all(z < -3 for z in z_list)
        print(f"  {label:>15s} {np.mean(z_list):+8.1f} {min(z_list):+8.1f} {max(z_list):+8.1f} {'YES' if all_lt2 else 'NO':>7s} {'YES' if all_lt3 else 'NO':>7s}")
    
    # Which grouping wins most often?
    print(f"\n  WHICH GROUPING IS STRONGEST PER LANGUAGE?")
    win_counts = Counter()
    for lang in results:
        scores = {}
        for key in ['triads', 'positions', 'diag_main', 'diag_anti', 'data_driven']:
            if results[lang].get(key) and results[lang][key] is not None:
                scores[key] = results[lang][key]['z_score']
        if scores:
            winner = min(scores, key=scores.get)
            win_counts[winner] += 1
    
    for key in ['triads', 'positions', 'diag_main', 'diag_anti', 'data_driven']:
        print(f"    {key:>15s}: strongest in {win_counts.get(key, 0)}/{len(results)} languages")
    
    # Key comparisons
    print(f"\n  HEAD-TO-HEAD COMPARISONS (which axis stronger per language):")
    for a, b in [('triads', 'positions'), ('triads', 'diag_main'), ('positions', 'diag_main'), ('triads', 'diag_anti'), ('positions', 'diag_anti')]:
        a_wins = 0
        b_wins = 0
        for lang in results:
            ra = results[lang].get(a)
            rb = results[lang].get(b)
            if ra and rb:
                if ra['z_score'] < rb['z_score']:
                    a_wins += 1
                else:
                    b_wins += 1
        print(f"    {a:>15s} wins {a_wins:2d} | {b:<15s} wins {b_wins:2d}")
    
    # Visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(14, max(6, len(results) * 0.4)))
        
        langs_sorted = sorted(results.keys(), 
                             key=lambda x: results[x]['positions']['z_score'] if results[x].get('positions') else 0)
        
        y = np.arange(len(langs_sorted))
        height = 0.16
        
        for i, (key, color, label) in enumerate([
            ('triads', '#e74c3c', 'Triads (rows)'),
            ('positions', '#3498db', 'Positions (cols)'),
            ('diag_main', '#2ecc71', 'Main diagonal'),
            ('diag_anti', '#9b59b6', 'Anti-diagonal'),
            ('data_driven', '#e67e22', 'Data-driven'),
        ]):
            z_vals = []
            for lang in langs_sorted:
                r = results[lang].get(key)
                z_vals.append(r['z_score'] if r else 0)
            
            ax.barh(y + i * height, z_vals, height=height, color=color, alpha=0.8, label=label)
        
        ax.set_yticks(y + height * 2)
        ax.set_yticklabels([f"{l} [{results[l]['family']}]" for l in langs_sorted], fontsize=7)
        ax.set_xlabel("Z-score (more negative = stronger separation)")
        ax.axvline(0, color='black', linewidth=0.5)
        ax.axvline(-1.96, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.legend(loc='lower left', fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Geometric Separation by Grouping Scheme (All 5 Axes)\nAcross 27 Languages (native definition embeddings)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ, "crossling_all_axes.png"), dpi=150)
        plt.close()
        print(f"\n  ✓ crossling_all_axes.png")
    except ImportError:
        pass
    
    # Save
    report = {}
    for lang, r in results.items():
        report[lang] = {}
        for key in ['operators', 'triads', 'positions', 'diag_main', 'diag_anti', 'data_driven']:
            if r.get(key) and r[key] is not None:
                report[lang][key] = r[key]['z_score']
            else:
                report[lang][key] = None
        report[lang]['family'] = r['family']
        report[lang]['era'] = r['era']
        report[lang]['n_verbs'] = r['n_verbs']
    
    with open(os.path.join(OUT, "crossling_all_axes.json"), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Saved crossling_all_axes.json")


if __name__ == '__main__':
    main()
