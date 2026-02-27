#!/usr/bin/env python3
"""
09f_embed_centroids.py — Compute 27-phasepost centroids in embedding space
===========================================================================

For each language (or all combined):
  1. Load embeddings + EO classifications + referent classifications
  2. Assign each verb to its 27-cell
  3. Compute centroid of each cell's embeddings
  4. PCA to 3D
  5. Output JSON for 3D visualization

Usage:
  python scripts/09f_embed_centroids.py                # all languages combined
  python scripts/09f_embed_centroids.py --lang English  # single language
  python scripts/09f_embed_centroids.py --all           # each language separately + combined
"""

import json, os, sys, argparse
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling")
OUT  = os.path.join(BASE, "output")

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']

TRIADS = {
    'NUL':'Existence','DES':'Existence','INS':'Existence',
    'SEG':'Structure','CON':'Structure','SYN':'Structure',
    'ALT':'Interpretation','SUP':'Interpretation','REC':'Interpretation',
}
POSITIONS = {
    'NUL':'Differentiate','SEG':'Differentiate','ALT':'Differentiate',
    'DES':'Relate','CON':'Relate','SUP':'Relate',
    'INS':'Generate','SYN':'Generate','REC':'Generate',
}

POS_ORDER = ['Differentiate','Relate','Generate']
TRI_ORDER = ['Existence','Structure','Interpretation']
REF_ORDER = ['FIGURE','PATTERN','GROUND']

LANGUAGES = [
    "Ancient_Greek","Arabic","Basque","Classical_Chinese","Coptic",
    "English","Finnish","French","German","Gothic","Hindi",
    "Indonesian","Japanese","Korean","Latin","Naija",
    "Old_Church_Slavonic","Old_East_Slavic","Persian","Russian",
    "Sanskrit","Tagalog","Tamil","Turkish","Uyghur",
    "Vietnamese","Wolof"
]


def load_language(lang):
    """Load embeddings, classifications, and referent axis for one language.
    Returns list of {verb, operator, referent, embedding}."""
    lang_dir = os.path.join(DATA, lang)
    if not os.path.exists(lang_dir):
        print(f"✗ dir not found: {lang_dir}")
        return None

    # Load embeddings (.npz format)
    emb_file = os.path.join(lang_dir, "definition_embeddings.npz")
    if not os.path.exists(emb_file):
        # Try JSON fallback
        emb_json = os.path.join(lang_dir, "embeddings.json")
        if not os.path.exists(emb_json):
            print(f"✗ no embeddings file")
            return None
        with open(emb_json) as f:
            emb_data = json.load(f)
        emb_map = {}
        for item in emb_data:
            v = item.get('verb', '').strip()
            e = item.get('embedding')
            if v and e:
                emb_map[v] = np.array(e, dtype=np.float32)
    else:
        npz = np.load(emb_file, allow_pickle=True)
        # Detect format: could be verbs+embeddings arrays, or a dict
        if 'verbs' in npz and 'embeddings' in npz:
            verbs_arr = npz['verbs']
            embs_arr = npz['embeddings']
            emb_map = {}
            for i, v in enumerate(verbs_arr):
                emb_map[str(v).strip()] = embs_arr[i].astype(np.float32)
        elif 'arr_0' in npz:
            # Single array - try to parse
            data = npz['arr_0']
            if isinstance(data, np.ndarray) and data.dtype == object:
                # Array of dicts
                emb_map = {}
                for item in data:
                    if isinstance(item, dict):
                        v = str(item.get('verb', '')).strip()
                        e = item.get('embedding')
                        if v and e is not None:
                            emb_map[v] = np.array(e, dtype=np.float32)
            else:
                return None
        else:
            # Try all keys
            keys = list(npz.keys())
            if len(keys) >= 2:
                # Guess: first is verbs, second is embeddings
                emb_map = {}
                try:
                    verbs_arr = npz[keys[0]]
                    embs_arr = npz[keys[1]]
                    for i, v in enumerate(verbs_arr):
                        emb_map[str(v).strip()] = embs_arr[i].astype(np.float32)
                except:
                    return None
            else:
                return None

    # Load classifications
    class_file = os.path.join(lang_dir, "classified.json")
    if not os.path.exists(class_file):
        print(f"✗ no classified.json")
        return None

    with open(class_file) as f:
        class_data = json.load(f)

    # Build verb -> operator map (handle both formats)
    op_map = {}
    if isinstance(class_data, dict) and 'classifications' in class_data:
        items = class_data['classifications']
    elif isinstance(class_data, list):
        items = class_data
    elif isinstance(class_data, dict):
        # Maybe keys are verbs?
        items = [{'verb': k, 'operator': v} for k, v in class_data.items() if isinstance(v, str)]
        if not items:
            # Try other dict formats
            for k, v in class_data.items():
                if isinstance(v, list):
                    items = v
                    break
    else:
        return None

    for c in items:
        if isinstance(c, dict):
            v = str(c.get('verb', '')).strip()
            op = str(c.get('operator', '')).upper().strip()
            if v and op in HELIX:
                op_map[v] = op

    # Load referent axis
    ref_file = os.path.join(lang_dir, "referent_axis.json")
    if not os.path.exists(ref_file):
        print(f"✗ no referent_axis.json")
        return None

    with open(ref_file) as f:
        ref_data = json.load(f)

    # Build verb -> referent map (handle both list and dict formats)
    ref_map = {}
    if isinstance(ref_data, list):
        items = ref_data
    elif isinstance(ref_data, dict):
        items = ref_data.get('classifications', ref_data.get('verbs', []))
        if not items and all(isinstance(v, str) for v in ref_data.values()):
            items = [{'verb': k, 'referent': v} for k, v in ref_data.items()]
    else:
        return None

    for r in items:
        if isinstance(r, dict):
            v = str(r.get('verb', '')).strip()
            ref = str(r.get('referent', '')).upper().strip()
            if v and ref in REF_ORDER:
                ref_map[v] = ref

    # Merge: only verbs with all three
    results = []
    for v in emb_map:
        if v in op_map and v in ref_map:
            results.append({
                'verb': v,
                'operator': op_map[v],
                'referent': ref_map[v],
                'embedding': emb_map[v],
            })

    if not results:
        print(f"✗ no verbs with all three (emb={len(emb_map)}, op={len(op_map)}, ref={len(ref_map)})")
        return None

    return results


def compute_centroids_3d(verbs, label=""):
    """Compute 27-cell centroids and PCA to 3D. Returns dict for JSON."""

    # Group by cell
    cells = defaultdict(list)
    for v in verbs:
        op = v['operator']
        pos = POSITIONS[op]
        tri = TRIADS[op]
        ref = v['referent']
        key = f"{pos}|{tri}|{ref}"
        cells[key].append(v['embedding'])

    # Also collect ALL embeddings for PCA basis
    all_embs = np.array([v['embedding'] for v in verbs])

    # PCA on all embeddings to get 3D basis
    mean = all_embs.mean(axis=0)
    centered = all_embs - mean
    # Use SVD for efficiency
    # For very large matrices, compute covariance on subset
    if len(centered) > 5000:
        idx = np.random.choice(len(centered), 5000, replace=False)
        subset = centered[idx]
    else:
        subset = centered

    U, S, Vt = np.linalg.svd(subset, full_matrices=False)
    basis = Vt[:3]  # top 3 principal components

    # Variance explained
    total_var = (S**2).sum()
    var_explained = [(S[i]**2 / total_var * 100) for i in range(min(3, len(S)))]

    # Compute centroids in 3D
    points = []
    for pos in POS_ORDER:
        for tri in TRI_ORDER:
            for ref in REF_ORDER:
                key = f"{pos}|{tri}|{ref}"
                op = [o for o in HELIX if POSITIONS[o]==pos and TRIADS[o]==tri][0]
                embs = cells.get(key, [])
                n = len(embs)

                if n == 0:
                    points.append({
                        'pos': pos, 'tri': tri, 'ref': ref, 'op': op,
                        'n': 0, 'x': 0, 'y': 0, 'z': 0,
                        'sample_verbs': [],
                    })
                else:
                    centroid = np.mean(embs, axis=0)
                    projected = (centroid - mean) @ basis.T
                    # Get sample verbs for this cell
                    cell_verbs = [v['verb'] for v in verbs
                                  if POSITIONS[v['operator']]==pos
                                  and TRIADS[v['operator']]==tri
                                  and v['referent']==ref][:10]
                    points.append({
                        'pos': pos, 'tri': tri, 'ref': ref, 'op': op,
                        'n': n,
                        'x': float(projected[0]),
                        'y': float(projected[1]),
                        'z': float(projected[2]),
                        'sample_verbs': cell_verbs,
                    })

    # Also project individual verbs for optional scatter
    # (subsample if too many)
    scatter_verbs = []
    sample_n = min(len(verbs), 3000)
    if sample_n < len(verbs):
        idx = np.random.choice(len(verbs), sample_n, replace=False)
        sample = [verbs[i] for i in idx]
    else:
        sample = verbs

    for v in sample:
        proj = (v['embedding'] - mean) @ basis.T
        scatter_verbs.append({
            'x': float(proj[0]),
            'y': float(proj[1]),
            'z': float(proj[2]),
            'op': v['operator'],
            'ref': v['referent'],
        })

    return {
        'label': label,
        'n_verbs': len(verbs),
        'n_cells_populated': sum(1 for p in points if p['n'] > 0),
        'variance_explained': var_explained,
        'centroids': points,
        'scatter': scatter_verbs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, help='Single language')
    parser.add_argument('--all', action='store_true', help='All languages separately + combined')
    parser.add_argument('--diag', action='store_true', help='Diagnose file formats')
    args = parser.parse_args()

    if args.diag:
        # Print what files exist and their format
        lang_dir = os.path.join(DATA, "English")
        print(f"Checking {lang_dir}:")
        for f in os.listdir(lang_dir):
            fp = os.path.join(lang_dir, f)
            print(f"  {f} ({os.path.getsize(fp)} bytes)")
            if f.endswith('.npz'):
                npz = np.load(fp, allow_pickle=True)
                print(f"    keys: {list(npz.keys())}")
                for k in list(npz.keys())[:3]:
                    arr = npz[k]
                    print(f"    {k}: shape={arr.shape}, dtype={arr.dtype}")
                    if arr.dtype == object and len(arr) > 0:
                        print(f"    sample[0] type={type(arr[0])}")
                        if isinstance(arr[0], dict):
                            print(f"    sample[0] keys={list(arr[0].keys())[:5]}")
                    elif len(arr.shape) > 1:
                        print(f"    sample[0][:3]={arr[0][:3]}")
            elif f.endswith('.json'):
                with open(fp) as fh:
                    d = json.load(fh)
                if isinstance(d, dict):
                    print(f"    dict keys: {list(d.keys())[:5]}")
                    for k in list(d.keys())[:2]:
                        v = d[k]
                        if isinstance(v, list) and len(v) > 0:
                            print(f"    {k}[0] = {str(v[0])[:120]}")
                elif isinstance(d, list):
                    print(f"    list len={len(d)}")
                    if len(d) > 0:
                        print(f"    [0] = {str(d[0])[:120]}")
        return

    results = {}

    if args.lang:
        langs = [args.lang]
    elif args.all:
        langs = LANGUAGES
    else:
        langs = LANGUAGES  # default: combine all

    # Load and compute per language
    all_verbs_combined = []

    for lang in langs:
        print(f"  Loading {lang}...", end=" ", flush=True)
        verbs = load_language(lang)
        if verbs is None:
            continue
        print(f"{len(verbs)} verbs", end=" ", flush=True)

        if args.all or args.lang:
            result = compute_centroids_3d(verbs, label=lang)
            results[lang] = result
            print(f"→ {result['n_cells_populated']}/27 cells, PCA var={sum(result['variance_explained']):.1f}%")
        else:
            print()

        all_verbs_combined.extend(verbs)

    # Combined
    if not args.lang and len(all_verbs_combined) > 0:
        print(f"\n  Computing combined ({len(all_verbs_combined)} verbs)...", end=" ", flush=True)
        combined = compute_centroids_3d(all_verbs_combined, label="All languages")
        results['__combined__'] = combined
        print(f"→ {combined['n_cells_populated']}/27 cells, PCA var={sum(combined['variance_explained']):.1f}%")

    # Save
    out_path = os.path.join(OUT, "phasepost_embeddings_3d.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved {out_path}")
    print(f"    {len(results)} datasets")

    # Print centroid positions for combined
    if '__combined__' in results:
        print(f"\n  COMBINED CENTROIDS (PCA 3D):")
        print(f"  Variance explained: {', '.join(f'PC{i+1}={v:.1f}%' for i,v in enumerate(results['__combined__']['variance_explained']))}")
        for p in sorted(results['__combined__']['centroids'], key=lambda x: -x['n']):
            if p['n'] > 0:
                print(f"    {p['pos']:>15s} × {p['tri']:>15s} × {p['ref']:>8s} [{p['op']:>3s}]  n={p['n']:5d}  ({p['x']:+.3f}, {p['y']:+.3f}, {p['z']:+.3f})")
            else:
                print(f"    {p['pos']:>15s} × {p['tri']:>15s} × {p['ref']:>8s} [{p['op']:>3s}]  EMPTY")


if __name__ == '__main__':
    main()
