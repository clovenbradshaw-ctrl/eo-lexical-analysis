#!/usr/bin/env python3
"""
Step 10b: Wide K-Sweep Within Each Operator
============================================
Sweeps k=2..60 (or n//4, whichever is smaller) for each operator,
prints ALL values, saves everything.
"""

import json, os, numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

print("Loading data...")
with open(os.path.join(DATA, "llm_classifications.json")) as f:
    cls = json.load(f)["classifications"]
embs = np.load(os.path.join(DATA, "reembed_combined.npz"))["embeddings"]

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']
op_indices = {op: [] for op in HELIX}
for i, c in enumerate(cls):
    if c['operator'] in HELIX:
        op_indices[c['operator']].append(i)

all_results = {}

for op in HELIX:
    idx = op_indices[op]
    n = len(idx)
    
    if n < 30:
        print(f"\n{op}: {n} verbs, skipping wide sweep")
        all_results[op] = {'n_verbs': n, 'sweeps': [], 'best_k': None, 'best_sil': None}
        continue

    op_embs = embs[idx]
    max_k = min(60, n // 4)
    
    print(f"\n{'='*60}")
    print(f"  {op} ({n} verbs), sweep k=2..{max_k}")
    print(f"{'='*60}")
    
    sweeps = []
    best_k, best_sil = 2, -1
    
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(op_embs)
        
        counts = Counter(labels)
        min_cluster = min(counts.values())
        if min_cluster < 2:
            sweeps.append({'k': k, 'sil': None, 'min_cluster': min_cluster, 'skipped': True})
            continue
        
        sil = silhouette_score(op_embs, labels, metric='cosine', sample_size=min(5000, n))
        
        marker = ''
        if sil > best_sil:
            best_sil = sil
            best_k = k
            marker = ' ★'
        
        bar = '█' * int(sil * 100)
        print(f"  k={k:3d}: {sil:.4f} {bar}{marker}")
        
        sweeps.append({
            'k': k, 
            'sil': float(sil), 
            'min_cluster': int(min_cluster),
            'max_cluster': int(max(counts.values())),
            'skipped': False,
        })
    
    print(f"  BEST: k={best_k} (sil={best_sil:.4f})")
    
    all_results[op] = {
        'n_verbs': n,
        'max_k_tested': max_k,
        'best_k': best_k,
        'best_sil': float(best_sil),
        'sweeps': sweeps,
        'still_climbing': sweeps[-1]['sil'] is not None and best_k >= max_k - 3,
    }

# Save
report_path = os.path.join(OUT, "recursive_ksweep.json")
with open(report_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for op in HELIX:
    r = all_results[op]
    if r['best_k'] is None:
        print(f"  {op:3s}: too few verbs")
        continue
    climbing = " ⚠ STILL CLIMBING" if r.get('still_climbing') else ""
    print(f"  {op:3s} ({r['n_verbs']:5d} verbs): best k={r['best_k']:3d}  sil={r['best_sil']:.4f}{climbing}")

print(f"\n  Saved: {report_path}")
