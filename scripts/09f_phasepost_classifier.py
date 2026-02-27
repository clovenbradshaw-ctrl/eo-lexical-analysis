#!/usr/bin/env python3
"""
09f_phasepost_classifier.py — Classify verbs by embedding proximity to phasepost centroids
============================================================================================

Two modes:

1. --build    Compute full-dimensional centroids from existing data, save to JSON
2. --classify Take new words, embed them via OpenAI, find nearest phaseposts

Usage:
  # First: build centroids from your classified+embedded data
  python scripts/09f_phasepost_classifier.py --build

  # Then: classify new words
  python scripts/09f_phasepost_classifier.py --classify "shimmer" "yearn" "coalesce" "gestate"

  # Classify from a file (one word per line)
  python scripts/09f_phasepost_classifier.py --classify-file my_verbs.txt

  # Interactive mode
  python scripts/09f_phasepost_classifier.py --interactive

  # Test accuracy against known classifications
  python scripts/09f_phasepost_classifier.py --test
"""

import json, os, sys, argparse, time
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

OP_NAMES = {
    'NUL':'Nullification','DES':'Designation','INS':'Instantiation',
    'SEG':'Segmentation','CON':'Connection','SYN':'Synthesis',
    'ALT':'Alternation','SUP':'Superposition','REC':'Reconstitution',
}

POS_ORDER = ['Differentiate','Relate','Generate']
TRI_ORDER = ['Existence','Structure','Interpretation']
REF_ORDER = ['FIGURE','PATTERN','GROUND']

CENTROID_FILE = os.path.join(OUT, "phasepost_centroids_full.npz")
META_FILE = os.path.join(OUT, "phasepost_centroids_meta.json")

LANGUAGES = [
    "Ancient_Greek","Arabic","Basque","Classical_Chinese","Coptic",
    "English","Finnish","French","German","Gothic","Hindi",
    "Indonesian","Japanese","Korean","Latin","Naija",
    "Old_Church_Slavonic","Old_East_Slavic","Persian","Russian",
    "Sanskrit","Tagalog","Tamil","Turkish","Uyghur",
    "Vietnamese","Wolof"
]


def load_language(lang):
    """Load embeddings + classifications + referent for one language."""
    lang_dir = os.path.join(DATA, lang)
    if not os.path.exists(lang_dir):
        return None

    # Embeddings
    emb_file = os.path.join(lang_dir, "definition_embeddings.npz")
    if not os.path.exists(emb_file):
        return None
    npz = np.load(emb_file, allow_pickle=True)
    if 'verbs' in npz and 'embeddings' in npz:
        emb_map = {str(v).strip(): npz['embeddings'][i].astype(np.float32)
                    for i, v in enumerate(npz['verbs'])}
    elif 'arr_0' in npz:
        data = npz['arr_0']
        if isinstance(data, np.ndarray) and data.dtype == object:
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
        return None

    # Classifications
    class_file = os.path.join(lang_dir, "classified.json")
    if not os.path.exists(class_file):
        return None
    with open(class_file) as f:
        class_data = json.load(f)
    op_map = {}
    items = class_data.get('classifications', class_data) if isinstance(class_data, dict) else class_data
    if isinstance(items, dict):
        items = items.get('classifications', [])
    for c in items:
        if isinstance(c, dict):
            v = str(c.get('verb', '')).strip()
            op = str(c.get('operator', '')).upper().strip()
            if v and op in HELIX:
                op_map[v] = op

    # Referent
    ref_file = os.path.join(lang_dir, "referent_axis.json")
    if not os.path.exists(ref_file):
        return None
    with open(ref_file) as f:
        ref_data = json.load(f)
    ref_map = {}
    ritems = ref_data if isinstance(ref_data, list) else ref_data.get('classifications', [])
    for r in ritems:
        if isinstance(r, dict):
            v = str(r.get('verb', '')).strip()
            ref = str(r.get('referent', '')).upper().strip()
            if v and ref in REF_ORDER:
                ref_map[v] = ref

    # Merge
    results = []
    for v in emb_map:
        if v in op_map and v in ref_map:
            results.append({
                'verb': v,
                'operator': op_map[v],
                'referent': ref_map[v],
                'embedding': emb_map[v],
            })
    return results if results else None


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_centroids():
    """Compute full-dimensional centroids for all 27 phaseposts."""
    print("Building phasepost centroids from all languages...\n")

    # Collect embeddings by cell
    cells = defaultdict(list)
    cell_verbs = defaultdict(list)
    total = 0

    for lang in LANGUAGES:
        print(f"  {lang}...", end=" ", flush=True)
        verbs = load_language(lang)
        if verbs is None:
            print("skip")
            continue
        print(f"{len(verbs)} verbs")
        total += len(verbs)

        for v in verbs:
            op = v['operator']
            pos = POSITIONS[op]
            tri = TRIADS[op]
            ref = v['referent']
            key = f"{pos}|{tri}|{ref}"
            cells[key].append(v['embedding'])
            if lang == 'English':
                cell_verbs[key].append(v['verb'])

    print(f"\n  Total: {total} verbs across {len(cells)} populated cells")

    # Compute centroids
    dim = None
    centroid_keys = []
    centroid_vecs = []
    meta = {}

    for pos in POS_ORDER:
        for tri in TRI_ORDER:
            for ref in REF_ORDER:
                key = f"{pos}|{tri}|{ref}"
                op = [o for o in HELIX if POSITIONS[o]==pos and TRIADS[o]==tri][0]
                embs = cells.get(key, [])
                n = len(embs)

                if n > 0:
                    arr = np.array(embs)
                    centroid = arr.mean(axis=0)
                    # Also compute std for confidence estimation
                    std = arr.std(axis=0).mean()
                    if dim is None:
                        dim = centroid.shape[0]
                else:
                    centroid = np.zeros(dim or 3072, dtype=np.float32)
                    std = 0.0

                centroid_keys.append(key)
                centroid_vecs.append(centroid)
                meta[key] = {
                    'pos': pos, 'tri': tri, 'ref': ref, 'op': op,
                    'op_name': OP_NAMES[op],
                    'n': n,
                    'mean_std': float(std),
                    'sample_verbs': cell_verbs.get(key, [])[:20],
                }

    # Save
    centroids_arr = np.array(centroid_vecs, dtype=np.float32)
    np.savez_compressed(CENTROID_FILE, centroids=centroids_arr, keys=np.array(centroid_keys))

    with open(META_FILE, 'w') as f:
        json.dump({'dim': dim, 'total_verbs': total, 'cells': meta}, f, indent=2)

    print(f"\n  Saved {CENTROID_FILE}")
    print(f"  Saved {META_FILE}")
    print(f"  {len([k for k in meta if meta[k]['n']>0])}/27 cells populated")
    print(f"  Embedding dimension: {dim}")

    # Print summary
    print(f"\n  CENTROIDS:")
    for key in centroid_keys:
        m = meta[key]
        if m['n'] > 0:
            print(f"    {m['pos']:>15s} x {m['tri']:>15s} x {m['ref']:>8s} [{m['op']:>3s}]  n={m['n']:5d}  std={m['mean_std']:.4f}")
        else:
            print(f"    {m['pos']:>15s} x {m['tri']:>15s} x {m['ref']:>8s} [{m['op']:>3s}]  EMPTY")


def embed_words(words):
    """Embed words using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: pip install openai")
        sys.exit(1)

    client = OpenAI()
    # Build definition-style prompts matching training format
    prompts = []
    for w in words:
        prompts.append(f"The English verb '{w}': to {w}")

    # Batch embed
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=prompts,
    )
    embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
    return embeddings


def classify_embedding(embedding, centroids, keys, meta, top_k=5):
    """Classify a single embedding by proximity to centroids."""
    sims = []
    for i, key in enumerate(keys):
        m = meta['cells'][key]
        if m['n'] == 0:
            continue
        sim = cosine_sim(embedding, centroids[i])
        sims.append((sim, key, m))

    sims.sort(key=lambda x: -x[0])
    return sims[:top_k]


def load_centroids():
    """Load precomputed centroids."""
    if not os.path.exists(CENTROID_FILE) or not os.path.exists(META_FILE):
        print("Error: centroids not built yet. Run with --build first.")
        sys.exit(1)

    npz = np.load(CENTROID_FILE)
    centroids = npz['centroids']
    keys = list(npz['keys'])

    with open(META_FILE) as f:
        meta = json.load(f)

    return centroids, keys, meta


def classify_words(words, show_top=5):
    """Classify a list of words."""
    centroids, keys, meta = load_centroids()

    print(f"Embedding {len(words)} words...", flush=True)
    embeddings = embed_words(words)

    print()
    for word, emb in zip(words, embeddings):
        results = classify_embedding(emb, centroids, keys, meta, top_k=show_top)
        top = results[0]
        m = top[2]
        print(f"  {word}")
        print(f"    → {m['op']} ({m['op_name']}) x {m['ref']}")
        print(f"      {m['pos']} x {m['tri']} x {m['ref']}")
        print(f"      cosine={top[0]:.4f}")
        if m['sample_verbs']:
            print(f"      neighbors: {', '.join(m['sample_verbs'][:8])}")

        # Show runner-ups
        for sim, key, rm in results[1:]:
            gap = top[0] - sim
            print(f"      #{results.index((sim,key,rm))+1}: {rm['op']} x {rm['ref']} (cos={sim:.4f}, gap={gap:.4f})")
        print()


def test_accuracy():
    """Test geometric classifier against LLM classifications across all languages."""
    centroids, keys, meta = load_centroids()

    print("Testing geometric classifier against LLM classifications...\n")

    # Collect all verbs from all languages
    all_verbs = []
    lang_stats = {}

    for lang in LANGUAGES:
        verbs = load_language(lang)
        if not verbs:
            continue
        all_verbs.extend([(v, lang) for v in verbs])
        lang_stats[lang] = len(verbs)
        print(f"  Loaded {lang}: {len(verbs)} verbs")

    print(f"\n  Total: {len(all_verbs)} verbs across {len(lang_stats)} languages\n")

    # Classify each verb by centroid proximity
    correct_27 = 0
    correct_op = 0
    correct_pos = 0
    correct_tri = 0
    correct_ref = 0
    total = 0

    # Per-operator tracking
    op_correct = defaultdict(int)
    op_total = defaultdict(int)

    # Per-language tracking
    lang_correct_op = defaultdict(int)
    lang_total = defaultdict(int)

    # Per-referent tracking
    ref_correct = defaultdict(int)
    ref_total = defaultdict(int)

    for v, lang in all_verbs:
        emb = v['embedding']
        true_op = v['operator']
        true_ref = v['referent']
        true_pos = POSITIONS[true_op]
        true_tri = TRIADS[true_op]
        true_key = f"{true_pos}|{true_tri}|{true_ref}"

        results = classify_embedding(emb, centroids, keys, meta, top_k=1)
        if not results:
            continue

        pred_key = results[0][1]
        pred_meta = results[0][2]
        pred_op = pred_meta['op']
        pred_pos = pred_meta['pos']
        pred_tri = pred_meta['tri']
        pred_ref = pred_meta['ref']

        total += 1
        if pred_key == true_key: correct_27 += 1
        if pred_op == true_op: correct_op += 1
        if pred_pos == true_pos: correct_pos += 1
        if pred_tri == true_tri: correct_tri += 1
        if pred_ref == true_ref: correct_ref += 1

        op_total[true_op] += 1
        if pred_op == true_op: op_correct[true_op] += 1

        lang_total[lang] += 1
        if pred_op == true_op: lang_correct_op[lang] += 1

        ref_total[true_ref] += 1
        if pred_ref == true_ref: ref_correct[true_ref] += 1

    print(f"  === GLOBAL ACCURACY ({total} verbs, {len(lang_stats)} languages) ===\n")
    print(f"  27-cell exact match:  {correct_27}/{total} = {100*correct_27/total:.1f}%")
    print(f"  Operator (9-class):   {correct_op}/{total} = {100*correct_op/total:.1f}%")
    print(f"  Position (3-class):   {correct_pos}/{total} = {100*correct_pos/total:.1f}%")
    print(f"  Triad (3-class):      {correct_tri}/{total} = {100*correct_tri/total:.1f}%")
    print(f"  Referent (3-class):   {correct_ref}/{total} = {100*correct_ref/total:.1f}%")

    print(f"\n  Chance baselines:")
    print(f"    27-cell: {100/27:.1f}%  |  Operator: {100/9:.1f}%  |  3-class: {100/3:.1f}%")

    print(f"\n  === PER-OPERATOR ACCURACY ===\n")
    for op in HELIX:
        t = op_total.get(op, 0)
        c = op_correct.get(op, 0)
        if t > 0:
            print(f"    {op} ({OP_NAMES[op]:>15s}): {c:5d}/{t:5d} = {100*c/t:.1f}%")

    print(f"\n  === PER-REFERENT ACCURACY ===\n")
    for ref in REF_ORDER:
        t = ref_total.get(ref, 0)
        c = ref_correct.get(ref, 0)
        if t > 0:
            print(f"    {ref:>8s}: {c:5d}/{t:5d} = {100*c/t:.1f}%")

    print(f"\n  === PER-LANGUAGE ACCURACY (operator) ===\n")
    for lang in sorted(lang_stats.keys()):
        t = lang_total.get(lang, 0)
        c = lang_correct_op.get(lang, 0)
        if t > 0:
            print(f"    {lang:>25s}: {c:5d}/{t:5d} = {100*c/t:.1f}%")


def interactive_mode():
    """Interactive classification loop."""
    centroids, keys, meta = load_centroids()
    print("Phasepost Geometric Classifier")
    print(f"Centroids loaded: {meta['dim']}-dimensional, {meta['total_verbs']} training verbs")
    print("Type a verb (or 'quit' to exit):\n")

    while True:
        try:
            word = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not word or word.lower() in ('quit', 'exit', 'q'):
            break

        try:
            embeddings = embed_words([word])
            results = classify_embedding(embeddings[0], centroids, keys, meta, top_k=5)
            top = results[0]
            m = top[2]
            print(f"    → {m['op']} ({m['op_name']}) × {m['ref']}  [cos={top[0]:.4f}]")
            print(f"      {m['pos']} × {m['tri']} × {m['ref']}")
            if m['sample_verbs']:
                print(f"      neighbors: {', '.join(m['sample_verbs'][:6])}")
            for sim, key, rm in results[1:3]:
                print(f"      also: {rm['op']} × {rm['ref']} (cos={sim:.4f})")
            print()
        except Exception as e:
            print(f"    Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Phasepost geometric classifier")
    parser.add_argument('--build', action='store_true', help='Build centroids from training data')
    parser.add_argument('--classify', nargs='+', help='Classify words')
    parser.add_argument('--classify-file', type=str, help='Classify words from file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--test', action='store_true', help='Test accuracy against LLM classifications')
    args = parser.parse_args()

    if args.build:
        build_centroids()
    elif args.classify:
        classify_words(args.classify)
    elif args.classify_file:
        with open(args.classify_file) as f:
            words = [line.strip() for line in f if line.strip()]
        classify_words(words)
    elif args.interactive:
        interactive_mode()
    elif args.test:
        test_accuracy()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
