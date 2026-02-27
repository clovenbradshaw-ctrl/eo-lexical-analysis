#!/usr/bin/env python3
"""
09f_crossling_referent.py — Classify all verbs across 27 languages on the referent axis
========================================================================================

For each language:
  1. Load classified verbs (from 09b)
  2. Classify each as FIGURE / PATTERN / GROUND referent via Claude
  3. Output JSON organized by 27-cell position (Position × Triad × Referent)

Usage:
  python scripts/09f_crossling_referent.py              # classify all
  python scripts/09f_crossling_referent.py --lang Korean # one language
  python scripts/09f_crossling_referent.py --dump        # dump JSON (skip classification)
"""

import json, os, sys, time, re, argparse
from collections import defaultdict, Counter
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling")
OUT  = os.path.join(BASE, "output")

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']

TRIADS = {
    'NUL': 'Existence', 'DES': 'Existence', 'INS': 'Existence',
    'SEG': 'Structure', 'CON': 'Structure', 'SYN': 'Structure',
    'ALT': 'Interpretation', 'SUP': 'Interpretation', 'REC': 'Interpretation',
}
POSITIONS = {
    'NUL': 'Differentiate', 'SEG': 'Differentiate', 'ALT': 'Differentiate',
    'DES': 'Relate', 'CON': 'Relate', 'SUP': 'Relate',
    'INS': 'Generate', 'SYN': 'Generate', 'REC': 'Generate',
}

TRIAD_LABELS = ['Existence', 'Structure', 'Interpretation']
POS_LABELS = ['Differentiate', 'Relate', 'Generate']
REF_LABELS = ['FIGURE', 'PATTERN', 'GROUND']

LANGUAGES = [
    "Ancient_Greek", "Arabic", "Basque", "Classical_Chinese", "Coptic",
    "English", "Finnish", "French", "German", "Gothic", "Hindi",
    "Indonesian", "Japanese", "Korean", "Latin", "Naija",
    "Old_Church_Slavonic", "Old_East_Slavic", "Persian", "Russian",
    "Sanskrit", "Tagalog", "Tamil", "Turkish", "Uyghur",
    "Vietnamese", "Wolof"
]


def classify_language(lang, client):
    """Classify one language's verbs on the referent axis."""
    lang_dir = os.path.join(DATA, lang)
    classified_file = os.path.join(lang_dir, "classified.json")
    output_file = os.path.join(lang_dir, "referent_axis.json")

    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
        print(f"  {lang}: already done ({len(existing)} verbs)")
        return

    if not os.path.exists(classified_file):
        print(f"  {lang}: ✗ no classified.json")
        return

    with open(classified_file) as f:
        data = json.load(f)

    verbs = []
    for c in data['classifications']:
        op = c.get('operator', '').upper().strip()
        if op not in HELIX:
            continue
        verb = c.get('verb', '').strip()
        gloss = c.get('gloss', '').strip()
        if verb:
            verbs.append({'verb': verb, 'operator': op, 'gloss': gloss})

    if not verbs:
        print(f"  {lang}: ✗ no valid verbs")
        return

    print(f"  {lang}: classifying {len(verbs)} verbs...")

    system_prompt = """You are a cognitive linguist classifying verbs along a single dimension:
what kind of REFERENT does the verb point to?

Three categories:

FIGURE: The verb is about particulars, bounded entities, discrete things, specific objects
or events. The referent is something you could point to, count, or isolate.
Examples: "kick" (specific action on specific thing), "build" (creates discrete object),
"arrive" (bounded event), "name" (picks out particular), "kill" (targets specific entity)

PATTERN: The verb is about relationships, regularities, structures, or connections between
things. The referent is not a thing but a relation, rule, similarity, or recurring form.
Examples: "resemble" (relationship), "correlate" (regularity), "govern" (structural relation),
"alternate" (pattern of change), "rank" (ordering relation)

GROUND: The verb is about contexts, substrates, conditions, backgrounds, states, or the
medium in which things happen. The referent is not a thing or relation but a condition,
atmosphere, or enabling context.
Examples: "exist" (condition), "rain" (environmental state), "persist" (ongoing condition),
"underlie" (substrate), "permeate" (diffuse presence), "afford" (enabling condition)

You will receive verbs that may be from any language. Each verb will have an English gloss
to help you understand its meaning. Classify based on the MEANING (via the gloss), not the
surface form.

Classify each verb. When uncertain, ask: would you describe the verb's referent as
"a thing/event" (FIGURE), "a relationship/regularity" (PATTERN), or "a condition/context" (GROUND)?

Respond ONLY as a JSON array: [{"verb": "...", "referent": "FIGURE|PATTERN|GROUND"}]
No explanations."""

    all_results = []
    batch_size = 80

    for batch_start in range(0, len(verbs), batch_size):
        batch = verbs[batch_start:batch_start + batch_size]

        verb_list = "\n".join(
            f"  {v['verb']}" + (f" ({v['gloss']})" if v['gloss'] else "")
            for v in batch
        )

        prompt = f"""Classify these {lang.replace('_', ' ')} verbs as FIGURE, PATTERN, or GROUND based on their meaning:

{verb_list}

Return JSON array: [{{"verb": "...", "referent": "FIGURE|PATTERN|GROUND"}}]"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )

                text = response.content[0].text.strip()
                if '```' in text:
                    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
                    if match:
                        text = match.group(1).strip()

                batch_results = json.loads(text)

                # Merge with operator info
                verb_op_map = {v['verb']: v['operator'] for v in batch}
                verb_gloss_map = {v['verb']: v['gloss'] for v in batch}
                for r in batch_results:
                    v = r.get('verb', '')
                    r['operator'] = verb_op_map.get(v, '')
                    r['gloss'] = verb_gloss_map.get(v, '')
                    r['referent'] = r.get('referent', '').upper().strip()

                all_results.extend(batch_results)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    retry {attempt+1}: {e}")
                    time.sleep(5)
                else:
                    print(f"    ✗ batch {batch_start} failed: {e}")

        n_done = min(batch_start + batch_size, len(verbs))
        print(f"    {n_done}/{len(verbs)}", end="\r", flush=True)
        time.sleep(0.5)

    # Save
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    ref_counts = Counter(r.get('referent', '') for r in all_results)
    total = sum(ref_counts.values())
    parts = []
    for ref in ['FIGURE', 'PATTERN', 'GROUND']:
        n = ref_counts.get(ref, 0)
        parts.append(f"{ref}={n}({100*n/total:.0f}%)")
    print(f"    ✓ {len(all_results)} verbs: {', '.join(parts)}")


def dump_json():
    """Load all languages and output combined 27-cell JSON."""

    all_data = {}

    for lang in LANGUAGES:
        lang_dir = os.path.join(DATA, lang)
        ref_file = os.path.join(lang_dir, "referent_axis.json")

        if not os.path.exists(ref_file):
            print(f"  {lang}: ✗ no referent_axis.json, skipping")
            continue

        with open(ref_file) as f:
            referents = json.load(f)

        # Build cells
        cells = defaultdict(list)
        valid = 0
        for r in referents:
            verb = r.get('verb', '').strip()
            ref = r.get('referent', '').upper().strip()
            op = r.get('operator', '').upper().strip()
            gloss = r.get('gloss', '').strip()

            if not verb or ref not in REF_LABELS or op not in HELIX:
                continue

            pos = POSITIONS[op]
            tri = TRIADS[op]
            cell_key = f"{pos}|{tri}|{ref}"

            cells[cell_key].append({
                'verb': verb,
                'operator': op,
                'gloss': gloss,
            })
            valid += 1

        # Build structured output for this language
        lang_output = {
            'language': lang,
            'total_verbs': valid,
            'distribution': {},
            'cells': {}
        }

        # Distribution summary
        ref_counts = Counter()
        for r in referents:
            ref = r.get('referent', '').upper().strip()
            if ref in REF_LABELS:
                ref_counts[ref] += 1
        lang_output['distribution']['referent'] = dict(ref_counts)

        op_counts = Counter()
        for r in referents:
            op = r.get('operator', '').upper().strip()
            if op in HELIX:
                op_counts[op] += 1
        lang_output['distribution']['operator'] = dict(op_counts)

        # 27 cells
        for pos in POS_LABELS:
            for tri in TRIAD_LABELS:
                for ref in REF_LABELS:
                    key = f"{pos}|{tri}|{ref}"
                    cell_verbs = cells.get(key, [])
                    cell_name = f"{pos} × {tri} × {ref}"
                    lang_output['cells'][cell_name] = {
                        'count': len(cell_verbs),
                        'verbs': cell_verbs
                    }

        all_data[lang] = lang_output
        print(f"  {lang}: {valid} verbs across {sum(1 for k, v in cells.items() if v)} cells")

    # Cross-linguistic summary
    summary = {
        'languages': len(all_data),
        'total_verbs': sum(d['total_verbs'] for d in all_data.values()),
        'cell_populations': {},
        'empty_cells_by_language': {},
    }

    # Which cells are empty across languages?
    for pos in POS_LABELS:
        for tri in TRIAD_LABELS:
            for ref in REF_LABELS:
                cell_name = f"{pos} × {tri} × {ref}"
                counts = {}
                for lang, d in all_data.items():
                    c = d['cells'].get(cell_name, {}).get('count', 0)
                    counts[lang] = c
                summary['cell_populations'][cell_name] = {
                    'total': sum(counts.values()),
                    'mean': sum(counts.values()) / len(counts) if counts else 0,
                    'empty_in': [l for l, c in counts.items() if c == 0],
                    'n_empty': sum(1 for c in counts.values() if c == 0),
                }

    # Empty cells per language
    for lang, d in all_data.items():
        empty = [name for name, cell in d['cells'].items() if cell['count'] == 0]
        summary['empty_cells_by_language'][lang] = empty

    output = {
        'summary': summary,
        'languages': all_data,
    }

    out_path = os.path.join(OUT, "crossling_27_cells.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved {out_path}")
    print(f"    {summary['languages']} languages, {summary['total_verbs']} total verbs")

    # Print universally empty/sparse cells
    print(f"\n  UNIVERSALLY SPARSE CELLS (empty in >50% of languages):")
    for cell_name, info in sorted(summary['cell_populations'].items(),
                                   key=lambda x: -x[1]['n_empty']):
        if info['n_empty'] > len(all_data) // 2:
            print(f"    {cell_name:>45s}: empty in {info['n_empty']}/{len(all_data)} languages, total={info['total']}")

    print(f"\n  MOST POPULATED CELLS:")
    for cell_name, info in sorted(summary['cell_populations'].items(),
                                   key=lambda x: -x[1]['total'])[:10]:
        print(f"    {cell_name:>45s}: {info['total']:5d} verbs across languages")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, help='Classify single language')
    parser.add_argument('--dump', action='store_true', help='Dump JSON only (skip classification)')
    args = parser.parse_args()

    if args.dump:
        print(f"{'='*70}")
        print(f"  DUMPING 27-CELL JSON")
        print(f"{'='*70}")
        dump_json()
        return

    # Classification phase
    try:
        import anthropic
    except ImportError:
        print("pip install anthropic")
        return

    client = anthropic.Anthropic()

    langs = [args.lang] if args.lang else LANGUAGES

    print(f"{'='*70}")
    print(f"  CLASSIFYING VERBS ON REFERENT AXIS")
    print(f"{'='*70}")

    for lang in langs:
        if lang not in LANGUAGES and not args.lang:
            continue
        classify_language(lang, client)

    # Auto-dump after classification
    print(f"\n{'='*70}")
    print(f"  DUMPING 27-CELL JSON")
    print(f"{'='*70}")
    dump_json()


if __name__ == '__main__':
    main()
