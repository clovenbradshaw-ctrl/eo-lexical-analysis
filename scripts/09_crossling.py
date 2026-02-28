#!/usr/bin/env python3
"""
Step 9: Cross-Linguistic EO Operator Analysis
==============================================

Uses Universal Dependencies treebanks to extract verb inventories from
typologically diverse languages across global north/south, modernity/antiquity,
then classifies them with the EO operator framework and compares distributions.

NO PRIORS: The classification prompt contains only operator definitions
derived from centroid analysis. No frequency expectations, no assumptions
about which operators will be rare or common.

Pipeline:
  Phase 1: Download UD treebanks + extract verb lemmas
  Phase 2: Classify verbs via Anthropic API (batched)
  Phase 3: Analyze and compare distributions

Usage:
  python scripts/09_crossling.py --phase 1      # download + extract
  python scripts/09_crossling.py --phase 2      # classify (needs API key)
  python scripts/09_crossling.py --phase 3      # analyze
  python scripts/09_crossling.py --phase all     # everything

Requirements:
  pip install conllu anthropic tqdm numpy
  export ANTHROPIC_API_KEY=sk-...
"""

import os, json, sys, time, argparse, re, tarfile, zipfile, glob
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling")
OUT  = os.path.join(BASE, "output")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  LANGUAGE SELECTION
# ══════════════════════════════════════════════════════════════════
# Typologically diverse sample covering:
#   - Language families: Indo-European, Sino-Tibetan, Afro-Asiatic,
#     Altaic/Turkic, Japonic, Niger-Congo, Uralic, Dravidian,
#     Austronesian, Uto-Aztecan, isolates
#   - Time: ancient (>1000 years) and modern
#   - Geography: every inhabited continent
#   - Morphological type: isolating, agglutinative, fusional, polysynthetic
#
# Each entry: (language_name, ud_treebank_name, family, era, region, morph_type)

LANGUAGES = [
    # ── ANCIENT ──────────────────────────────────────────────────
    ("Ancient_Greek",    "UD_Ancient_Greek-Perseus",    "IE-Hellenic",     "ancient", "Mediterranean",  "fusional"),
    ("Latin",            "UD_Latin-Perseus",             "IE-Italic",       "ancient", "Mediterranean",  "fusional"),
    ("Classical_Chinese","UD_Classical_Chinese-Kyoto",   "Sino-Tibetan",    "ancient", "East Asia",      "isolating"),
    ("Sanskrit",         "UD_Sanskrit-Vedic",            "IE-Indo-Aryan",   "ancient", "South Asia",     "fusional"),
    ("Old_Church_Slavonic","UD_Old_Church_Slavonic-PROIEL","IE-Slavic",     "ancient", "Eastern Europe", "fusional"),
    ("Gothic",           "UD_Gothic-PROIEL",             "IE-Germanic",     "ancient", "Northern Europe","fusional"),
    ("Old_French",       "UD_Old_French-SRCMF",          "IE-Romance",     "medieval","Western Europe", "fusional"),
    ("Old_East_Slavic",  "UD_Old_East_Slavic-TOROT",     "IE-Slavic",      "medieval","Eastern Europe", "fusional"),

    # ── MODERN: GLOBAL NORTH ─────────────────────────────────────
    ("English",          "UD_English-EWT",               "IE-Germanic",     "modern",  "Global North",   "fusional"),
    ("German",           "UD_German-GSD",                "IE-Germanic",     "modern",  "Global North",   "fusional"),
    ("Russian",          "UD_Russian-SynTagRus",         "IE-Slavic",       "modern",  "Global North",   "fusional"),
    ("French",           "UD_French-GSD",                "IE-Romance",      "modern",  "Global North",   "fusional"),
    ("Finnish",          "UD_Finnish-TDT",               "Uralic",          "modern",  "Global North",   "agglutinative"),

    # ── MODERN: GLOBAL SOUTH / NON-WESTERN ───────────────────────
    ("Japanese",         "UD_Japanese-GSD",              "Japonic",         "modern",  "East Asia",      "agglutinative"),
    ("Mandarin",         "UD_Chinese-GSDSimp",           "Sino-Tibetan",    "modern",  "East Asia",      "isolating"),
    ("Korean",           "UD_Korean-Kaist",              "Koreanic",        "modern",  "East Asia",      "agglutinative"),
    ("Arabic",           "UD_Arabic-PADT",               "Afro-Asiatic",    "modern",  "Middle East",    "fusional"),
    ("Hindi",            "UD_Hindi-HDTB",                "IE-Indo-Aryan",   "modern",  "South Asia",     "fusional"),
    ("Turkish",          "UD_Turkish-BOUN",              "Turkic",          "modern",  "West Asia",      "agglutinative"),
    ("Indonesian",       "UD_Indonesian-GSD",            "Austronesian",    "modern",  "Southeast Asia", "agglutinative"),
    ("Vietnamese",       "UD_Vietnamese-VTB",            "Austroasiatic",   "modern",  "Southeast Asia", "isolating"),
    ("Tamil",            "UD_Tamil-TTB",                 "Dravidian",       "modern",  "South Asia",     "agglutinative"),
    ("Yoruba",           "UD_Yoruba-YTB",               "Niger-Congo",     "modern",  "West Africa",    "isolating"),
    ("Wolof",            "UD_Wolof-WTB",                "Niger-Congo",     "modern",  "West Africa",    "agglutinative"),
    ("Naija",            "UD_Naija-NSC",                "Creole-English",  "modern",  "West Africa",    "isolating"),
    ("Tagalog",          "UD_Tagalog-TRG",              "Austronesian",    "modern",  "Southeast Asia", "agglutinative"),
    ("Persian",          "UD_Persian-PerDT",            "IE-Iranian",      "modern",  "West Asia",      "fusional"),
    ("Uyghur",           "UD_Uyghur-UDT",               "Turkic",          "modern",  "Central Asia",   "agglutinative"),
    ("Coptic",           "UD_Coptic-Scriptorium",       "Afro-Asiatic",    "ancient", "North Africa",   "agglutinative"),
    ("Basque",           "UD_Basque-BDT",               "Isolate",         "modern",  "Western Europe", "agglutinative"),
]

# ══════════════════════════════════════════════════════════════════
#  OPERATOR DEFINITIONS (from centroid analysis, no priors)
# ══════════════════════════════════════════════════════════════════

OPERATOR_DEFINITIONS = """
You are classifying verb meanings into nine transformation types.
For each verb, determine which single operator best describes the
transformation the verb enacts. Consider what exists BEFORE the verb
acts and what exists AFTER.

THE NINE OPERATORS:

NUL (∅) — MARK ABSENCE
Transform a state by making the absence of something the salient fact.
Something that was present — energy, substance, connection, presence —
is gone, and the goneness itself is the new condition.
Examples across domains: deplete, silence, evacuate, boycott, fade, die out.

DES (⊡) — DRAW DISTINCTION
Register something as different from its ground. Before: undifferentiated.
After: something has become something rather than anything. A distinction
has been drawn where there wasn't one.
Examples: designate, define, distinguish, elect, legalize, certify, must, should.

INS (△) — SOMETHING APPEARS
Create a new event, entity, or state in the world. Before: it was not.
After: it is. Pure appearance — the most basic transformation.
Examples: run, sing, build, cry, whisper, heat, accompany.

SEG (|) — ONE BECOMES MANY
Transform a unity into parts by introducing a boundary. Before: whole.
After: pieces with a division between them.
Examples: split, cut, analyze, isolate, shatter, disentangle, steal.

CON (⋈) — CREATE PERSISTENT LINK
Establish a relationship between separate identities that persists and
changes what both are. Not momentary contact but structural bond.
Examples: marry, chain, correlate, adopt, imprison, promise, affiliate.

SYN (∨) — MANY BECOME ONE NEW THING
Combine separate elements into a unified whole that transcends its parts.
The product is something none of the inputs were. Emergent novelty.
Examples: synthesize, braid, bake, amalgamate, total, compromise.

ALT (∿) — CHANGE STATE
Same entity, different state. The entity persists; its state changes.
Toggle, switch, convert. The thing itself is unchanged; what changes
is which state it occupies.
Examples: toggle, freeze, alternate, convert, americanize, clarify, translate.

SUP (∥) — HOLD INCOMPATIBLE WITHOUT RESOLUTION
Maintain multiple mutually exclusive values simultaneously without
collapsing into any one. Stable coexistence of contradiction.
Examples: vacillate, contradict, straddle, doubt, haunt, equivocate.

REC (⟳) — REBUILD AROUND NEW CENTER
Take an existing structure and reorganize it around a different organizing
principle. The material persists; the architecture transforms.
Examples: restructure, evolve, democratize, metamorphose, modernize, repent.

CLASSIFICATION RULES:
- Choose the SINGLE best operator for the verb's primary meaning
- If uncertain, note your confidence (high/medium/low) and best alternative
- Consider the verb's most common usage, not rare or metaphorical senses
- The transformation type is about WHAT CHANGES, not the domain it operates in
"""

# ══════════════════════════════════════════════════════════════════
#  PHASE 1: DOWNLOAD AND EXTRACT
# ══════════════════════════════════════════════════════════════════

def phase1_download_and_extract():
    """Download UD treebanks and extract verb lemmas."""
    try:
        import subprocess
    except ImportError:
        pass

    print("="*70)
    print("  PHASE 1: Download UD Treebanks & Extract Verbs")
    print("="*70)

    # We'll use the UD release from GitHub
    UD_BASE = "https://raw.githubusercontent.com/UniversalDependencies"

    results = {}

    for lang_name, treebank, family, era, region, morph in LANGUAGES:
        print(f"\n  Processing {lang_name} ({treebank})...")

        lang_dir = os.path.join(DATA, lang_name)
        os.makedirs(lang_dir, exist_ok=True)

        conllu_file = os.path.join(lang_dir, "combined.conllu")
        verbs_file = os.path.join(lang_dir, "verbs.json")

        # Check if already extracted
        if os.path.exists(verbs_file):
            with open(verbs_file) as f:
                vdata = json.load(f)
            print(f"    Already extracted: {vdata['n_unique']} unique verbs")
            results[lang_name] = vdata
            continue

        # Try to find/download the conllu files
        # User may need to download manually - provide instructions
        if not os.path.exists(conllu_file):
            # Look for any .conllu files in the directory
            conllu_files = glob.glob(os.path.join(lang_dir, "*.conllu"))
            if not conllu_files:
                print(f"    ⚠ No .conllu file found for {lang_name}")
                print(f"    Download from: https://github.com/UniversalDependencies/{treebank}")
                print(f"    Place .conllu files in: {lang_dir}/")

                # Try automatic download
                try:
                    import urllib.request
                    # Try the dev branch for latest
                    for split in ['train', 'test', 'dev']:
                        # Construct filename from treebank name
                        tb_short = treebank.replace("UD_", "").lower().replace("-", "_").replace("__", "_")
                        # UD naming convention: xx_name-ud-split.conllu
                        # Get language code
                        parts = treebank.split("-")
                        lang_part = parts[0].replace("UD_", "")

                        # Try common patterns
                        possible_names = []
                        # Convert language name to code
                        lang_codes = {
                            "Ancient_Greek": "grc", "Latin": "la", "Classical_Chinese": "lzh",
                            "Sanskrit": "sa", "Old_Church_Slavonic": "cu", "Gothic": "got",
                            "Old_French": "fro", "Old_East_Slavic": "orv",
                            "English": "en", "German": "de", "Russian": "ru", "French": "fr",
                            "Finnish": "fi", "Japanese": "ja", "Chinese": "zh",
                            "Korean": "ko", "Arabic": "ar", "Hindi": "hi", "Turkish": "tr",
                            "Indonesian": "id", "Vietnamese": "vi", "Tamil": "ta",
                            "Yoruba": "yo", "Wolof": "wo", "Naija": "pcm", "Tagalog": "tl",
                            "Persian": "fa", "Uyghur": "ug", "Coptic": "cop", "Basque": "eu",
                        }
                        code = lang_codes.get(lang_name, "xx")
                        tb_name = treebank.split("-")[-1].lower() if "-" in treebank else ""

                        fname = f"{code}_{tb_name}-ud-{split}.conllu"
                        url = f"{UD_BASE}/{treebank}/master/{fname}"

                        dest = os.path.join(lang_dir, fname)
                        try:
                            urllib.request.urlretrieve(url, dest)
                            print(f"    ✓ Downloaded {fname}")
                        except Exception:
                            # Try alternate name patterns
                            pass

                    conllu_files = glob.glob(os.path.join(lang_dir, "*.conllu"))
                except Exception as e:
                    print(f"    Auto-download failed: {e}")
                    continue

            if conllu_files:
                # Combine all conllu files
                with open(conllu_file, 'w', encoding='utf-8') as out:
                    for cf in sorted(conllu_files):
                        if cf != conllu_file:
                            with open(cf, encoding='utf-8') as inp:
                                out.write(inp.read())
                                out.write('\n')

        if not os.path.exists(conllu_file):
            print(f"    ✗ Skipping {lang_name} — no data")
            continue

        # Extract verbs using conllu parser
        verbs = extract_verbs_from_conllu(conllu_file, lang_name)

        vdata = {
            'language': lang_name,
            'family': family,
            'era': era,
            'region': region,
            'morph_type': morph,
            'treebank': treebank,
            'n_unique': len(verbs),
            'verbs': verbs,  # {lemma: {'count': N, 'forms': [...]}}
        }

        with open(verbs_file, 'w', encoding='utf-8') as f:
            json.dump(vdata, f, ensure_ascii=False, indent=2)

        print(f"    ✓ Extracted {len(verbs)} unique verb lemmas")
        results[lang_name] = vdata

    # Summary
    print(f"\n{'='*70}")
    print(f"  PHASE 1 SUMMARY")
    print(f"{'='*70}")
    for lang_name, vdata in sorted(results.items()):
        print(f"  {lang_name:25s} {vdata['n_unique']:6d} verbs  [{vdata['family']}, {vdata['era']}, {vdata['region']}]")

    return results


def extract_verbs_from_conllu(filepath, lang_name):
    """Extract unique verb lemmas and their frequencies from a CoNLL-U file."""
    verbs = defaultdict(lambda: {'count': 0, 'forms': set()})

    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            fields = line.split('\t')
            if len(fields) < 10:
                continue

            # CoNLL-U format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            try:
                token_id = fields[0]
                # Skip multi-word tokens and empty nodes
                if '-' in token_id or '.' in token_id:
                    continue

                form = fields[1]
                lemma = fields[2]
                upos = fields[3]

                if upos == 'VERB':
                    # Normalize lemma
                    lemma_clean = lemma.lower().strip()
                    if lemma_clean and lemma_clean != '_':
                        verbs[lemma_clean]['count'] += 1
                        verbs[lemma_clean]['forms'].add(form.lower())

            except (IndexError, ValueError):
                continue

    # Convert sets to lists for JSON serialization
    result = {}
    for lemma, data in verbs.items():
        result[lemma] = {
            'count': data['count'],
            'forms': sorted(list(data['forms']))[:10],  # keep top 10 forms
        }

    return result


# ══════════════════════════════════════════════════════════════════
#  PHASE 2: CLASSIFY
# ══════════════════════════════════════════════════════════════════

def phase2_classify():
    """Classify extracted verbs using Anthropic API."""
    print("="*70)
    print("  PHASE 2: Classify Verbs with EO Operators")
    print("="*70)

    try:
        import anthropic
    except ImportError:
        print("  ✗ pip install anthropic")
        return

    client = anthropic.Anthropic()

    # Load all extracted verb files
    for lang_dir in sorted(Path(DATA).iterdir()):
        if not lang_dir.is_dir():
            continue

        verbs_file = lang_dir / "verbs.json"
        classified_file = lang_dir / "classified.json"

        if not verbs_file.exists():
            continue

        if classified_file.exists():
            with open(classified_file) as f:
                existing = json.load(f)
            print(f"\n  {lang_dir.name}: already classified ({len(existing.get('classifications',[]))} verbs)")
            continue

        with open(verbs_file) as f:
            vdata = json.load(f)

        lang_name = vdata['language']
        verbs = vdata['verbs']

        print(f"\n  Classifying {lang_name}: {len(verbs)} verbs")

        # Sort by frequency, take top N
        # For languages with very many verbs, sample intelligently
        sorted_verbs = sorted(verbs.items(), key=lambda x: -x[1]['count'])

        # Take all verbs with count >= 2, or top 3000, whichever is smaller
        candidates = [(lemma, data) for lemma, data in sorted_verbs
                      if data['count'] >= 2]
        if len(candidates) > 3000:
            candidates = candidates[:3000]
        if len(candidates) < 100:
            candidates = sorted_verbs[:min(500, len(sorted_verbs))]

        print(f"    Classifying {len(candidates)} verbs (freq >= 2 or top 500)")

        # Batch classification
        classifications = []
        batch_size = 50

        for batch_start in range(0, len(candidates), batch_size):
            batch = candidates[batch_start:batch_start + batch_size]

            verb_list = []
            for lemma, data in batch:
                forms_str = ", ".join(data['forms'][:5])
                freq = data['count']
                verb_list.append(f"  {lemma} (forms: {forms_str}; freq: {freq})")

            prompt = f"""Classify these {lang_name} verbs into EO operators.

For each verb, provide:
- The verb lemma
- The operator (NUL, DES, INS, SEG, CON, SYN, ALT, SUP, REC)
- Confidence (high, medium, low)
- Brief English gloss if the verb is not English
- Alternative operator if confidence is not high

VERBS:
{chr(10).join(verb_list)}

Respond in JSON array format:
[{{"verb": "...", "operator": "...", "confidence": "...", "gloss": "...", "alternative": ""}}]

Classify based on the verb's PRIMARY, MOST COMMON meaning.
Do not explain — just the JSON array."""

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=OPERATOR_DEFINITIONS,
                    messages=[{"role": "user", "content": prompt}]
                )

                text = response.content[0].text.strip()

                # Extract JSON from response
                # Handle markdown code blocks
                if '```' in text:
                    text = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
                    if text:
                        text = text.group(1).strip()
                    else:
                        text = response.content[0].text.strip()

                batch_results = json.loads(text)
                classifications.extend(batch_results)

                n_done = min(batch_start + batch_size, len(candidates))
                print(f"    {n_done}/{len(candidates)} classified", end='\r')

            except Exception as e:
                print(f"\n    ✗ Batch {batch_start}: {e}")
                time.sleep(5)
                continue

            # Rate limiting
            time.sleep(1)

        # Save
        result = {
            'language': lang_name,
            'family': vdata['family'],
            'era': vdata['era'],
            'region': vdata['region'],
            'morph_type': vdata['morph_type'],
            'n_classified': len(classifications),
            'n_total_verbs': len(verbs),
            'classifications': classifications,
        }

        with open(classified_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n    ✓ Classified {len(classifications)} verbs for {lang_name}")


# ══════════════════════════════════════════════════════════════════
#  PHASE 3: ANALYZE
# ══════════════════════════════════════════════════════════════════

def phase3_analyze():
    """Compare EO distributions across languages."""
    print("="*70)
    print("  PHASE 3: Cross-Linguistic Analysis")
    print("="*70)

    HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']
    TRIADS = {
        'Existence': ['NUL','DES','INS'],
        'Structure': ['SEG','CON','SYN'],
        'Interpretation': ['ALT','SUP','REC'],
    }

    all_langs = {}

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

        # Count operators
        op_counts = Counter()
        for c in cls:
            op = c.get('operator', '').upper().strip()
            if op in HELIX:
                op_counts[op] += 1

        total = sum(op_counts.values())
        if total == 0:
            continue

        op_pcts = {op: op_counts[op] / total * 100 for op in HELIX}

        # Triad percentages
        triad_pcts = {}
        for tname, tops in TRIADS.items():
            triad_pcts[tname] = sum(op_pcts.get(op, 0) for op in tops)

        # Confidence distribution
        conf_counts = Counter(c.get('confidence', 'unknown') for c in cls)

        all_langs[lang] = {
            'family': data['family'],
            'era': data['era'],
            'region': data['region'],
            'morph_type': data['morph_type'],
            'n_classified': total,
            'op_counts': dict(op_counts),
            'op_pcts': op_pcts,
            'triad_pcts': triad_pcts,
            'confidence': dict(conf_counts),
        }

    if not all_langs:
        print("  No classified languages found. Run phases 1 and 2 first.")
        return

    # ── REPORT ─────────────────────────────────────────────────────
    print(f"\n  Languages analyzed: {len(all_langs)}")

    # Distribution table
    print(f"\n  {'Language':25s} {'n':>5s}", end="")
    for op in HELIX:
        print(f" {op:>6s}", end="")
    print(f"  {'Exist':>6s} {'Struc':>6s} {'Inter':>6s}")
    print("  " + "-"*110)

    for lang in sorted(all_langs.keys(), key=lambda x: (
        {'ancient':0,'medieval':1,'modern':2}.get(all_langs[x]['era'], 3),
        all_langs[x]['region'],
        x
    )):
        d = all_langs[lang]
        era_marker = {'ancient': '†', 'medieval': '‡', 'modern': ' '}.get(d['era'], '?')
        print(f"  {era_marker}{lang:24s} {d['n_classified']:5d}", end="")
        for op in HELIX:
            pct = d['op_pcts'].get(op, 0)
            print(f" {pct:5.1f}%", end="")
        for tname in ['Existence', 'Structure', 'Interpretation']:
            print(f" {d['triad_pcts'][tname]:5.1f}%", end="")
        print(f"  [{d['family']}]")

    # ── SUP ANALYSIS ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUP (∥) ACROSS LANGUAGES")
    print(f"{'='*70}")

    sup_data = [(lang, d['op_pcts'].get('SUP', 0), d['op_counts'].get('SUP', 0), d['n_classified'])
                for lang, d in all_langs.items()]
    sup_data.sort(key=lambda x: -x[1])

    for lang, pct, count, total in sup_data:
        d = all_langs[lang]
        bar = '█' * int(pct * 5)
        print(f"  {lang:25s} {pct:5.1f}% ({count:4d}/{total:5d}) {bar:30s} [{d['family']}, {d['era']}]")

    # ── FAMILY COMPARISONS ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  OPERATOR DISTRIBUTIONS BY LANGUAGE FAMILY")
    print(f"{'='*70}")

    families = defaultdict(list)
    for lang, d in all_langs.items():
        families[d['family']].append(d)

    for family in sorted(families.keys()):
        members = families[family]
        if len(members) < 1:
            continue
        print(f"\n  {family} ({len(members)} languages):")
        avg_pcts = {op: sum(m['op_pcts'].get(op, 0) for m in members) / len(members) for op in HELIX}
        print(f"    ", end="")
        for op in HELIX:
            print(f"{op}:{avg_pcts[op]:4.1f}%  ", end="")
        print()

    # ── ERA COMPARISONS ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  OPERATOR DISTRIBUTIONS BY ERA")
    print(f"{'='*70}")

    eras = defaultdict(list)
    for lang, d in all_langs.items():
        eras[d['era']].append(d)

    for era in ['ancient', 'medieval', 'modern']:
        members = eras.get(era, [])
        if not members:
            continue
        print(f"\n  {era.upper()} ({len(members)} languages):")
        avg_pcts = {op: sum(m['op_pcts'].get(op, 0) for m in members) / len(members) for op in HELIX}
        print(f"    ", end="")
        for op in HELIX:
            print(f"{op}:{avg_pcts[op]:4.1f}%  ", end="")
        print()

        # Triad averages
        for tname, tops in TRIADS.items():
            avg_triad = sum(avg_pcts[op] for op in tops)
            print(f"    {tname}: {avg_triad:.1f}%")

    # ── MORPHOLOGICAL TYPE COMPARISONS ────────────────────────────
    print(f"\n{'='*70}")
    print("  OPERATOR DISTRIBUTIONS BY MORPHOLOGICAL TYPE")
    print(f"{'='*70}")

    morph_types = defaultdict(list)
    for lang, d in all_langs.items():
        morph_types[d['morph_type']].append(d)

    for mtype in ['isolating', 'agglutinative', 'fusional']:
        members = morph_types.get(mtype, [])
        if not members:
            continue
        print(f"\n  {mtype.upper()} ({len(members)} languages):")
        avg_pcts = {op: sum(m['op_pcts'].get(op, 0) for m in members) / len(members) for op in HELIX}
        print(f"    ", end="")
        for op in HELIX:
            print(f"{op}:{avg_pcts[op]:4.1f}%  ", end="")
        print()

    # ── REGION COMPARISONS ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  OPERATOR DISTRIBUTIONS BY REGION")
    print(f"{'='*70}")

    regions = defaultdict(list)
    for lang, d in all_langs.items():
        regions[d['region']].append(d)

    for region in sorted(regions.keys()):
        members = regions[region]
        print(f"\n  {region} ({len(members)} languages):")
        avg_pcts = {op: sum(m['op_pcts'].get(op, 0) for m in members) / len(members) for op in HELIX}
        # Just show interesting variations from overall mean
        overall_mean = {op: sum(d['op_pcts'].get(op, 0) for d in all_langs.values()) / len(all_langs)
                       for op in HELIX}
        deviations = {op: avg_pcts[op] - overall_mean[op] for op in HELIX}
        notable = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        notable_str = ", ".join(f"{op}:{deviations[op]:+.1f}%" for op, _ in notable)
        print(f"    Notable deviations from mean: {notable_str}")

    # ── SAVE FULL REPORT ──────────────────────────────────────────
    report = {
        'n_languages': len(all_langs),
        'languages': {lang: {
            'family': d['family'],
            'era': d['era'],
            'region': d['region'],
            'morph_type': d['morph_type'],
            'n_classified': d['n_classified'],
            'op_pcts': d['op_pcts'],
            'triad_pcts': d['triad_pcts'],
        } for lang, d in all_langs.items()},
    }

    report_path = os.path.join(OUT, "crossling_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Saved {report_path}")

    # ── VISUALIZATION ─────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        viz_dir = os.path.join(OUT, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Fig 1: Heatmap of operator distributions across languages
        langs_sorted = sorted(all_langs.keys(), key=lambda x: (
            {'ancient':0,'medieval':1,'modern':2}.get(all_langs[x]['era'], 3), x
        ))

        matrix = np.array([[all_langs[l]['op_pcts'].get(op, 0) for op in HELIX]
                          for l in langs_sorted])

        fig, ax = plt.subplots(figsize=(14, max(8, len(langs_sorted) * 0.4)))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(9))
        ax.set_xticklabels(HELIX, fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(langs_sorted)))

        ylabels = []
        for l in langs_sorted:
            d = all_langs[l]
            era_marker = {'ancient': '†', 'medieval': '‡', 'modern': ''}.get(d['era'], '')
            ylabels.append(f"{era_marker}{l} [{d['family']}]")
        ax.set_yticklabels(ylabels, fontsize=9)

        # Add values
        for i in range(len(langs_sorted)):
            for j in range(9):
                val = matrix[i, j]
                color = 'white' if val > 30 else 'black'
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=7, color=color)

        plt.colorbar(im, label='% of verbs')
        ax.set_title("EO Operator Distribution Across Languages", fontsize=14)

        # Draw triad separators
        for x in [2.5, 5.5]:
            ax.axvline(x, color='black', linewidth=2)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "crossling_heatmap.png"), dpi=150)
        plt.close()
        print(f"  ✓ crossling_heatmap.png")

        # Fig 2: SUP percentage across languages
        fig, ax = plt.subplots(figsize=(12, max(6, len(sup_data) * 0.3)))

        langs_sup = [s[0] for s in sup_data]
        pcts_sup = [s[1] for s in sup_data]
        colors = []
        for s in sup_data:
            d = all_langs[s[0]]
            if d['era'] == 'ancient':
                colors.append('#e74c3c')
            elif d['era'] == 'medieval':
                colors.append('#e67e22')
            else:
                colors.append('#3498db')

        bars = ax.barh(range(len(langs_sup)), pcts_sup, color=colors)
        ax.set_yticks(range(len(langs_sup)))
        ax.set_yticklabels([f"{l} [{all_langs[l]['family']}]" for l in langs_sup], fontsize=9)
        ax.set_xlabel("SUP (∥) as % of all verbs")
        ax.set_title("Superposition Operator Across Languages\n(red=ancient, orange=medieval, blue=modern)")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "crossling_sup.png"), dpi=150)
        plt.close()
        print(f"  ✓ crossling_sup.png")

        # Fig 3: Triad proportions across languages
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(langs_sorted) * 0.3)))
        for ax_i, tname in enumerate(['Existence', 'Structure', 'Interpretation']):
            ax = axes[ax_i]
            vals = [all_langs[l]['triad_pcts'][tname] for l in langs_sorted]
            colors = ['#e74c3c' if all_langs[l]['era'] == 'ancient'
                     else '#e67e22' if all_langs[l]['era'] == 'medieval'
                     else '#3498db' for l in langs_sorted]
            ax.barh(range(len(langs_sorted)), vals, color=colors)
            ax.set_yticks(range(len(langs_sorted)))
            if ax_i == 0:
                ax.set_yticklabels([l for l in langs_sorted], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_title(tname)
            ax.set_xlabel("%")
            ax.invert_yaxis()

        plt.suptitle("Triad Proportions Across Languages", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "crossling_triads.png"), dpi=150)
        plt.close()
        print(f"  ✓ crossling_triads.png")

    except ImportError:
        print("  (matplotlib not available, skipping visualizations)")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-linguistic EO analysis")
    parser.add_argument('--phase', default='all', choices=['1','2','3','all'],
                       help="Which phase to run")
    args = parser.parse_args()

    if args.phase in ('1', 'all'):
        phase1_download_and_extract()

    if args.phase in ('2', 'all'):
        phase2_classify()

    if args.phase in ('3', 'all'):
        phase3_analyze()


if __name__ == '__main__':
    main()
