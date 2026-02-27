#!/usr/bin/env python3
"""
Step 09d: Cross-Linguistic Definition Embedding & Z-Score Analysis
====================================================================

For each language's classified verbs:
  1. Use Claude to generate brief dictionary definitions in the NATIVE language
  2. Embed those definitions using OpenAI text-embedding-3-large (multilingual)
  3. Compute z-scores per language

This replicates the English methodology: we embedded English dictionary
definitions, now we embed native-language dictionary definitions.

Requires:
  pip install anthropic openai numpy scikit-learn
  export ANTHROPIC_API_KEY=sk-...
  export OPENAI_API_KEY=sk-...

Usage:
  python scripts/09d_embed.py                  # all phases
  python scripts/09d_embed.py --define         # generate definitions only
  python scripts/09d_embed.py --embed          # embed definitions only
  python scripts/09d_embed.py --analyze        # analyze only
"""

import json, os, sys, time, argparse, re
import numpy as np
from collections import Counter
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling")
OUT  = os.path.join(BASE, "output")
VIZ  = os.path.join(OUT, "visualizations")
os.makedirs(VIZ, exist_ok=True)

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']

# Language display names and writing guidance
LANG_INFO = {
    'Ancient_Greek':      {'native': 'Ancient Greek', 'script': 'Greek', 'note': 'Use classical Greek, not modern'},
    'Latin':              {'native': 'Latin', 'script': 'Latin', 'note': ''},
    'Classical_Chinese':  {'native': '古漢語', 'script': 'Chinese characters', 'note': 'Use classical literary Chinese, not modern Mandarin'},
    'Sanskrit':           {'native': 'संस्कृतम्', 'script': 'Devanagari', 'note': 'Use classical Sanskrit'},
    'Old_Church_Slavonic':{'native': 'Old Church Slavonic', 'script': 'Cyrillic', 'note': 'Use Old Church Slavonic vocabulary'},
    'Gothic':             {'native': 'Gothic', 'script': 'Latin transliteration', 'note': 'Gothic is extinct; provide definition in reconstructed Gothic or Latin transliteration'},
    'Coptic':             {'native': 'Coptic', 'script': 'Coptic/Greek script', 'note': 'Use Sahidic Coptic where possible'},
    'Old_French':         {'native': 'ancien français', 'script': 'Latin', 'note': 'Use Old French vocabulary and forms'},
    'Old_East_Slavic':    {'native': 'Old East Slavic', 'script': 'Cyrillic', 'note': 'Use Old East Slavic vocabulary'},
    'English':            {'native': 'English', 'script': 'Latin', 'note': ''},
    'German':             {'native': 'Deutsch', 'script': 'Latin', 'note': ''},
    'Russian':            {'native': 'русский', 'script': 'Cyrillic', 'note': ''},
    'French':             {'native': 'français', 'script': 'Latin', 'note': ''},
    'Finnish':            {'native': 'suomi', 'script': 'Latin', 'note': ''},
    'Japanese':           {'native': '日本語', 'script': 'Japanese', 'note': 'Use Japanese, mixing kanji/hiragana as natural'},
    'Mandarin':           {'native': '中文', 'script': 'Simplified Chinese', 'note': 'Use modern Mandarin Chinese'},
    'Korean':             {'native': '한국어', 'script': 'Hangul', 'note': ''},
    'Arabic':             {'native': 'العربية', 'script': 'Arabic', 'note': 'Use Modern Standard Arabic'},
    'Hindi':              {'native': 'हिन्दी', 'script': 'Devanagari', 'note': ''},
    'Turkish':            {'native': 'Türkçe', 'script': 'Latin', 'note': ''},
    'Indonesian':         {'native': 'Bahasa Indonesia', 'script': 'Latin', 'note': ''},
    'Vietnamese':         {'native': 'tiếng Việt', 'script': 'Latin (Vietnamese)', 'note': ''},
    'Tamil':              {'native': 'தமிழ்', 'script': 'Tamil', 'note': ''},
    'Yoruba':             {'native': 'Yorùbá', 'script': 'Latin (Yoruba)', 'note': ''},
    'Wolof':              {'native': 'Wolof', 'script': 'Latin', 'note': ''},
    'Naija':              {'native': 'Naija (Nigerian Pidgin)', 'script': 'Latin', 'note': 'Use Nigerian Pidgin English'},
    'Tagalog':            {'native': 'Tagalog', 'script': 'Latin', 'note': ''},
    'Persian':            {'native': 'فارسی', 'script': 'Persian/Arabic', 'note': ''},
    'Uyghur':             {'native': 'ئۇيغۇرچە', 'script': 'Arabic (Uyghur)', 'note': ''},
    'Basque':             {'native': 'euskara', 'script': 'Latin', 'note': ''},
}


# ══════════════════════════════════════════════════════════════
#  PHASE 1: GENERATE NATIVE-LANGUAGE DEFINITIONS
# ══════════════════════════════════════════════════════════════

def phase1_define():
    """Generate dictionary definitions in each language's native tongue."""
    try:
        import anthropic
    except ImportError:
        print("  ✗ pip install anthropic")
        return
    
    client = anthropic.Anthropic()
    
    for lang_dir in sorted(Path(DATA).iterdir()):
        if not lang_dir.is_dir():
            continue
        
        classified_file = lang_dir / "classified.json"
        definitions_file = lang_dir / "native_definitions.json"
        
        if not classified_file.exists():
            continue
        
        if definitions_file.exists():
            with open(definitions_file) as f:
                existing = json.load(f)
            print(f"  {lang_dir.name}: already defined ({len(existing)} verbs)")
            continue
        
        with open(classified_file) as f:
            data = json.load(f)
        
        lang = data['language']
        cls = data['classifications']
        
        # Get language info
        info = LANG_INFO.get(lang, {'native': lang, 'script': 'Latin', 'note': ''})
        
        # Collect valid verbs
        verbs_to_define = []
        for c in cls:
            op = c.get('operator', '').upper().strip()
            if op not in HELIX:
                continue
            verb = c.get('verb', '').strip()
            if not verb:
                continue
            gloss = c.get('gloss', '').strip()
            verbs_to_define.append({
                'verb': verb,
                'operator': op,
                'gloss': gloss,
            })
        
        if not verbs_to_define:
            continue
        
        print(f"  {lang}: defining {len(verbs_to_define)} verbs in {info['native']}...")
        
        system_prompt = f"""You are a lexicographer writing brief dictionary definitions.

For each verb provided, write a SHORT dictionary-style definition (5-15 words)
in {info['native']} using {info['script']} script.

RULES:
- Write the definition ENTIRELY in {info['native']}. Do NOT use English.
- Keep definitions brief: one short phrase or sentence, like a dictionary entry.
- Focus on the verb's primary, most common meaning.
- Do not include the verb itself in the definition.
{f'- Note: {info["note"]}' if info['note'] else ''}

Respond ONLY as a JSON array:
[{{"verb": "...", "definition": "..."}}]

No explanations. Just the JSON."""
        
        all_definitions = []
        batch_size = 50
        
        for batch_start in range(0, len(verbs_to_define), batch_size):
            batch = verbs_to_define[batch_start:batch_start + batch_size]
            
            verb_list = []
            for v in batch:
                gloss_hint = f" (English: {v['gloss']})" if v['gloss'] and lang != 'English' else ""
                verb_list.append(f"  {v['verb']}{gloss_hint}")
            
            prompt = f"""Define these {lang} verbs in {info['native']}:

{chr(10).join(verb_list)}

Return JSON array: [{{"verb": "...", "definition": "..."}}]
Definitions must be in {info['native']} only."""
            
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                text = response.content[0].text.strip()
                
                # Extract JSON
                if '```' in text:
                    match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
                    if match:
                        text = match.group(1).strip()
                
                batch_defs = json.loads(text)
                
                # Merge with operator info
                verb_op_map = {v['verb']: v['operator'] for v in batch}
                for d in batch_defs:
                    d['operator'] = verb_op_map.get(d.get('verb', ''), '')
                
                all_definitions.extend(batch_defs)
                
                n_done = min(batch_start + batch_size, len(verbs_to_define))
                print(f"    {n_done}/{len(verbs_to_define)}", end="\r", flush=True)
                
            except Exception as e:
                print(f"\n    ✗ Batch {batch_start}: {e}")
                time.sleep(5)
                continue
            
            time.sleep(1)  # rate limit
        
        # Save
        with open(definitions_file, 'w', encoding='utf-8') as f:
            json.dump(all_definitions, f, ensure_ascii=False, indent=2)
        
        print(f"\n    ✓ {len(all_definitions)} definitions for {lang}")


# ══════════════════════════════════════════════════════════════
#  PHASE 2: EMBED NATIVE DEFINITIONS
# ══════════════════════════════════════════════════════════════

def phase2_embed():
    """Embed native-language definitions via OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  ✗ pip install openai")
        return
    
    client = OpenAI()
    
    print(f"\n{'='*70}")
    print("  PHASE 2: Embed Native Definitions")
    print(f"{'='*70}")
    
    for lang_dir in sorted(Path(DATA).iterdir()):
        if not lang_dir.is_dir():
            continue
        
        definitions_file = lang_dir / "native_definitions.json"
        embeddings_file = lang_dir / "definition_embeddings.npz"
        
        if not definitions_file.exists():
            continue
        
        if embeddings_file.exists():
            emb = np.load(embeddings_file)
            print(f"  {lang_dir.name}: already embedded ({emb['embeddings'].shape[0]} definitions)")
            continue
        
        with open(definitions_file, encoding='utf-8') as f:
            defs = json.load(f)
        
        # Filter: need verb, definition, and valid operator
        valid = []
        for d in defs:
            op = d.get('operator', '').upper().strip()
            defn = d.get('definition', '').strip()
            verb = d.get('verb', '').strip()
            if op in HELIX and defn and verb:
                valid.append(d)
        
        if not valid:
            print(f"  {lang_dir.name}: no valid definitions")
            continue
        
        texts = [d['definition'] for d in valid]
        operators = [d['operator'] for d in valid]
        verbs = [d['verb'] for d in valid]
        
        print(f"  {lang_dir.name}: embedding {len(texts)} definitions...", end="", flush=True)
        
        # Batch embed
        all_embeddings = []
        batch_size = 2048
        
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            
            try:
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=batch
                )
                batch_embs = [r.embedding for r in response.data]
                all_embeddings.extend(batch_embs)
            except Exception as e:
                print(f"\n    ✗ Batch {batch_start}: {e}")
                time.sleep(5)
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-large",
                        input=batch
                    )
                    batch_embs = [r.embedding for r in response.data]
                    all_embeddings.extend(batch_embs)
                except Exception as e2:
                    print(f"    ✗ Retry failed: {e2}")
                    continue
            
            time.sleep(0.5)
        
        if len(all_embeddings) != len(texts):
            print(f" ✗ got {len(all_embeddings)}/{len(texts)}")
            continue
        
        emb_array = np.array(all_embeddings, dtype=np.float32)
        
        np.savez_compressed(
            embeddings_file,
            embeddings=emb_array,
            verbs=np.array(verbs, dtype=object),
            operators=np.array(operators, dtype=object),
            definitions=np.array(texts, dtype=object),
        )
        
        print(f" ✓ {emb_array.shape}")


# ══════════════════════════════════════════════════════════════
#  PHASE 3: COMPUTE Z-SCORES
# ══════════════════════════════════════════════════════════════

def compute_z_score(embeddings, labels, n_perm=500):
    """Z-score of inter-centroid separation vs random permutations."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    unique = sorted(set(labels))
    n_groups = len(unique)
    if n_groups < 2:
        return None
    
    label_map = {u: i for i, u in enumerate(unique)}
    mapped = np.array([label_map[l] for l in labels])
    
    # Centroids
    centroids = []
    for i in range(n_groups):
        mask = mapped == i
        if mask.sum() == 0:
            centroids.append(np.zeros(embeddings.shape[1]))
        else:
            centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.array(centroids)
    
    # Real mean pairwise similarity
    sim = cosine_similarity(centroids)
    pairs = [sim[i,j] for i in range(n_groups) for j in range(i+1, n_groups)]
    real_mean = np.mean(pairs)
    
    # Permutation baseline
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


def phase3_analyze():
    """Compute z-scores for each language."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    print(f"\n{'='*70}")
    print("  PHASE 3: Z-SCORES PER LANGUAGE")
    print(f"{'='*70}")
    
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
        
        # Filter valid
        valid_mask = [op in HELIX for op in operators]
        embeddings = embeddings[valid_mask]
        operators = [op for op, v in zip(operators, valid_mask) if v]
        
        if len(operators) < 30:
            print(f"  {lang}: only {len(operators)} verbs, skipping")
            continue
        
        op_counts = Counter(operators)
        present_ops = [op for op in HELIX if op_counts.get(op, 0) >= 2]
        
        print(f"  {lang}: {len(operators)} definitions, {len(present_ops)} operators...", end="", flush=True)
        
        # Z-score: all operators
        z_all = compute_z_score(embeddings, operators, n_perm=500)
        
        # Z-score: excluding INS
        non_ins_mask = [op != 'INS' for op in operators]
        if sum(non_ins_mask) >= 30:
            z_no_ins = compute_z_score(
                embeddings[non_ins_mask],
                [op for op, m in zip(operators, non_ins_mask) if m],
                n_perm=500
            )
        else:
            z_no_ins = None
        
        # Z-score: triads
        triad_labels = []
        for op in operators:
            if op in ['NUL','DES','INS']:
                triad_labels.append('Existence')
            elif op in ['SEG','CON','SYN']:
                triad_labels.append('Structure')
            else:
                triad_labels.append('Interpretation')
        z_triad = compute_z_score(embeddings, triad_labels, n_perm=500)
        
        # Z-score: random baseline (scrambled operators, same distribution)
        rng = np.random.RandomState(99)
        random_z_scores = []
        for _ in range(20):
            shuffled = list(operators)
            rng.shuffle(shuffled)
            rz = compute_z_score(embeddings, shuffled, n_perm=100)
            if rz:
                random_z_scores.append(rz['z_score'])
        
        results[lang] = {
            'family': meta['family'],
            'era': meta['era'],
            'region': meta.get('region', ''),
            'morph_type': meta.get('morph_type', ''),
            'n_verbs': len(operators),
            'n_operators': len(present_ops),
            'z_all': z_all,
            'z_no_ins': z_no_ins,
            'z_triad': z_triad,
            'random_z_mean': float(np.mean(random_z_scores)) if random_z_scores else 0,
            'random_z_std': float(np.std(random_z_scores)) if random_z_scores else 0,
        }
        
        z_ni_str = f"{z_no_ins['z_score']:+.1f}" if z_no_ins else "n/a"
        rz_str = f"{np.mean(random_z_scores):+.1f}" if random_z_scores else "n/a"
        print(f" z={z_all['z_score']:+.1f} (no-INS: {z_ni_str}, triad: {z_triad['z_score']:+.1f}, random: {rz_str})")
    
    return results


# ══════════════════════════════════════════════════════════════
#  PHASE 4: COMPARISON & REPORTING
# ══════════════════════════════════════════════════════════════

def phase4_report(results):
    """Cross-linguistic comparison of z-scores."""
    
    print(f"\n{'='*70}")
    print("  CROSS-LINGUISTIC Z-SCORE COMPARISON")
    print(f"{'='*70}")
    
    sorted_langs = sorted(results.keys(), key=lambda x: results[x]['z_all']['z_score'])
    
    print(f"\n  {'Language':25s} {'n':>5s} {'ops':>4s} {'z(all)':>8s} {'z(no INS)':>10s} {'z(triad)':>9s} {'z(rand)':>8s} {'Family'}")
    print("  " + "-"*95)
    
    for lang in sorted_langs:
        r = results[lang]
        z_all = r['z_all']['z_score']
        z_ni = r['z_no_ins']['z_score'] if r['z_no_ins'] else float('nan')
        z_tri = r['z_triad']['z_score']
        z_rnd = r['random_z_mean']
        print(f"  {lang:25s} {r['n_verbs']:5d} {r['n_operators']:4d} {z_all:+8.1f} {z_ni:+10.1f} {z_tri:+9.1f} {z_rnd:+8.1f}  [{r['family']}]")
    
    # Summary
    all_z = [results[l]['z_all']['z_score'] for l in results]
    no_ins_z = [results[l]['z_no_ins']['z_score'] for l in results if results[l]['z_no_ins']]
    triad_z = [results[l]['z_triad']['z_score'] for l in results]
    rand_z = [results[l]['random_z_mean'] for l in results]
    
    print(f"\n  {'Metric':>25s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'All <-2?':>9s}")
    print("  " + "-"*65)
    
    for label, zlist in [
        ('z(all operators)', all_z),
        ('z(excluding INS)', no_ins_z),
        ('z(triads)', triad_z),
        ('z(random labels)', rand_z),
    ]:
        if not zlist:
            continue
        all_sig = all(z < -1.96 for z in zlist)
        print(f"  {label:>25s} {np.mean(zlist):+8.1f} {min(zlist):+8.1f} {max(zlist):+8.1f} {'YES' if all_sig else 'NO':>9s}")
    
    # Bias checks
    n_verbs = [results[l]['n_verbs'] for l in results]
    z_scores = [results[l]['z_all']['z_score'] for l in results]
    corr_nz = np.corrcoef(n_verbs, z_scores)[0,1]
    print(f"\n  BIAS: corr(n_verbs, z_all) = {corr_nz:.3f}")
    
    # By era
    print(f"\n  BY ERA:")
    for era in ['ancient', 'medieval', 'modern']:
        era_z = [results[l]['z_all']['z_score'] for l in results if results[l]['era'] == era]
        if era_z:
            print(f"    {era:10s}: mean z = {np.mean(era_z):+.1f} (n={len(era_z)})")
    
    # Key finding
    print(f"\n  KEY FINDINGS:")
    n_sig = sum(1 for z in all_z if z < -3)
    print(f"  {n_sig}/{len(all_z)} languages: z < -3 (strong separation)")
    if no_ins_z:
        n_sig_ni = sum(1 for z in no_ins_z if z < -3)
        print(f"  {n_sig_ni}/{len(no_ins_z)} languages: z < -3 even excluding INS")
    n_sig_tri = sum(1 for z in triad_z if z < -3)
    print(f"  {n_sig_tri}/{len(triad_z)} languages: z(triad) < -3")
    
    # Visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(results) * 0.35)))
        
        for ax_i, (title, z_key) in enumerate([
            ("All Operators", 'z_all'),
            ("Excluding INS", 'z_no_ins'),
            ("Triads", 'z_triad'),
        ]):
            ax = axes[ax_i]
            plot_data = []
            for lang in sorted_langs:
                r = results[lang]
                zr = r[z_key]
                if zr is None:
                    continue
                z = zr['z_score'] if isinstance(zr, dict) else zr
                color = '#e74c3c' if r['era'] == 'ancient' else '#e67e22' if r['era'] == 'medieval' else '#3498db'
                plot_data.append((lang, z, color))
            
            ax.barh(range(len(plot_data)), [d[1] for d in plot_data], 
                   color=[d[2] for d in plot_data])
            ax.set_yticks(range(len(plot_data)))
            if ax_i == 0:
                ax.set_yticklabels([f"{d[0]} [{results[d[0]]['family']}]" for d in plot_data], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("Z-score")
            ax.set_title(title)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.axvline(-1.96, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
            ax.invert_yaxis()
        
        plt.suptitle("EO Operator Geometric Separation: Native-Language Definition Embeddings\n(red=ancient, orange=medieval, blue=modern)", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ, "crossling_zscores.png"), dpi=150)
        plt.close()
        print(f"\n  ✓ crossling_zscores.png")
    except ImportError:
        pass
    
    # Save
    report = {}
    for lang, r in results.items():
        report[lang] = {
            'family': r['family'],
            'era': r['era'],
            'n_verbs': r['n_verbs'],
            'z_all': r['z_all']['z_score'],
            'z_no_ins': r['z_no_ins']['z_score'] if r['z_no_ins'] else None,
            'z_triad': r['z_triad']['z_score'],
            'random_z': r['random_z_mean'],
        }
    with open(os.path.join(OUT, "crossling_zscores.json"), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Saved crossling_zscores.json")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--define', action='store_true', help='Generate definitions only')
    parser.add_argument('--embed', action='store_true', help='Embed definitions only')
    parser.add_argument('--analyze', action='store_true', help='Analyze only')
    args = parser.parse_args()
    
    run_all = not (args.define or args.embed or args.analyze)
    
    if args.define or run_all:
        print(f"{'='*70}")
        print("  PHASE 1: Generate Native-Language Definitions")
        print(f"{'='*70}")
        phase1_define()
    
    if args.embed or run_all:
        phase2_embed()
    
    if args.analyze or run_all:
        results = phase3_analyze()
        if results:
            phase4_report(results)


if __name__ == '__main__':
    main()
