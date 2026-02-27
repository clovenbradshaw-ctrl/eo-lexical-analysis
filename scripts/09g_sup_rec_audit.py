#!/usr/bin/env python3
"""
09g_sup_rec_audit.py — Is SUP/REC poverty real or a classifier artifact?
=========================================================================

Two tests:

TEST 1: SECOND-CHOICE ANALYSIS
  For every verb, ask the classifier for its top 3 operator choices with
  confidence scores. If SUP/REC frequently appear as runner-up (score > 0.2),
  the poverty is partly artifact — the classifier is uncertain in that region.
  If SUP/REC almost never appear even as second choice, the poverty is real.

TEST 2: TARGETED RE-CLASSIFICATION
  Take all ALT verbs and specifically ask: "Is this really ALT (reframing),
  or is it SUP (holding contradiction)?" Do the same for INS vs REC boundary.
  Use a more targeted prompt that explains the distinction carefully.

Uses English verbs. Requires ANTHROPIC_API_KEY.

Usage:
  python scripts/09g_sup_rec_audit.py                  # run both tests
  python scripts/09g_sup_rec_audit.py --test1           # second-choice only
  python scripts/09g_sup_rec_audit.py --test2           # re-classify only
  python scripts/09g_sup_rec_audit.py --analyze         # analyze saved results
"""

import json, os, sys, time, re, argparse
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling", "English")
OUT  = os.path.join(BASE, "output")

HELIX = ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']


# ══════════════════════════════════════════════════════════════
#  TEST 1: SECOND-CHOICE ANALYSIS
# ══════════════════════════════════════════════════════════════

def test1_second_choices(client):
    """Ask classifier for top-3 choices with confidence for every verb."""

    classified_file = os.path.join(DATA, "classified.json")
    output_file = os.path.join(DATA, "operator_confidence.json")

    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
        print(f"  Already done: {len(existing)} verbs")
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

    print(f"  Getting confidence scores for {len(verbs)} verbs...")

    system_prompt = """You are a cognitive linguist classifying verbs into Experiential Ontology operators.

The nine operators:
  NUL: Nullification — removing, negating, making absent, voiding
  DES: Designation — naming, typing, categorizing, assigning identity
  INS: Instantiation — bringing into presence, creating, manifesting, performing
  SEG: Segmentation — cutting, dividing, separating, breaking apart
  CON: Connection — linking, joining, relating, binding together
  SYN: Synthesis — merging, combining, fusing into unified whole
  ALT: Alternation — reframing, changing perspective, shifting interpretation
  SUP: Superposition — holding contradictory states simultaneously, maintaining productive tension between incompatible truths
  REC: Reconstitution — restructuring foundations, rebuilding from ground up, fundamental reorganization (not just restoration)

For each verb, provide your TOP 3 operator choices with confidence scores (0.0-1.0, must sum to ≤ 1.0).

Key distinctions to attend to:
  ALT vs SUP: ALT changes FROM one frame TO another. SUP holds BOTH frames at once without resolving.
  INS vs REC: INS brings something new into being. REC restructures something existing at its foundations.
  NUL vs SUP: NUL removes or negates. SUP holds absence and presence simultaneously.

Respond ONLY as JSON array:
[{"verb": "...", "choices": [{"op": "...", "score": 0.X}, {"op": "...", "score": 0.X}, {"op": "...", "score": 0.X}]}]"""

    all_results = []
    batch_size = 60

    for batch_start in range(0, len(verbs), batch_size):
        batch = verbs[batch_start:batch_start + batch_size]

        verb_list = "\n".join(
            f"  {v['verb']}" + (f" ({v['gloss']})" if v['gloss'] else "")
            for v in batch
        )

        prompt = f"""Give top-3 operator classifications with confidence scores for each verb:

{verb_list}

JSON array: [{{"verb": "...", "choices": [{{"op": "...", "score": 0.X}}, ...]}}]"""

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

                # Attach original operator
                verb_op_map = {v['verb']: v['operator'] for v in batch}
                for r in batch_results:
                    r['original_op'] = verb_op_map.get(r.get('verb', ''), '')

                all_results.extend(batch_results)
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    retry {attempt+1}: {e}")
                    time.sleep(5)
                else:
                    print(f"    ✗ batch {batch_start}: {e}")

        n_done = min(batch_start + batch_size, len(verbs))
        print(f"    {n_done}/{len(verbs)}", end="\r", flush=True)
        time.sleep(0.5)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n    ✓ Saved {len(all_results)} confidence records")


# ══════════════════════════════════════════════════════════════
#  TEST 2: TARGETED RE-CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def test2_reclassify(client):
    """Re-examine ALT verbs for SUP, and INS verbs for REC."""

    classified_file = os.path.join(DATA, "classified.json")
    output_file = os.path.join(DATA, "sup_rec_reclass.json")

    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
        print(f"  Already done: {len(existing)} verbs")
        return

    with open(classified_file) as f:
        data = json.load(f)

    # Collect ALT verbs (potential SUP) and INS verbs (potential REC)
    alt_verbs = []
    ins_verbs = []
    nul_verbs = []
    for c in data['classifications']:
        op = c.get('operator', '').upper().strip()
        verb = c.get('verb', '').strip()
        gloss = c.get('gloss', '').strip()
        if not verb or op not in HELIX:
            continue
        if op == 'ALT':
            alt_verbs.append({'verb': verb, 'gloss': gloss, 'original': 'ALT'})
        elif op == 'INS':
            ins_verbs.append({'verb': verb, 'gloss': gloss, 'original': 'INS'})
        elif op == 'NUL':
            nul_verbs.append({'verb': verb, 'gloss': gloss, 'original': 'NUL'})

    print(f"  Re-examining: {len(alt_verbs)} ALT, {len(ins_verbs)} INS, {len(nul_verbs)} NUL")

    # Test ALT vs SUP
    alt_sup_prompt = """You are performing a targeted audit of verb classifications.

These verbs were all classified as ALT (alternation — reframing, changing perspective).
Your job: determine if any are actually SUP (superposition — holding contradictory states simultaneously).

The key distinction:
  ALT: The verb describes CHANGING from one frame/state to another. There is movement between perspectives. At any moment, one frame is active.
  SUP: The verb describes HOLDING two or more incompatible states at the SAME TIME. Both are simultaneously true. There is no resolution, no movement — just sustained tension.

Examples:
  "convert" = ALT (changes from one state to another)
  "straddle" = SUP (occupies two positions at once)
  "reframe" = ALT (shifts perspective)
  "juggle" = SUP (maintains multiple things simultaneously)
  "oscillate" = could be either — if it implies being in one state then another, ALT; if it implies a standing wave of both, SUP

For each verb, respond:
  KEEP = correctly ALT
  CHANGE = should be SUP
  UNCERTAIN = genuinely ambiguous

JSON array: [{"verb": "...", "verdict": "KEEP|CHANGE|UNCERTAIN", "reasoning": "brief"}]"""

    # Test INS vs REC
    ins_rec_prompt = """You are performing a targeted audit of verb classifications.

These verbs were all classified as INS (instantiation — bringing into presence, creating, manifesting).
Your job: determine if any are actually REC (reconstitution — fundamentally restructuring, rebuilding from foundations).

The key distinction:
  INS: The verb describes bringing something NEW into phenomenal presence. It wasn't there, now it is. Creation, appearance, manifestation, performance.
  REC: The verb describes taking something that EXISTS and restructuring it at a fundamental level. Not just changing it (that's ALT) but rebuilding its foundations, reorganizing its deep structure.

Examples:
  "create" = INS (brings new thing into being)
  "rebuild" = REC (restructures existing thing from ground up)
  "perform" = INS (brings action into presence)
  "develop" = REC (fundamentally transforms existing structure)
  "grow" = could be either — if emphasis is on appearing/increasing, INS; if on fundamental structural transformation, REC

For each verb, respond:
  KEEP = correctly INS
  CHANGE = should be REC
  UNCERTAIN = genuinely ambiguous

JSON array: [{"verb": "...", "verdict": "KEEP|CHANGE|UNCERTAIN", "reasoning": "brief"}]"""

    # Test NUL vs SUP
    nul_sup_prompt = """You are performing a targeted audit of verb classifications.

These verbs were all classified as NUL (nullification — removing, negating, voiding).
Your job: determine if any are actually SUP (superposition — holding contradictory states simultaneously).

The key distinction:
  NUL: The verb describes REMOVING something, making it absent, negating it. The thing is gone.
  SUP: The verb describes a state where something is both present AND absent, or both true AND false. The contradiction is maintained, not resolved.

Examples:
  "delete" = NUL (removes completely)
  "suspend" = could be SUP (thing exists in a state of being both active and inactive)
  "hide" = could be SUP (thing exists but is not present to perception)
  "deny" = NUL (negates) or SUP (the denied thing persists alongside the denial)

For each verb, respond:
  KEEP = correctly NUL
  CHANGE = should be SUP
  UNCERTAIN = genuinely ambiguous

JSON array: [{"verb": "...", "verdict": "KEEP|CHANGE|UNCERTAIN", "reasoning": "brief"}]"""

    all_results = []

    for verbs, prompt_system, boundary in [
        (alt_verbs, alt_sup_prompt, "ALT→SUP"),
        (ins_verbs, ins_rec_prompt, "INS→REC"),
        (nul_verbs, nul_sup_prompt, "NUL→SUP"),
    ]:
        print(f"\n  Testing {boundary} boundary ({len(verbs)} verbs)...")
        batch_size = 80

        for batch_start in range(0, len(verbs), batch_size):
            batch = verbs[batch_start:batch_start + batch_size]
            verb_list = "\n".join(
                f"  {v['verb']}" + (f" ({v['gloss']})" if v['gloss'] else "")
                for v in batch
            )

            prompt = f"""Audit these verbs:\n\n{verb_list}\n\nJSON array:"""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=8192,
                        system=prompt_system,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    text = response.content[0].text.strip()
                    if '```' in text:
                        match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
                        if match:
                            text = match.group(1).strip()

                    batch_results = json.loads(text)

                    for r in batch_results:
                        r['boundary'] = boundary
                        r['original'] = batch[0]['original'] if batch else ''
                        # find matching original
                        for v in batch:
                            if v['verb'] == r.get('verb', ''):
                                r['original'] = v['original']
                                r['gloss'] = v.get('gloss', '')
                                break

                    all_results.extend(batch_results)
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(5)
                    else:
                        print(f"    ✗ batch {batch_start}: {e}")

            n_done = min(batch_start + batch_size, len(verbs))
            print(f"    {n_done}/{len(verbs)}", end="\r", flush=True)
            time.sleep(0.5)

        # Quick summary
        verdicts = Counter(r.get('verdict', '').upper() for r in all_results if r.get('boundary') == boundary)
        print(f"    {boundary}: KEEP={verdicts.get('KEEP',0)} CHANGE={verdicts.get('CHANGE',0)} UNCERTAIN={verdicts.get('UNCERTAIN',0)}")

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Saved {len(all_results)} reclass records")


# ══════════════════════════════════════════════════════════════
#  ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyze():
    """Analyze results from both tests."""

    print(f"\n{'='*70}")
    print(f"  TEST 1: SECOND-CHOICE ANALYSIS")
    print(f"{'='*70}")

    conf_file = os.path.join(DATA, "operator_confidence.json")
    if os.path.exists(conf_file):
        with open(conf_file) as f:
            confidence = json.load(f)

        # How often does SUP/REC appear as ANY of the top 3?
        sup_appearances = defaultdict(int)  # keyed by rank (1st, 2nd, 3rd)
        rec_appearances = defaultdict(int)
        total_verbs = 0
        sup_as_runner_up = []  # verbs where SUP is 2nd or 3rd
        rec_as_runner_up = []

        for r in confidence:
            choices = r.get('choices', [])
            total_verbs += 1
            for i, c in enumerate(choices):
                op = c.get('op', '').upper().strip()
                score = c.get('score', 0)
                rank = i + 1
                if op == 'SUP':
                    sup_appearances[rank] += 1
                    if rank > 1 and score >= 0.1:
                        sup_as_runner_up.append({
                            'verb': r.get('verb', ''),
                            'original': r.get('original_op', ''),
                            'rank': rank,
                            'score': score,
                            'top_op': choices[0].get('op', '') if choices else '',
                            'top_score': choices[0].get('score', 0) if choices else 0,
                        })
                if op == 'REC':
                    rec_appearances[rank] += 1
                    if rank > 1 and score >= 0.1:
                        rec_as_runner_up.append({
                            'verb': r.get('verb', ''),
                            'original': r.get('original_op', ''),
                            'rank': rank,
                            'score': score,
                            'top_op': choices[0].get('op', '') if choices else '',
                            'top_score': choices[0].get('score', 0) if choices else 0,
                        })

        print(f"\n  Total verbs with confidence data: {total_verbs}")
        print(f"\n  SUP appearances in top-3 choices:")
        for rank in [1, 2, 3]:
            n = sup_appearances.get(rank, 0)
            print(f"    Rank {rank}: {n} verbs ({100*n/total_verbs:.1f}%)")

        print(f"\n  REC appearances in top-3 choices:")
        for rank in [1, 2, 3]:
            n = rec_appearances.get(rank, 0)
            print(f"    Rank {rank}: {n} verbs ({100*n/total_verbs:.1f}%)")

        # How often is each operator a runner-up?
        print(f"\n  All operators as runner-up (2nd or 3rd choice, score >= 0.1):")
        runner_up_counts = Counter()
        for r in confidence:
            choices = r.get('choices', [])
            for i, c in enumerate(choices):
                if i > 0:  # not first choice
                    op = c.get('op', '').upper().strip()
                    score = c.get('score', 0)
                    if score >= 0.1 and op in HELIX:
                        runner_up_counts[op] += 1

        for op in HELIX:
            n = runner_up_counts.get(op, 0)
            print(f"    {op:>5s}: {n:5d} ({100*n/total_verbs:.1f}%)")

        # Show SUP runner-up verbs
        if sup_as_runner_up:
            print(f"\n  Verbs where SUP is runner-up (score >= 0.1):")
            for v in sorted(sup_as_runner_up, key=lambda x: -x['score'])[:30]:
                print(f"    {v['verb']:20s} classified={v['original']:>5s}  SUP score={v['score']:.2f}  (top: {v['top_op']} {v['top_score']:.2f})")

        if rec_as_runner_up:
            print(f"\n  Verbs where REC is runner-up (score >= 0.1):")
            for v in sorted(rec_as_runner_up, key=lambda x: -x['score'])[:30]:
                print(f"    {v['verb']:20s} classified={v['original']:>5s}  REC score={v['score']:.2f}  (top: {v['top_op']} {v['top_score']:.2f})")

        # Mean confidence by operator
        print(f"\n  Mean top-choice confidence by original operator:")
        op_scores = defaultdict(list)
        for r in confidence:
            choices = r.get('choices', [])
            orig = r.get('original_op', '').upper().strip()
            if choices and orig in HELIX:
                op_scores[orig].append(choices[0].get('score', 0))

        for op in HELIX:
            scores = op_scores.get(op, [])
            if scores:
                print(f"    {op:>5s}: mean={np.mean(scores):.3f}  std={np.std(scores):.3f}  n={len(scores)}")

    else:
        print("  ✗ No confidence data — run --test1 first")

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  TEST 2: TARGETED RE-CLASSIFICATION")
    print(f"{'='*70}")

    reclass_file = os.path.join(DATA, "sup_rec_reclass.json")
    if os.path.exists(reclass_file):
        with open(reclass_file) as f:
            reclass = json.load(f)

        for boundary in ["ALT→SUP", "INS→REC", "NUL→SUP"]:
            subset = [r for r in reclass if r.get('boundary') == boundary]
            if not subset:
                continue

            verdicts = Counter(r.get('verdict', '').upper() for r in subset)
            total = len(subset)
            print(f"\n  {boundary} ({total} verbs):")
            for v in ['KEEP', 'CHANGE', 'UNCERTAIN']:
                n = verdicts.get(v, 0)
                print(f"    {v:12s}: {n:4d} ({100*n/total:.1f}%)")

            # List the CHANGE verbs
            changed = [r for r in subset if r.get('verdict', '').upper() == 'CHANGE']
            if changed:
                print(f"\n    Verbs that should be reclassified:")
                for r in changed:
                    print(f"      {r.get('verb', ''):20s} — {r.get('reasoning', '')}")

            # List the UNCERTAIN verbs
            uncertain = [r for r in subset if r.get('verdict', '').upper() == 'UNCERTAIN']
            if uncertain:
                print(f"\n    Genuinely ambiguous verbs:")
                for r in uncertain[:20]:
                    print(f"      {r.get('verb', ''):20s} — {r.get('reasoning', '')}")

    else:
        print("  ✗ No reclass data — run --test2 first")

    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    if os.path.exists(conf_file) and os.path.exists(reclass_file):
        with open(conf_file) as f:
            confidence = json.load(f)
        with open(reclass_file) as f:
            reclass = json.load(f)

        # Count potential SUP/REC gains
        sup_gained = sum(1 for r in reclass
                        if r.get('verdict', '').upper() == 'CHANGE'
                        and 'SUP' in r.get('boundary', ''))
        rec_gained = sum(1 for r in reclass
                        if r.get('verdict', '').upper() == 'CHANGE'
                        and 'REC' in r.get('boundary', ''))

        # Current counts
        current_sup = sum(1 for r in confidence
                         if r.get('original_op', '').upper() == 'SUP')
        current_rec = sum(1 for r in confidence
                         if r.get('original_op', '').upper() == 'REC')

        total = len(confidence)

        print(f"\n  Current:  SUP = {current_sup} ({100*current_sup/total:.1f}%)  REC = {current_rec} ({100*current_rec/total:.1f}%)")
        print(f"  Gained:   SUP +{sup_gained}  REC +{rec_gained}")
        print(f"  Revised:  SUP = {current_sup + sup_gained} ({100*(current_sup+sup_gained)/total:.1f}%)  REC = {current_rec + rec_gained} ({100*(current_rec+rec_gained)/total:.1f}%)")

        if (current_sup + sup_gained) / total < 0.05 and (current_rec + rec_gained) / total < 0.05:
            print(f"\n  → Poverty CONFIRMED even after targeted audit.")
            print(f"    Even generous reclassification keeps SUP+REC below 5%.")
        else:
            print(f"\n  → Poverty PARTIALLY ARTIFACT.")
            print(f"    Reclassification meaningfully increases SUP/REC counts.")

    # Save summary
    summary = {}
    if os.path.exists(conf_file):
        with open(conf_file) as f:
            confidence = json.load(f)
        runner_up = Counter()
        for r in confidence:
            for i, c in enumerate(r.get('choices', [])):
                if i > 0 and c.get('score', 0) >= 0.1:
                    op = c.get('op', '').upper().strip()
                    if op in HELIX:
                        runner_up[op] += 1
        summary['runner_up_counts'] = dict(runner_up)

    if os.path.exists(reclass_file):
        with open(reclass_file) as f:
            reclass = json.load(f)
        for boundary in ["ALT→SUP", "INS→REC", "NUL→SUP"]:
            subset = [r for r in reclass if r.get('boundary') == boundary]
            verdicts = Counter(r.get('verdict', '').upper() for r in subset)
            summary[boundary] = dict(verdicts)

    with open(os.path.join(OUT, "sup_rec_audit.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Saved sup_rec_audit.json")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test1', action='store_true', help='Second-choice analysis only')
    parser.add_argument('--test2', action='store_true', help='Targeted reclassification only')
    parser.add_argument('--analyze', action='store_true', help='Analyze saved results')
    args = parser.parse_args()

    run_all = not (args.test1 or args.test2 or args.analyze)

    if args.test1 or args.test2 or run_all:
        try:
            import anthropic
        except ImportError:
            print("pip install anthropic")
            return
        client = anthropic.Anthropic()

    if args.test1 or run_all:
        print(f"{'='*70}")
        print(f"  TEST 1: SECOND-CHOICE CONFIDENCE ANALYSIS")
        print(f"{'='*70}")
        test1_second_choices(client)

    if args.test2 or run_all:
        print(f"\n{'='*70}")
        print(f"  TEST 2: TARGETED RE-CLASSIFICATION")
        print(f"{'='*70}")
        test2_reclassify(client)

    if args.analyze or run_all:
        analyze()


if __name__ == '__main__':
    main()
