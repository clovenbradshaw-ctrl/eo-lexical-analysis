"""
Step 4: LLM Classification of verbs into EO operators.

Run: python scripts/04_llm_classify.py [--backend openai|anthropic] [--batch_size 40]
Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY
Outputs: data/llm_classifications.json
"""

import argparse, json, os, sys, time, re
from typing import List, Dict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
SCRIPTS_DIR = os.path.dirname(__file__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, SCRIPTS_DIR)
from operator_definitions import OPERATORS, HELIX_ORDER

def log(msg):
    print(msg, flush=True)


SYSTEM_PROMPT = """You are classifying verbs into nine primitive transformation operators. These describe how ANYTHING changes — physical processes, social acts, perceptions, emotions, information. Every verb enacts one primary transformation.

The operators form a HELIX: Existence (what appears) → Structure (how it's organized) → Interpretation (how it's understood). Most verbs live in Existence. Fewer reach Structure. Fewest reach Interpretation.

ONE CRITICAL PRINCIPLE: classify by what the verb DOES as a transformation, not by what it's ABOUT. Ask: what is the input state, what is the output state, what changed structurally?

THE NINE OPERATORS:

1. NUL (∅) — Void: absence becomes explicit
   Input: a state where something is expected. Output: same state with absence marked.
   The verb makes a GAP, VOID, or MISSING thing visible. Something that should be there isn't.
   Yes: omit, lack, vanish, forget, deplete, silence, starve, deprive, empty, erase, void
   No: "displease" (nothing is missing), "break" (that's partition → SEG)

2. DES (⊡) — Designate: something gets named or typed
   Input: an unnamed or untyped thing. Output: that thing now has a category, label, or identity.
   The verb assigns WHAT something IS. It draws a definitional boundary.
   Yes: name, define, label, classify, diagnose, brand, title, baptize, declare, identify, dub
   No: "describe" in the sense of adding details (that adds observations → INS)

3. INS (△) — Instantiate: something new appears
   Input: the world as it is. Output: the world with one more thing in it.
   The verb brings a new phenomenon into existence. Additive — never replaces, only expands.
   Yes: create, build, emit, sing, pour, write, bloom, erupt, produce, moo, wiggle, cry
   THIS IS THE DEFAULT FOR EMBODIED VERBS. A movement, sound, gesture, or feeling arising
   is a new observation entering the world. Emotions appearing in an experiencer are INS —
   anger, dread, joy, calm are new states arising, not links or structures being built.

4. SEG (|) — Segment: one becomes many
   Input: a whole. Output: parts. A boundary is drawn THROUGH something, dividing it.
   The verb partitions, splits, filters, or separates. It creates subsets from a set.
   Yes: split, cut, filter, sort, crack, chip, sever, dice, dissect, peel, separate, sift
   No: "remove" (that's making something absent → NUL, not dividing a whole into parts)

5. CON (⋈) — Connect: a link is built between two things
   Input: two unlinked entities. Output: those entities with a new persistent structural link.
   The verb CONSTRUCTS a connection that PERSISTS after the event. A new edge in the graph.
   Yes: marry, hire, ally, befriend, cite, reference, attach, wire, assign, bind, introduce
   No: "accompany" (temporary co-presence, no persistent link built — that's INS, an event)
   No: "dread" (internal state change, no link constructed between entities — that's INS)
   The test: after the verb completes, does a structural relationship exist that didn't before?

6. SYN (∨) — Synthesize: many become one
   Input: multiple things. Output: a new whole that transcends its parts. Opposite of SEG.
   The verb combines, merges, fuses. The result is not reducible to its inputs.
   Yes: merge, fuse, synthesize, blend, brew, cook, compose, weave, summarize, compile, mix
   No: "gather" (collecting without transforming into new whole — that's INS, adding to a set)

7. ALT (∿) — Alternate: the frame shifts
   Input: thing + old frame. Output: same thing + new frame. The thing doesn't change; the lens does.
   The verb reinterprets, translates, or shifts perspective. Reversible.
   Yes: translate, reinterpret, convert, reframe, disguise, decode, rethink, paraphrase, adapt
   No: "restructure" (that changes the thing itself, not just the view — that's REC)

8. SUP (∥) — Superpose: contradictions coexist
   Input: a resolved state. Output: an unresolved state where incompatible values are held together.
   The verb creates or maintains a condition where MULTIPLE INCOMPATIBLE things are simultaneously true.
   Yes: contradict, conflict, equivocate, straddle, oscillate, haunt, paradox, complicate
   "Grieve" can be SUP — simultaneously holding love and loss, presence and absence.
   SUP is genuinely rare in natural language. Few verbs primarily do this.

9. REC (⟳) — Recurse: the structure rebuilds itself
   Input: an existing structure. Output: that structure reorganized around a new center.
   The verb doesn't just switch the view (ALT) — it REBUILDS the architecture.
   Yes: restructure, evolve, metamorphose, revolutionize, pivot, transcend, reform, regenerate
   No: "translate" (the structure stays, only the reading changes — that's ALT)

GUIDELINES:
- Every verb gets exactly one operator. Choose the PRIMARY transformation.
- If ambiguous, note the alternative. But commit to one.
- Most verbs → Existence triad (NUL/DES/INS). This is expected and correct.
- INS is the largest category. Physical actions, sounds, gestures, emotions arising — all INS.
- CON requires a PERSISTENT STRUCTURAL LINK being built. High bar. Most interactions are events (INS).
- SUP is the smallest category. Genuine contradiction-holding is rare.

Return ONLY valid JSON."""


def make_batch_prompt(verbs_with_defs: List[Dict]) -> str:
    lines = []
    for v in verbs_with_defs:
        entry = f'- "{v["verb"]}"'
        if v.get("definition"):
            entry += f': {v["definition"]}'
        if v.get("frames"):
            entry += f' [Frames: {", ".join(v["frames"][:2])}]'
        lines.append(entry)
    
    verbs_block = "\n".join(lines)
    
    return f"""Classify each verb into one operator (NUL, DES, INS, SEG, CON, SYN, ALT, SUP, REC).

For each verb ask:
1. What is the INPUT state and OUTPUT state?
2. What STRUCTURALLY changed? (gap appeared? new thing? parts? link? whole? frame? contradiction? rebuild?)
3. At what SCALE? (physical/social/informational/psychological)

VERBS:
{verbs_block}

Return a JSON array. Each element:
- "verb": the verb
- "operator": NUL/DES/INS/SEG/CON/SYN/ALT/SUP/REC
- "confidence": "high"/"medium"/"low"
- "reason": one sentence — name the input, output, and what changed
- "alternative": second-best operator if confidence is not high, else null
- "scale": "physical"/"social"/"informational"/"psychological"

ONLY the JSON array. No markdown, no preamble."""


# ─── Backends ─────────────────────────────────────────────────────

def classify_openai(prompt: str, system: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content

def classify_anthropic(prompt: str, system: str) -> str:
    from anthropic import Anthropic
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

BACKENDS = {
    "openai": classify_openai,
    "anthropic": classify_anthropic,
}


# ─── Corpus Loading ──────────────────────────────────────────────

def load_verbs_with_context():
    log("Loading corpus with context...")
    with open(os.path.join(DATA_DIR, "combined_corpus.json")) as f:
        corpus = json.load(f)
    
    verbs = []
    for entry in corpus:
        v = entry["verb"]
        definition = None
        wn = entry["sources"].get("wordnet", {})
        defs = wn.get("definitions", [])
        if defs:
            good_defs = [d for d in defs if len(d) > 10]
            if good_defs:
                definition = min(good_defs, key=len)
            elif defs:
                definition = defs[0]
        
        frames = list(dict.fromkeys(entry["sources"].get("framenet", {}).get("frames", [])))[:2]
        verbs.append({"verb": v, "definition": definition, "frames": frames})
    
    log(f"  {len(verbs)} verbs loaded")
    log(f"  {sum(1 for v in verbs if v['definition'])} have definitions")
    log(f"  {sum(1 for v in verbs if v['frames'])} have FrameNet frames")
    return verbs


def parse_response(text: str) -> List[Dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, list): return result
        elif isinstance(result, dict): return [result]
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try: return json.loads(match.group())
            except: pass
    return []


# ─── Main Pipeline ────────────────────────────────────────────────

def run(backend_name: str, batch_size: int = 40, max_verbs: int = None):
    log("=" * 60)
    log(f"STEP 4: LLM CLASSIFICATION (backend: {backend_name})")
    log("=" * 60)
    
    classify_fn = BACKENDS[backend_name]
    verbs = load_verbs_with_context()
    
    if max_verbs:
        verbs = verbs[:max_verbs]
        log(f"  Limited to {max_verbs} verbs")
    
    total = len(verbs)
    n_batches = (total - 1) // batch_size + 1
    
    log(f"\n  Total verbs:    {total}")
    log(f"  Batch size:     {batch_size}")
    log(f"  Total batches:  {n_batches}")
    log(f"  Backend:        {backend_name}")
    if backend_name == "openai":
        log(f"  Est. cost:      ~${total * 0.0003:.2f}")
    elif backend_name == "anthropic":
        log(f"  Est. cost:      ~${total * 0.0015:.2f} (claude-sonnet-4.5)")
    
    all_results = []
    failed_batches = []
    t_start = time.time()
    running_dist = {op: 0 for op in HELIX_ORDER}
    running_scales = {}
    
    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, total)
        batch = verbs[start:end]
        
        elapsed = time.time() - t_start
        rate = len(all_results) / elapsed if elapsed > 0 else 0
        eta = (total - len(all_results)) / rate if rate > 0 else 0
        
        if bi > 0 and bi % 10 == 0:
            log(f"\n  ── Running distribution (n={len(all_results)}) ──")
            for op in HELIX_ORDER:
                c = running_dist[op]
                pct = c / len(all_results) * 100 if all_results else 0
                bar = "█" * int(pct / 2)
                log(f"    {op:3s}: {c:5d} ({pct:5.1f}%) {bar}")
            if running_scales:
                log(f"  Scales: {dict(sorted(running_scales.items(), key=lambda x: -x[1]))}")
            log("")
        
        log(f"  Batch {bi+1}/{n_batches} | verbs {start+1}-{end}/{total} | "
            f"done: {len(all_results)} | rate: {rate:.1f}/s | ETA: {eta:.0f}s")
        
        prompt = make_batch_prompt(batch)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                bt = time.time()
                response_text = classify_fn(prompt, SYSTEM_PROMPT)
                call_time = time.time() - bt
                
                results = parse_response(response_text)
                
                if len(results) == 0:
                    log(f"    ⚠ Empty parse (attempt {attempt+1}), preview: {response_text[:200]}")
                    if attempt < max_retries - 1: continue
                    else:
                        log(f"    ✗ Failed after {max_retries} attempts")
                        failed_batches.append(bi)
                        break
                
                valid = 0
                for r in results:
                    op = r.get("operator", "").upper()
                    if op in HELIX_ORDER:
                        r["operator"] = op
                        running_dist[op] += 1
                        valid += 1
                    else:
                        r["operator"] = "UNKNOWN"
                    scale = r.get("scale", "unknown")
                    running_scales[scale] = running_scales.get(scale, 0) + 1
                
                all_results.extend(results)
                log(f"    ✓ {len(results)} classified ({valid} valid) in {call_time:.1f}s")
                
                for r in results[:3]:
                    conf = r.get("confidence", "?")
                    reason = r.get("reason", "")[:60]
                    alt = r.get("alternative")
                    scale = r.get("scale", "?")
                    alt_str = f" (alt: {alt})" if alt else ""
                    log(f"      {r['verb']:20s} → {r['operator']:3s} [{conf}/{scale}]{alt_str}  {reason}...")
                
                break
                
            except Exception as e:
                log(f"    ✗ Error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    failed_batches.append(bi)
        
        time.sleep(0.1)
    
    elapsed = time.time() - t_start
    
    log(f"\n{'='*60}")
    log(f"CLASSIFICATION COMPLETE")
    log(f"{'='*60}")
    log(f"  Total classified: {len(all_results)}/{total}")
    log(f"  Failed batches:   {len(failed_batches)}")
    log(f"  Time:             {elapsed:.1f}s ({elapsed/60:.1f}m)")
    log(f"  Rate:             {len(all_results)/elapsed:.1f} verbs/s")
    
    log(f"\n  ── Final Distribution ──")
    max_c = max(running_dist.values()) if running_dist else 1
    for op in HELIX_ORDER:
        c = running_dist[op]
        pct = c / len(all_results) * 100 if all_results else 0
        bar = "█" * int(c / max_c * 30)
        log(f"    {op:3s}: {c:5d} ({pct:5.1f}%) {bar}")
    
    log(f"\n  ── Triad Distribution ──")
    existence = sum(running_dist[op] for op in ["NUL","DES","INS"])
    structure = sum(running_dist[op] for op in ["SEG","CON","SYN"])
    interpretation = sum(running_dist[op] for op in ["ALT","SUP","REC"])
    for name, count in [("Existence (NUL+DES+INS)", existence), ("Structure (SEG+CON+SYN)", structure), ("Interpretation (ALT+SUP+REC)", interpretation)]:
        pct = count / len(all_results) * 100 if all_results else 0
        log(f"    {name:35s}: {count:5d} ({pct:.1f}%)")
    
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for r in all_results:
        c = r.get("confidence", "unknown").lower()
        if c in conf_counts: conf_counts[c] += 1
    
    log(f"\n  ── Confidence ──")
    for conf, c in sorted(conf_counts.items(), key=lambda x: -x[1]):
        pct = c / len(all_results) * 100 if all_results else 0
        log(f"    {conf:8s}: {c:5d} ({pct:.1f}%)")
    
    log(f"\n  ── Scale Distribution ──")
    for scale, c in sorted(running_scales.items(), key=lambda x: -x[1]):
        pct = c / len(all_results) * 100 if all_results else 0
        log(f"    {scale:15s}: {c:5d} ({pct:.1f}%)")
    
    log(f"\n  ── Scale × Triad ──")
    scale_triad = {}
    for r in all_results:
        op = r.get("operator", "")
        scale = r.get("scale", "unknown")
        if op in ["NUL","DES","INS"]: triad = "existence"
        elif op in ["SEG","CON","SYN"]: triad = "structure"
        elif op in ["ALT","SUP","REC"]: triad = "interpretation"
        else: continue
        key = (scale, triad)
        scale_triad[key] = scale_triad.get(key, 0) + 1
    
    scales = sorted(set(s for s, _ in scale_triad.keys()))
    triads = ["existence", "structure", "interpretation"]
    log(f"    {'':15s} {'exist':>8s} {'struct':>8s} {'interp':>8s}")
    for scale in scales:
        vals = [scale_triad.get((scale, t), 0) for t in triads]
        log(f"    {scale:15s} {vals[0]:8d} {vals[1]:8d} {vals[2]:8d}")
    
    alts = [(r.get("operator"), r.get("alternative")) for r in all_results if r.get("alternative")]
    alt_pairs = {}
    for op, alt in alts:
        if alt and alt.upper() in HELIX_ORDER:
            pair = tuple(sorted([op, alt.upper()]))
            alt_pairs[pair] = alt_pairs.get(pair, 0) + 1
    
    if alt_pairs:
        log(f"\n  ── Top confused boundaries ──")
        for (a, b), c in sorted(alt_pairs.items(), key=lambda x: -x[1])[:10]:
            log(f"    {a:3s} ↔ {b:3s}: {c:4d} verbs")
    
    output = {
        "metadata": {
            "backend": backend_name, "total_verbs": total,
            "classified": len(all_results), "failed_batches": len(failed_batches),
            "elapsed_seconds": elapsed, "batch_size": batch_size,
        },
        "distribution": running_dist,
        "triad_distribution": {"existence": existence, "structure": structure, "interpretation": interpretation},
        "confidence": conf_counts,
        "scale_distribution": running_scales,
        "scale_triad": {f"{s}_{t}": c for (s, t), c in scale_triad.items()},
        "alternative_pairs": {f"{a}-{b}": c for (a, b), c in sorted(alt_pairs.items(), key=lambda x: -x[1])},
        "classifications": all_results,
    }
    
    outpath = os.path.join(DATA_DIR, "llm_classifications.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\n  ✓ Saved → {outpath}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), default="anthropic")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--max_verbs", type=int, default=None)
    args = parser.parse_args()
    run(args.backend, args.batch_size, args.max_verbs)
