"""
Step 1: Extract verb corpora from VerbNet, FrameNet, and WordNet.
Run: python scripts/01_extract_corpora.py
"""

import json, os, sys, time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)

def ensure_nltk_data():
    import nltk
    for cid, name in [("verbnet3","VerbNet3"),("verbnet","VerbNet"),("framenet_v17","FrameNet"),("wordnet","WordNet"),("omw-1.4","OMW")]:
        log(f"  Downloading {name}...")
        try:
            nltk.download(cid, quiet=True)
            log(f"  ✓ {name}")
        except Exception as e:
            log(f"  ✗ {name}: {e}")

def extract_verbnet():
    log("\n" + "="*50)
    log("VERBNET")
    log("="*50)
    try:
        from nltk.corpus import verbnet as vn
    except:
        try:
            from nltk.corpus import verbnet3 as vn
        except:
            log("✗ Not available"); return []
    
    cids = vn.classids()
    log(f"Found {len(cids)} classes")
    verb_classes = {}
    t0 = time.time()
    for ci, cid in enumerate(cids):
        if (ci+1) % 50 == 0 or ci == len(cids)-1:
            log(f"  class {ci+1}/{len(cids)} ... {len(verb_classes)} verbs so far [{time.time()-t0:.1f}s]")
        try:
            for lemma in vn.lemmas(cid):
                if lemma not in verb_classes:
                    verb_classes[lemma] = {"verb": lemma, "classes": [], "source": "verbnet"}
                try: roles = [str(t) for t in vn.themroles(cid)]
                except: roles = []
                verb_classes[lemma]["classes"].append({"class_id": cid, "base_class": cid.split("-")[0], "thematic_roles": roles})
        except: pass
    
    result = list(verb_classes.values())
    log(f"✓ {len(result)} unique verbs [{time.time()-t0:.1f}s]")
    with open(os.path.join(OUTPUT_DIR, "verbnet_verbs.json"), "w") as f: json.dump(result, f, indent=2)
    return result

def extract_framenet():
    log("\n" + "="*50)
    log("FRAMENET")
    log("="*50)
    try:
        from nltk.corpus import framenet as fn
    except:
        log("✗ Not available"); return []
    
    frames = fn.frames()
    log(f"Found {len(frames)} frames")
    records, verb_frames = [], {}
    t0 = time.time()
    for fi, frame in enumerate(frames):
        if (fi+1) % 100 == 0 or fi == len(frames)-1:
            log(f"  frame {fi+1}/{len(frames)} ({frame.name}) [{time.time()-t0:.1f}s]")
        try:
            fe_list = []
            if hasattr(frame, "FE") and frame.FE:
                for n, d in frame.FE.items():
                    e = {"name": n}
                    if hasattr(d, "definition"): e["definition"] = str(d.definition)[:200]
                    if hasattr(d, "coreType"): e["core_type"] = str(d.coreType)
                    fe_list.append(e)
            lu_list = []
            if hasattr(frame, "lexUnit") and frame.lexUnit:
                for n, d in frame.lexUnit.items():
                    lu_list.append({"name": n, "POS": d.POS if hasattr(d, "POS") else None})
            verb_lus = [lu for lu in lu_list if lu.get("POS") == "V"]
            records.append({"frame_name": frame.name, "definition": str(getattr(frame, "definition", "")), "frame_elements": fe_list, "verb_lexical_units": verb_lus, "source": "framenet"})
            for lu in verb_lus:
                v = lu["name"].split(".")[0]
                verb_frames.setdefault(v, []).append(frame.name)
        except: pass
    
    log(f"✓ {len(records)} frames, {len(verb_frames)} unique verbs [{time.time()-t0:.1f}s]")
    with open(os.path.join(OUTPUT_DIR, "framenet_frames.json"), "w") as f: json.dump(records, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "framenet_verb_frames.json"), "w") as f: json.dump(verb_frames, f, indent=2)
    return records

def extract_wordnet():
    log("\n" + "="*50)
    log("WORDNET")
    log("="*50)
    try:
        from nltk.corpus import wordnet as wn
    except:
        log("✗ Not available"); return []
    
    synsets = list(wn.all_synsets(pos=wn.VERB))
    log(f"Found {len(synsets)} verb synsets")
    records = []
    t0 = time.time()
    for si, ss in enumerate(synsets):
        if (si+1) % 2000 == 0 or si == len(synsets)-1:
            log(f"  synset {si+1}/{len(synsets)} [{time.time()-t0:.1f}s]")
        try:
            records.append({
                "synset": ss.name(), "lemmas": [l.name() for l in ss.lemmas()],
                "definition": ss.definition(), "examples": ss.examples()[:3],
                "hypernyms": [h.name() for h in ss.hypernyms()],
                "entailments": [e.name() for e in ss.entailments()],
                "source": "wordnet",
            })
        except: pass
    
    all_verbs = set()
    for r in records:
        for l in r["lemmas"]: all_verbs.add(l.replace("_"," "))
    log(f"✓ {len(records)} synsets, {len(all_verbs)} unique lemmas [{time.time()-t0:.1f}s]")
    with open(os.path.join(OUTPUT_DIR, "wordnet_verbs.json"), "w") as f: json.dump(records, f, indent=2)
    return records

def build_combined(vn_data, fn_data, wn_data):
    log("\n" + "="*50)
    log("COMBINING")
    log("="*50)
    combined = {}
    for entry in vn_data:
        v = entry["verb"].lower().replace("_"," ")
        combined.setdefault(v, {"verb": v, "sources": {}})
        combined[v]["sources"]["verbnet"] = {"classes": [c["class_id"] for c in entry.get("classes",[])]}
    log(f"  After VerbNet: {len(combined)} verbs")
    
    for frame in fn_data:
        for lu in frame.get("verb_lexical_units", []):
            v = lu["name"].split(".")[0].lower().replace("_"," ")
            combined.setdefault(v, {"verb": v, "sources": {}})
            combined[v]["sources"].setdefault("framenet", {"frames": []})
            combined[v]["sources"]["framenet"]["frames"].append(frame["frame_name"])
    log(f"  After FrameNet: {len(combined)} verbs")
    
    for entry in wn_data:
        for lemma in entry["lemmas"]:
            v = lemma.lower().replace("_"," ")
            combined.setdefault(v, {"verb": v, "sources": {}})
            combined[v]["sources"].setdefault("wordnet", {"synsets": [], "definitions": []})
            combined[v]["sources"]["wordnet"]["synsets"].append(entry["synset"])
            combined[v]["sources"]["wordnet"]["definitions"].append(entry["definition"])
    log(f"  After WordNet: {len(combined)} verbs")
    
    result = list(combined.values())
    multi = sum(1 for v in result if len(v["sources"]) > 1)
    log(f"  Multi-source: {multi}")
    
    with open(os.path.join(OUTPUT_DIR, "combined_corpus.json"), "w") as f: json.dump(result, f, indent=2)
    
    # Build embedding texts
    log("\n  Building embedding texts...")
    texts = []
    for entry in result:
        v = entry["verb"]
        parts = [v]
        wn = entry["sources"].get("wordnet", {})
        defs = list(dict.fromkeys(wn.get("definitions",[])))[:3]
        parts.extend(defs)
        fn = entry["sources"].get("framenet", {})
        frames = list(dict.fromkeys(fn.get("frames",[])))[:3]
        if frames: parts.append("Frames: " + ", ".join(frames))
        vn = entry["sources"].get("verbnet", {})
        classes = list(dict.fromkeys(vn.get("classes",[])))[:3]
        if classes: parts.append("Classes: " + ", ".join(classes))
        texts.append({"verb": v, "embed_text": ". ".join(parts), "bare_verb": v})
    
    with open(os.path.join(OUTPUT_DIR, "embedding_texts.json"), "w") as f: json.dump(texts, f, indent=2)
    log(f"✓ {len(texts)} embedding texts ready")
    return result

if __name__ == "__main__":
    t0 = time.time()
    log("="*60)
    log("STEP 1: CORPUS EXTRACTION")
    log("="*60)
    ensure_nltk_data()
    vn = extract_verbnet()
    fn = extract_framenet()
    wn = extract_wordnet()
    build_combined(vn, fn, wn)
    log(f"\nStep 1 done in {time.time()-t0:.1f}s")
