"""
Step 2: Embed verbs and operator definitions using OpenAI.

Requires: OPENAI_API_KEY environment variable set.
Fallback: sentence-transformers (free, local).

Run: python scripts/02_embed.py [--backend openai|sentence-transformers]
"""

import argparse, json, os, sys, time
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCRIPTS_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPTS_DIR)
from operator_definitions import OPERATORS, HELIX_ORDER

def log(msg):
    print(msg, flush=True)

def load_corpus():
    path = os.path.join(DATA_DIR, "embedding_texts.json")
    if not os.path.exists(path):
        log(f"ERROR: {path} not found. Run 01_extract_corpora.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    log(f"Loaded {len(data)} verb entries from corpus")
    return data

def build_operator_texts():
    texts = {}
    for op in HELIX_ORDER:
        o = OPERATORS[op]
        texts[op] = {"name": op, "short_def": o["short_def"], "full_spec": o["full_spec"], "seed_verbs": ", ".join(o["verbs_seed"])}
    return texts


# ─── OpenAI Backend ───────────────────────────────────────────────

def embed_openai(texts, model="text-embedding-3-large"):
    from openai import OpenAI
    client = OpenAI()
    
    batch_size = 2048
    all_embs = []
    total = len(texts)
    t0 = time.time()
    
    log(f"\n  OpenAI model: {model}")
    log(f"  Total texts: {total}")
    log(f"  Batch size: {batch_size}")
    log(f"  Estimated batches: {(total-1)//batch_size + 1}")
    log("")
    
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total - 1) // batch_size + 1
        
        log(f"  Batch {batch_num}/{total_batches}: texts {i+1}-{min(i+batch_size, total)} of {total}...")
        
        bt = time.time()
        response = client.embeddings.create(input=batch, model=model)
        elapsed = time.time() - bt
        
        batch_embs = [item.embedding for item in response.data]
        all_embs.extend(batch_embs)
        
        total_elapsed = time.time() - t0
        rate = len(all_embs) / total_elapsed
        remaining = (total - len(all_embs)) / rate if rate > 0 else 0
        
        log(f"    ✓ {len(batch)} embeddings in {elapsed:.1f}s | "
            f"Total: {len(all_embs)}/{total} | "
            f"Rate: {rate:.0f}/s | "
            f"ETA: {remaining:.0f}s")
        
        # Show token usage if available
        if hasattr(response, 'usage') and response.usage:
            log(f"    Tokens: {response.usage.total_tokens}")
    
    log(f"\n  ✓ All {total} embeddings complete in {time.time()-t0:.1f}s")
    return np.array(all_embs)


# ─── Sentence-Transformers Backend ────────────────────────────────

def embed_st(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    
    log(f"\n  Model: {model_name}")
    log(f"  Loading model...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    log(f"  ✓ Model loaded in {time.time()-t0:.1f}s")
    
    log(f"  Embedding {len(texts)} texts...")
    t0 = time.time()
    embs = model.encode(texts, show_progress_bar=True, batch_size=256)
    log(f"  ✓ Done in {time.time()-t0:.1f}s")
    return np.array(embs)


BACKENDS = {"openai": embed_openai, "sentence-transformers": embed_st}


# ─── Main ─────────────────────────────────────────────────────────

def run(backend_name):
    log("="*60)
    log(f"STEP 2: EMBEDDING (backend: {backend_name})")
    log("="*60)
    
    embed_fn = BACKENDS[backend_name]
    corpus = load_corpus()
    op_texts = build_operator_texts()
    
    # Prepare texts
    verb_texts = [e["embed_text"] for e in corpus]
    verb_names = [e["verb"] for e in corpus]
    bare_texts = [e["bare_verb"] for e in corpus]
    op_short = [op_texts[op]["short_def"] for op in HELIX_ORDER]
    op_full = [op_texts[op]["full_spec"] for op in HELIX_ORDER]
    op_seed = [op_texts[op]["seed_verbs"] for op in HELIX_ORDER]
    
    log(f"\nTexts to embed:")
    log(f"  Enriched verb texts:  {len(verb_texts)}")
    log(f"  Bare verb names:      {len(bare_texts)}")
    log(f"  Operator short defs:  {len(op_short)}")
    log(f"  Operator full specs:  {len(op_full)}")
    log(f"  Operator seed verbs:  {len(op_seed)}")
    log(f"  Total API calls:      5 rounds")
    
    # Embed each set
    log(f"\n{'─'*40}")
    log(f"Round 1/5: Enriched verb texts ({len(verb_texts)} texts)")
    log(f"{'─'*40}")
    verb_embs = embed_fn(verb_texts)
    log(f"  Shape: {verb_embs.shape}")
    
    log(f"\n{'─'*40}")
    log(f"Round 2/5: Bare verb names ({len(bare_texts)} texts)")
    log(f"{'─'*40}")
    bare_embs = embed_fn(bare_texts)
    log(f"  Shape: {bare_embs.shape}")
    
    log(f"\n{'─'*40}")
    log(f"Round 3/5: Operator short definitions (9 texts)")
    log(f"{'─'*40}")
    op_short_embs = embed_fn(op_short)
    log(f"  Shape: {op_short_embs.shape}")
    
    log(f"\n{'─'*40}")
    log(f"Round 4/5: Operator full specifications (9 texts)")
    log(f"{'─'*40}")
    op_full_embs = embed_fn(op_full)
    log(f"  Shape: {op_full_embs.shape}")
    
    log(f"\n{'─'*40}")
    log(f"Round 5/5: Operator seed verb lists (9 texts)")
    log(f"{'─'*40}")
    op_seed_embs = embed_fn(op_seed)
    log(f"  Shape: {op_seed_embs.shape}")
    
    # Save
    log(f"\nSaving embeddings...")
    np.savez_compressed(os.path.join(DATA_DIR, "verb_embeddings.npz"), enriched=verb_embs, bare=bare_embs)
    log(f"  ✓ verb_embeddings.npz")
    
    np.savez_compressed(os.path.join(DATA_DIR, "operator_embeddings.npz"), short_def=op_short_embs, full_spec=op_full_embs, seed_verbs=op_seed_embs)
    log(f"  ✓ operator_embeddings.npz")
    
    meta = {
        "backend": backend_name, "embedding_dim": int(verb_embs.shape[1]),
        "num_verbs": len(verb_names), "verb_names": verb_names,
        "operator_order": HELIX_ORDER,
        "operator_texts": {op: op_texts[op] for op in HELIX_ORDER},
    }
    with open(os.path.join(DATA_DIR, "embedding_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  ✓ embedding_metadata.json")
    
    log(f"\n✓ Step 2 complete. Embedding dim: {verb_embs.shape[1]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=list(BACKENDS.keys()), default="openai")
    args = parser.parse_args()
    run(args.backend)
