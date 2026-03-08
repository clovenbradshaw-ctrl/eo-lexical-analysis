"""
CoNLL-U parser and clause extractor for the EO Notebook Experiment App.
Extracted from scripts/09_crossling.py — adapted for Pyodide (no file I/O).

Functions accept raw text and return structured data.
"""

from collections import defaultdict, Counter


def parse_conllu(text):
    """Parse CoNLL-U formatted text into sentences.

    Args:
        text: raw CoNLL-U text content

    Returns:
        list of sentences, each a list of token dicts with keys:
        id, form, lemma, upos, xpos, feats, head, deprel, deps, misc
    """
    sentences = []
    current = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) < 10:
            continue

        token_id = fields[0]
        # Skip multi-word tokens and empty nodes
        if "-" in token_id or "." in token_id:
            continue

        try:
            current.append({
                "id": int(token_id),
                "form": fields[1],
                "lemma": fields[2],
                "upos": fields[3],
                "xpos": fields[4],
                "feats": fields[5],
                "head": int(fields[6]) if fields[6] != "_" else 0,
                "deprel": fields[7],
                "deps": fields[8],
                "misc": fields[9],
            })
        except (ValueError, IndexError):
            continue

    if current:
        sentences.append(current)

    return sentences


def extract_verbs(text):
    """Extract unique verb lemmas and frequencies from CoNLL-U text.

    Args:
        text: raw CoNLL-U text content

    Returns:
        dict: {lemma: {count: N, forms: [form1, form2, ...]}}
    """
    verbs = defaultdict(lambda: {"count": 0, "forms": set()})

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")
        if len(fields) < 10:
            continue

        token_id = fields[0]
        if "-" in token_id or "." in token_id:
            continue

        upos = fields[3]
        if upos == "VERB":
            lemma = fields[2].lower().strip()
            form = fields[1].lower()
            if lemma and lemma != "_":
                verbs[lemma]["count"] += 1
                verbs[lemma]["forms"].add(form)

    result = {}
    for lemma, data in verbs.items():
        result[lemma] = {
            "count": data["count"],
            "forms": sorted(list(data["forms"]))[:10],
        }

    return result


def extract_clauses(text):
    """Extract predicate-argument structures (clauses) from CoNLL-U text.

    Parses dependency relations to find:
    - nsubj (nominal subject)
    - obj (direct object)
    - iobj (indirect object)
    - obl (oblique nominal)
    - xcomp/ccomp (clausal complements)

    Args:
        text: raw CoNLL-U text content

    Returns:
        list of clause dicts: {verb, subject, object, indirect_object, oblique, complement, full_text}
    """
    sentences = parse_conllu(text)
    clauses = []

    for sent in sentences:
        # Build token lookup by id
        tokens = {t["id"]: t for t in sent}

        # Find all verbs (VERB upos)
        verb_tokens = [t for t in sent if t["upos"] == "VERB"]

        for verb in verb_tokens:
            clause = {
                "verb": verb["lemma"].lower(),
                "verb_form": verb["form"].lower(),
                "subject": None,
                "object": None,
                "indirect_object": None,
                "oblique": [],
                "complement": None,
            }

            # Find dependents of this verb
            for t in sent:
                if t["head"] == verb["id"]:
                    deprel = t["deprel"].split(":")[0]  # strip subtypes

                    if deprel == "nsubj":
                        clause["subject"] = t["lemma"].lower()
                    elif deprel == "obj":
                        clause["object"] = t["lemma"].lower()
                    elif deprel == "iobj":
                        clause["indirect_object"] = t["lemma"].lower()
                    elif deprel == "obl":
                        clause["oblique"].append(t["lemma"].lower())
                    elif deprel in ("xcomp", "ccomp"):
                        clause["complement"] = t["lemma"].lower()

            # Build readable text representation
            parts = []
            if clause["subject"]:
                parts.append(clause["subject"])
            parts.append(clause["verb"])
            if clause["object"]:
                parts.append(clause["object"])
            if clause["indirect_object"]:
                parts.append("to " + clause["indirect_object"])
            for obl in clause["oblique"]:
                parts.append("(obl: " + obl + ")")
            if clause["complement"]:
                parts.append("[comp: " + clause["complement"] + "]")

            clause["full_text"] = " ".join(parts)
            clauses.append(clause)

    return clauses


def summarize_clauses(clauses):
    """Summarize extracted clauses into frequency statistics.

    Args:
        clauses: list of clause dicts from extract_clauses()

    Returns:
        dict with clause pattern frequencies, verb-argument statistics
    """
    verb_counts = Counter()
    pattern_counts = Counter()
    arg_structure_counts = Counter()

    for c in clauses:
        verb_counts[c["verb"]] += 1

        # Argument structure pattern
        args = []
        if c["subject"]:
            args.append("S")
        args.append("V")
        if c["object"]:
            args.append("O")
        if c["indirect_object"]:
            args.append("IO")
        if c["oblique"]:
            args.append("OBL")
        if c["complement"]:
            args.append("COMP")

        pattern = "-".join(args)
        pattern_counts[pattern] += 1

        # Simplified: intransitive / transitive / ditransitive
        if c["object"] and c["indirect_object"]:
            arg_structure_counts["ditransitive"] += 1
        elif c["object"]:
            arg_structure_counts["transitive"] += 1
        else:
            arg_structure_counts["intransitive"] += 1

    return {
        "n_clauses": len(clauses),
        "n_unique_verbs": len(verb_counts),
        "top_verbs": dict(verb_counts.most_common(30)),
        "pattern_counts": dict(pattern_counts.most_common(20)),
        "arg_structure": dict(arg_structure_counts),
        "sample_clauses": [c["full_text"] for c in clauses[:50]],
    }
