#!/usr/bin/env python3
"""
Build consolidated JSON data files for the EO Lexical Analysis minisite.
Reads from output/ and produces data/ JSON files sized for client-side use.
"""

import json
import os
import sys

OUT = "output"
DATA = "data"
os.makedirs(DATA, exist_ok=True)


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def save(name, obj):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    size = os.path.getsize(path)
    print(f"  {name}: {size/1024:.1f} KB")


def build_operators():
    """Consolidate per-operator data from explore files + topology."""
    operators = {}
    op_order = ["NUL", "DES", "INS", "SEG", "CON", "SYN", "ALT", "SUP", "REC"]
    symbols = {"NUL": "∅", "DES": "⊡", "INS": "△", "SEG": "｜", "CON": "⋈",
               "SYN": "∨", "ALT": "∿", "SUP": "∥", "REC": "⟳"}
    triads = {
        "NUL": "Existence", "DES": "Existence", "INS": "Existence",
        "SEG": "Structure", "CON": "Structure", "SYN": "Structure",
        "ALT": "Interpretation", "SUP": "Interpretation", "REC": "Interpretation",
    }
    roles = {
        "NUL": 1, "DES": 2, "INS": 3,
        "SEG": 1, "CON": 2, "SYN": 3,
        "ALT": 1, "SUP": 2, "REC": 3,
    }
    short_defs = {
        "NUL": "Removal, negation, voiding — making something absent",
        "DES": "Distinguishing, marking difference — drawing distinctions",
        "INS": "Creating, appearing, bringing into being — new existence",
        "SEG": "Dividing, splitting, differentiating — creating boundaries",
        "CON": "Linking, relating, joining — building persistent connections",
        "SYN": "Combining, merging, unifying — creating wholes from parts",
        "ALT": "Changing state, toggling — same entity, different state",
        "SUP": "Holding contradictions simultaneously — unresolved tension",
        "REC": "Self-referential restructuring — rebuilding from held tension",
    }

    topo = load("topology_report.json")
    topo_by_op = {t["operator"]: t for t in topo["operators"]}

    for op in op_order:
        explore = load(f"explore_{op}.json")
        t = topo_by_op[op]

        # Gather top 5 prototype verbs
        prototypes = []
        for p in t["prototypes"][:5]:
            prototypes.append({
                "verb": p["verb"],
                "margin": round(p["margin"], 3),
                "scale": p["scale"],
                "reason": p["reason"],
            })

        # Horizon verbs (boundary)
        horizon = []
        for h in t["horizon_verbs"][:5]:
            horizon.append({
                "verb": h["verb"],
                "neighbor": h["nearest_other_op"],
                "margin": round(h["margin"], 3),
            })

        # Nearest operators
        neighbors = []
        for n in t["nearest_neighbors"][:3]:
            neighbors.append({
                "op": n["operator"],
                "sim": round(n["centroid_similarity"], 4),
            })

        operators[op] = {
            "symbol": symbols[op],
            "triad": triads[op],
            "role": roles[op],
            "def": short_defs[op],
            "total": explore["total_verbs"],
            "scales": explore["scales"],
            "confidence": explore["confidence"],
            "territory": t["territory"],
            "margin_mean": round(t["margin_stats"]["mean"], 4),
            "prototypes": prototypes,
            "horizon": horizon,
            "neighbors": neighbors,
            "empirical_def": t["empirical_definition"],
        }

    return operators


def build_confusion():
    """Build confusion matrix data."""
    conf = load("llm_confusion.json")
    return conf


def build_boundaries():
    """Build boundary verb examples."""
    b = load("llm_boundaries.json")
    return b["boundaries"]


def build_metrics():
    """Build metrics comparison across embedding modes."""
    r = load("reembed_report.json")
    modes = []
    op_order = ["NUL", "DES", "INS", "SEG", "CON", "SYN", "ALT", "SUP", "REC"]
    for m in r["results"]:
        mode_data = {
            "name": m["mode"],
            "n_verbs": m["shape"][0],
            "n_dims": m["shape"][1],
            "ari": round(m["kmeans_k9"]["ari"], 4),
            "nmi": round(m["kmeans_k9"]["nmi"], 4),
            "silhouette": round(m["kmeans_k9"]["silhouette"], 4),
            "matched_accuracy": round(m["kmeans_k9"]["matched_accuracy"], 4),
            "z_score": round(m["centroid_separation"]["z_score"], 1),
            "best_k": m["best_k"],
            "sweep": m["sweep"],
            "centroid_matrix": [[round(v, 4) for v in row] for row in m["centroid_matrix"]],
        }
        modes.append(mode_data)

    # Taxonomy comparison
    tax = load("taxonomy_comparison.json")

    # Falsification
    fals = load("falsification_report.json")

    return {
        "modes": modes,
        "op_order": op_order,
        "taxonomy": tax,
        "falsification": fals,
    }


def build_verbs():
    """Build verb browser data from explore files + crosslinguistic data."""
    op_order = ["NUL", "DES", "INS", "SEG", "CON", "SYN", "ALT", "SUP", "REC"]
    all_verbs = []

    # English verbs from explore files (with full classification detail)
    for op in op_order:
        explore = load(f"explore_{op}.json")
        for v in explore["all_verbs"]:
            all_verbs.append({
                "v": v["verb"],
                "op": op,
                "c": v["confidence"][0].upper(),  # H/M/L
                "s": v["scale"][:4] if v.get("scale") else "?",  # phys/soci/psyc/info
                "r": v.get("reason", ""),
                "a": v.get("alternative"),
                "l": "English",
            })

    # Crosslinguistic verbs from 27-cell analysis (all languages)
    try:
        cells = load("crossling_27_cells.json")
        seen = set()  # deduplicate English verbs already added
        for v in all_verbs:
            seen.add(("English", v["v"]))

        for lang, ldata in cells.get("languages", {}).items():
            for cell_name, cell in ldata.get("cells", {}).items():
                for v in cell.get("verbs", []):
                    key = (lang, v["verb"])
                    if key in seen:
                        continue
                    seen.add(key)
                    all_verbs.append({
                        "v": v["verb"],
                        "op": v.get("operator", "?"),
                        "c": "?",
                        "s": "?",
                        "r": v.get("gloss", ""),
                        "a": None,
                        "l": lang,
                    })
        print(f"    (includes {len(all_verbs)} verbs across {len(set(v['l'] for v in all_verbs))} languages)")
    except Exception as e:
        print(f"    (crosslinguistic data not available: {e})")

    # Sort by language then verb name
    all_verbs.sort(key=lambda x: (x["l"].lower(), x["v"].lower()))
    return all_verbs


def build_embeddings_3d():
    """Build 3D embedding points for scatter plot."""
    emb = load("phasepost_embeddings_3d.json")
    meta = load("phasepost_centroids_meta.json")

    # Extract English scatter for 3D explorer
    eng = emb.get("English", {})
    scatter = []
    for pt in eng.get("scatter", []):
        scatter.append({
            "x": round(pt["x"], 4),
            "y": round(pt["y"], 4),
            "z": round(pt["z"], 4),
            "op": pt["op"],
            "ref": pt.get("ref", ""),
        })

    # Extract centroids for English
    centroids = []
    for c in eng.get("centroids", []):
        centroids.append({
            "pos": c["pos"],
            "tri": c["tri"],
            "ref": c["ref"],
            "op": c["op"],
            "n": c["n"],
            "x": round(c["x"], 4),
            "y": round(c["y"], 4),
            "z": round(c["z"], 4),
            "sample_verbs": c.get("sample_verbs", [])[:8],
        })

    return {
        "scatter": scatter,
        "centroids": centroids,
        "variance_explained": [round(v, 2) for v in eng.get("variance_explained", [])],
        "n_verbs": eng.get("n_verbs", 0),
        "cells_meta": meta["cells"],
    }


def build_crossling():
    """Build crosslinguistic z-scores data."""
    zscores = load("crossling_zscores.json")
    positions = load("crossling_positions.json")
    report = load("crossling_report.json")
    falsification = load("crossling_falsification.json")
    covariation = load("crossling_covariation.json")

    # Merge into a single crosslinguistic dataset
    langs = {}
    for lang, data in zscores.items():
        langs[lang] = {
            "z_all": round(data["z_all"], 2) if data.get("z_all") else None,
            "z_no_ins": round(data["z_no_ins"], 2) if data.get("z_no_ins") else None,
            "z_triad": round(data["z_triad"], 2) if data.get("z_triad") else None,
            "random_z": round(data["random_z"], 2) if data.get("random_z") else None,
            "family": data.get("family", ""),
            "era": data.get("era", ""),
            "n_verbs": data.get("n_verbs", 0),
        }
        if lang in positions:
            p = positions[lang]
            langs[lang]["z_positions"] = round(p.get("positions", 0), 2) if p.get("positions") else None
            langs[lang]["z_data_driven"] = round(p.get("data_driven", 0), 2) if p.get("data_driven") else None

    # Add op_pcts from report
    if "languages" in report:
        for lang, rdata in report["languages"].items():
            if lang in langs:
                langs[lang]["op_pcts"] = {k: round(v, 1) for k, v in rdata.get("op_pcts", {}).items()}

    return {
        "languages": langs,
        "falsification": falsification,
        "covariation": covariation,
    }


def build_influence():
    """Build influence/gravitational pull data."""
    inf = load("influence_report.json")
    # Round the matrices
    excess = {}
    for op, row in inf["excess_matrix"].items():
        excess[op] = {k: round(v, 4) for k, v in row.items()}
    return {
        "global_mean_pull": {k: round(v, 4) for k, v in inf["global_mean_pull"].items()},
        "excess_matrix": excess,
    }


def main():
    print("Building site data files...")

    print("\n1. Operators overview")
    operators = build_operators()
    save("operators.json", operators)

    print("\n2. Confusion matrix")
    confusion = build_confusion()
    save("confusion.json", confusion)

    print("\n3. Boundary verbs")
    boundaries = build_boundaries()
    save("boundaries.json", boundaries)

    print("\n4. Metrics & modes")
    metrics = build_metrics()
    save("metrics.json", metrics)

    print("\n5. Verb browser data")
    verbs = build_verbs()
    save("verbs.json", verbs)

    print("\n6. 3D embeddings / centroids")
    emb = build_embeddings_3d()
    save("embeddings.json", emb)

    print("\n7. Crosslinguistic data")
    crossling = build_crossling()
    save("crossling.json", crossling)

    print("\n8. Influence data")
    influence = build_influence()
    save("influence.json", influence)

    print("\nDone! All data files in data/")


if __name__ == "__main__":
    main()
