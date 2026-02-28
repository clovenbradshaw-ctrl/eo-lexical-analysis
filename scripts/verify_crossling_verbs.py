#!/usr/bin/env python3
"""
Verify cross-linguistic verb coverage across all crossling outputs.

Audits every language defined in the pipeline against all output files
to report which languages have complete verb data and which are missing
or inconsistent.

Usage:
  python scripts/verify_crossling_verbs.py
"""

import json
import os
import sys
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")

# Languages defined in 09_crossling.py LANGUAGES list
EXPECTED_LANGUAGES = [
    "Ancient_Greek", "Latin", "Classical_Chinese", "Sanskrit",
    "Old_Church_Slavonic", "Gothic", "Old_French", "Old_East_Slavic",
    "English", "German", "Russian", "French", "Finnish",
    "Japanese", "Mandarin", "Korean", "Arabic", "Hindi", "Turkish",
    "Indonesian", "Vietnamese", "Tamil", "Yoruba", "Wolof", "Naija",
    "Tagalog", "Persian", "Uyghur", "Coptic", "Basque",
]

LOW_VERB_THRESHOLD = 200


def load_json(path):
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def check_crossling_json():
    """Check data/crossling.json for language coverage."""
    path = os.path.join(DATA, "crossling.json")
    data = load_json(path)
    if data is None:
        return None, "FILE NOT FOUND"
    langs = data.get("languages", {})
    return langs, None


def check_crossling_report():
    """Check output/crossling_report.json for language coverage."""
    path = os.path.join(OUT, "crossling_report.json")
    data = load_json(path)
    if data is None:
        return None, "FILE NOT FOUND"
    langs = data.get("languages", {})
    return langs, None


def check_crossling_cells():
    """Check output/crossling_27_cells.json for language coverage."""
    path = os.path.join(OUT, "crossling_27_cells.json")
    data = load_json(path)
    if data is None:
        return None, "FILE NOT FOUND"
    langs = data.get("languages", {})
    return langs, None


def check_crossling_zscores():
    """Check output/crossling_zscores.json for language coverage."""
    path = os.path.join(OUT, "crossling_zscores.json")
    data = load_json(path)
    if data is None:
        return None, "FILE NOT FOUND"
    if isinstance(data, dict):
        return data, None
    return None, "UNEXPECTED FORMAT"


def check_crossling_positions():
    """Check output/crossling_positions.json for language coverage."""
    path = os.path.join(OUT, "crossling_positions.json")
    data = load_json(path)
    if data is None:
        return None, "FILE NOT FOUND"
    if isinstance(data, dict):
        return data, None
    return None, "UNEXPECTED FORMAT"


def check_raw_data():
    """Check data/crossling/ for per-language verb and classified files."""
    crossling_dir = os.path.join(DATA, "crossling")
    results = {}
    for lang in EXPECTED_LANGUAGES:
        lang_dir = os.path.join(crossling_dir, lang)
        verbs_file = os.path.join(lang_dir, "verbs.json")
        classified_file = os.path.join(lang_dir, "classified.json")
        results[lang] = {
            "dir_exists": os.path.isdir(lang_dir),
            "verbs_json": os.path.exists(verbs_file),
            "classified_json": os.path.exists(classified_file),
        }
    return results


def main():
    print("=" * 72)
    print("  CROSSLING VERB COVERAGE AUDIT")
    print("=" * 72)

    # Load all sources
    crossling_json, cj_err = check_crossling_json()
    report_json, rj_err = check_crossling_report()
    cells_json, cells_err = check_crossling_cells()
    zscores_json, zs_err = check_crossling_zscores()
    positions_json, pos_err = check_crossling_positions()
    raw_data = check_raw_data()

    sources = {
        "crossling.json": (crossling_json, cj_err),
        "crossling_report.json": (report_json, rj_err),
        "crossling_27_cells.json": (cells_json, cells_err),
        "crossling_zscores.json": (zscores_json, zs_err),
        "crossling_positions.json": (positions_json, pos_err),
    }

    # File status
    print("\n  OUTPUT FILES:")
    for name, (data, err) in sources.items():
        if err:
            print(f"    {name:35s}  {err}")
        else:
            n = len(data) if data else 0
            print(f"    {name:35s}  {n} languages")

    # Raw data status
    n_raw_dirs = sum(1 for v in raw_data.values() if v["dir_exists"])
    n_raw_verbs = sum(1 for v in raw_data.values() if v["verbs_json"])
    n_raw_classified = sum(1 for v in raw_data.values() if v["classified_json"])
    print(f"\n    data/crossling/ subdirectories:    {n_raw_dirs}/{len(EXPECTED_LANGUAGES)}")
    print(f"    data/crossling/*/verbs.json:       {n_raw_verbs}/{len(EXPECTED_LANGUAGES)}")
    print(f"    data/crossling/*/classified.json:  {n_raw_classified}/{len(EXPECTED_LANGUAGES)}")

    # Per-language audit
    print(f"\n{'=' * 72}")
    print("  PER-LANGUAGE STATUS")
    print(f"{'=' * 72}")

    header = f"  {'Language':25s} {'crossling':>10s} {'report':>10s} {'cells':>10s} {'raw':>8s} {'n_verbs':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    missing = []
    inconsistent = []
    low_count = []

    for lang in sorted(EXPECTED_LANGUAGES):
        in_cj = lang in (crossling_json or {})
        in_rj = lang in (report_json or {})
        in_cells = lang in (cells_json or {})
        has_raw = raw_data[lang]["classified_json"]

        cj_mark = "yes" if in_cj else "---"
        rj_mark = "yes" if in_rj else "---"
        cells_mark = "yes" if in_cells else "---"
        raw_mark = "yes" if has_raw else "---"

        # Get verb count from best available source
        n_verbs = "?"
        if in_cj:
            n_verbs = str(crossling_json[lang].get("n_verbs", "?"))
        elif in_rj:
            n_verbs = str(report_json[lang].get("n_classified", "?"))

        is_missing = not in_cj and not in_rj and not in_cells
        is_inconsistent = (in_cj != in_cells) or (in_rj and not in_cj)

        flag = ""
        if is_missing:
            flag = " ** MISSING"
            missing.append(lang)
        elif is_inconsistent:
            flag = " * INCONSISTENT"
            inconsistent.append(lang)

        if n_verbs != "?" and int(n_verbs) < LOW_VERB_THRESHOLD:
            flag += f" (LOW: {n_verbs})"
            low_count.append((lang, int(n_verbs)))

        print(f"  {lang:25s} {cj_mark:>10s} {rj_mark:>10s} {cells_mark:>10s} {raw_mark:>8s} {n_verbs:>8s}{flag}")

    # Summary
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")

    total_expected = len(EXPECTED_LANGUAGES)
    total_in_cj = len(crossling_json or {})
    total_in_rj = len(report_json or {})
    total_in_cells = len(cells_json or {})

    print(f"\n  Expected languages (from pipeline):  {total_expected}")
    print(f"  In crossling.json:                   {total_in_cj}")
    print(f"  In crossling_report.json:            {total_in_rj}")
    print(f"  In crossling_27_cells.json:          {total_in_cells}")

    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for lang in missing:
            print(f"    - {lang}")

    if inconsistent:
        print(f"\n  INCONSISTENT ({len(inconsistent)}):")
        for lang in inconsistent:
            in_cj = lang in (crossling_json or {})
            in_rj = lang in (report_json or {})
            in_cells = lang in (cells_json or {})
            present_in = []
            absent_from = []
            for name, present in [("crossling.json", in_cj), ("report", in_rj), ("cells", in_cells)]:
                (present_in if present else absent_from).append(name)
            print(f"    - {lang}: in [{', '.join(present_in)}], missing from [{', '.join(absent_from)}]")

    if low_count:
        print(f"\n  LOW VERB COUNT (< {LOW_VERB_THRESHOLD}):")
        for lang, n in sorted(low_count, key=lambda x: x[1]):
            print(f"    - {lang}: {n} verbs")

    # Verb count summary
    if crossling_json:
        counts = [v.get("n_verbs", 0) for v in crossling_json.values()]
        total_verbs = sum(counts)
        print(f"\n  Total verbs across {total_in_cj} languages: {total_verbs:,}")
        print(f"  Mean verbs per language:             {total_verbs / total_in_cj:.0f}")
        print(f"  Min: {min(counts)} ({[k for k, v in crossling_json.items() if v.get('n_verbs') == min(counts)][0]})")
        print(f"  Max: {max(counts)} ({[k for k, v in crossling_json.items() if v.get('n_verbs') == max(counts)][0]})")

    # Exit code
    if missing or inconsistent:
        print(f"\n  STATUS: INCOMPLETE — {len(missing)} missing, {len(inconsistent)} inconsistent")
        return 1
    else:
        print(f"\n  STATUS: ALL LANGUAGES PRESENT")
        return 0


if __name__ == "__main__":
    sys.exit(main())
