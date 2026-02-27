#!/usr/bin/env python3
"""Dump verbs assigned to each of the 27 cells: Position × Triad × Referent"""

import json, os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling", "English")

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

# Load referent classifications
with open(os.path.join(DATA, "referent_axis.json")) as f:
    referents = json.load(f)

ref_map = {}
for r in referents:
    v = r.get('verb', '').strip()
    ref = r.get('referent', '').upper().strip()
    op = r.get('operator', '').upper().strip()
    gloss = r.get('gloss', '').strip()
    if v and ref in ['FIGURE','PATTERN','GROUND'] and op in HELIX:
        ref_map[v] = {'referent': ref, 'operator': op, 'gloss': gloss}

# Build 27 cells
cells = defaultdict(list)
for verb, info in sorted(ref_map.items()):
    op = info['operator']
    pos = POSITIONS[op]
    tri = TRIADS[op]
    ref = info['referent']
    cell_key = (pos, tri, ref)
    cells[cell_key].append((verb, op, info['gloss']))

# Print
POS_ORDER = ['Differentiate', 'Relate', 'Generate']
TRI_ORDER = ['Existence', 'Structure', 'Interpretation']
REF_ORDER = ['FIGURE', 'PATTERN', 'GROUND']

lines = []
lines.append("# The 27 Cells: Position × Triad × Referent\n")
lines.append("Each verb is listed with its EO operator in brackets.\n")

for pos in POS_ORDER:
    for tri in TRI_ORDER:
        for ref in REF_ORDER:
            key = (pos, tri, ref)
            verbs = cells.get(key, [])
            label = f"{pos} × {tri} × {ref}"
            lines.append(f"\n## {label} ({len(verbs)} verbs)\n")
            if not verbs:
                lines.append("*(empty)*\n")
            else:
                for verb, op, gloss in sorted(verbs):
                    g = f" — {gloss}" if gloss else ""
                    lines.append(f"- {verb} [{op}]{g}")
                lines.append("")

output = "\n".join(lines)
out_path = os.path.join(BASE, "output", "27_cells.md")
with open(out_path, 'w') as f:
    f.write(output)
print(f"✓ Wrote {out_path}")
print(f"  {len(ref_map)} verbs across {len([k for k in cells if cells[k]])} populated cells")
