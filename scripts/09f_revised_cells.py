#!/usr/bin/env python3
"""Generate updated 27-cell verb list with SUP/REC audit reclassifications applied."""

import json, os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "crossling", "English")
OUT  = os.path.join(BASE, "output")

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

POS_ORDER = ['Differentiate', 'Relate', 'Generate']
TRI_ORDER = ['Existence', 'Structure', 'Interpretation']
REF_ORDER = ['FIGURE', 'PATTERN', 'GROUND']

# Reclassifications from audit
RECLASS = {
    # NUL → SUP
    'hide': 'SUP', 'ignore': 'SUP', 'deny': 'SUP', 'censor': 'SUP',
    'pause': 'SUP', 'suppress': 'SUP', 'overlook': 'SUP', 'interrupt': 'SUP',
    'neglect': 'SUP', 'repress': 'SUP', 'blackline': 'SUP', 'unreport': 'SUP',
    'stifle': 'SUP', 'disregard': 'SUP',
    # ALT → SUP
    'range': 'SUP', 'suspend': 'SUP', 'accommodate': 'SUP', 'vary': 'SUP',
    # INS → REC
    'establish': 'REC', 'study': 'REC', 'design': 'REC', 'implement': 'REC',
    'revive': 'REC', 'reestablish': 'REC', 'foster': 'REC', 'incubate': 'REC',
}

# Load referent classifications
with open(os.path.join(DATA, "referent_axis.json")) as f:
    referents = json.load(f)

# Build verb list with reclassifications applied
cells = defaultdict(list)
verb_count = 0
reclass_count = 0

for r in referents:
    verb = r.get('verb', '').strip()
    ref = r.get('referent', '').upper().strip()
    op = r.get('operator', '').upper().strip()
    gloss = r.get('gloss', '').strip()

    if not verb or ref not in REF_ORDER or op not in HELIX:
        continue

    # Apply reclassification
    original_op = op
    if verb in RECLASS:
        op = RECLASS[verb]
        reclass_count += 1

    pos = POSITIONS[op]
    tri = TRIADS[op]
    cell_key = (pos, tri, ref)

    reclassed = " ⟵ was " + original_op if verb in RECLASS else ""
    cells[cell_key].append((verb, op, gloss, reclassed))
    verb_count += 1

# Generate markdown
lines = []
lines.append("# The 27 Cells: Position × Triad × Referent")
lines.append(f"\n*{verb_count} English verbs. {reclass_count} reclassified after SUP/REC audit.*")
lines.append("\nEach verb is listed with its EO operator. Reclassified verbs marked with ⟵.\n")

# Summary table
lines.append("## Distribution Summary\n")
lines.append(f"| {'Cell':50s} | {'n':>5s} |")
lines.append(f"|{'-'*51}|{'-'*6}:|")

total_by_ref = defaultdict(int)
total_by_pos = defaultdict(int)
total_by_tri = defaultdict(int)

for pos in POS_ORDER:
    for tri in TRI_ORDER:
        for ref in REF_ORDER:
            key = (pos, tri, ref)
            n = len(cells.get(key, []))
            label = f"{pos} × {tri} × {ref}"
            lines.append(f"| {label:50s} | {n:5d} |")
            total_by_ref[ref] += n
            total_by_pos[pos] += n
            total_by_tri[tri] += n

lines.append(f"|{'':51s}|{'':6s}|")
lines.append(f"| **Total** | **{verb_count}** |")
lines.append("")
lines.append(f"**By referent:** FIGURE={total_by_ref['FIGURE']}, PATTERN={total_by_ref['PATTERN']}, GROUND={total_by_ref['GROUND']}")
lines.append(f"**By position:** Differentiate={total_by_pos['Differentiate']}, Relate={total_by_pos['Relate']}, Generate={total_by_pos['Generate']}")
lines.append(f"**By triad:** Existence={total_by_tri['Existence']}, Structure={total_by_tri['Structure']}, Interpretation={total_by_tri['Interpretation']}")

# Operator counts after reclassification
op_counts = defaultdict(int)
for key, verbs in cells.items():
    for v, op, g, rc in verbs:
        op_counts[op] += 1
lines.append("\n**Operator counts (post-audit):**")
for op in HELIX:
    lines.append(f"  {op}: {op_counts.get(op, 0)} ({100*op_counts.get(op,0)/verb_count:.1f}%)")

# Verb lists
lines.append("\n---\n")

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
                for verb, op, gloss, reclassed in sorted(verbs):
                    g = f" — {gloss}" if gloss else ""
                    lines.append(f"- {verb} [{op}]{g}{reclassed}")
                lines.append("")

output = "\n".join(lines)
out_path = os.path.join(OUT, "27_cells_revised.md")
with open(out_path, 'w') as f:
    f.write(output)
print(f"✓ Wrote {out_path}")
print(f"  {verb_count} verbs, {reclass_count} reclassified, across {len([k for k in cells if cells[k]])} populated cells")

# Count empty cells
empty = [(pos, tri, ref) for pos in POS_ORDER for tri in TRI_ORDER for ref in REF_ORDER
         if not cells.get((pos, tri, ref), [])]
if empty:
    print(f"\n  Empty cells ({len(empty)}):")
    for pos, tri, ref in empty:
        print(f"    {pos} × {tri} × {ref}")
