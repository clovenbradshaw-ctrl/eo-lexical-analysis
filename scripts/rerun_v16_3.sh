#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# rerun_v16_3.sh — Re-run EO Lexical Analysis with v16.3 corrections
# ═══════════════════════════════════════════════════════════════
#
# This script archives old results and re-runs the full analysis
# pipeline with the corrected DES (Distinction) and ALT (State Change)
# operator definitions from Handbook PATCH v16.3.
#
# Usage:
#   ANTHROPIC_API_KEY=<your-key> OPENAI_API_KEY=<your-key> bash scripts/rerun_v16_3.sh
#
# Or export first:
#   export ANTHROPIC_API_KEY=<your-anthropic-key>
#   export OPENAI_API_KEY=<your-openai-key>
#   bash scripts/rerun_v16_3.sh
#
# Options:
#   --skip-archive    Skip archiving old results
#   --skip-english    Skip English reclassification (steps 1-6)
#   --skip-crossling  Skip cross-linguistic reclassification (steps 7-9)
#   --dry-run         Show what would be done without doing it
#
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Parse arguments ──────────────────────────────────────────
SKIP_ARCHIVE=false
SKIP_ENGLISH=false
SKIP_CROSSLING=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --skip-archive)   SKIP_ARCHIVE=true ;;
        --skip-english)   SKIP_ENGLISH=true ;;
        --skip-crossling) SKIP_CROSSLING=true ;;
        --dry-run)        DRY_RUN=true ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Configuration ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/output"
DATA_DIR="$PROJECT_DIR/data"
ARCHIVE_DIR="$OUTPUT_DIR/pre_v16_3"
CHANGELOG="$PROJECT_DIR/CHANGELOG_v16_3.md"
TIMESTAMP=$(date '+%Y-%m-%d_%H%M%S')
LOG_FILE="$OUTPUT_DIR/rerun_v16_3_${TIMESTAMP}.log"

cd "$PROJECT_DIR"

# ── Helpers ──────────────────────────────────────────────────
log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

run_step() {
    local step_num="$1"
    local step_name="$2"
    shift 2
    local cmd="$*"

    echo ""
    log "═══════════════════════════════════════════════════"
    log "  STEP $step_num: $step_name"
    log "═══════════════════════════════════════════════════"

    if [ "$DRY_RUN" = true ]; then
        log "  [DRY RUN] Would execute: $cmd"
        return 0
    fi

    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        log "  ✓ $step_name — DONE"
        return 0
    else
        log "  ✗ $step_name — FAILED (see $LOG_FILE)"
        return 1
    fi
}

# ── Pre-flight checks ───────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "  EO Lexical Analysis — v16.3 Re-Run Pipeline"
echo "  DES: Designation → Distinction"
echo "  ALT: Frame-Switching → State Change"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check API keys
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "⚠  ANTHROPIC_API_KEY not set."
    echo "   Set it with: export ANTHROPIC_API_KEY=<your-key>"
    echo "   Required for: verb classification (steps 1, 7)"
    if [ "$SKIP_ENGLISH" = true ] && [ "$SKIP_CROSSLING" = true ]; then
        echo "   (Skipping classification steps — proceeding without API key)"
    else
        exit 1
    fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "⚠  OPENAI_API_KEY not set."
    echo "   Set it with: export OPENAI_API_KEY=<your-key>"
    echo "   Required for: embeddings (steps 3, 8)"
    echo "   Proceeding — some steps may fail."
fi

mkdir -p "$OUTPUT_DIR"
touch "$LOG_FILE"
log "Pipeline started at $TIMESTAMP"
log "Project directory: $PROJECT_DIR"

# ═══════════════════════════════════════════════════════════
#  PHASE 0: ARCHIVE OLD RESULTS
# ═══════════════════════════════════════════════════════════

if [ "$SKIP_ARCHIVE" = false ]; then
    log ""
    log "═══════════════════════════════════════════════════"
    log "  PHASE 0: ARCHIVING OLD RESULTS"
    log "═══════════════════════════════════════════════════"

    mkdir -p "$ARCHIVE_DIR"

    # Archive English classification output
    if [ -f "$DATA_DIR/llm_classifications.json" ]; then
        cp "$DATA_DIR/llm_classifications.json" "$ARCHIVE_DIR/llm_classifications.json"
        log "  Archived data/llm_classifications.json"
    fi

    # Archive output files that will be regenerated
    for f in \
        llm_boundaries.json llm_completeness.json llm_confusion.json \
        reembed_report.json topology_report.json topology_summary.txt \
        falsification_report.json sup_rec_audit.json \
        crossling_report.json crossling_27_cells.json crossling_all_axes.json \
        crossling_covariation.json crossling_falsification.json \
        crossling_positions.json crossling_zscores.json \
        phasepost_centroids_meta.json phasepost_embeddings_3d.json \
        third_axis.json influence_report.json taxonomy_comparison.json \
        27_cells.md 27_cells_revised.md; do
        if [ -f "$OUTPUT_DIR/$f" ]; then
            cp "$OUTPUT_DIR/$f" "$ARCHIVE_DIR/$f"
            log "  Archived output/$f"
        fi
    done

    # Archive explore files
    for op in NUL DES INS SEG CON SYN ALT SUP REC; do
        if [ -f "$OUTPUT_DIR/explore_${op}.json" ]; then
            cp "$OUTPUT_DIR/explore_${op}.json" "$ARCHIVE_DIR/explore_${op}.json"
            log "  Archived output/explore_${op}.json"
        fi
    done

    # Archive site data files
    for f in operators.json embeddings.json verbs.json boundaries.json confusion.json metrics.json; do
        if [ -f "$DATA_DIR/$f" ]; then
            cp "$DATA_DIR/$f" "$ARCHIVE_DIR/data_${f}"
            log "  Archived data/$f → pre_v16_3/data_${f}"
        fi
    done

    # Archive crossling classified files (per language)
    if [ -d "$DATA_DIR/crossling" ]; then
        for lang_dir in "$DATA_DIR/crossling"/*/; do
            lang_name=$(basename "$lang_dir")
            if [ "$lang_name" = "index.html" ]; then continue; fi
            if [ -f "$lang_dir/classified.json" ]; then
                mkdir -p "$ARCHIVE_DIR/crossling/$lang_name"
                cp "$lang_dir/classified.json" "$ARCHIVE_DIR/crossling/$lang_name/classified.json"
                log "  Archived crossling/$lang_name/classified.json"
            fi
        done
    fi

    log "  ✓ Archive complete → output/pre_v16_3/"
else
    log "  Skipping archive (--skip-archive)"
fi


# ═══════════════════════════════════════════════════════════
#  PHASE 1: ENGLISH RE-CLASSIFICATION
# ═══════════════════════════════════════════════════════════

if [ "$SKIP_ENGLISH" = false ]; then

    # Remove old classification to force re-run
    if [ "$DRY_RUN" = false ] && [ -f "$DATA_DIR/llm_classifications.json" ]; then
        rm "$DATA_DIR/llm_classifications.json"
        log "  Removed old data/llm_classifications.json to trigger reclassification"
    fi

    # Step 1: Reclassify English verbs with updated DES/ALT definitions
    run_step 1 "Reclassify English verbs (04_llm_classify.py)" \
        "python scripts/04_llm_classify.py --backend anthropic"

    # Step 2: Analyze LLM classifications
    run_step 2 "Analyze classifications (05_analyze_llm.py)" \
        "python scripts/05_analyze_llm.py"

    # Step 3: Re-embed with updated classifications
    run_step 3 "Re-embed verbs (06_reembed.py)" \
        "python scripts/06_reembed.py --backend openai"

    # Step 4: Re-explore operator boundaries
    run_step 4 "Explore operator boundaries (07_explore.py)" \
        "python scripts/07_explore.py"

    # Step 5: Topology analysis
    run_step 5 "Topology analysis (08_topology.py)" \
        "python scripts/08_topology.py"

    # Step 6: Falsification tests
    run_step 6 "Falsification tests (11_falsify.py)" \
        "python scripts/11_falsify.py"

else
    log "  Skipping English re-classification (--skip-english)"
fi


# ═══════════════════════════════════════════════════════════
#  PHASE 2: CROSS-LINGUISTIC RE-CLASSIFICATION
# ═══════════════════════════════════════════════════════════

if [ "$SKIP_CROSSLING" = false ]; then

    # Remove old crossling classified files to force re-classification
    if [ "$DRY_RUN" = false ] && [ -d "$DATA_DIR/crossling" ]; then
        for lang_dir in "$DATA_DIR/crossling"/*/; do
            lang_name=$(basename "$lang_dir")
            if [ "$lang_name" = "index.html" ]; then continue; fi
            if [ -f "$lang_dir/classified.json" ]; then
                rm "$lang_dir/classified.json"
                log "  Removed old crossling/$lang_name/classified.json"
            fi
        done
    fi

    # Step 7: Re-classify all languages
    run_step 7 "Cross-linguistic classification (09_crossling.py --phase 2)" \
        "python scripts/09_crossling.py --phase 2"

    # Step 8: Cross-linguistic analysis
    run_step 8 "Cross-linguistic analysis (09_crossling.py --phase 3)" \
        "python scripts/09_crossling.py --phase 3"

    # Step 9: Extended crossling analysis
    run_step 9a "Crossling detailed analysis (09b_crossling_analysis.py)" \
        "python scripts/09b_crossling_analysis.py" || true

    run_step 9b "Crossling covariation (09c_covariation.py)" \
        "python scripts/09c_covariation.py" || true

    run_step 9c "Crossling embeddings (09d_embed.py)" \
        "python scripts/09d_embed.py" || true

    run_step 9d "Crossling positions (09e_positions.py)" \
        "python scripts/09e_positions.py" || true

    # Step 10: Phasepost and third axis
    run_step 10a "Phasepost classifier (09f_phasepost_classifier.py)" \
        "python scripts/09f_phasepost_classifier.py" || true

    run_step 10b "Third axis analysis (09f_third_axis.py)" \
        "python scripts/09f_third_axis.py" || true

    run_step 10c "Crossling referent analysis (09f_crossling_referent.py)" \
        "python scripts/09f_crossling_referent.py" || true

    run_step 10d "Revised cells (09f_revised_cells.py)" \
        "python scripts/09f_revised_cells.py" || true

    # Step 11: SUP/REC audit with updated definitions
    # Remove old audit files first
    if [ "$DRY_RUN" = false ]; then
        for f in "$DATA_DIR/crossling/English/operator_confidence.json" \
                 "$DATA_DIR/crossling/English/sup_rec_reclass.json"; do
            [ -f "$f" ] && rm "$f" && log "  Removed old $(basename "$f")"
        done
    fi

    run_step 11 "SUP/REC audit (09g_sup_rec_audit.py)" \
        "python scripts/09g_sup_rec_audit.py" || true

else
    log "  Skipping cross-linguistic re-classification (--skip-crossling)"
fi


# ═══════════════════════════════════════════════════════════
#  PHASE 3: ADDITIONAL ANALYSES
# ═══════════════════════════════════════════════════════════

run_step 12 "Recursive analysis (10_recursive.py)" \
    "python scripts/10_recursive.py" || true

run_step 13 "K-sweep (10b_ksweep_wide.py)" \
    "python scripts/10b_ksweep_wide.py" || true

run_step 14 "Influence analysis (10c_influence.py)" \
    "python scripts/10c_influence.py" || true

run_step 15 "Taxonomy comparison (12_taxonomy.py)" \
    "python scripts/12_taxonomy.py" || true


# ═══════════════════════════════════════════════════════════
#  PHASE 4: REBUILD SITE DATA
# ═══════════════════════════════════════════════════════════

run_step 16 "Rebuild site data (build_site_data.py)" \
    "python scripts/build_site_data.py"


# ═══════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════

echo ""
log "═══════════════════════════════════════════════════════"
log "  PIPELINE COMPLETE"
log "═══════════════════════════════════════════════════════"
log ""
log "  Old results archived in: output/pre_v16_3/"
log "  New results in:          output/"
log "  Full log:                $LOG_FILE"
log ""
log "  Next steps:"
log "    1. Compare old vs new DES/ALT counts"
log "    2. Update CHANGELOG_v16_3.md with results"
log "    3. Review verb migrations between operators"
log ""

# Quick comparison if both old and new exist
if [ -f "$ARCHIVE_DIR/data_operators.json" ] && [ -f "$DATA_DIR/operators.json" ]; then
    log "  Quick comparison (old → new operator counts):"
    python3 -c "
import json
with open('$ARCHIVE_DIR/data_operators.json') as f:
    old = json.load(f)
with open('$DATA_DIR/operators.json') as f:
    new = json.load(f)

old_counts = {op: d.get('count', d.get('verb_count', '?')) for op, d in old.items()} if isinstance(old, dict) else {}
new_counts = {op: d.get('count', d.get('verb_count', '?')) for op, d in new.items()} if isinstance(new, dict) else {}

for op in ['NUL','DES','INS','SEG','CON','SYN','ALT','SUP','REC']:
    o = old_counts.get(op, '?')
    n = new_counts.get(op, '?')
    marker = ' ←' if o != n else ''
    print(f'    {op}: {o} → {n}{marker}')
" 2>/dev/null || log "  (Could not generate comparison — check manually)"
fi

log ""
log "Done at $(date '+%Y-%m-%d %H:%M:%S')"
