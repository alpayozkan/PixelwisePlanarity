#!/bin/bash
# Re-run the 4-step postprocessing chain
#   1. evaluate_gt_moge_zeroplane_benchmark.py --aggregate_only
#   2. make_benchmark_summary.py
#   3. prettify_benchmark_summary.py
#   4. make_benchmark_latex.py
# for one or more benchmark EXP dirs, with the union of all methods that
# actually have per-scene CSVs in each dir.
#
# Use this when you've stacked multiple `sbatch submit_eval_gt_moge_zp_
# benchmark.sh` invocations into the same EXP (e.g. moge_ep2 then metric3d
# under gt_moge_zp_benchmark_kunscaled_new). Each submission's built-in
# aggregator only knows about its own --methods and overwrites summary.csv,
# so the final cross-method tables are missing rows.
#
# Auto-discovery:
#   - EXPs: any subdir of $EVAL_ROOT matching gt_moge_zp_benchmark*
#   - Methods: any subdir of <exp>/ that contains at least one
#     <method>/<dataset>/<scene>/results.csv (i.e. real per-scene output)
#   - --kscaled / --rivoisc_ver: parsed from the EXP name suffix
#       _kscaled_*    → --kscaled
#       _kunscaled_*  → --no-kscaled
#       *_new         → --rivoisc_ver new
#       *_old         → --rivoisc_ver old
#     EXPs without these suffixes are skipped (legacy layout).
#
# Usage:
#   bash aggregate_all_benchmark_exps.sh                   # all auto-discovered EXPs
#   bash aggregate_all_benchmark_exps.sh EXP1 EXP2 ...     # explicit EXP names
#   DRY_RUN=1 bash aggregate_all_benchmark_exps.sh         # print commands, don't run
#   EVAL_ROOT=/path/to/eval bash aggregate_all_benchmark_exps.sh

set -euo pipefail

EVAL_ROOT="${EVAL_ROOT:-/cluster/scratch/aoezkan/planeseg/eval}"
PROJECT_ROOT="${PROJECT_ROOT:-/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono}"
DRY_RUN="${DRY_RUN:-0}"
DATASETS=(${DATASETS:-scannetpp nyuv2 sevenscenes})

EVAL_PY="$PROJECT_ROOT/evaluation/quantitative/evaluate_gt_moge_zeroplane_benchmark.py"
SUMMARY_PY="$PROJECT_ROOT/evaluation/quantitative/make_benchmark_summary.py"
PRETTIFY_PY="$PROJECT_ROOT/evaluation/quantitative/prettify_benchmark_summary.py"
LATEX_PY="$PROJECT_ROOT/evaluation/quantitative/make_benchmark_latex.py"

run() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '  [DRY] %s\n' "$*"
    else
        "$@"
    fi
}

discover_methods() {
    # echo space-separated method names under $1 that have at least one
    # results.csv at depth method/dataset/scene/results.csv
    local exp_dir="$1"
    local m
    for m in "$exp_dir"/*/; do
        [[ -d "$m" ]] || continue
        local name=$(basename "$m")
        # any per-scene results.csv anywhere inside?
        if compgen -G "$m"*/*/results.csv > /dev/null; then
            printf '%s\n' "$name"
        fi
    done
}

parse_flags_from_exp() {
    # echo "kscaled_flag rivoisc_ver" e.g. "--kscaled new" or "--no-kscaled old"
    # exit 1 if EXP name doesn't carry both suffixes (legacy layout)
    local exp="$1"
    local kflag="" rivoisc=""
    case "$exp" in
        *_kscaled_*)   kflag="--kscaled" ;;
        *_kunscaled_*) kflag="--no-kscaled" ;;
        *) return 1 ;;
    esac
    case "$exp" in
        *_new) rivoisc="new" ;;
        *_old) rivoisc="old" ;;
        *) return 1 ;;
    esac
    printf '%s %s\n' "$kflag" "$rivoisc"
}

# ── pick EXPs ─────────────────────────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    EXPS=("$@")
else
    EXPS=()
    for d in "$EVAL_ROOT"/gt_moge_zp_benchmark*/; do
        [[ -d "$d" ]] || continue
        EXPS+=("$(basename "$d")")
    done
fi

if [[ ${#EXPS[@]} -eq 0 ]]; then
    echo "No EXP dirs found under $EVAL_ROOT (pattern: gt_moge_zp_benchmark*)" >&2
    exit 1
fi

# ── per-EXP loop ──────────────────────────────────────────────────────────
SKIPPED=()
PROCESSED=()
for EXP in "${EXPS[@]}"; do
    EXP_DIR="$EVAL_ROOT/$EXP"
    echo "================================================================"
    echo "  EXP: $EXP"
    echo "  dir: $EXP_DIR"

    if [[ ! -d "$EXP_DIR" ]]; then
        echo "  → SKIP (dir does not exist)"
        SKIPPED+=("$EXP (no dir)")
        continue
    fi

    if ! flags=$(parse_flags_from_exp "$EXP"); then
        echo "  → SKIP (legacy EXP name without _kscaled/_kunscaled and _new/_old suffixes)"
        SKIPPED+=("$EXP (legacy name)")
        continue
    fi
    KFLAG=$(echo "$flags" | awk '{print $1}')
    RIVOISC=$(echo "$flags" | awk '{print $2}')

    METHODS_LIST=$(discover_methods "$EXP_DIR")
    if [[ -z "$METHODS_LIST" ]]; then
        echo "  → SKIP (no method/dataset/scene/results.csv under $EXP_DIR)"
        SKIPPED+=("$EXP (no method results)")
        continue
    fi
    # turn into a flat space-separated string
    METHODS=$(echo "$METHODS_LIST" | tr '\n' ' ')

    echo "  flags: $KFLAG --rivoisc_ver $RIVOISC"
    echo "  methods (auto-discovered): $METHODS"
    echo "  datasets: ${DATASETS[*]}"
    echo "----------------------------------------------------------------"

    echo "[1/4] aggregate_only"
    run python "$EVAL_PY" \
        --exp "$EXP" \
        --methods $METHODS \
        --datasets "${DATASETS[@]}" \
        --eval_root "$EVAL_ROOT" \
        $KFLAG \
        --rivoisc_ver "$RIVOISC" \
        --aggregate_only

    echo "[2/4] make_benchmark_summary.py"
    run python "$SUMMARY_PY" --exp_dir "$EXP_DIR"

    echo "[3/4] prettify_benchmark_summary.py"
    run python "$PRETTIFY_PY" \
        --input  "$EXP_DIR/summary.csv" \
        --output "$EXP_DIR/summary_pretty.xlsx"

    echo "[4/4] make_benchmark_latex.py"
    run python "$LATEX_PY" --xlsx "$EXP_DIR/summary_complete_picked.xlsx"

    PROCESSED+=("$EXP")
done

echo "================================================================"
echo "Done."
echo "  processed (${#PROCESSED[@]}): ${PROCESSED[*]:-none}"
echo "  skipped   (${#SKIPPED[@]}):"
for s in "${SKIPPED[@]}"; do echo "    - $s"; done
