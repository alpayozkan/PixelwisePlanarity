#!/bin/bash
# Submit `save_moge_signals_planarity.py` for ScanNet++, NYU-v2, and 7-Scenes
# as 3 independent GPU SLURM jobs. Each writes to a dataset-specific subdir
# under the same OUTPUT_ROOT_BASE, so they cannot collide.
#
# Output layout (same H5 schema across datasets, see save_moge_signals_planarity.py):
#     <OUTPUT_ROOT_BASE>/scannetpp/<scene>/moge_signals.h5      (42 scenes, frame_step=25)
#     <OUTPUT_ROOT_BASE>/nyuv2/nyuv2/moge_signals.h5            (single virtual scene)
#     <OUTPUT_ROOT_BASE>/sevenscenes/<chess|fire|.../>moge_signals.h5
#
# Usage:
#   bash submit_moge_signals_all_datasets.sh
#   MODEL_PATH=/path/to/model.pt bash submit_moge_signals_all_datasets.sh
#   DATASETS="nyuv2 sevenscenes" bash submit_moge_signals_all_datasets.sh   # subset

set -euo pipefail

# ── Configuration (override via env) ─────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/cluster/scratch/ayavuz/moge_all_output_bce_476644/model_epoch2.pt}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep2}"
DATASETS="${DATASETS:-scannetpp nyuv2 sevenscenes}"

# Per-dataset knobs (used only when that dataset is in $DATASETS)
SCANNETPP_SCENES="${SCANNETPP_SCENES:-/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono/splits/scannetpp/test.txt}"
SCANNETPP_FRAME_STEP="${SCANNETPP_FRAME_STEP:-25}"
SCANNETPP_BATCH_SIZE="${SCANNETPP_BATCH_SIZE:-8}"

NYUV2_BATCH_SIZE="${NYUV2_BATCH_SIZE:-16}"
SEVENSCENES_BATCH_SIZE="${SEVENSCENES_BATCH_SIZE:-16}"

NUM_TOKENS="${NUM_TOKENS:-1600}"
RESOLUTION="${RESOLUTION:-480x640}"

# SLURM resources (same per-job for all 3)
TIME="${TIME:-4:00:00}"
CPUS="${CPUS:-8}"
MEM_PER_CPU="${MEM_PER_CPU:-16G}"
GPU="${GPU:-rtx_3090:1}"

PROJECT_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity/planamono"
PY_SCRIPT="$PROJECT_ROOT/inference/planarity/save_moge_signals_planarity.py"
LOG_DIR="/cluster/scratch/aoezkan/planeseg/logs/moge_signals"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT_BASE"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi

echo "================================================================"
echo "  save_moge_signals_planarity — multi-dataset GPU submission"
echo "================================================================"
echo "  model:          $MODEL_PATH"
echo "  output base:    $OUTPUT_ROOT_BASE"
echo "  datasets:       $DATASETS"
echo "  num_tokens:     $NUM_TOKENS    resolution: $RESOLUTION"
echo "  resources:      ${TIME}, ${CPUS} CPU, ${MEM_PER_CPU}/cpu, GPU=${GPU}"
echo "================================================================"

submit_one() {
    local NAME="$1"      # job name suffix
    local CMD="$2"       # python command (single line)
    local OUT_DIR="$3"   # output dir for the job
    mkdir -p "$OUT_DIR"

    local JOB_ID
    JOB_ID=$(/cluster/apps/slurm/bin/sbatch --parsable \
        --time="$TIME" \
        --cpus-per-task="$CPUS" \
        --mem-per-cpu="$MEM_PER_CPU" \
        --gpus="$GPU" \
        --job-name="msig_${NAME}" \
        --output="${LOG_DIR}/moge_signals_${NAME}_%j.out" \
        --error="${LOG_DIR}/moge_signals_${NAME}_%j.err" \
        <<EOF
#!/bin/bash
set -uo pipefail
source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh
conda activate planeseg

echo "==== save_moge_signals — ${NAME} ===="
echo " output: ${OUT_DIR}"
echo ""

set -x
${CMD}
set +x

echo ""
echo "==== ${NAME} done ===="
EOF
    )
    echo "  ${NAME} → job $JOB_ID  (output: $OUT_DIR)"
}

for DS in $DATASETS; do
    case "$DS" in
        scannetpp)
            OUT="$OUTPUT_ROOT_BASE/scannetpp"
            CMD="python $PY_SCRIPT --dataset scannetpp --scenes $SCANNETPP_SCENES --output_root $OUT --model_path $MODEL_PATH --frame_step $SCANNETPP_FRAME_STEP --batch_size $SCANNETPP_BATCH_SIZE --num_tokens $NUM_TOKENS --resolution $RESOLUTION"
            submit_one scannetpp "$CMD" "$OUT"
            ;;
        nyuv2)
            OUT="$OUTPUT_ROOT_BASE/nyuv2"
            CMD="python $PY_SCRIPT --dataset nyuv2 --output_root $OUT --model_path $MODEL_PATH --batch_size $NYUV2_BATCH_SIZE --num_tokens $NUM_TOKENS --resolution $RESOLUTION"
            submit_one nyuv2 "$CMD" "$OUT"
            ;;
        sevenscenes)
            OUT="$OUTPUT_ROOT_BASE/sevenscenes"
            CMD="python $PY_SCRIPT --dataset sevenscenes --output_root $OUT --model_path $MODEL_PATH --batch_size $SEVENSCENES_BATCH_SIZE --num_tokens $NUM_TOKENS --resolution $RESOLUTION"
            submit_one sevenscenes "$CMD" "$OUT"
            ;;
        *)
            echo "  [skip] unknown dataset: $DS" >&2
            continue
            ;;
    esac
done

echo ""
echo "================================================================"
echo "Queue snapshot (jobs named msig_*):"
/cluster/apps/slurm/bin/squeue -u "$USER" -o "%.12i %.12j %.8T %R" 2>/dev/null | grep -E "msig_|JOBID" || true
echo ""
echo "Logs:    $LOG_DIR/moge_signals_*.{out,err}"
echo "Outputs: $OUTPUT_ROOT_BASE/{scannetpp,nyuv2,sevenscenes}/"
echo "Cancel:  scancel --name=msig_scannetpp; scancel --name=msig_nyuv2; scancel --name=msig_sevenscenes"
echo "================================================================"
