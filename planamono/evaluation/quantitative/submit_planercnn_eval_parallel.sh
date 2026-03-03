#!/bin/bash
# Submit PlaneRCNN evaluation with 10 parallel jobs per dataset.
#
# ScanNet++: 42 matched scenes → 10 shards of ~4-5 scenes each
# Hypersim:  68 matched scenes → 10 shards of ~7 scenes each
#
# After all shards complete, an aggregation job merges shard CSVs
# into the final planercnn_v1/ directory per dataset.
#
# Usage:
#   bash submit_planercnn_eval_parallel.sh                    # Both datasets
#   bash submit_planercnn_eval_parallel.sh scannetpp          # ScanNet++ only
#   bash submit_planercnn_eval_parallel.sh hypersim           # Hypersim only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGS_DIR="/cluster/scratch/aoezkan/planeseg/logs/eval_planercnn_parallel"
ENV_ACTIVATE="source /cluster/scratch/aoezkan/miniconda3/etc/profile.d/conda.sh && conda activate planamono"

mkdir -p "$LOGS_DIR"

N_JOBS=10

# Scene counts: matched scenes in our test split that also have PlaneRCNN H5.
# ScanNet++: 42 test scenes (PlaneRCNN has 50 H5s but 8 are outside our splits)
# Hypersim:  68 test scenes (all have PlaneRCNN H5s)
declare -A SCENE_COUNTS
SCENE_COUNTS[scannetpp]=42
SCENE_COUNTS[hypersim]=68

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(scannetpp hypersim)
fi

echo "============================================"
echo "PlaneRCNN Parallel Evaluation"
echo "  Datasets: ${DATASETS[*]}"
echo "  Jobs per dataset: ${N_JOBS}"
echo "============================================"

ALL_JOB_IDS=()

for DS in "${DATASETS[@]}"; do
    TOTAL=${SCENE_COUNTS[$DS]}
    CHUNK=$(( (TOTAL + N_JOBS - 1) / N_JOBS ))  # ceiling division

    echo ""
    echo "--- ${DS}: ${TOTAL} scenes, ${N_JOBS} shards of ~${CHUNK} ---"

    DS_JOB_IDS=()

    for i in $(seq 0 $((N_JOBS - 1))); do
        START=$((i * CHUNK))
        END=$(( (i + 1) * CHUNK ))
        if [ $END -gt $TOTAL ]; then
            END=$TOTAL
        fi
        if [ $START -ge $TOTAL ]; then
            break
        fi

        JOB_ID=$(sbatch --parsable \
            --job-name="prcnn_${DS}_${i}" \
            --time=8:00:00 \
            --cpus-per-task=16 \
            --mem-per-cpu=8G \
            --output="${LOGS_DIR}/${DS}_shard${i}_%j.out" \
            --error="${LOGS_DIR}/${DS}_shard${i}_%j.err" \
            --wrap="bash -c '${ENV_ACTIVATE} && cd ${SCRIPT_DIR} && python evaluate_planercnn.py --datasets ${DS} --scene-start ${START} --scene-end ${END}'")

        DS_JOB_IDS+=("$JOB_ID")
        ALL_JOB_IDS+=("$JOB_ID")
        echo "  shard ${i} [${START}:${END}): job ${JOB_ID}"
        sleep 0.2
    done

    # Per-dataset merge job: combine shard CSVs into planercnn_v1/
    DEP_STR=$(IFS=:; echo "${DS_JOB_IDS[*]}")
    MERGE_CMD="python -c \"
import pandas as pd
from pathlib import Path
from planamono.evaluation.quantitative.eval_utils import save_results_csv
import ast, csv

eval_root = Path('/cluster/scratch/aoezkan/planeseg/${DS}/eval')
shard_dirs = sorted(eval_root.glob('planercnn_v1_shard_*'))
print(f'Found {len(shard_dirs)} shard directories')

all_rows = []
for sd in shard_dirs:
    csv_path = sd / 'results.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        all_rows.append(df)
        print(f'  {sd.name}: {len(df)} frames')

if not all_rows:
    print('ERROR: no shard results found')
    exit(1)

merged = pd.concat(all_rows, ignore_index=True)
out_dir = eval_root / 'planercnn_v1'
out_dir.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_dir / 'results.csv', index=False)

# Aggregate per-scene and dataset-level
scene_groups = merged.groupby('scene_id')
scene_agg = []
for scene_id, grp in scene_groups:
    row = {'scene_id': scene_id, 'num_frames': len(grp)}
    for col in grp.columns:
        if col in ('scene_id', 'frame_idx'):
            continue
        try:
            row[col + '_mean'] = grp[col].mean()
            row[col + '_std'] = grp[col].std()
        except:
            pass
    scene_agg.append(row)
scene_df = pd.DataFrame(scene_agg)
scene_df.to_csv(out_dir / 'results_per_scene.csv', index=False)

# Dataset-level
ds_row = {'num_scenes': len(scene_df), 'num_frames_total': len(merged)}
for col in merged.columns:
    if col in ('scene_id', 'frame_idx'):
        continue
    try:
        ds_row[col + '_mean'] = merged[col].mean()
        ds_row[col + '_std'] = merged[col].std()
    except:
        pass
pd.DataFrame([ds_row]).to_csv(out_dir / 'results_dataset.csv', index=False)
print(f'Merged: {len(merged)} frames from {len(scene_df)} scenes -> {out_dir}')
\""

    MERGE_JOB_ID=$(sbatch --parsable \
        --job-name="prcnn_merge_${DS}" \
        --time=00:10:00 \
        --cpus-per-task=1 \
        --mem-per-cpu=4G \
        --output="${LOGS_DIR}/${DS}_merge_%j.out" \
        --error="${LOGS_DIR}/${DS}_merge_%j.err" \
        --dependency="afterok:${DEP_STR}" \
        --wrap="bash -c '${ENV_ACTIVATE} && ${MERGE_CMD}'")

    ALL_JOB_IDS+=("$MERGE_JOB_ID")
    echo "  merge: job ${MERGE_JOB_ID} (after ${DEP_STR})"
done

echo ""
echo "============================================"
echo "All jobs submitted: ${#ALL_JOB_IDS[@]} total"
echo "Monitor: squeue -u \$USER | grep prcnn"
echo "============================================"
