#!/bin/bash
# Submit Metric3D evaluation jobs to SLURM.
#
# Shards evaluation across SLURM array jobs, one job per scene shard.
# After all jobs finish, run with --aggregate-only to merge results.
#
# Usage:
#   # Quick test (no SLURM, 2 scenes):
#   conda activate planeseg && python evaluate_metric3d.py --max-scenes 2
#
#   # Full evaluation (single job, no sharding):
#   sbatch --wrap="conda activate planeseg && python evaluate_metric3d.py --datasets scannetpp" ...
#
#   # Sharded evaluation (recommended for large datasets):
#   bash submit_metric3d_eval_jobs.sh scannetpp  # 42 scenes → 6 shards of 8
#   bash submit_metric3d_eval_jobs.sh hypersim   # 68 scenes → 7 shards of 10
#
# After all jobs finish:
#   python evaluate_metric3d.py --aggregate-only

set -euo pipefail

DATASET="${1:-scannetpp}"
REPO_ROOT="/cluster/home/aoezkan/planeseg/PixelwisePlanarity"
SCRIPT="$REPO_ROOT/planamono/evaluation/quantitative/evaluate_metric3d.py"
LOGS_DIR="$REPO_ROOT/logs"
mkdir -p "$LOGS_DIR"

# Shard sizes (tune to keep each job under 2h)
# ScanNet++: 42 scenes total → ~8 scenes/shard → 6 shards
# Hypersim:  68 scenes total → ~10 scenes/shard → 7 shards
if [ "$DATASET" == "scannetpp" ]; then
    TOTAL_SCENES=42
    SHARD_SIZE=8
elif [ "$DATASET" == "hypersim" ]; then
    TOTAL_SCENES=68
    SHARD_SIZE=10
else
    echo "Unknown dataset: $DATASET (use 'scannetpp' or 'hypersim')"
    exit 1
fi

echo "Submitting $DATASET evaluation: $TOTAL_SCENES scenes, shard_size=$SHARD_SIZE"

SCENE_START=0
JOB_IDX=0
while [ "$SCENE_START" -lt "$TOTAL_SCENES" ]; do
    SCENE_END=$((SCENE_START + SHARD_SIZE))
    if [ "$SCENE_END" -gt "$TOTAL_SCENES" ]; then
        SCENE_END=$TOTAL_SCENES
    fi

    sbatch \
        --job-name="metric3d_${DATASET}_${JOB_IDX}" \
        --output="$LOGS_DIR/metric3d_${DATASET}_${JOB_IDX}_%j.out" \
        --time=4:00:00 \
        --cpus-per-task=16 \
        --mem-per-cpu=4G \
        --wrap="
            source /cluster/home/aoezkan/.bashrc
            conda activate planeseg
            cd $REPO_ROOT
            python $SCRIPT \
                --datasets $DATASET \
                --scene-start $SCENE_START \
                --scene-end $SCENE_END
        "

    echo "  Submitted shard [$SCENE_START:$SCENE_END)"
    SCENE_START=$SCENE_END
    JOB_IDX=$((JOB_IDX + 1))
done

echo "Submitted $JOB_IDX jobs for $DATASET."
echo ""
echo "After all jobs finish, merge shards with:"
echo "  python $SCRIPT --datasets $DATASET --aggregate-only"
