# Evaluation Scripts

Shell scripts for running plane segmentation evaluation.

## Single Method Evaluation

```bash
./run_evaluation.sh <method> <split> <output_dir>
```

**Example:**
```bash
# Evaluate MoGe on test set
./run_evaluation.sh moge test ./results/moge_test
```

**Methods:**
- `gt` - Ground truth baseline (upper bound)
- `moge` - MoGe monocular depth/normal prediction
- `planercnn` - PlaneRCNN baseline
- `monoplane` - MonoPlane baseline

**Splits:**
- `train` - Training set
- `val` - Validation set
- `test` - Test set

## Batch Evaluation

Evaluate all methods at once:

```bash
./batch_evaluate.sh test ./results
```

This will:
1. Evaluate GT, MoGe, PlaneRCNN, MonoPlane
2. Save results to `./results/<method>_test/`
3. Generate CSV files with metrics
4. Create visualization outputs

## Metrics Computed

### Plane Fitting Metrics
- **Precision@1cm/2cm/5cm** - Fraction of predicted points that are inliers
- **Recall@1cm/2cm/5cm** - Fraction of GT points explained
- **Inlier Ratio** - Per-plane quality measure

### Segmentation Metrics
- **Rand Index** - Clustering quality
- **Variation of Information (VOI)** - Segmentation consistency
- **Segmentation Covering (SC)** - GT coverage

### Per-Scene Results
CSV with columns:
- `scene_id`, `frame_id`
- `precision@1cm`, `precision@2cm`, `precision@5cm`
- `recall@1cm`, `recall@2cm`, `recall@5cm`
- `rand_index`, `voi`, `seg_covering`
- `num_planes_pred`, `num_planes_gt`

## Configuration

### Environment Variables

Set before running:

```bash
# MoGe model path
export MOGE_MODEL_PATH=/path/to/moge/checkpoint.pth
export MOGE_MODEL_SIZE=large  # or base, small

# Dataset paths
export SCANNETPP_ROOT=/path/to/scannetpp/data
export PLANE_GT_ROOT=/path/to/plane/ground/truth
```

### SLURM Configuration

Scripts include SLURM directives (commented). Uncomment for cluster:

```bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1  # For MoGe inference
```

## Output Format

```
results/
├── moge_test/
│   ├── metrics.csv          # Per-scene metrics
│   ├── summary.txt          # Aggregated statistics
│   └── visualizations/      # Optional qualitative results
├── planercnn_test/
│   └── ...
└── gt_test/
    └── ...
```

## Qualitative Visualization

Generate side-by-side comparisons:

```bash
python ../qualitative/visualize_comparison.py \
    --results_dir ./results \
    --output_dir ./visualizations \
    --split test
```

Creates comparison videos and images.

## Troubleshooting

**GPU Out of Memory (MoGe):**
```bash
# Reduce batch size or image resolution
python ../run_evaluation.py --method moge --batch_size 1 --img_size 512
```

**Missing predictions:**
```bash
# Check if inference completed successfully
ls results/moge_test/*.npy | wc -l
```

**Slow evaluation:**
```bash
# Run on subset first
head -10 test_scenes.txt > quick_test.txt
./run_evaluation.sh moge quick_test ./quick_results
```
