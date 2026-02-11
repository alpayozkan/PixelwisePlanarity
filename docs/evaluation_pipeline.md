# Metric Evaluation Pipeline Analysis

## Overview

The metric evaluation system computes **2D segmentation metrics** (SC, RI, VOI) and **3D geometric metrics** (Precision/Recall @ distance thresholds) for plane segmentation predictions.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

Input Sources:
├── Ground Truth Data (ScanNetPPPlaneDataset)
│   ├── RGB images: /cluster/project/cvg/Shared_datasets/scannet++/data
│   ├── GT plane labels: /cluster/scratch/aoezkan/planeseg/dataset/scannetpp
│   ├── GT depth maps: (from rendered planes)
│   └── Camera intrinsics/poses: K, c2w
│
└── Prediction Data (H5 files)
    └── /cluster/scratch/aoezkan/planeseg/scannetpp/inference/<method>_h5/
        └── <scene_id>/planes.h5
            ├── planes: (N_frames, H, W) uint16 - predicted plane labels
            └── frame_ids: (N_frames,) - frame identifiers

                                ↓

┌─────────────────────────────────────────────────────────────────────┐
│                      BATCH PROCESSING                                │
└─────────────────────────────────────────────────────────────────────┘

DataLoader (batch_size=32)
    ↓
For each batch:
    1. Load GT: plane labels, depth, intrinsics, poses
    2. Load predictions from H5 (LazyH5SceneLoader - one scene in memory)
    3. Resize predictions to match GT resolution (INTER_NEAREST)
    4. Apply label offset (e.g., ZeroPlane: +1)

                                ↓

┌─────────────────────────────────────────────────────────────────────┐
│                   PARALLEL FRAME EVALUATION                          │
│                   (joblib, N_JOBS=16, backend=loky)                  │
└─────────────────────────────────────────────────────────────────────┘

For each frame (evaluate_single_frame_v1):

    1. SEGMENTATION METRICS (image-to-image, no 3D data)
       ├── Rand Index (sklearn.metrics.rand_score)
       ├── Variation of Information (skimage.metrics)
       └── Segmentation Covering (segmentation_covering_fast)

    2. 3D PLANE METRICS (if COMPUTE_PLANE_METRICS=True)
       ├── Backproject depth to 3D world coordinates
       │   └── backproject_v1(depth, K, c2w, labels) → pts_world, pt_labels
       │
       ├── For each evaluation threshold (e.g., 0.001, 0.005, 0.01 meters):
       │   ├── RANSAC plane fitting @ threshold
       │   │   └── fit_planes_per_label_v1(pts_world, labels, threshold, 200 iters)
       │   │       └── Returns: {segment_id: plane_params (a,b,c,d)}
       │   │
       │   └── Compute Precision/Recall
       │       └── compute_inliers_at_threshold(pts_world, labels, plane_params, threshold)
       │           ├── For each segment:
       │           │   ├── Count points within threshold of fitted plane
       │           │   └── Apply inlier_ratio_gate (0.9): segments with <90% inliers contribute 0
       │           └── Return: {precision, recall}
       │               ├── precision = total_inliers / total_predicted_points
       │               └── recall = total_inliers / all_scene_points
       │
       └── Returns: {prec@0.1cm, rec@0.1cm, prec@0.5cm, rec@0.5cm, ...}

                                ↓

┌─────────────────────────────────────────────────────────────────────┐
│                        RESULTS AGGREGATION                           │
└─────────────────────────────────────────────────────────────────────┘

Output: /cluster/scratch/aoezkan/planeseg/scannetpp/eval/<exp_name>/

├── results.csv                    # Per-frame metrics
│   Columns: scene_id, frame_idx, rand_index, voi, sc,
│            prec@0.1cm, rec@0.1cm, prec@0.5cm, rec@0.5cm, prec@1.0cm, rec@1.0cm
│
├── results_per_scene.csv          # Per-scene aggregated (mean of frames)
│   Columns: scene_id, num_frames, <all metrics>_mean
│
├── results_dataset.csv            # Dataset-level summary (mean across scenes)
│   Columns: num_scenes, num_frames_total, <metric>_mean, <metric>_std
│
└── runtime_breakdown.csv          # Profiling data
    Columns: stage, time_seconds, time_hms, calls, avg_ms
```

---

## Key Scripts and Their Roles

### 1. `evaluate_all_baselines.py` (Unified Evaluator)

**Purpose**: Evaluate multiple methods from pre-saved H5 predictions.

**Input**:
- H5 predictions: `/cluster/scratch/aoezkan/planeseg/scannetpp/inference/<method>_h5/`
- GT dataset: `ScanNetPPPlaneDataset` with split="val"

**Configuration**:
```python
METHODS = {
    "ours": {"h5_folder": "moge_ours_v2_h5", "label_offset": 0},
    "zeroplane": {"h5_folder": "zeroplane_h5", "label_offset": 1},
    "gtseg": {"h5_folder": "gtseg_v1_h5", "label_offset": 0},
    # ... more methods
}
THRESHOLDS = (0.001, 0.005, 0.01)  # 1mm, 5mm, 10mm
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
```

**Output**:
- Per-method CSVs in `/cluster/scratch/aoezkan/planeseg/scannetpp/eval/<exp_name>/`
- Aggregated tables:
  - `table_precision_recall_baselines.csv`
  - `table_segmentation_baselines.csv`
  - `table_combined_baselines.csv`

**Usage**:
```bash
# Evaluate all methods
python evaluate_all_baselines.py

# Evaluate specific methods
python evaluate_all_baselines.py --methods ours zeroplane

# Only aggregate existing results
python evaluate_all_baselines.py --aggregate-only
```

---

### 2. `evaluate_scannetpp_fast.py` (Full Pipeline)

**Purpose**: Run full inference + evaluation pipeline for "our method" (MoGe planarity + segmentation).

**Pipeline**:
1. **Batch GPU Inference** (BATCH_SIZE=32):
   - MoGe 4-head inference: planarity, depth, normals
   - Vectorized segmentation: `compute_vectorized_planar_segments_v4`
   - Outputs: predicted plane labels

2. **Parallel CPU Evaluation** (N_JOBS=16):
   - Per-frame metric computation
   - RANSAC plane fitting at multiple thresholds

3. **Save Results**:
   - H5 predictions: `/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_ours_v1_h5/`
   - CSV metrics: `/cluster/scratch/aoezkan/planeseg/scannetpp/eval/moge_ours_v1/`

**Key Differences from evaluate_all_baselines.py**:
- Runs inference (not just evaluation)
- Only evaluates a single method
- Saves H5 predictions for later use

---

### 3. `eval_utils.py` (Shared Utilities)

**Key Functions**:

| Function | Purpose | Returns |
|----------|---------|---------|
| `evaluate_single_frame_v1()` | Evaluate one frame | `(metrics_dict, labels)` |
| `compute_clustering_metrics()` | SC, RI, VOI | `{rand_index, voi, sc}` |
| `compute_plane_metrics_v1()` | Precision/Recall @ thresholds | `{prec@Xcm, rec@Xcm}` |
| `save_results_csv()` | Write 3 CSV files | `(df_frames, df_scenes, df_dataset)` |
| `save_predictions_h5()` | Write H5 predictions | None |
| `Timer` | Profiling infrastructure | timing stats |

**LazyH5SceneLoader**:
- Memory-efficient: only loads one scene at a time
- O(1) frame lookup via index dictionary
- Handles label offsets and resizing

---

## Data Flow Details

### Input: Ground Truth Dataset

**ScanNetPPPlaneDataset** returns:
```python
{
    "rgb_path": str,                    # Path to RGB image
    "scene_id": str,                    # Scene identifier
    "frame_idx": str,                   # Frame identifier
    "plane": torch.Tensor (1, H, W),    # GT plane labels
    "depth": torch.Tensor (1, H, W),    # GT depth map (meters)
    "K": torch.Tensor (3, 3),           # Intrinsics
    "c2w": torch.Tensor (4, 4),         # Camera pose
}
```

**Source paths** (configured in dataset):
- RGB: `/cluster/project/cvg/Shared_datasets/scannet++/data/<scene_id>/dslr/resized_images_1296/<frame>.JPG`
- Planes/Depth: `/cluster/scratch/aoezkan/planeseg/dataset/scannetpp/<scene_id>/planes.h5`
- Split: `planamono/splits/scannetpp/val.txt` (42 scenes, ~14,439 frames)

---

### Input: Prediction H5 Files

**Structure**: `<h5_root>/<scene_id>/planes.h5`
```python
with h5py.File(h5_path) as f:
    planes = f["planes"][:]        # (N_frames, H, W) uint16
    frame_ids = f["frame_ids"][:]  # (N_frames,) bytes
```

**H5 Root Locations** (from `METHODS` in evaluate_all_baselines.py):
```
/cluster/scratch/aoezkan/planeseg/scannetpp/inference/
├── moge_ours_v2_h5/         # Our full pipeline
├── zeroplane_h5/            # ZeroPlane baseline
├── gtseg_v1_h5/             # GT segmentation (upper bound)
├── gtplanarity_ourseg_h5/   # GT planarity + our seg
└── ourplanarity_gtseg_h5/   # Our planarity + GT seg
```

---

### Output: Evaluation Results

**Directory**: `/cluster/scratch/aoezkan/planeseg/scannetpp/eval/<exp_name>/`

**Files**:

1. **results.csv** (per-frame, ~14,439 rows):
   ```csv
   scene_id,frame_idx,rand_index,voi,sc,prec@0.1cm,rec@0.1cm,prec@0.5cm,...
   0a5c013435,000010,0.89,1.23,0.76,0.34,0.29,0.68,0.62,...
   ```

2. **results_per_scene.csv** (per-scene aggregated, ~42 rows):
   ```csv
   scene_id,num_frames,rand_index,voi,sc,prec@0.1cm,...
   0a5c013435,344,0.87,1.45,0.73,0.32,...
   ```

3. **results_dataset.csv** (dataset summary, 1 row):
   ```csv
   num_scenes,num_frames_total,rand_index_mean,rand_index_std,voi_mean,...
   42,14439,0.85,0.12,1.38,...
   ```

4. **runtime_breakdown.csv**:
   ```csv
   stage,time_seconds,time_hms,calls,avg_ms
   evaluation_pipeline,1234.56,00:20:34.560,1,1234560
   _gpu_inference,234.12,00:03:54.120,451,518.9
   ```

---

## Metric Computation Details

### 2D Segmentation Metrics

**Computed by**: `compute_clustering_metrics()` in [eval_utils.py:275-311](eval_utils.py#L275-L311)

| Metric | Function | Source | Range | Better |
|--------|----------|--------|-------|--------|
| Rand Index (RI) | `rand_score()` | sklearn.metrics | [0, 1] | Higher |
| Variation of Information (VOI) | `variation_of_information()` | skimage.metrics | [0, ∞) | Lower |
| Segmentation Covering (SC) | `segmentation_covering_fast()` | planamono.shared.plane_fitting.metrics | [0, 1] | Higher |

**No 3D data required** - pure image-to-image comparison.

---

### 3D Plane Metrics (Precision/Recall)

**Computed by**: `compute_plane_metrics_v1()` in [eval_utils.py:356-433](eval_utils.py#L356-L433)

**Algorithm**:
1. Backproject depth to 3D: `pts_world, pt_labels = backproject_v1(depth, K, c2w, labels)`
2. For each threshold τ (e.g., 0.001, 0.005, 0.01 meters):
   - **RANSAC Plane Fitting** @ threshold τ:
     - `fit_planes_per_label_v1(pts_world, labels, threshold=τ, num_iterations=200)`
     - Returns: `{segment_id: (a, b, c, d)}` plane parameters

   - **Inlier Counting**:
     - For each segment: count points within τ of fitted plane
     - Apply **inlier ratio gate** (0.9):
       - If `inliers/total_points >= 0.9`: count inliers
       - Else: count 0 inliers (but points still in denominator)

   - **Metrics**:
     - `precision = total_inliers / total_predicted_points`
     - `recall = total_inliers / all_scene_points`

**Critical**: The `_v1` version uses **threshold-consistent RANSAC** (RANSAC threshold = evaluation threshold). The original version used fixed 2cm RANSAC for all thresholds.

---

## v1 vs Original Metric Computation

**Key Difference**: See [METRIC_INCONSISTENCY_ANALYSIS.md](../planamono/evaluation/quantitative/METRIC_INCONSISTENCY_ANALYSIS.md)

| Version | RANSAC Threshold | Evaluation Threshold | Use Case |
|---------|------------------|---------------------|----------|
| **Original** (`compute_plane_metrics`) | Fixed 2cm | Variable (0.1, 0.5, 1.0 cm) | "How precise is a robustly-fit plane at stricter thresholds?" |
| **v1** (`compute_plane_metrics_v1`) | = Evaluation threshold | Same | "How well can planes be fit AND evaluated at threshold X?" |

**Current scripts use v1** for consistency between visualization and evaluation.

---

## Performance Optimizations

### 1. Batch GPU Inference
- **BATCH_SIZE=32**: Process 32 frames at once on GPU
- **Speedup**: ~10x vs sequential inference

### 2. Lazy H5 Loading
- **Memory**: Only one scene in memory at a time
- **Lookup**: O(1) frame access via index dictionary

### 3. Parallel CPU Evaluation
- **N_JOBS=16**: joblib parallel evaluation
- **Backend**: `loky` (safer than threading/multiprocessing)
- **Speedup**: ~10x vs sequential

### 4. Fast Metric Functions
- `segmentation_covering_fast()`: Vectorized contingency matrix (~10x speedup)
- `backproject_v2()`: Optimized backprojection (~10x speedup)
- `compute_vectorized_planar_segments_v5()`: GPU-accelerated segmentation (~3-5x speedup)

### 5. Reduced RANSAC Iterations
- **200 iterations** (sufficient for evaluation)
- vs 2000 in original PlaneRCNN implementation

---

## Example: Running Full Evaluation

```bash
# 1. Generate predictions for all baselines (if not already done)
cd planamono/evaluation/quantitative

# Run our method (inference + evaluation)
python evaluate_scannetpp_fast.py

# Run ZeroPlane baseline (assumes predictions exist)
python evaluate_scannetpp_zeroplane_fast.py

# Run GT segmentation upper bound
python evaluate_scannetpp_gtseg_fast.py

# 2. Unified evaluation from H5 predictions
python evaluate_all_baselines.py --methods ours zeroplane gtseg

# 3. Generate aggregated tables
python evaluate_all_baselines.py --aggregate-only

# Output tables:
# - table_precision_recall_baselines.csv
# - table_segmentation_baselines.csv
# - table_combined_baselines.csv
```

---

## Configuration Parameters

### Global Settings (evaluate_all_baselines.py)

```python
COMPUTE_PLANE_METRICS = True       # Set False to skip 3D metrics (faster)
RANSAC_ITERATIONS = 200            # Sufficient for evaluation
INLIER_RATIO_GATE = 0.9            # Segments with <90% inliers contribute 0
THRESHOLDS = (0.001, 0.005, 0.01)  # Distance thresholds in meters
BATCH_SIZE = 32                    # DataLoader batch size
N_JOBS = min(16, os.cpu_count())   # Parallel workers
```

### Method-Specific Settings

**Label Offset**: Some methods don't use label 0 for background
- `ours`, `gtseg`, `gtplanarity_ourseg`: offset = 0
- `zeroplane`: offset = 1 (add 1 to all labels)

---

## Common Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| Missing H5 files | Predictions not generated | Run method-specific evaluation script first |
| Shape mismatch | GT and pred different resolutions | H5 loader handles resizing with INTER_NEAREST |
| Low precision | Inlier ratio gate too strict | Reduce INLIER_RATIO_GATE (default 0.9) |
| Slow evaluation | Too many workers/large batch | Reduce N_JOBS or BATCH_SIZE |
| OOM errors | Loading all scenes at once | Use LazyH5SceneLoader (already default) |
| Metric inconsistency | Using v1 eval with original viz | Use `visualize_scannetpp_all_baselines_v1.py` |

---

## Related Files

| File | Purpose |
|------|---------|
| [METRICS.md](../planamono/evaluation/quantitative/METRICS.md) | Detailed metric definitions with formulas |
| [METRIC_INCONSISTENCY_ANALYSIS.md](../planamono/evaluation/quantitative/METRIC_INCONSISTENCY_ANALYSIS.md) | v1 vs original differences |
| [metrics.py](../planamono/shared/plane_fitting/metrics.py) | Core metric implementations |
| [planefit.py](../planamono/shared/plane_fitting/planefit.py) | RANSAC plane fitting |
| [scannetpp_dataset.py](../planamono/shared/datasets/scannetpp.py) | Dataset loader |
