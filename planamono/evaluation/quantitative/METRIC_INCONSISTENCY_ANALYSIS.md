# Metric Calculation Inconsistency Analysis

## Overview

This document analyzes the inconsistency between metric calculations in two scripts:
- `evaluate_all_baselines.py` - Main evaluation script for computing quantitative metrics
- `visualize_scannetpp_all_baselines.py` - Visualization script for qualitative inspection

Both scripts are intended to evaluate plane segmentation quality, but they compute precision/recall differently.

---

## Background: How Plane Metrics Work

### The Goal
Given a predicted plane segmentation, measure how well each predicted segment fits an actual 3D plane.

### The Pipeline
1. **Backproject** depth image to 3D world coordinates
2. **Fit a plane** to each predicted segment using RANSAC
3. **Count inliers** - points within a distance threshold of the fitted plane
4. **Compute metrics** - precision (inliers / segment points) and recall (inliers / all points)

### Key Parameters
- **RANSAC distance threshold**: Used during plane fitting to determine which points "vote" for a plane hypothesis
- **Evaluation distance threshold**: Used after fitting to count how many points are within X distance of the plane
- **Inlier ratio gate**: Minimum fraction of inliers required to consider a segment "valid" (filters out poor fits)

---

## Script 1: `evaluate_all_baselines.py`

### Configuration
```python
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
THRESHOLDS = (0.001, 0.005, 0.01)  # 0.1cm, 0.5cm, 1.0cm
```

### Call Chain
```
evaluate_all_baselines.py
  └── evaluate_method()
        └── evaluate_single_frame()  [eval_utils.py:201]
              └── compute_plane_metrics()  [eval_utils.py:314]
                    └── fit_planes_and_evaluate_multi_threshold()  [metrics.py:131]
                          ├── fit_planes_per_label_v1(distance_threshold=0.02)  # RANSAC @ 2cm
                          └── compute_inliers_at_threshold(threshold=0.001/0.005/0.01)  # Evaluate @ 0.1/0.5/1.0cm
```

### How It Works

#### Step 1: Fit planes with RANSAC (base_threshold = 2cm)
```python
# metrics.py:157-164
results, df = fit_planes_per_label_v1(
    pts_world,
    labels,
    ignore_labels=(0,),
    distance_threshold=base_threshold,  # HARDCODED: 0.02 (2cm)
    num_iterations=num_iterations,
    min_support=min_support
)
```

This calls Open3D's `segment_plane()` with 2cm threshold:
```python
# planefit.py:242-246
plane_model, inliers_local = pcd.segment_plane(
    distance_threshold=distance_threshold,  # 2cm
    ransac_n=ransac_n,
    num_iterations=num_iterations
)
```

**Output**: Plane parameters `(a, b, c, d)` where `ax + by + cz + d = 0`

#### Step 2: Extract plane parameters
```python
# metrics.py:169-173
plane_params = {}
for pid, data in results.items():
    if "plane_model_refined" in data:
        plane_params[pid] = data["plane_model_refined"]
```

#### Step 3: Evaluate at each threshold (0.1cm, 0.5cm, 1.0cm)
```python
# metrics.py:178-183
for thr in thresholds:
    metrics[thr] = compute_inliers_at_threshold(
        pts_world, labels, plane_params, thr, inlier_ratio_gate
    )
```

#### Step 4: Count inliers at evaluation threshold
```python
# metrics.py:108-127
for pid, params in plane_params.items():
    mask = (labels == pid)
    pts_plane = pts_world[mask]
    n_pts = pts_plane.shape[0]

    a, b, c, d = params
    distances = np.abs(pts_plane @ np.array([a, b, c]) + d)
    n_inliers = np.sum(distances < threshold)  # Uses EVALUATION threshold (0.1/0.5/1.0cm)

    # Quality gate applied at EVALUATION threshold
    if n_inliers / n_pts >= inlier_ratio_gate:
        total_inliers += n_inliers
        total_points += n_pts

precision = total_inliers / total_points
recall = total_inliers / len(labels)
```

### Summary for evaluate_all_baselines.py
| Stage | Threshold Used |
|-------|----------------|
| RANSAC plane fitting | Fixed 2cm (`base_threshold=0.02`) |
| Inlier counting | Variable (0.1cm, 0.5cm, 1.0cm) |
| Quality gate check | Variable (0.1cm, 0.5cm, 1.0cm) |

---

## Script 2: `visualize_scannetpp_all_baselines.py`

### Configuration
```python
# Imports from evaluate_all_baselines.py
from planamono.evaluation.quantitative.evaluate_all_baselines import (
    THRESHOLDS,          # (0.001, 0.005, 0.01)
    INLIER_RATIO_GATE,   # 0.9
    RANSAC_ITERATIONS,   # 200
)

INLIER_RATIO_THRESHOLD = INLIER_RATIO_GATE  # 0.9
```

### Call Chain
```
visualize_scannetpp_all_baselines.py
  └── main()
        └── for distance_threshold in THRESHOLDS:  # Loops over 0.1cm, 0.5cm, 1.0cm
              └── visualize_frame()
                    └── compute_inlier_mask(distance_threshold=0.001/0.005/0.01)
                          ├── fit_planes_per_label_v1(distance_threshold=0.001/0.005/0.01)  # RANSAC @ evaluation threshold
                          └── mark_planes_below_threshold_as_outliers()
```

### How It Works

#### Step 1: Loop over evaluation thresholds
```python
# visualize_scannetpp_all_baselines.py:602
for distance_threshold in THRESHOLDS:  # 0.001, 0.005, 0.01
    ...
    frame_result = visualize_frame(
        ...
        distance_threshold=distance_threshold,  # Passed to compute_inlier_mask
    )
```

#### Step 2: Fit planes with RANSAC using EVALUATION threshold
```python
# visualize_scannetpp_all_baselines.py:135-142
results, df = fit_planes_per_label_v1(
    pts_world,
    labels,
    ignore_labels=(0,),
    distance_threshold=distance_threshold,  # VARIABLE: 0.1cm, 0.5cm, or 1.0cm
    num_iterations=RANSAC_ITERATIONS,
    min_support=100
)
```

#### Step 3: Apply quality gate based on RANSAC results
```python
# visualize_scannetpp_all_baselines.py:149-152
results, df = mark_planes_below_threshold_as_outliers(
    results, df, inlier_ratio_threshold  # Uses inlier_ratio from RANSAC fit
)
```

#### Step 4: Use RANSAC's inlier count directly
```python
# visualize_scannetpp_all_baselines.py:169-172
total_predicted = df["num_points"].sum()
total_inliers = df["refined_inlier_num_points"].sum()  # From RANSAC, NOT recounted
precision = total_inliers / total_predicted
recall = total_inliers / pts_world.shape[0]
```

### Summary for visualize_scannetpp_all_baselines.py
| Stage | Threshold Used |
|-------|----------------|
| RANSAC plane fitting | Variable (0.1cm, 0.5cm, 1.0cm) |
| Inlier counting | Same as RANSAC (from DataFrame) |
| Quality gate check | Based on RANSAC's inlier ratio |

---

## The Inconsistencies

### Inconsistency 1: RANSAC Threshold

| Script | RANSAC Threshold |
|--------|------------------|
| `evaluate_all_baselines.py` | Fixed 2cm for all evaluations |
| `visualize_scannetpp_all_baselines.py` | Variable (matches evaluation threshold) |

**Impact**: Different plane equations `(a, b, c, d)` may be found.

- RANSAC @ 2cm: More points vote → plane influenced by points up to 2cm away
- RANSAC @ 0.1cm: Fewer points vote → plane fits tighter to core points, but may fail if noise > 0.1cm

### Inconsistency 2: Inlier Counting Method

| Script | How Inliers Are Counted |
|--------|------------------------|
| `evaluate_all_baselines.py` | Recounted fresh at each evaluation threshold |
| `visualize_scannetpp_all_baselines.py` | Read from DataFrame (`refined_inlier_num_points`) |

**Impact**:
- `evaluate_all_baselines.py`: Inlier count varies by evaluation threshold
- `visualize_scannetpp_all_baselines.py`: Inlier count is fixed per RANSAC run

### Inconsistency 3: Quality Gate Application

| Script | When Gate Is Applied |
|--------|---------------------|
| `evaluate_all_baselines.py` | At each evaluation threshold (re-checked) |
| `visualize_scannetpp_all_baselines.py` | Once, based on RANSAC inlier ratio |

**Impact**: A segment might pass the gate in one script but fail in the other.

Example:
- Segment has 80% inliers at 2cm (RANSAC), but only 70% at 0.1cm (evaluation)
- `evaluate_all_baselines.py`: Segment fails gate at 0.1cm (70% < 90%)
- `visualize_scannetpp_all_baselines.py`: Depends on which threshold RANSAC used

---

## Concrete Example

Consider a plane segment with 1000 points:

### evaluate_all_baselines.py
```
1. RANSAC @ 2cm → finds plane (a, b, c, d)
2. At 0.1cm threshold:
   - Recount: 600 points within 0.1cm
   - Inlier ratio: 600/1000 = 60% < 90% gate → REJECTED
3. At 0.5cm threshold:
   - Recount: 850 points within 0.5cm
   - Inlier ratio: 850/1000 = 85% < 90% gate → REJECTED
4. At 1.0cm threshold:
   - Recount: 950 points within 1.0cm
   - Inlier ratio: 950/1000 = 95% ≥ 90% gate → ACCEPTED
   - Precision contribution: 950/950 = 100%
```

### visualize_scannetpp_all_baselines.py
```
1. At 0.1cm threshold:
   - RANSAC @ 0.1cm → finds plane, 600 inliers
   - Inlier ratio: 60% < 90% gate → REJECTED

2. At 0.5cm threshold:
   - RANSAC @ 0.5cm → finds plane, 850 inliers (possibly DIFFERENT plane equation)
   - Inlier ratio: 85% < 90% gate → REJECTED

3. At 1.0cm threshold:
   - RANSAC @ 1.0cm → finds plane, 920 inliers (possibly DIFFERENT plane equation)
   - Inlier ratio: 92% ≥ 90% gate → ACCEPTED
   - Precision contribution: 920/1000 = 92%
```

**Result**: Even when both accept the segment, precision differs (100% vs 92%) because:
1. Different plane equations were found
2. Inlier counting method differs

---

## Implications

### For Paper/Publication
If metrics from both scripts are reported or compared, they are **not directly comparable**.

### For Debugging
Visualizations may show different inlier patterns than what the evaluation metrics reflect.

### For Threshold Selection
The 2cm base threshold in `evaluate_all_baselines.py` is hardcoded and not configurable via command line.

---

## Recommendations

### Option A: Make visualize consistent with evaluate
Modify `visualize_scannetpp_all_baselines.py` to:
1. Use fixed 2cm RANSAC threshold
2. Recount inliers at evaluation threshold
3. Apply gate at evaluation threshold

```python
# In compute_inlier_mask(), change:
results, df = fit_planes_per_label_v1(
    pts_world, labels,
    distance_threshold=0.02,  # Fixed 2cm, like evaluate
    ...
)

# Then recount inliers at evaluation threshold
for pid, data in results.items():
    a, b, c, d = data["plane_model_refined"]
    distances = np.abs(pts_plane @ np.array([a, b, c]) + d)
    n_inliers = np.sum(distances < evaluation_threshold)
    ...
```

### Option B: Make evaluate consistent with visualize
Modify `evaluate_all_baselines.py` to use evaluation threshold for RANSAC:
```python
# In eval_utils.py compute_plane_metrics(), change:
base_threshold=0.02  →  base_threshold=threshold  # Use evaluation threshold
```

### Option C: Make base_threshold configurable
Add `BASE_THRESHOLD` to configuration and use consistently in both scripts.

---

## File Locations

| File | Path |
|------|------|
| evaluate_all_baselines.py | `planamono/evaluation/quantitative/evaluate_all_baselines.py` |
| visualize_scannetpp_all_baselines.py | `planamono/evaluation/quantitative/visualize_scannetpp_all_baselines.py` |
| eval_utils.py | `planamono/evaluation/quantitative/eval_utils.py` |
| metrics.py | `planamono/shared/plane_fitting/metrics.py` |
| planefit.py | `planamono/shared/plane_fitting/planefit.py` |

---

## Summary Table (Original Inconsistency)

| Aspect | evaluate_all_baselines.py (old) | visualize_scannetpp_all_baselines.py |
|--------|---------------------------|--------------------------------------|
| RANSAC threshold | Fixed 2cm | Variable (0.1/0.5/1.0cm) |
| Plane equation | One per segment | Different per threshold |
| Inlier counting | Recounted at eval threshold | From RANSAC DataFrame |
| Gate application | Per eval threshold | Per RANSAC run |
| RANSAC runs per frame | 1 | 3 (one per threshold) |

---

## Resolution: v1 Versions

The inconsistency has been resolved by creating v1 versions that use a **consistent approach**:
- RANSAC threshold = evaluation threshold (same threshold for fitting and evaluation)
- Inliers are recounted at the evaluation threshold
- Quality gate is applied based on recounted inlier ratio

### What Was Changed

#### 1. `eval_utils.py` - New Functions Added

**`compute_plane_metrics_v1()`** (line 356+):
- Uses evaluation threshold for RANSAC fitting
- Recounts inliers at the same threshold
- Each threshold gets its own plane fit

**`evaluate_single_frame_v1()`** (line 436+):
- Wrapper that uses `compute_plane_metrics_v1`

```python
# Key difference from original:
for thr in thresholds:
    # Fit planes with RANSAC at this threshold
    fit_results, df = fit_planes_per_label_v1(
        pts_world, labels,
        distance_threshold=thr,  # Use evaluation threshold for RANSAC
        ...
    )
    # Evaluate at the same threshold
    metrics = compute_inliers_at_threshold(
        pts_world, labels, plane_params, thr, inlier_ratio_gate
    )
```

#### 2. `evaluate_all_baselines.py` - Updated

Now imports and uses `evaluate_single_frame_v1`:
```python
from planamono.evaluation.quantitative.eval_utils import (
    ...
    evaluate_single_frame_v1,
)

# In eval_frame_wrapper:
return evaluate_single_frame_v1(...)
```

#### 3. `visualize_scannetpp_all_baselines_v1.py` - New File

New visualization script with `compute_inlier_mask_v1()` that is consistent with evaluation:
- RANSAC at evaluation threshold
- Recounts inliers at same threshold
- Applies gate based on recounted inlier ratio

### Summary Table (After Fix)

| Aspect | evaluate_all_baselines.py (v1) | visualize_scannetpp_all_baselines_v1.py |
|--------|-------------------------------|----------------------------------------|
| RANSAC threshold | Variable (0.1/0.5/1.0cm) | Variable (0.1/0.5/1.0cm) |
| Plane equation | Different per threshold | Different per threshold |
| Inlier counting | Recounted at eval threshold | Recounted at eval threshold |
| Gate application | Per eval threshold | Per eval threshold |
| RANSAC runs per frame | 3 (one per threshold) | 3 (one per threshold) |

### Version Comparison

| Function | Original | v1 |
|----------|----------|-----|
| `compute_plane_metrics` | RANSAC @ 2cm fixed | - |
| `compute_plane_metrics_v1` | - | RANSAC @ eval threshold |
| `evaluate_single_frame` | Uses `compute_plane_metrics` | - |
| `evaluate_single_frame_v1` | - | Uses `compute_plane_metrics_v1` |
| `compute_inlier_mask` | Uses `mark_planes_below_threshold_as_outliers` | - |
| `compute_inlier_mask_v1` | - | Recounts inliers, applies gate consistently |

### Which Version to Use

**Use v1 versions when:**
- You want to measure "What is the precision/recall when planes are fit at threshold X?"
- You want visualization and evaluation to be consistent
- You want a single, interpretable threshold meaning

**Use original versions when:**
- You want to measure "Given a robustly-fit plane (at 2cm), how precise is it at stricter thresholds?"
- You need backward compatibility with previous results

### File Locations (Updated)

| File | Path | Notes |
|------|------|-------|
| evaluate_all_baselines.py | `planamono/evaluation/quantitative/evaluate_all_baselines.py` | Now uses v1 |
| visualize_scannetpp_all_baselines.py | `planamono/evaluation/quantitative/visualize_scannetpp_all_baselines.py` | Original (inconsistent) |
| visualize_scannetpp_all_baselines_v1.py | `planamono/evaluation/quantitative/visualize_scannetpp_all_baselines_v1.py` | **New**: Consistent with evaluation |
| eval_utils.py | `planamono/evaluation/quantitative/eval_utils.py` | Contains both original and v1 functions |
| metrics.py | `planamono/shared/plane_fitting/metrics.py` | Unchanged |
| planefit.py | `planamono/shared/plane_fitting/planefit.py` | Unchanged |
