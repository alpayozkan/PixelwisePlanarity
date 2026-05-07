# Plane-Segmentation Evaluation Metrics (`metrics_planes.py`)

Lives at: `planamono/shared/plane_fitting/metrics_planes.py`

Pixel-aligned evaluation metrics for plane segmentation. Three metric
families plus helpers, all in one module:

| Family | Functions | What it measures |
|---|---|---|
| Segmentation         | `compute_segmentation_metrics`                | Label-map agreement: RI, VOI, SC |
| Plane recall         | `plane_recall_at_depth`, `plane_recall_at_normal` | Per-GT-plane recall under depth/normal thresholds |
| Direct param error   | `per_plane_error_stats`                       | Per-plane normal-angle and offset error (mean/median/std) |
| Helpers              | `compute_gt_normals_from_depth_labels`, `match_planes_by_overlap`, `aggregate_plane_normals`, `aggregate_plane_depths` | GT-normal recovery + matching + per-plane reductions |
| One-shot driver      | `evaluate_plane_predictions`                  | Runs everything, returns a flat dict |

The module is the missing piece between the per-pixel inference H5 outputs
(`labels`, `depth`, `normal`) and a CSV row of metrics. Reuses
`segmentation_covering_fast` from `metrics.py` and depends on
`sklearn.metrics.rand_score` + `skimage.metrics.variation_of_information`
(already used by `eval_utils.py:303–304`, no new deps).

## When to use this vs. the existing eval utilities

| Need | Use |
|---|---|
| 2D label-map metrics only (RI/VOI/SC) | `eval_utils.compute_clustering_metrics` (already exists) or this module's `compute_segmentation_metrics` |
| 3D point-cloud precision/recall at distance thresholds (with RANSAC) | `metrics.fit_planes_and_evaluate_multi_threshold` (existing, faster — point cloud already built) |
| Plane-level recall comparing **mean** predicted vs GT depth/normal per plane | **this module** — `plane_recall_at_depth`, `plane_recall_at_normal` |
| Per-plane angle/offset error summaries | **this module** — `per_plane_error_stats` |
| End-to-end CSV row from inference H5 outputs | **this module** — `evaluate_plane_predictions` |

This module's plane-recall is **per-plane**, not per-pixel — matched in
label space and compared on aggregated statistics. The existing
`metrics.compute_inliers_at_threshold` is **per-point** (every pixel of
every fitted plane gets an inlier/outlier vote). Different question, both
useful. See "Glossary" at the bottom.

## Inputs

```
depth_pred / depth_gt        (H, W)      float32/64    meters, positive, finite
normals_pred / normals_gt    (H, W, 3)   float32/64    unit vectors
labels_pred / labels_gt      (H, W)      int           0 = background, 1+ = plane IDs
K                            (3, 3)      float64       camera intrinsics
                                                       (only required if normals_gt is not provided)
```

If you don't have `normals_gt`, pass `K` and the module fits one normal per
GT plane via SVD on backprojected points (the recipe from your spec).

## Output

`evaluate_plane_predictions` returns a flat dict with keys:

```
rand_index, voi, sc                                    segmentation
plane_recall_d_<mm>mm                                  one per depth threshold (m → mm)
plane_recall_n_<deg>deg                                one per normal threshold
normal_err_deg_{mean,median,std,n}                     per-plane normal angle stats
offset_err_m_{mean,median,std,n}                       per-plane depth offset stats
n_gt_planes, n_pred_planes                             counts (background-excluded)
```

The individual functions return their own narrower dicts (see signatures).

## Plane matching

`match_planes_by_overlap` does a simple argmax-overlap match: for each
non-background GT label, find the predicted label with the highest pixel
count overlap (excluding background labels on either side). One match per
GT plane; predicted planes can be matched to multiple GT planes
(no Hungarian / no exclusivity).

This matches the user-spec recipe (and it's what most plane benchmarks do
informally). For competitive papers a Hungarian assignment based on IoU is
common — easy to swap in by replacing this function only.

## API

### Segmentation metrics

```python
from planamono.shared.plane_fitting import compute_segmentation_metrics
m = compute_segmentation_metrics(labels_gt, labels_pred)
# {"rand_index": 0.91, "voi": 1.43, "sc": 0.78}
```

Both inputs must be `(H, W)` of the same shape. No background filtering is
done — pass pre-filtered maps if you want to ignore label 0.

### Plane recall @ depth

```python
from planamono.shared.plane_fitting import plane_recall_at_depth
r = plane_recall_at_depth(
    depth_pred, depth_gt, labels_pred, labels_gt, threshold=0.05,  # 5 cm
)
# {"recall": 0.62, "n_total": 21, "n_matched": 18, "n_within": 13}
```

`recall = n_within / n_total`. `n_total` is GT-plane count after
`ignore_labels_gt`; `n_matched` is GT planes that found a pred match;
`n_within` is matched GT planes whose pred mean-depth is within threshold.
Strict inequality (`<`).

### Plane recall @ normal

```python
from planamono.shared.plane_fitting import plane_recall_at_normal
r = plane_recall_at_normal(
    normals_pred, normals_gt, labels_pred, labels_gt, threshold_deg=5.0,
)
```

Sign-agnostic via `|dot|` (so a flipped normal is still considered a match).
Threshold is on the angle in degrees; internally compares
`|n̂_pred · n̂_gt| ≥ cos(threshold_deg)`.

### Per-plane error statistics

```python
from planamono.shared.plane_fitting import per_plane_error_stats
s = per_plane_error_stats(
    depth_pred, depth_gt, normals_pred, normals_gt,
    labels_pred, labels_gt,
)
# normal_err_deg_mean / _median / _std / _n
# offset_err_m_mean   / _median / _std / _n
```

Errors are computed only on matched GT planes that have valid pred
aggregates on the corresponding side.

### GT normals from depth + labels (when you don't have GT normals)

```python
from planamono.shared.plane_fitting import compute_gt_normals_from_depth_labels
normals_gt = compute_gt_normals_from_depth_labels(
    depth_gt, labels_gt, K, ignore_labels=(0,), orient_positive_z=True,
)
# (H, W, 3); pixels outside any plane are zero
```

For each non-background GT label, backprojects pixels via `K`, runs SVD on
the centered point cloud, takes the smallest right singular vector as the
plane normal, then broadcasts that normal to every pixel of the label.
Disable `orient_positive_z` if your camera/normal convention differs.

### One-shot driver

```python
from planamono.shared.plane_fitting import evaluate_plane_predictions

row = evaluate_plane_predictions(
    depth_pred, normals_pred, labels_pred,
    depth_gt, labels_gt,
    normals_gt=None,                 # auto-computed via SVD if K provided
    K=K,
    depth_thresholds_m=(0.05, 0.1, 0.2),
    normal_thresholds_deg=(5.0, 10.0, 20.0),
)
# row is a flat dict — drop straight into a pandas DataFrame:
import pandas as pd
df = pd.DataFrame([row])
```

## Empirical example

Synthetic 80×80 frame with two planes side-by-side:
- plane 1: clean
- plane 2: predicted depth offset by **+5 cm**
- normals_pred for plane 1 tilted by **7°** about the X axis

```
                 rand_index : 1.0      ← labels are identical
                        voi : 0.0
                         sc : 1.0
        plane_recall_d_10mm : 0.5      ← only plane 1 within 10 mm
        plane_recall_d_50mm : 0.5      ← 50 mm threshold strict; plane 2 = 50 mm
       plane_recall_d_100mm : 1.0
        plane_recall_n_5deg : 0.5      ← plane 1 is 7° off
       plane_recall_n_10deg : 1.0
       plane_recall_n_20deg : 1.0
        normal_err_deg_mean : 3.50     ← (0 + 7) / 2
       normal_err_deg_median: 3.50
          normal_err_deg_std: 3.50
            normal_err_deg_n: 2
          offset_err_m_mean : 0.025    ← (0 + 0.05) / 2
        offset_err_m_median : 0.025
           offset_err_m_std : 0.025
             offset_err_m_n : 2
                n_gt_planes : 2
              n_pred_planes : 2
```

Numbers move as expected for each threshold; recall is monotone non-decreasing
in the threshold within each family.

## Compatibility with existing repo outputs

| Source | Shape | Direct input to this module? |
|---|---|---|
| `inference_to_h5.py` planes.h5 → `f['plane_labels'][cam_idx]` | `(H, W)` int | ✓ as `labels_pred` / `labels_gt` |
| `save_moge_raw.py` → `f['depth'][cam_idx]` | `(H, W)` float | ✓ as `depth_pred` |
| `save_moge_raw.py` → `f['normal'][cam_idx]` | `(H, W, 3)` float | ✓ as `normals_pred` |
| ScanNet++ / Hypersim dataset GT → `depth`, `plane`, `K` | per `DATASETS.md` | ✓ as `depth_gt`, `labels_gt`, `K` |
| `compute_plane_params(...)` output | `Dict[int, (4,)]` | not direct — see below |

To use plane-parameter outputs from `compute_plane_params` with this module,
render them to per-pixel maps first: for each `pid → (a,b,c,d)`, set
`normals_out[mask] = (a,b,c)` and `depth_out[mask] = -d / (a·(u-cx)/fx + b·(v-cy)/fy + c)`.
That's not currently a helper — say the word and it gets one.

## Caveats

1. **Affine vs. metric depth.** If `depth_pred` came from
   `save_moge_raw.py` without `--metric_depth`, it is affine and `d` lives
   in affine units. Offset errors will be biased per-frame by `1/metric_scale`
   (this scene's effective error threshold drifts with the frame).
   See `compare_metric_depth.py` and the analysis in
   `compare_metric_depth.md` (if/when written) — or run with metric depth.

2. **Argmax-overlap matching is not Hungarian.** Two GT planes can match
   the same pred plane. For comparison against benchmarks that use IoU
   bipartite matching, swap `match_planes_by_overlap` for a Hungarian
   match (e.g. `scipy.optimize.linear_sum_assignment` on `-iou_matrix`).

3. **`compute_gt_normals_from_depth_labels` orientation default
   (`orient_positive_z=True`)** matches the user-spec pseudocode (flip if
   `n[2] < 0`). Some camera conventions want the opposite (normals pointing
   *at* the camera, i.e. `n[2] < 0` for points with `z > 0`). Toggle via
   the flag if your downstream code expects camera-facing normals.

4. **Mean depth is not a plane offset.** This module compares
   `mean(depth)` per plane mask, which is only a proxy for the true plane
   `d`. The proxy is exact for fronto-parallel planes and drifts with
   tilt/perspective. For strict plane-offset metrics, fit `[a,b,c,d]` per
   plane (e.g. via `compute_plane_params`) and compare `d` directly.

## Glossary — per-pixel vs. per-plane recall

| | Per-pixel (existing `metrics.py`) | Per-plane (this module) |
|---|---|---|
| Question | Of all predicted pixels, how many are within τ of their fitted plane? | Of all GT planes, how many have a matched pred plane within τ? |
| Numerator | Sum of inlier *pixels* across all planes | Count of *planes* that pass |
| Denominator | All pixels (or all GT pixels for recall) | All GT planes |
| Used in | `evaluate_all_baselines.py` (P@1mm, R@1mm, etc.) | NEW — paper-style "plane recall @ depth/normal" |
| Speed | RANSAC + multi-threshold (3× speedup helper exists) | One-pass, no fitting |

Both are valid; they answer different questions and tell different stories
about the same predictions.

## See also

- `planamono/shared/plane_fitting/metrics.py` — `segmentation_covering_fast`
  (reused), `compute_inliers_at_threshold`, `fit_planes_and_evaluate_multi_threshold`
- `planamono/shared/segmentation/compute_plane_params.py` — produces
  `Dict[int, [a,b,c,d]]` if you want plane params; see
  `docs/compute_plane_params.md`
- `planamono/evaluation/quantitative/eval_utils.py` — has the existing
  `compute_clustering_metrics` (RI/VOI/SC bundle) used by per-frame eval scripts
- `planamono/inference/planarity/compare_metric_depth.py` — diagnostic for
  the affine-vs-metric depth pitfall mentioned in caveats
- `planamono/evaluation/quantitative/METRICS.md` — definitions for the
  existing per-point precision/recall family
