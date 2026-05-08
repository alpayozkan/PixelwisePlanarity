# `evaluate_gt_moge_zeroplane.py` — output columns

This script writes per-frame, per-scene, and dataset-level CSVs under
`<eval_root>/<exp>/<method>/<dataset>/`. The dataset-level
`aggregate_dataset.csv` has many columns with similar-looking names. This
document explains the structure.

## Output layout

```
<eval_root>/<exp>/                       # e.g. /cluster/scratch/.../eval/smoke_all
├── <method>/                            # gt | moge | zeroplane
│   └── <dataset>/                       # scannetpp | nyuv2 | sevenscenes
│       ├── <scene_id>/
│       │   ├── results.csv              # one row per frame, all metrics
│       │   └── summary.csv              # one row: scene_id, num_frames, <metric>_mean/_std
│       ├── aggregate_results.csv        # concat of all per-scene results.csv
│       ├── aggregate_per_scene.csv      # concat of all per-scene summary.csv
│       ├── aggregate_dataset.csv        # one row: dataset-level mean/std across scenes
│       └── runtime.csv                  # wall_time_seconds, num_frames, frames_per_second
└── summary.csv                          # one row per (method, dataset) pair
```

## Naming convention — two-level aggregation

Every metric ships in two columns: `<metric>_mean` and `<metric>_std`. These
are the **mean and std across scenes** (i.e. across rows in
`aggregate_per_scene.csv`). For a single-scene run the `_std` columns are
blank.

For per-plane error stats the metric is itself already a per-frame summary
(mean/median/std across the planes in that frame), so the column name carries
**two suffixes**:

| column | meaning |
|---|---|
| `normal_err_deg_mean_mean` | mean-across-scenes of (mean-across-planes-in-frame of normal angular error in deg) |
| `normal_err_deg_median_mean` | mean-across-scenes of (median-across-planes-in-frame of normal angular error in deg) |
| `normal_err_deg_std_mean` | mean-across-scenes of (std-across-planes-in-frame of normal angular error in deg) |
| `normal_err_deg_n_mean` | mean-across-scenes of (number of matched planes per frame) |

## The five metric families

### 1. Pixel-level 2D segmentation

Standard region-comparison metrics over the predicted-vs-GT label maps.

| column | meaning |
|---|---|
| `sc` | Segmentation Covering |
| `rand_index` | Rand Index |
| `voi` | Variation of Information (lower = better) |

### 2. RANSAC plane precision / recall

For each predicted plane segment, fit a plane via RANSAC at threshold τ, count
3D inliers, aggregate across the frame. Computed at three thresholds.

| column | threshold |
|---|---|
| `prec@0.1cm`, `rec@0.1cm` | 0.001 m |
| `prec@0.5cm`, `rec@0.5cm` | 0.005 m |
| `prec@1.0cm`, `rec@1.0cm` | 0.010 m |

### 3. Binary planarity classification (`bp_*`)

Collapse the per-pixel labels to a binary planar/non-planar mask
(label > 0 vs label = 0), then evaluate as a binary classifier.

| column | meaning |
|---|---|
| `bp_accuracy` | per-pixel accuracy |
| `bp_precision` | precision over predicted-planar pixels |
| `bp_recall` | recall over GT-planar pixels |
| `bp_f1` | harmonic mean |
| `bp_iou` | intersection-over-union |

### 4. Per-plane matched recall

GT planes are matched to predicted planes by mask overlap. Each matched pair
is then checked against an error threshold. The bare metric reports the
**fraction of GT planes that match AND have error under threshold**. Two
flavors of error:

| column family | error type | thresholds |
|---|---|---|
| `plane_recall_d_<τ>mm` | depth error (m) | 50, 100, 600 mm |
| `plane_recall_n_<τ>deg` | normal angle error (deg) | 5, 10, 30 deg |

Each comes with two bookkeeping companions:

| column | meaning |
|---|---|
| `..._n_total` | number of GT planes in the frame |
| `..._n_matched` | number of GT planes that matched a prediction (regardless of error) |
| `...` (bare) | fraction matched AND under threshold |

So `plane_recall_d_50mm = 0.23` and `_n_matched = 9` and `_n_total = 13.5`
means: ~13.5 GT planes per frame, ~9 matched a prediction, but only ~3 of
those were within 50 mm depth.

### 5. Per-plane error stats

For each matched plane pair, compute the per-plane angular or offset error,
then summarize within the frame. Then aggregate across scenes (giving the
double-suffix columns described above).

| column family | meaning |
|---|---|
| `normal_err_deg_{mean, median, std, n}` | normal angular error in degrees, summarized across matched planes in the frame |
| `offset_err_m_{mean, median, std, n}` | plane offset (D in `aX+bY+cZ+D=0`) error in meters, summarized across matched planes in the frame |

## The three different "recall" concepts

This is the most common source of confusion. They are **not the same**:

| name | unit of analysis | what's the "positive"? |
|---|---|---|
| `rec@<τ>cm` | per-pixel | predicted plane RANSAC inliers under τ |
| `bp_recall` | per-pixel | GT-planar pixels (vs non-planar) |
| `plane_recall_*` | per-plane | matched GT planes within error threshold |

## Numbers that don't depend on the method

A few `_n_total` columns just describe the dataset:

- `plane_recall_d_*_n_total_mean = 13.5` — there are on average ~13.5 GT planes per frame in the evaluated scene
- `plane_recall_n_*_n_total_mean = 13.1` — of those, ~13.1 also have valid GT normals (a few planes too small for stable normal computation are excluded)

These are identical across all methods (they describe GT, not predictions).

## Reading the smoke-test row

For scene `0d2ee665be`, 10 frames, scannetpp:

| family | gt | moge | zeroplane |
|---|---|---|---|
| sc | 1.000 | 0.761 | 0.574 |
| prec@1cm | 0.967 | 0.892 | 0.562 |
| rec@1cm | 0.687 | 0.543 | 0.562 |
| bp_f1 | 1.000 | 0.862 | 0.791 |
| plane_recall_d_50mm | 1.000 | 0.233 | 0.009 |
| plane_recall_n_5deg | 1.000 | 0.038 | 0.014 |
| normal_err_deg_mean_mean | 0.0 | 36.98 | 40.50 |
| offset_err_m_mean_mean | 0.0 | 0.137 | 0.407 |
| normal_err_deg_n_mean | 13.1 | 9.0 | 13.1 |

The last row says: gt and zeroplane have a matched prediction for all 13.1
GT planes (gt trivially, zeroplane because its non-planar label 20 is small),
while moge only matches 9 of them — its planarity head misses ~30% of the
GT planes outright on this scene.

## Why are normal errors ~40°?

That `normal_err_deg_mean_mean ≈ 37–40` for both moge and zeroplane is high
enough to suggest a frame/convention mismatch rather than genuine prediction
quality. Likely suspects:

1. K scaling in `_scale_K_to_hw` vs the H5 maps' resolution (ScanNet++ K is at
   iPhone native res; H5 depth/plane maps are 480×640).
2. Predicted normals in camera frame vs GT normals in a different frame — the
   GT normals here are computed from `(depth_gt, labels_gt, K)` via
   `compute_gt_normals_from_depth_labels`, which produces camera-frame
   normals.
3. Sign convention (inward vs outward facing) — but this would give errors
   near 180°, not 40°.

Worth investigating before trusting full-split numbers.
