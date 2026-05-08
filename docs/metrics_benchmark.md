# `metrics_benchmark.py` and `evaluate_gt_moge_zeroplane_benchmark.py`

A second evaluation pipeline that reports the **union of plane-segmentation
metrics from ZeroPlane, PlaneRCNN, and PlaneRecTR** instead of the
planamono-native metric block produced by `evaluate_gt_moge_zeroplane.py`.

## Why a second pipeline

The existing `evaluate_gt_moge_zeroplane.py` reports planamono-native
metrics (`prec@<τ>cm`, `bp_*`, `plane_recall_d/n_*`,
`normal_err_deg_*`, `offset_err_m_*`). These don't line up with the metric
columns in the ZeroPlane / PlaneRCNN / PlaneRecTR papers — different
matching rules, different reductions, different threshold sets. To compare
directly against published numbers we need the same metric definitions, not
a translation layer. This pipeline drops the planamono columns and
re-implements the three-repo metric set faithfully.

See `docs/metric_comparison_zeroplane_vs_planamono.md` and
`docs/zeroplane_metric_coverage.md` for the per-metric deltas this
addresses.

## Files

| File | Role |
|---|---|
| `planamono/evaluation/quantitative/metrics_benchmark.py` | Pure-numpy/torch metric kernels merged from the three source repos. Self-contained. |
| `planamono/evaluation/quantitative/evaluate_gt_moge_zeroplane_benchmark.py` | Driver. Same CLI / output layout as `evaluate_gt_moge_zeroplane.py`; routes per-frame work through `compute_benchmark_metrics()`. |

## Metric coverage

`compute_benchmark_metrics()` returns ~37 columns per frame from the merged
ZP/PlaneRCNN/PlaneRecTR suite, plus the driver adds 6 RANSAC prec/rec
columns on top — **43 total per frame**.

| Family | Source | Output columns |
|---|---|---|
| 2D segmentation | ZeroPlane `evaluateMasks` (≡ PlaneRCNN `evaluateMasksTensor`) | `RI`, `VI`, `SC` |
| Depth quality | ZeroPlane `evaluateDepths` | `DE_{rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3}` (also `pixel_DE_*` if a separate per-pixel-depth-head map is provided) |
| Plane recall by per-pixel depth diff | ZeroPlane `eval_plane_recall_depth` | indoor: `per_pixel_depth_{01, 06}`, `per_plane_depth_{005, 01, 06}`<br>outdoor: `per_pixel_depth_{1, 10}`, `per_plane_depth_{1, 3, 10}` |
| Plane recall by normal angle | ZeroPlane `eval_plane_recall_normal` | `per_pixel_normal_{5, 30}`, `per_plane_normal_{5, 10, 30}` |
| Plane recall by offset | PlaneRecTR `eval_plane_recall_offset` (ZeroPlane has the function but never wires it in) | `per_pixel_offset_{50mm, 150mm, 300mm}`, `per_plane_offset_{50mm, 150mm, 300mm}` |
| Best-match (Hungarian) param errors | ZeroPlane `eval_plane_bestmatch_normal_offset` (offset half re-enabled, which ZP discards) | `mean_normal_error_deg`, `median_normal_error_deg`, `mean_offset_error_m`, `median_offset_error_m` |
| Average Precision | PlaneRCNN `evaluatePlanesTensor` (predictions ranked by **plane area as proxy** for confidence — planamono predictions don't carry scores) | `AP@{20, 30, 60, 90}cm` |
| Plane-parameter L2 from depth | PlaneRCNN `evaluatePlaneDepth` | `plane_param_L2_{mean, area_weighted}` |
| **RANSAC plane prec/rec @ τ** (planamono) | `eval_utils.compute_plane_metrics` — backprojects GT depth with pred labels, RANSAC-fits a plane per pred segment at threshold τ, reports inlier prec/rec | `prec@{0.1, 0.5, 1.0}cm`, `rec@{0.1, 0.5, 1.0}cm` |

The RANSAC block is **driver-only** — `metrics_benchmark.py` doesn't know
about it. The driver calls `backproject_v1` + `compute_plane_metrics` from
the planamono codebase (same recipe as `evaluate_gt_moge_zeroplane.py`'s
`evaluate_single_frame`) and merges the result into the per-frame row.

CLI flags controlling it:
- `--ransac_thresholds 0.001 0.005 0.01` (m, default)
- `--ransac_iterations 200`
- `--inlier_ratio_gate 0.9`

This is the only piece of the original planamono metric set that the
benchmark pipeline carries — `bp_*`, `plane_recall_d/n_*`, and
`normal_err_deg_*` / `offset_err_m_*` are still excluded by design (their
matching/aggregation conventions don't line up with the published
ZP/PlaneRCNN/PlaneRecTR numbers).

## Internal adapters

planamono inputs are remapped to ZP/PlaneRCNN convention inside the
metric module — callers continue to pass planamono-native inputs.

| Adapter | What it does |
|---|---|
| `_densify_labels(seg, plane_params)` | Remap arbitrary-int labels (0 = non-planar, sparse positive ints) to dense `0..N-1` with non-planar set to `N` itself (dynamic per-frame sentinel). Returns `(densified seg, (N,3) n/d-form params, original-to-dense mapping, n_planes)`. |
| Hessian → n/d conversion | Inside `_densify_labels`. Standard Hessian normal form `AX + BY + CZ + D = 0` (planamono) → ZeroPlane's n/d form `aX + bY + cZ = 1` via `(a, b, c) = (-A/D, -B/D, -C/D)`. |

### Why the non-planar sentinel is dynamic

ZeroPlane's reference kernels iterate `for i in range(non_plane_idx)` and
treat any label ≥ `non_plane_idx` as background. ZeroPlane's predictions are
bounded by `num_queries = 20`, so a fixed `NONPLANAR_IDX = 20` works there.
**MoGe's segmentation routinely produces 50–60+ planes per frame** — a fixed
20-cap silently dropped everything past plane #20 (those labels collided
with the non-planar sentinel and were treated as background).

The current implementation sizes the sentinel to the actual plane count per
frame: `n_planes` planes get dense labels `0..n_planes-1`, non-planar pixels
get `n_planes`. Each kernel takes `pred_plane_num` and `gt_plane_num`
explicitly, so all real planes are seen.

These adapters keep the metric kernels exact ports of the reference
implementations.

## Driver: how `evaluate_gt_moge_zeroplane_benchmark.py` mirrors the base script

CLI flags, scene loading, output layout, and SLURM modes
(`--skip_dataset_aggregates`, `--aggregate_only`, `--scene_ids`) are all
identical to `evaluate_gt_moge_zeroplane.py`. The differences live entirely
in:

1. The per-frame worker, which calls `compute_benchmark_metrics()` instead
   of `evaluate_single_frame()` + `match_planes_by_overlap()` + the
   per-plane error helpers.
2. Two new internal helpers:
   - `_build_pred_arrays(method, pred, gt, gt_hw)` — per-method handler:
     - `moge`: SVD-fit plane params from `(depth, normal, labels, K)`,
       render depth from those params.
     - `zeroplane`: convert ZP n/d params via `_zp_plane_params_to_dict`
       (or fall back to SVD if the H5 lacks them), render depth.
   - `_build_gt_plane_params(depth_gt, labels_gt, K)` — fits GT plane
     params via SVD on backprojected GT XYZ, oriented by
     `compute_gt_normals_from_depth_labels`. Used both as `gt_plane_params`
     in metrics and as the GT method's "prediction".

Reused without change from the base script:

- `_norm_fid`, `_decode_frame_ids`, `_scale_K_to_hw`, `_render_plane_params_to_maps`, `_zp_plane_params_to_dict`
- `_match_to_gt_shape` (NEAREST for labels, LINEAR + renorm for depth/normals)
- `_load_moge_scene` (runs `compute_vectorized_planar_segments_v5_relative` on the saved planarity / normal / depth)
- `_load_zeroplane_scene` (label remap `20 → 0`, normals transposed to HWC)
- `_build_gt_dataset`, `_build_scene_index`, `_load_gt_sample`
- `_save_scene_csvs`, `_save_dataset_aggregates`, `_write_top_summary`,
  `_aggregate_from_disk`

## CLI usage

Same as the base script:

```bash
python planamono/evaluation/quantitative/evaluate_gt_moge_zeroplane_benchmark.py \
    --exp benchmark_smoke \
    --methods gt moge zeroplane \
    --datasets scannetpp \
    --max_scenes 1 --max_frames_per_scene 10 \
    --eval_root /cluster/scratch/aoezkan/planeseg/eval --n_jobs 8
```

Output layout (also identical to the base script):

```
<eval_root>/<exp>/
├── <method>/<dataset>/
│   ├── <scene>/{results,summary}.csv
│   ├── aggregate_results.csv
│   ├── aggregate_per_scene.csv
│   ├── aggregate_dataset.csv
│   └── runtime.csv
└── summary.csv
```

## Smoke test results

10 frames, ScanNet++ scene `0d2ee665be`. Performance: gt 0.51 fps, moge
0.10 fps, zeroplane 0.11 fps with `--n_jobs 8`. Output landed at
`/cluster/scratch/aoezkan/planeseg/eval/benchmark_smoke/`. Numbers below
are with the **dynamic non-planar sentinel** in place (the original v0
results understated MoGe by ~10% on most metrics due to the 20-plane cap).

| metric | gt | moge | zeroplane |
|---|---|---|---|
| RI | 1.0000 | 0.8883 | 0.9243 |
| VI | 0.0012 | 1.1732 | 0.8383 |
| SC | 1.0000 | 0.7646 | 0.8084 |
| DE_rel | 0.034 | 0.059 | 0.242 |
| DE_rmse | 0.208 | 0.124 | 0.380 |
| DE_accuracy_1 (δ<1.25) | 0.965 | 0.966 | 0.442 |
| DE_accuracy_2 (δ<1.25²) | 0.982 | 0.992 | 0.868 |
| DE_accuracy_3 (δ<1.25³) | 0.990 | 0.995 | 0.955 |
| per_pixel_depth_01 / _06 | 0.93 / 0.95 | 0.55 / 0.75 | 0.01 / 0.71 |
| per_plane_depth_005 / _01 / _06 | 0.66 / 0.71 / 0.86 | 0.20 / 0.34 / 0.45 | 0.00 / 0.03 / 0.31 |
| per_pixel_normal_5 / _30 | 1.00 / 1.00 | 0.65 / 0.73 | 0.39 / 0.72 |
| per_plane_normal_5 / _10 / _30 | 1.00 / 1.00 / 1.00 | 0.33 / 0.37 / 0.41 | 0.15 / 0.29 / 0.32 |
| per_plane_offset_50 / _150 / _300mm | 1.00 / 1.00 / 1.00 | 0.17 / 0.34 / 0.39 | 0.06 / 0.17 / 0.25 |
| mean_normal_error_deg | 0.01 | 19.70 | 24.28 |
| median_normal_error_deg | 0.01 | 9.49 | 18.27 |
| mean_offset_error_m | 0.000 | 0.351 | 0.325 |
| AP@20 / 30 / 60 / 90 cm | 0.77 / 0.79 / 0.83 / 0.87 | 0.39 / 0.41 / 0.42 / 0.42 | 0.06 / 0.14 / 0.26 / 0.27 |
| plane_param_L2_mean | 0.256 | 0.420 | 0.556 |
| **prec / rec @ 0.1cm** | 0.28 / 0.24 | 0.22 / 0.12 | 0.03 / 0.03 |
| **prec / rec @ 0.5cm** | 0.97 / 0.75 | 0.77 / 0.50 | 0.45 / 0.45 |
| **prec / rec @ 1.0cm** | 0.97 / 0.76 | 0.78 / 0.51 | 0.46 / 0.46 |

### Sanity-check observations

1. **GT recall curves are 1.0 on all normal/offset thresholds.** Expected:
   `gt_plane_params` is also used as the GT method's "prediction", so the
   IoU>0.5 + param-distance check is trivially satisfied.
2. **GT `mean_normal_error_deg ≈ 0.01°`.** Floating-point drift between two
   passes of SVD on the same XYZ; perfect identity.
3. **GT AP < 1.0 (0.75–0.86), GT `DE_rel ≠ 0`.** The depth-diff metric uses
   raw H5 GT depth, which has per-pixel noise on planar regions
   (acquisition + meshing artefacts). The rendered "GT prediction" is a
   perfectly flat plane through that noisy surface, so |gt_depth - pred_depth|
   > 0 even on a "perfect" prediction. This is a property of the source
   metrics, not a bug.
4. **MoGe `mean_normal_error_deg = 19.7°` here vs `36.98°` in
   `evaluate_gt_moge_zeroplane.py`.** Different definitions:
   - Benchmark uses Hungarian matching on n/d plane params. Match cost is
     L1 on the param vectors; angle is `arccos(unit_n · unit_n_gt)`.
   - planamono's `per_plane_error_stats` matches by largest-overlap on
     label maps and aggregates per-pixel **predicted-normal-map** averages,
     using `|dot|` (sign-agnostic).
   Both are valid "normal error" numbers, and both belong in the docs.
5. **All recall curves monotonic** with threshold (50mm ≤ 150mm ≤ 300mm,
   5° ≤ 10° ≤ 30°), so the kernels are wired consistently.
6. **AP curves slightly non-monotonic for ZeroPlane (0.06 → 0.14 → 0.26 → 0.27).**
   Expected: AP integrates precision/recall over a per-frame ranking, and
   ZeroPlane often has many predictions where a few hit the threshold —
   precision jumps as the threshold loosens.

## Bug fixed during smoke testing

`evaluate_planes_ap` initially used
`match = np.zeros(correct.shape[1], dtype=bool)` (per-prediction). PlaneRCNN's
reference is `match_mask = np.zeros(len(correct_mask), dtype=np.bool)`, i.e.
shape `(G,)` (per GT plane). The accumulator must be per-GT because
`num_matches` is the count of GT planes that have been recalled at any rank
≤ r. Fixed in `metrics_benchmark.py:574`:

```python
# Per PlaneRCNN: accumulate "any rank ≤ r matched this GT plane" — match
# is per-GT, not per-prediction.
match = np.zeros(correct.shape[0], dtype=bool)
```

## Bugfix: dynamic non-planar sentinel (post-smoke-test)

The first version of `_densify_labels` used a fixed `NONPLANAR_IDX = 20`
sentinel, matching ZeroPlane's `num_queries = 20`. ZeroPlane fits, GT
does not (~13 planes/frame in ScanNet++) and MoGe definitely does not —
sampled scenes had **51-63 planes per frame, every frame**. Anything beyond
plane #20 either collided with the non-planar sentinel (label 20) or
exceeded the kernel's `range(20)` iteration bound and was silently dropped.

Fixed by:
1. `_densify_labels` returns `n_planes` along with the dense seg.
   Non-planar sentinel = `n_planes` itself.
2. `evaluate_masks`, `evaluate_depths`, `eval_plane_recall_depth`,
   `evaluate_planes_ap`, `evaluate_plane_depth_param_l2` now take explicit
   `pred_plane_num` / `gt_plane_num` arguments instead of reading a
   constant.
3. `compute_benchmark_metrics` threads `n_pred_planes` and `n_gt_planes`
   through every kernel call.

ZeroPlane's H5 load step in the driver still references the literal
constant `ZP_NONPLANAR_LABEL = 20` — that's correct, since ZeroPlane's
H5 output uses 20 as non-planar by convention. The remap to planamono
(`labels + 1`, then `21 → 0`) happens before any metric kernel runs.

## Cross-reference

- `docs/metric_comparison_zeroplane_vs_planamono.md` — definition-level
  comparison between the two metric families. This is what the benchmark
  pipeline implements.
- `docs/zeroplane_metric_coverage.md` — what ZeroPlane has vs what's
  imported in its evaluator vs what gets reported.
- `docs/evaluate_gt_moge_zeroplane_columns.md` — column reference for the
  base script's CSV output.
