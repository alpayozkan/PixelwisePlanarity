# ZeroPlane: metric coverage summary

What metrics ZeroPlane has implementations for, broken down by:
1. **Function exists** — code is in `ZeroPlane/utils/`
2. **Wired into evaluator** — `planeSeg_evaluation.py` actually calls it
3. **Reported as output key** — appears in the result dict

This synthesises three comparison docs:
- `docs/metric_comparison_zeroplane_vs_planamono.md`
- `ZeroPlane/docs/zero_vs_planercnn_metrics.md`
- `ZeroPlane/docs/zero_vs_planerectr_metric.md`

## ZeroPlane metric inventory

| Metric family | Function (file:line) | Wired in evaluator? | Output keys |
|---|---|---|---|
| **Rand Index / VOI / SC** | `evaluateMasks` (`metrics.py:202`) | yes | `RI`, `VI`, `SC` |
| **Plane recall by depth diff** | `eval_plane_recall_depth` (`metrics.py:29`) | yes | indoor: `per_pixel_depth_01/06`, `per_plane_depth_005/01/06`<br>outdoor: `per_pixel_depth_1/10`, `per_plane_depth_1/3/10` |
| **Plane recall by normal angle** | `eval_plane_recall_normal` (`metrics.py:89`) | yes | `per_pixel_normal_5/30`, `per_plane_normal_5/10/30` |
| **Plane recall by offset (mm)** | `eval_plane_recall_offset` (`metrics.py:146`) | **NO** — function defined but never imported by `planeSeg_evaluation.py` | none (PlaneRecTR exposed `per_plane_offset_*`, ZeroPlane dropped it) |
| **Best-match params (Hungarian on L1 cost)** | `eval_plane_bestmatch_normal_offset` (`metrics_onlyparams.py:5`) | yes (only the normal half) | `mean_normal_error`, `median_normal_error` (offset return value is discarded) |
| **Pixel depth quality** (rel, log10, rmse, rmse_log, δ < 1.25^k for k=1,2,3) | `evaluateDepths` (`metrics_de.py:7`) | yes — called **twice** per frame | `DE_*` (over `plane_depth`)<br>`pixel_DE_*` (over `seg_depth`) |
| **IoU helper** | `eval_iou` (`metrics.py:5`) | indirect (used inside the recall functions) | — |

## Reported output keys (always populated when `not infer_only`)

```
# 2D segmentation
RI, VI, SC

# Best-match (Hungarian on plane params)
mean_normal_error, median_normal_error

# Plane-rendered depth quality (over plane_depth)
DE_rel, DE_rel_sqr, DE_log10, DE_rmse, DE_rmse_log,
DE_accuracy_1, DE_accuracy_2, DE_accuracy_3

# Per-pixel depth-head quality (over seg_depth)
pixel_DE_rel, pixel_DE_rel_sqr, pixel_DE_log10, pixel_DE_rmse, pixel_DE_rmse_log,
pixel_DE_accuracy_1, pixel_DE_accuracy_2, pixel_DE_accuracy_3

# Plane recall — indoor depth thresholds
per_pixel_depth_01, per_pixel_depth_06
per_plane_depth_005, per_plane_depth_01, per_plane_depth_06

# Plane recall — outdoor depth thresholds (alternate set)
per_pixel_depth_1, per_pixel_depth_10
per_plane_depth_1, per_plane_depth_3, per_plane_depth_10

# Plane recall — normal angle
per_pixel_normal_5, per_pixel_normal_30
per_plane_normal_5, per_plane_normal_10, per_plane_normal_30
```

## What ZeroPlane does NOT compute

| Metric | Where it lives | Why ZeroPlane skips it |
|---|---|---|
| **Average Precision (AP@τ)** for detection | PlaneRCNN's `evaluatePlanesTensor` | ZeroPlane reports recall-only, no rank-based AP |
| **Plane-parameter L2 from depth** | PlaneRCNN's `evaluatePlaneDepth` | ZeroPlane uses raw plane params directly, decoupled from depth |
| **Offset recall curve / `offset_error`** | PlaneRecTR's `eval_plane_recall_offset` | function copied over but evaluator does not import it |
| **3D RANSAC plane prec/rec @ τ** (`prec@<τ>cm`) | planamono's `compute_plane_metrics` | not implemented |
| **Per-plane mean-depth recall** (planamono's `plane_recall_d_*`) | planamono's `metrics_planes.py` | different concept — ZeroPlane uses per-pixel intersection diff |
| **Per-plane normal recall from pixel-mean** | planamono's `plane_recall_at_normal` | ZeroPlane's normal recall uses param-derived normals, not pixel-mean of a normal map |
| **Binary planarity classification** (`bp_accuracy/precision/recall/f1/iou`) | planamono's `compute_binary_planarity_metrics` | not implemented |

## Cross-doc consistency

The three comparison docs agree on what's implemented in ZeroPlane and disagree only on the framing:

- **vs PlaneRCNN**: ZeroPlane adds normal-recall + offset-recall (function only) + Hungarian param matching; drops AP and PlaneRCNN's plane-param-from-depth L2.
- **vs PlaneRecTR**: shared kernels; ZeroPlane drops offset recall keys + `offset_error`, adds `median_normal_error`, splits `evaluateDepths` into a `DE_*` (plane_depth) + `pixel_DE_*` (seg_depth) pair, adds outdoor recall thresholds.
- **vs planamono**: ZeroPlane has depth quality metrics (REL/RMSE/δ) and Hungarian param matching; lacks RANSAC plane prec/rec, binary planarity, per-plane mean-depth recall, and scene-level aggregation.

## Latent functions

Two computations exist in ZeroPlane code but are never reported during a normal evaluation run:

1. **`eval_plane_recall_offset`** — defined in `metrics.py:146`, never imported by `planeSeg_evaluation.py`. To enable it you would need to import it and add a third recall curve alongside the depth and normal curves.
2. **Offset half of `eval_plane_bestmatch_normal_offset`** — `offset_error` is returned but discarded by the evaluator. Adding `res["offset_error"] = offset_error` next to the existing `mean_normal_error` would surface it.

If you need offset-error numbers from ZeroPlane code, one of these has to be wired in.
