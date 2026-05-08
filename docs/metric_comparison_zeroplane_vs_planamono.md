# Metrics: `evaluate_gt_moge_zeroplane.py` (planamono) vs `planeSeg_evaluation.py` (ZeroPlane)

Side-by-side comparison of how the two evaluators compute the metrics they
share. Both produce numbers under similar names (RI / VI / SC, plane recall at
depth / normal, mean normal error), but the **definitions differ** in ways
that matter when comparing across papers/scripts.

## Files

| Side | Driver | Metric helpers |
|---|---|---|
| **planamono** | `planamono/evaluation/quantitative/evaluate_gt_moge_zeroplane.py` | `eval_utils.py` (driver), `metrics_planes.py` (per-plane recall, error stats) |
| **ZeroPlane** | `ZeroPlane/ZeroPlane/evaluation/planeSeg_evaluation.py` | `utils/metrics.py` (RI/VOI/SC, plane recall), `utils/metrics_de.py` (depth metrics), `utils/metrics_onlyparams.py` (Hungarian param matching) |

## Cheat-sheet of metric correspondences

| Concept | planamono key | ZeroPlane key |
|---|---|---|
| Rand Index | `rand_index` | `RI` |
| Variation of Information | `voi` | `VI` |
| Segmentation Covering | `sc` | `SC` |
| 3D plane prec/rec @ τ (RANSAC) | `prec@<τ>cm`, `rec@<τ>cm` | — (not computed) |
| Binary planarity classification | `bp_accuracy / bp_precision / bp_recall / bp_f1 / bp_iou` | — |
| Plane recall by depth-error | `plane_recall_d_<τ>mm` (50/100/600 mm) | `per_plane_depth_<τ>` (curve over 0..0.6 m, reports 005/01/06) |
| Plane recall by normal-angle | `plane_recall_n_<τ>deg` (5/10/30°) | `per_plane_normal_<τ>` (curve over 0..30°, reports 5/10/30) |
| Per-plane normal/offset errors | `normal_err_deg_*`, `offset_err_m_*` (largest-overlap match) | `mean_normal_error`, `median_normal_error` (Hungarian match on params) |
| Pixel-level depth metrics | — | `DE_rel / DE_rel_sqr / DE_log10 / DE_rmse / DE_rmse_log / DE_accuracy_{1,2,3}` |
| Pixel recall by depth/normal | — | `per_pixel_depth_<τ>`, `per_pixel_normal_<τ>` |

## 1. RI / VI / SC — same names, different masking

| | planamono `compute_clustering_metrics` | ZeroPlane `evaluateMasks` |
|---|---|---|
| Implementation | `sklearn.rand_score`, `skimage.variation_of_information`, `segmentation_covering_fast` | hand-rolled torch on per-plane mask stacks |
| Treatment of non-planar pixels | **Included** — label 0 is treated as just another cluster | **Excluded** — `valid_mask = gt_planar_mask.max(0)` filters all non-planar GT pixels before reduction |
| Pred non-planar handling | Label 0 in pred is also a cluster | Pred non-planar appended as the `N+1`-th mask, but only pixels inside the valid (planar) GT mask are counted |
| Effect | A perfect non-planar prediction inflates RI/SC | Only planar regions contribute |

**Implication**: planamono's `sc=1.0` means perfect agreement *including* the
non-planar background. ZeroPlane's `SC=1.0` means perfect agreement
*restricted to GT planar pixels*. They are not directly comparable.

## 2. Per-plane recall at depth

| | planamono `plane_recall_at_depth` | ZeroPlane `eval_plane_recall_depth` |
|---|---|---|
| Matching rule | For each GT plane, take the pred label with **largest pixel overlap** (regardless of IoU) | For each GT plane, the pred is "matched" iff IoU > 0.5; among IoU>0.5 candidates, the one with smallest `planeDiff` |
| Depth quantity compared | `mean(depth_pred[label==pred_id])` vs `mean(depth_gt[label==gt_id])` — **mean of depths per plane** | `mean(\|depth_pred − depth_gt\|)` over the intersection mask — **per-pixel diff** |
| Threshold sweep | 3 fixed: 50 mm, 100 mm, 600 mm | 13 thresholds from 0 to 0.6 m at 0.05 m stride (indoor); reports 0.05/0.1/0.6 |
| Reported denominator | `n_total` = number of GT planes (after ignoring label 0) | `gtNumPlanes = len(unique(gt)) − 1` (treats label 20 as non-planar) |

**Implication**: ZeroPlane's `per_plane_depth_*` is sensitive to per-pixel
noise in `predDepths`; planamono's `plane_recall_d_*` only checks the
mean-depth of each plane region — much looser tolerance and a different
quantity.

## 3. Per-plane recall at normal

| | planamono `plane_recall_at_normal` | ZeroPlane `eval_plane_recall_normal` |
|---|---|---|
| Matching rule | Largest pixel overlap (no IoU gate) | First pred plane satisfying IoU > 0.5 |
| Normal source | Per-pixel **normal map** averaged per plane label, renormalized | Per-plane **parameter** `param[j]`, normalized as `param / \|param\|` (the "normal" of the n/d-form vector) |
| Comparison | `\|dot(n_pred, n_gt)\|` (sign-agnostic) | `arccos(dot(n_pred, n_gt))` (signed) |
| Threshold sweep | 3 fixed: 5°, 10°, 30° | 13 thresholds from 0 to 30°; reports 5/10/30 |

**Critical difference**: ZeroPlane's "normal" is the renormalized plane *param*
(`(a,b,c)` from `aX+bY+cZ=1`), which equals the unit normal direction by
construction. planamono's normal is the **mean of the per-pixel normal
predictions**, which is a different signal entirely — the param-derived normal
is the geometric normal of the *fitted* plane, while the pixel-mean is what
the network's normal head emits averaged spatially.

## 4. Mean normal error (best-match)

| | planamono `per_plane_error_stats` (`normal_err_deg_*`) | ZeroPlane `eval_plane_bestmatch_normal_offset` (`mean_normal_error`) |
|---|---|---|
| Matching rule | Largest pixel overlap on label maps | **Hungarian matching** on L1 cost between plane params |
| Normal source | Per-pixel normal map, mean per plane | Plane params `(a,b,c)` from `aX+bY+cZ=1`, normalized |
| Reduction across planes | mean / median / std / n (per frame), then mean across scenes | mean over Hungarian-matched pairs (one number per frame), then mean across frames |
| Sign convention | `\|dot\|` (treats opposite-facing planes as same) | signed `arccos(dot)` |

**Implication**: ZeroPlane's match doesn't require the planes to overlap
spatially at all — it pairs by parameter similarity. This makes the metric
robust to mis-localised but well-orientated predictions, which is the
opposite of what planamono measures.

## 5. Depth quality metrics

ZeroPlane reports a full set of **pixel-level depth-estimation metrics**
(`DE_rel`, `DE_rmse`, `DE_log10`, `DE_rmse_log`, `DE_accuracy_{1,2,3}` =
δ < 1.25^k) computed from `evaluateDepths`. These come in two flavors:
- `DE_*` — over the predicted-plane depth (`plane_depth`) restricted to
  pred-planar AND gt-planar pixels.
- `pixel_DE_*` — over the per-pixel depth output (`seg_depth`).

**planamono does not compute these.** Its depth signal is only used as input
to RANSAC plane fitting (`prec@<τ>cm` / `rec@<τ>cm`) — there's no direct
RMSE or δ-accuracy reporting. If you need δ < 1.25 for a paper table you
have to add it.

## 6. RANSAC plane fitting (planamono only)

planamono's `compute_plane_metrics` runs RANSAC on backprojected 3D points
**at the evaluation threshold** (e.g. 0.001 / 0.005 / 0.01 m), reporting:

- `prec@<τ>cm` = fraction of inlier pixels in pred-planar regions that lie
  within τ of their fitted plane.
- `rec@<τ>cm` = fraction of GT-planar pixels recovered.

ZeroPlane has **no equivalent metric**. Its 3D-plane error signal comes
indirectly through `eval_plane_recall_depth` (mean per-pixel depth diff over
intersection) and the `DE_*` family.

## 7. Plane parameter conventions

This bites silently when carrying numbers across the two tools.

| | ZeroPlane | planamono |
|---|---|---|
| Stored form | `a·X + b·Y + c·Z = 1` (n/d form) | `A·X + B·Y + C·Z + D = 0` (Hessian normal form) with ‖(A,B,C)‖ = 1 |
| Conversion | `unit_n = (a,b,c)/‖(a,b,c)‖`, `D = −1/‖(a,b,c)‖` | (already in standard form) |

`evaluate_gt_moge_zeroplane.py` does this conversion in `_zp_plane_params_to_dict()` before rendering ZeroPlane's plane params.

## 8. Aggregation level

| | planamono | ZeroPlane |
|---|---|---|
| Per-scene CSV | Yes — one row per frame, plus a one-row scene summary | No |
| Cross-scene aggregation | mean / std across scene-means (gives nested `*_mean_mean` columns) | Plain mean across all evaluated frames in the dataset |
| Threshold curve plots | No (3 fixed thresholds) | Yes (`plot_depth_recall_curve`, `plot_normal_recall_curve`) |

## What this means in practice

- Numbers like "SC = 0.76" from one script vs "SC = 0.84" from the other
  **cannot be compared**. The denominators differ (full image vs planar-only)
  and the matching rules differ (label-as-cluster vs per-plane masks).
- Same for `plane_recall_*`: planamono compares per-plane mean depths;
  ZeroPlane compares per-pixel depth diffs over the intersection. Even with
  identical predictions, expect different numbers.
- `mean_normal_error` differs by both the matching algorithm (Hungarian on
  params vs largest-overlap on masks) and the input (param-derived vs
  pixel-mean normal).
- ZeroPlane gives you depth quality (REL, RMSE, δ) for free; planamono gives
  you 3D-plane prec/rec at multiple thresholds for free. They cover
  complementary aspects.

## Suggested cross-mapping for a paper table

If you want a single table that reuses both:

| Reported metric | Compute via | Scope |
|---|---|---|
| Pixel-level seg (RI / VI / SC) | ZeroPlane `evaluateMasks` (planar-only mask) | "How well are GT planar regions partitioned?" |
| Pixel-level seg (RI / VI / SC, image-wide) | planamono `compute_clustering_metrics` | "How well do labels agree across the whole image, including non-planar?" |
| Per-plane recall (depth) | planamono `plane_recall_d_*` (mean diff) **or** ZeroPlane `per_plane_depth_*` (intersection diff) — pick one and label clearly | depends on whether you care about plane-level or pixel-level depth fit |
| Per-plane recall (normal) | planamono `plane_recall_n_*` if comparing **pixel-mean normals**; ZeroPlane `per_plane_normal_*` if comparing **fitted-plane normals** | choose to match the task |
| Best-match normal error | ZeroPlane `mean_normal_error` (Hungarian on params) | only meaningful if methods produce comparable plane params |
| Depth quality | ZeroPlane `DE_rel`, `DE_rmse`, `DE_accuracy_1` | needed for any "depth estimation" comparison |
| 3D plane prec/rec | planamono `prec@<τ>cm`, `rec@<τ>cm` | tightest 3D-geometric metric — preferred for plane fidelity |
