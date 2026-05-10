# Benchmark metric reference

Definitions, formulas, and source-repo attribution for every column emitted
by `evaluate_gt_moge_zeroplane_benchmark.py`.

For "what's the difference between these and v0's metrics?" see
`docs/metric_comparison_zeroplane_vs_planamono.md`. For ZP's coverage as a
standalone module, see `docs/zeroplane_metric_coverage.md`. This file is
the per-column dictionary.

## At a glance

43 columns per frame, organised in 12 groups. Direction column: ↑ = higher
is better, ↓ = lower is better.

| Group | Source repo (kernel) | # cols | Direction |
|---|---|---|---|
| 1. 2D segmentation (RI / VI / SC) | **`--rivoisc_ver`-selectable.** `new` (default): ZeroPlane (`evaluateMasks`) ≡ PlaneRCNN ≡ PlaneRecTR. `old`: planamono `compute_clustering_metrics` (used by `evaluate_all_baselines.py`). | 3 | ↑ ↓ ↑ |
| 2. Depth quality (DE_*) | ZeroPlane (`evaluateDepths`); PlaneRCNN/PlaneRecTR have the same kernel | 8 | mixed |
| 3. Per-pixel plane recall by depth | ZeroPlane (`eval_plane_recall_depth`) ≡ PlaneRecTR | 2 (indoor) | ↑ |
| 4. Per-plane recall by depth | same | 3 (indoor) | ↑ |
| 5. Per-pixel plane recall by normal | ZeroPlane (`eval_plane_recall_normal`) ≡ PlaneRecTR | 2 | ↑ |
| 6. Per-plane recall by normal | same | 3 | ↑ |
| 7. Per-pixel plane recall by offset | PlaneRecTR (`eval_plane_recall_offset`); ZP has the function but never wires it in | 3 | ↑ |
| 8. Per-plane recall by offset | same | 3 | ↑ |
| 9. Hungarian-matched param error | ZeroPlane (`eval_plane_bestmatch_normal_offset`) — offset half re-enabled (ZP discards it) | 4 | ↓ |
| 10. Average Precision (AP@τ) | PlaneRCNN (`evaluatePlanesTensor`) | 4 | ↑ |
| 11. Plane-parameter L2 from depth | PlaneRCNN (`evaluatePlaneDepth`) | 2 | ↓ |
| 12. RANSAC plane prec/rec @ τ | planamono (`eval_utils.compute_plane_metrics`); driver-only, not in `metrics_benchmark.py` | 6 | ↑ |

---

## 1. Two-dimensional segmentation (`RI`, `VI`, `SC`)

Mask-only metrics — depth and plane parameters are not used. **Two
implementations** are available, selected via `--rivoisc_ver`:

### `--rivoisc_ver new` (default) — ZeroPlane's `evaluateMasks`

ZeroPlane (= PlaneRCNN's `evaluateMasksTensor` = PlaneRecTR's
`evaluateMasks` — they're line-for-line ports of the same code).

The reduction is **masked to GT-planar pixels** (`valid_mask = gt_masks.max(0)`),
so non-planar background does not enter the numerator.

| Column | Direction | Definition |
|---|---|---|
| `RI` | ↑ | **Rand Index.** Probability that a random pair of pixels is in the same partition under both gt and pred (or in different partitions under both). `1 − ((Σ row²/2 + Σ col²/2 − Σ I²) / (N(N−1)/2))` where `I` is the (G+1, P+1) intersection matrix. |
| `VI` | ↓ | **Variation of Information.** `H(gt) + H(pred) − 2·MI(gt, pred)` in bits. Lower = pred and gt share more information. |
| `SC` | ↑ | **Segmentation Covering.** `½ · (Σ_g (\|g\| · max_p IoU(g,p)) + Σ_p (\|p\| · max_g IoU(g,p))) / N`. Both directions of the per-segment best-IoU mean, area-weighted. |

**Implementation note:** `metrics_benchmark.py` rewrote these to use a
bincount-based `(G+1, P+1)` matrix instead of the reference `(G+1, P+1, H, W)`
torch stack — same math, O(H·W + G·P) memory.

### `--rivoisc_ver old` — planamono's `compute_clustering_metrics`

The version that `evaluate_all_baselines.py` uses (defined in
`planamono/evaluation/quantitative/eval_utils.py:compute_clustering_metrics`).
Implementation:

```python
ri  = sklearn.metrics.rand_score(gt.flatten(), pred.flatten())
voi = sum(skimage.metrics.variation_of_information(gt, pred))
sc  = planamono.shared.plane_fitting.segmentation_covering_fast(gt, pred)
```

The columns emitted are still `RI`, `VI`, `SC` (key-renamed for column-name
consistency between the two modes), but the **numbers are not the same**:

| Aspect | `new` (ZP) | `old` (planamono) |
|---|---|---|
| Treatment of label 0 (non-planar) | excluded — only GT-planar pixels enter the reduction | **included** — treated as just another cluster, so non-planar agreement counts toward agreement |
| Implementation | bincount-based `(G+1, P+1)` torch reduction | sklearn / skimage / segmentation_covering_fast |
| What "SC = 1.0" means | "every GT-planar region is perfectly covered by exactly one pred region" | "the entire image (planar + non-planar) is perfectly partitioned" |

Smoke comparison on MoGe / scannetpp `0d2ee665be` (3 frames):

| metric | `new` | `old` | Δ |
|---|---|---|---|
| RI | 0.86 | 0.83 | +0.03 |
| VI | 1.21 | 1.55 | −0.35 |
| SC | 0.75 | 0.72 | +0.04 |

The two flavors are not directly comparable — each reflects a different
question. Use `new` to compare against ZP/PlaneRCNN/PlaneRecTR papers, `old`
to compare against `evaluate_all_baselines.py` numbers. The submit script
auto-suffixes the experiment name (`_old` / `_new`) so the two runs don't
clobber each other.

## 2. Depth quality (`DE_*`)

8 columns from ZeroPlane's `evaluateDepths` (= PlaneRecTR's; PlaneRCNN's is
nearly identical with no `max_depth` clamp). Computed over pixels that are
**both gt-planar AND pred-planar AND have valid GT depth (`gt > 1e-4`,
`gt < max_depth`)**.

The benchmark runs this on the **rendered plane depth** (per-plane fitted
parameters → re-rendered depth map). ZeroPlane runs it on its rendered plane
depth `plane_depth` (and a separate per-pixel head `seg_depth`); PlaneRecTR
runs it on `plane_depth` only. PlaneRCNN runs it on a single `depth_pred`.

| Column | Direction | Formula |
|---|---|---|
| `DE_rel` | ↓ | Mean Absolute Relative Error: `mean(\|pred − gt\| / max(gt, 1e-4))` |
| `DE_rel_sqr` | ↓ | Mean Squared Relative Error: `mean((pred − gt)² / max(gt, 1e-4))` |
| `DE_log10` | ↓ | Mean log-10 absolute error: `mean(\|log₁₀(pred) − log₁₀(gt)\|)` |
| `DE_rmse` | ↓ | Root-Mean-Square Error in meters: `sqrt(mean((pred − gt)²))` |
| `DE_rmse_log` | ↓ | RMS log error: `sqrt(mean((log(pred) − log(gt))²))` |
| `DE_accuracy_1` | ↑ | δ < 1.25: `mean(max(pred/gt, gt/pred) < 1.25)` |
| `DE_accuracy_2` | ↑ | δ < 1.25²: same with threshold 1.5625 |
| `DE_accuracy_3` | ↑ | δ < 1.25³: same with threshold 1.953125 |

If you supply a separate per-pixel depth head, `evaluate_depths(..., prefix='pixel_DE')`
gives a parallel 8 columns prefixed `pixel_DE_*` (the benchmark driver doesn't
populate this by default — pred has only one depth, the rendered one).

## 3-4. Plane recall by per-pixel depth diff

Source: ZeroPlane / PlaneRecTR `eval_plane_recall_depth`. PlaneRCNN's
`evaluatePlanesTensor` includes the same recall curve under a different
threshold grid (21 steps × 0.05 m vs 13 indoor steps × 0.05 m).

For each GT plane, find a pred plane with `IoU > 0.5`, take the **mean
absolute depth diff over their pixel intersection**, threshold against τ.

Indoor thresholds: 0..0.6 m at 0.05 m stride (13 steps). Reported columns
pick out specific indices: 0.05 m, 0.10 m, 0.6 m. Outdoor has its own
1.0-m-stride 0..12 m grid; indoor is the default and what's surfaced below.

| Column | Direction | Definition |
|---|---|---|
| `per_pixel_depth_01` | ↑ | Pixel-level recall at τ = 0.10 m. Numerator: pixels in matched (g, p) intersections whose mean depth-diff ≤ τ; denominator: total GT-planar pixels. |
| `per_pixel_depth_06` | ↑ | Same at τ = 0.60 m (the loosest indoor threshold). |
| `per_plane_depth_005` | ↑ | Plane-level recall at τ = 0.05 m. Fraction of GT planes whose **best** matched pred (min depth-diff over IoU>0.5 candidates) is within τ. |
| `per_plane_depth_01` | ↑ | Same at τ = 0.10 m. |
| `per_plane_depth_06` | ↑ | Same at τ = 0.60 m. |

**Implementation note:** rewritten with bincount-based contraction —
O(H·W + G·P) memory instead of the reference O(H·W·G·P) bool tensor.

## 5-6. Plane recall by normal angle

Source: ZeroPlane / PlaneRecTR `eval_plane_recall_normal`.

For each GT plane, find the first pred plane with `IoU > 0.5`, take the
**angular error** between their unit normals, threshold against τ° (13 steps,
0..30°).

Note: the "normal" here is `param / ‖param‖` where `param` is the n/d-form
plane vector (`a·X + b·Y + c·Z = 1`). It's the **fitted-plane normal**, not
a per-pixel normal-map mean. (planamono's `plane_recall_at_normal` uses the
pixel-mean and is different — see `metric_comparison_zeroplane_vs_planamono.md`.)

| Column | Direction | Definition |
|---|---|---|
| `per_pixel_normal_5` | ↑ | Pixel-level recall at τ = 5°. |
| `per_pixel_normal_30` | ↑ | Same at τ = 30°. |
| `per_plane_normal_5` | ↑ | Plane-level recall at τ = 5°. |
| `per_plane_normal_10` | ↑ | Same at τ = 10°. |
| `per_plane_normal_30` | ↑ | Same at τ = 30°. |

## 7-8. Plane recall by offset (mm)

Source: **PlaneRecTR** `eval_plane_recall_offset`. ZeroPlane copied the
function into `utils/metrics.py` but the evaluator never imports it — these
columns are absent from ZeroPlane's standard output. The benchmark
re-enables them.

For each GT plane, first pred with `IoU > 0.5`, take the **absolute offset
diff** between fitted planes, in millimeters: `|1/‖param_pred‖ − 1/‖param_gt‖| × 1000`.
13 thresholds linspace(0, 300) mm.

| Column | Direction | Definition |
|---|---|---|
| `per_pixel_offset_50mm` | ↑ | Pixel-level recall at τ = 50 mm offset. |
| `per_pixel_offset_150mm` | ↑ | Same at τ = 150 mm. |
| `per_pixel_offset_300mm` | ↑ | Same at τ = 300 mm (loosest). |
| `per_plane_offset_50mm` | ↑ | Plane-level recall at τ = 50 mm. |
| `per_plane_offset_150mm` | ↑ | Same at τ = 150 mm. |
| `per_plane_offset_300mm` | ↑ | Same at τ = 300 mm. |

## 9. Hungarian-matched parameter error

Source: ZeroPlane `eval_plane_bestmatch_normal_offset` (in
`metrics_onlyparams.py`). ZeroPlane's evaluator reports the normal half only
(`mean_normal_error`, `median_normal_error`); the offset half is computed but
discarded. The benchmark re-enables both.

Match algorithm: **Hungarian** (`scipy.optimize.linear_sum_assignment`) on L1
cost between n/d-form param vectors. Crucially this match has **no
spatial-overlap requirement** — it can pair planes that don't co-locate, so
it's robust to mis-localised but well-oriented predictions. Compare with the
recall-by-normal metrics above which need IoU > 0.5.

| Column | Direction | Formula |
|---|---|---|
| `mean_normal_error_deg` | ↓ | Mean over Hungarian-matched pairs of `arccos(unit_n_pred · unit_n_gt)` in degrees. |
| `median_normal_error_deg` | ↓ | Same, median. |
| `mean_offset_error_m` | ↓ | Mean over matched pairs of `\|1/‖param_pred‖ − 1/‖param_gt‖\|` in meters. |
| `median_offset_error_m` | ↓ | Same, median. |

## 10. Average Precision @ depth diff τ

Source: PlaneRCNN `evaluatePlanesTensor`. Standard rank-based AP integration.

Per frame:
1. Rank predictions by score (high → low). planamono predictions don't carry
   confidence scores, so we use **plane area as proxy** (largest-first).
2. A prediction at rank r is "correct" w.r.t. a GT plane iff
   `IoU > 0.5 AND mean intersection-depth-diff < τ`.
3. Compute precision[r] = correct / r, recall[r] = correct / num_gt_planes.
4. Smooth-AP integration:
   ```
   AP = Σ over decreasing-rank (prev_recall − recall) × max_so_far(precision)
   ```

Thresholds match PlaneRCNN: τ ∈ {0.2, 0.3, 0.6, 0.9} m.

| Column | Direction | Threshold |
|---|---|---|
| `AP@20cm` | ↑ | τ = 0.20 m |
| `AP@30cm` | ↑ | τ = 0.30 m |
| `AP@60cm` | ↑ | τ = 0.60 m |
| `AP@90cm` | ↑ | τ = 0.90 m |

## 11. Plane-parameter L2 from depth

Source: PlaneRCNN `evaluatePlaneDepth`. **No equivalent in ZeroPlane or
PlaneRecTR** — ZeroPlane uses raw predicted plane params decoupled from
depth, PlaneRCNN re-fits params from the predicted vs GT depth and measures
their L2 distance.

For each GT plane mask:
1. Backproject (depth_pred, depth_gt) into XYZ using GT K.
2. Per mask, fit `aX + bY + cZ = 1` parameters via least squares on each XYZ.
3. Re-normalise: `param ÷ max(‖param‖², 1e-4)` (PlaneRCNN's `n/d²·n` form).
4. L2 distance between (params_pred, params_gt).

| Column | Direction | Definition |
|---|---|---|
| `plane_param_L2_mean` | ↓ | Unweighted mean L2 over GT planes. |
| `plane_param_L2_area_weighted` | ↓ | Same, weighted by GT plane pixel area. |

PlaneRCNN's reference code crops to `[80:560]` (NYU 640×640 specific). The
benchmark evaluates the full frame.

## 12. RANSAC plane precision/recall @ distance τ

Source: **planamono** `eval_utils.compute_plane_metrics`. Not in
`metrics_benchmark.py` — added directly in the driver
(`evaluate_gt_moge_zeroplane_benchmark.py`) on top of the benchmark suite.
Recipe identical to the v0 driver's `evaluate_single_frame`.

Per frame:
1. Backproject GT depth + GT pose using K (see `--kscaled` below).
2. Partition the 3D points by **predicted** label.
3. For each segment with ≥ `min_support` points: RANSAC-fit a plane at
   threshold τ (200 iterations).
4. Count inliers (segment points within τ of fitted plane).
5. Apply `inlier_ratio_gate=0.9`: segments below the gate contribute 0
   inliers (filtered as bad fits).
6. `precision = total_inliers / total_pred-planar_points`,
   `recall = total_inliers / total_gt-planar_points`.

| Column | Direction | Threshold |
|---|---|---|
| `prec@0.1cm` | ↑ | τ = 1 mm |
| `rec@0.1cm` | ↑ | τ = 1 mm |
| `prec@0.5cm` | ↑ | τ = 5 mm |
| `rec@0.5cm` | ↑ | τ = 5 mm |
| `prec@1.0cm` | ↑ | τ = 1 cm |
| `rec@1.0cm` | ↑ | τ = 1 cm |

CLI flags controlling this block:
- `--ransac_thresholds 0.001 0.005 0.01` (m, default).
- `--ransac_iterations 200`.
- `--inlier_ratio_gate 0.9`.
- `--kscaled` / `--no-kscaled` — chooses the K source (see below).

### `--kscaled` and what it does to RANSAC

The RANSAC backprojection takes K. The flag picks between two values, both
derived from GT (no method-supplied K is ever used; see end of
`docs/metrics_benchmark.md` for why):

- `--kscaled` (default): K is `_scale_K_to_hw(gt["K"], gt_hw)`, scaled to
  match the H5 480×640 resolution. 3D points come out in true meters; τ in
  meters means meters.
- `--no-kscaled`: K is `gt["K"]`, the iPhone-native ~1920×1440 K. 3D points
  come out compressed in xy by ~3× (the K-scale bug v0's docstring already
  flagged). τ acts much tighter than advertised, so prec/rec are inflated,
  especially at the 1-mm threshold.

The submit script auto-suffixes the experiment name (`_kscaled` /
`_kunscaled`) so the two runs don't clobber each other.

---

## Sources cross-reference

| Repo | File | Functions used |
|---|---|---|
| ZeroPlane | `ZeroPlane/utils/metrics.py` | `evaluateMasks`, `eval_plane_recall_depth`, `eval_plane_recall_normal` |
| ZeroPlane | `ZeroPlane/utils/metrics_de.py` | `evaluateDepths` |
| ZeroPlane | `ZeroPlane/utils/metrics_onlyparams.py` | `eval_plane_bestmatch_normal_offset` |
| PlaneRecTR | `PlaneRecTR/utils/metrics.py` | `eval_plane_recall_offset` (ZP has it but doesn't import it) |
| PlaneRCNN | `planercnn/evaluate_utils.py` | `evaluatePlanesTensor` (AP), `evaluatePlaneDepth` (param L2) |
| planamono | `planamono/evaluation/quantitative/eval_utils.py` | `compute_plane_metrics` (RANSAC prec/rec) |

For a deeper compare-and-contrast on definitions across these repos see:
- `docs/metric_comparison_zeroplane_vs_planamono.md` — pairwise definitions, conventions, matching rules.
- `docs/zeroplane_metric_coverage.md` — what ZP has vs imports vs reports.
- `ZeroPlane/docs/zero_vs_planercnn_metrics.md` and `zero_vs_planerectr_metric.md` — ZP vs the upstream repos it forked from.
