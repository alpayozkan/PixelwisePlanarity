# RI / VI / SC: convention comparison between `evaluate_all_baselines.py` and `evaluate_gt_moge_zeroplane_benchmark.py`

Both scripts report Rand Index (RI ↑), Variation of Information (VI ↓), and Segmentation Covering (SC ↑) on the same kind of plane-segmentation predictions, but the two implementations are **not numerically equivalent** even on the same inputs. They use different sample spaces, a different SC formula, and different library implementations. This document explains where the gap comes from and which convention to prefer for each use case.

## TL;DR

| | `evaluate_all_baselines.py` | `evaluate_gt_moge_zeroplane_benchmark.py` |
|---|---|---|
| Sample space | every pixel of the image, **including non-planar (label 0)** | only **GT-planar** pixels (`valid = gt < non_planar_idx`) |
| SC formula | one-sided GT → pred, area-weighted by GT only | **symmetric**: average of GT → pred and pred → GT |
| Library | sklearn `rand_score`, skimage `variation_of_information`, custom `segmentation_covering_fast` | hand-rolled bincount + torch port of ZeroPlane `evaluateMasks` (= PlaneRCNN `evaluateMasksTensor`) |
| Convention | planamono-native | PlaneRCNN / PlaneTR / PlanRecTR / ZeroPlane canonical |
| Reported in papers? | no | **yes** — this is the protocol used in the ZeroPlane CVPR table |

For the same prediction on a typical ScanNet++ test frame (~50–60 % planar coverage), expect:

- `evaluate_all_baselines.py` → **higher RI**, **lower VI**, **higher SC**
- benchmark → **lower RI**, **higher VI**, **lower SC**

The two are not bugs of one another; they answer different questions. Use the benchmark script for any paper table that compares against published PlaneRCNN / PlaneTR / ZeroPlane numbers.

## Code paths

### `evaluate_all_baselines.py`

`evaluate_single_frame()` (`eval_utils.py:647`) calls `compute_clustering_metrics()` (`eval_utils.py:281`), which is a thin wrapper:

```python
from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information
from planamono.shared.plane_fitting import segmentation_covering_fast

def compute_clustering_metrics(gt_seg, pred_seg):
    ri = rand_score(gt_seg.flatten(), pred_seg.flatten())
    Hs, Hm = variation_of_information(gt_seg, pred_seg)
    voi = Hs + Hm
    sc = segmentation_covering_fast(gt_seg, pred_seg)
    return {"rand_index": ri, "voi": voi, "sc": sc}
```

`gt_seg` and `pred_seg` are the raw `(H, W)` int label maps with `0 = non-planar`. There is **no pre-filtering** — label 0 is treated as just another segment.

### `evaluate_gt_moge_zeroplane_benchmark.py`

`_eval_one_frame_benchmark()` calls `compute_benchmark_metrics()` (`metrics_benchmark.py:740`), which first densifies the labels:

```python
pred_dense, _, _, n_pred_planes = _densify_labels(pred_seg, pred_plane_params)
gt_dense,   _, _, n_gt_planes   = _densify_labels(gt_seg,   gt_plane_params)

out.update(evaluate_masks(
    pred_dense, gt_dense,
    pred_non_plane_idx=n_pred_planes,
    gt_non_plane_idx=n_gt_planes,
))
```

`_densify_labels()` (`metrics_benchmark.py:86`) renumbers planes to `0..N-1` and remaps the **non-planar bin to N** (sentinel one above the last plane). Then `evaluate_masks()` (`metrics_benchmark.py:145`) restricts the contingency to GT-planar pixels and applies the ZeroPlane formulas.

## Difference 1 — sample space (the dominant effect)

This is the source of most of the numerical gap.

`evaluate_all_baselines.py` includes every pixel:

```python
ri  = rand_score(gt_seg.flatten(), pred_seg.flatten())          # all H·W pixels
voi = sum(variation_of_information(gt_seg, pred_seg))           # all H·W pixels
sc  = segmentation_covering_fast(gt_seg, pred_seg)              # all H·W pixels
```

The benchmark masks to GT-planar pixels first:

```python
valid = gt_seg_dense < gt_non_plane_idx       # only GT-planar pixels
g_flat = gt_seg_dense[valid]
p_flat = pred_seg_dense[valid]
lin = g_flat * P + p_flat
intersection_np = np.bincount(lin, minlength=G * P).reshape(G, P)
N = intersection_t.sum()                      # = #GT-planar pixels (NOT H·W)
```

So `N` in the benchmark RI denominator `N(N-1)/2` is the GT-planar pixel count, while in the all_baselines path the underlying `n` is the full image size.

### Why this matters per metric

- **RI**: with all pixels included, the millions of (non-planar, non-planar) pixel pairs almost all "agree" — both partitions put them in cluster 0 — which inflates RI close to 1 even for mediocre predictions. Excluding them makes RI sensitive to plane-vs-plane disagreements.
- **VI**: a single huge non-planar cluster has very low marginal entropy. Including it in the joint distribution drives VI down. Excluding it raises VI.
- **SC**: the non-planar bin is one giant segment. If pred-0 covers most of GT-0, that one IoU dominates the area-weighted sum (because `gt_areas[0]` is huge) and pushes SC up. Excluding it removes that artificial inflation.

### Note: pred-non-planar column is NOT removed

After masking with `valid = gt < N_gt`, the GT-non-planar **row** of the contingency sums to zero. But pred-non-planar pixels at GT-planar locations still populate the pred-non-planar **column** — that's correct behavior (pred misses on GT planes count as misclassification). The doc-comment in `metrics_benchmark.py:161-164` saying "extra zero rows/cols ... contribute zero" only refers to the gt-non-planar row, not the pred-non-planar column.

## Difference 2 — SC formula (symmetric vs one-sided)

### `evaluate_all_baselines.py` — one-sided GT → pred

`segmentation_covering_fast()` (`metrics.py:21`):

```python
best_iou = iou_matrix.max(axis=1)                # best pred IoU for each GT segment
sc = (best_iou * gt_areas).sum() / total_area    # area-weighted by GT only
```

This is the classical BSDS-style covering: every GT segment claims its best-matching pred segment, and predictions that don't overlap any GT segment cost nothing.

### benchmark — symmetric average

```python
union = gt_areas[:, None] + pred_areas[None, :] - intersection
IOU = intersection / union.clamp(min=1)
sc_a = (IOU.max(-1)[0] * gt_areas.clamp(min=1e-4)).sum()  / N    # GT  → pred
sc_b = (IOU.max(0)[0]  * pred_areas.clamp(min=1e-4)).sum() / N    # pred → GT
SC = (sc_a + sc_b) / 2
```

`sc_b` punishes spurious / over-segmented predictions: a pred segment that doesn't overlap any GT plane lowers `sc_b`. `sc_a` does not see it. Therefore for the same prediction, **benchmark SC ≤ one-sided SC**.

Quirk to be aware of: `sc_b` is divided by `N = #GT-planar pixels` even though the numerator weights by `pred_areas`. That's the ZeroPlane convention — it keeps both halves on the same denominator and prevents pred-non-planar mass from blowing SC up disproportionately.

## Difference 3 — RI formula (algebraically equivalent on the same support)

The benchmark hand-implements the standard formula:

```
RI = 1 − ((Σ|gᵢ|² + Σ|pⱼ|²)/2 − Σ|gᵢ ∩ pⱼ|²) / (N(N−1)/2)
```

`sklearn.rand_score` computes the same quantity. Given identical flat label vectors they would agree exactly. So the only RI gap is from Difference 1 (sample space).

## Difference 4 — VI formula (subtle: skimage vs entropy on joint)

`skimage.metrics.variation_of_information` returns `(H(gt|pred), H(pred|gt))` in bits (log2) and the wrapper sums them → standard VI. The benchmark also uses log2 (`metrics_benchmark.py:205-208`) and computes `H_1 + H_2 − 2·MI`, which is mathematically equivalent (`H(X|Y) = H(X) − I(X;Y)`).

The benchmark clamps to `1e-8` inside `log2` to avoid `-inf` for unmatched bins; skimage handles this differently. For typical inputs this is sub-1 % noise — the dominant gap is again Difference 1.

## Difference 5 — non-planar idx convention inside the contingency

In the benchmark, after `_densify_labels`:

- GT non-planar lives at index `N_gt`
- Pred non-planar lives at index `N_pred`
- Contingency shape is `(N_gt + 1, N_pred + 1)`

`valid = gt < N_gt` removes the GT-non-planar row entirely. The pred-non-planar column is preserved and populated by GT-planar pixels that the model labelled as non-planar.

In `evaluate_all_baselines.py`, no remapping happens: label 0 is the non-planar bin and it's treated as a regular segment. The contingency is `(K_gt × K_pred)` where `K_gt = #unique GT labels including 0`.

## Worked example (intuitive)

100 × 100 image, GT has 5 planes covering 6000 pixels, 4000 non-planar. Pred has 5 planes covering 5000 pixels with reasonable overlap on GT planes, plus 5000 non-planar.

| | all_baselines | benchmark |
|---|---|---|
| Sample space | 10 000 pixels | 6 000 pixels |
| Pairs in RI denominator | ~5·10⁷ | ~1.8·10⁷ |
| Dominant agreement source | (non-planar, non-planar) pair → both = 0 | (GT plane gᵢ, GT plane gᵢ) pair → both same plane |
| Typical RI | 0.95+ | 0.7 – 0.9 |
| SC inflation from non-planar bin | yes (giant gt-area) | no (excluded) |
| SC penalty for spurious pred | no | yes (via `sc_b`) |

The qualitative direction (better prediction → higher RI, lower VI, higher SC) is the same in both. Absolute values are not comparable across the two scripts.

## Which one to use

- **Paper tables, comparisons against PlaneRCNN / PlaneTR / PlanRecTR / ZeroPlane**: use the benchmark script. The published numbers in those papers were computed with the same `evaluateMasks` formula.
- **Internal A/B comparisons across your own methods**: either is fine, as long as you hold the convention constant across all methods in the same table.
- **Mixing the two in one table is wrong** — the numbers are on different scales.

## Pointers to source

| Concept | File:line |
|---|---|
| `compute_clustering_metrics` (planamono) | `planamono/evaluation/quantitative/eval_utils.py:281` |
| `segmentation_covering_fast` (one-sided SC) | `planamono/shared/plane_fitting/metrics.py:21` |
| `evaluate_masks` (ZeroPlane port) | `planamono/evaluation/quantitative/metrics_benchmark.py:145` |
| `_densify_labels` | `planamono/evaluation/quantitative/metrics_benchmark.py:86` |
| `compute_benchmark_metrics` entry point | `planamono/evaluation/quantitative/metrics_benchmark.py:740` |
| Per-frame call site (all_baselines) | `planamono/evaluation/quantitative/evaluate_all_baselines.py:752` |
| Per-frame call site (benchmark) | `planamono/evaluation/quantitative/evaluate_gt_moge_zeroplane_benchmark.py:248` |

## Recipe to make the two agree

If you want `evaluate_all_baselines.py` to produce the same numbers as the benchmark, you'd need to:

1. Densify labels the same way (planes → `0..N-1`, non-planar → `N`).
2. Mask both label maps to GT-planar pixels before calling any metric.
3. Replace `segmentation_covering_fast` with the symmetric `(sc_a + sc_b)/2` formula and use `N = #GT-planar pixels` as the denominator.
4. Optionally swap `sklearn.rand_score` and `skimage.variation_of_information` for the bincount/log2 versions in `evaluate_masks` to remove the sub-1 % library-implementation noise.

That's a meaningful redesign of the planamono kernel — not a one-line tweak. Easier path: call `evaluate_masks` directly from `evaluate_all_baselines.py` for any prediction you also want to compare to the benchmark.
