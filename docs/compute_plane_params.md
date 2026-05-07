# Plane Parameter Estimation (`compute_plane_params.py`)

Lives at: `planamono/shared/segmentation/compute_plane_params.py`

Six algorithms for fitting a 4-vector plane parameter `[a, b, c, d]` per labeled
plane instance, given `(depth, normal, plane_label)` at H × W pixel resolution.
Output planes satisfy `ax + by + cz + d = 0` with `||(a,b,c)|| = 1`.

## When to use this vs. the existing `planefit.py`

| Need | Use |
|---|---|
| Given `(H,W)` depth + normal + labels, want `{plane_id: [a,b,c,d]}` | **this module** |
| Already have `(N,3)` point cloud + labels, want fast Open3D RANSAC | `planamono.shared.plane_fitting.planefit.fit_planes_per_label_v1` |
| Want to refine an existing plane on inliers | `refine_plane_least_squares` (planefit.py:230) |

The new module is the convenience layer that takes images and handles
backprojection internally. The existing `planefit.py` is faster when you
already have a 3D point cloud (uses Open3D's optimized RANSAC).

## Inputs

```
depth        (H, W)      float32/64   meters (or affine units)
normal       (H, W, 3)   float32/64   unit normals in camera coordinates
plane_label  (H, W)      int          0 = background/non-planar, 1+ = plane IDs
```

## Output

`Dict[int, np.ndarray]` mapping `plane_id → array shape (4,)` with
`sqrt(a² + b² + c²) = 1`. Plane-id 0 (and any other ids in `ignore_labels`)
are skipped, as are planes with fewer than `min_pixels` valid-depth pixels.

## API

Six top-level functions plus a dispatcher:

```python
from planamono.shared.segmentation import (
    fit_planes_normal_average,
    fit_planes_least_squares,
    fit_planes_svd,
    fit_planes_ransac,
    fit_planes_ransac_normal,
    fit_planes_ransac_mestimator,
    compute_plane_params,   # method=... dispatcher
)

params = compute_plane_params(
    depth, normal, plane_label,
    method="ransac_normal",      # one of: normal_average, least_squares,
                                 # svd, ransac, ransac_normal, ransac_mestimator
    K=K,                         # optional (3,3) intrinsics — see "Camera handling"
    residual_threshold=0.05,     # method-specific kwargs are forwarded
    max_trials=1000,
)
# params[1] -> np.array([a, b, c, d])
```

## Algorithm summary

| # | Method | Time | Robustness | Uses normals | Tunables |
|---|---|---|---|---|---|
| 1 | `normal_average`    | O(N)     | Medium    | ✓ | – |
| 2 | `least_squares`     | O(N)     | High      | ✓ | `normal_weight`, `max_nfev` |
| 3 | `svd`               | O(N)     | High      | only for sign disambiguation | `orient_with_predicted_normal` |
| 4 | `ransac`            | O(M·N)   | Very High | ✗ | `residual_threshold`, `max_trials` |
| 5 | `ransac_normal`     | O(M·N)   | Extreme   | ✓ | `residual_threshold`, `normal_threshold`, `max_trials` |
| 6 | `ransac_mestimator` | O(M·N)   | Extreme   | ✓ | `soft_threshold`, `max_trials` |

`N` = pixels in the plane, `M` = `max_trials`. Methods 1–3 use every pixel;
methods 4–6 sample 3 points per RANSAC iteration and score the rest.

### Algorithm 1 — Direct Normal Average

Fastest, no fitting on points. Computes the mean predicted normal, normalizes
it, then sets `d = −n · centroid(points)`. Sensitive to bad normals; if the
predicted normal map is noisy the resulting `d` will be biased even when the
points themselves are clean.

### Algorithm 2 — Least Squares

Levenberg–Marquardt over `[a,b,c,d]` minimizing two residual blocks:

- point-to-plane distance `(pts · n̂) + d/||n||`
- normal consistency `1 − |n_pred · n̂|`, scaled by `normal_weight`

Initial guess is mean predicted normal + centroid offset. Output normalized
to unit `(a,b,c)`. Exception during the LM solve falls through (plane skipped).

### Algorithm 3 — SVD

Orthogonal regression. Center points around centroid, take the smallest
right-singular vector as normal, set `d = −n · centroid`. Has no parameters
but the SVD normal is sign-ambiguous; we resolve sign by aligning to the
mean predicted normal (toggle off via `orient_with_predicted_normal=False`
to match upstream `refine_plane_least_squares` behavior).

### Algorithm 4 — RANSAC (points only)

Sample 3 points, fit a plane via cross-product, count inliers within
`residual_threshold` (meters), keep the model with the most inliers. Does
not use the predicted normal map at all.

### Algorithm 5 — RANSAC + Normal Consistency

Same as Algorithm 4 but inlier set is the AND of two gates:
- `|pts · n + d| < residual_threshold`
- `1 − |n_pred · n| < normal_threshold`

`normal_threshold = 0.1` ≈ allows ~26° angular deviation per pixel
(`arccos(1 − 0.1) ≈ 25.84°`). Set `0.05` for ~18°, `0.02` for ~11°.

### Algorithm 6 — RANSAC with Tukey M-Estimator

No hard threshold. For each candidate plane, score = `Σ wp · wn` where:

- `wp = TukeyBiweight(point_distance, soft_threshold)`
- `wn = TukeyBiweight(1 − |n_pred · n|, soft_threshold/2)`

Tukey biweight is `(1 − (r/c)²)²` for `|r| < c`, else 0 — smoothly
downweights outliers without a hard cliff.

## Empirical comparison (synthetic)

Two-plane 64×64 image, plane 2 corrupted with 30% depth outliers
(uniform `[−1, 1]` m):

```
                       plane 1 (clean)        plane 2 (30% depth outliers)
normal_average     :   exact                  exact (normals untouched)
least_squares      :   exact                  ✗ 9.2° / 0.5 m off
svd                :   exact                  ✗ 9.5° / 0.5 m off
ransac             :   exact                  ✓ exact
ransac_normal      :   exact                  ✓ exact
ransac_mestimator  :   exact                  ✓ exact
```

Takeaways:

- **LS and SVD are not robust** — depth outliers fold them by ~10° / ~50 cm
  even at 30% contamination. Do not use them as your only fitter on
  predicted depth.
- **All three RANSAC variants are robust** at this contamination level
  (300 trials).
- **`normal_average` is exact when the noise is in depth only** because
  it derives the normal from the predicted normal map and only uses
  points for the centroid offset (which medians out cleanly here).
  Flip the corruption pattern (clean depth, noisy normals) and direct
  averaging is the one that breaks first.

On a single clean plane with mild noise (depth σ=5 mm, normal σ=0.01),
all six methods agree to <1° / <12 mm; the three direct methods (1–3)
are essentially exact (<0.01°), RANSAC variants carry the expected
3-point-sampling jitter (0.1°–0.6°).

## Parameter guidance

| Param | Used by | Recommended start | Notes |
|---|---|---|---|
| `residual_threshold` | 4, 5 | 0.05 m | Inlier gate on point-to-plane distance |
| `normal_threshold`   | 5    | 0.10     | `1 − cos(angle)`, so 0.10 ≈ 26°; use 0.05 (~18°) for tighter |
| `soft_threshold`     | 6    | 0.05 m   | Tukey scale; no hard cliff |
| `max_trials`         | 4, 5, 6 | 1000  | RANSAC iterations; tune via `M = log(1−0.99) / log(1−w³)` |
| `normal_weight`      | 2    | 1.0      | Relative weighting of normal vs point residuals in LM |
| `max_nfev`           | 2    | 200      | LM iteration budget |
| `min_pixels`         | all  | 3        | Minimum valid-depth pixels per plane |
| `ignore_labels`      | all  | (0,)     | Labels to skip (typical: 0 = background) |

## Camera handling

```python
K = None        # treat (u, v, depth) as 3D coords (matches the design-doc
                # pseudocode). Output `d` is in pixel-mixed units, NOT metric.
                # Useful only for debugging / sanity checks.

K = (3,3) array # proper pinhole backprojection:
                #   X = (u − cx) z / fx
                #   Y = (v − cy) z / fy
                #   Z = z
                # Output `d` is in the same units as `depth` (meters if
                # `depth` is metric, affine units if `depth` is affine).
```

**Pass `K` whenever the goal is a metric `d`.** With `K=None` the normal
direction is still meaningful (it's a direction in pixel-and-depth-mixed
space), but the offset is essentially nonsense for any downstream metric
comparison. The repo defaults to `K`-aware backprojection everywhere
else — see `backproject_v1`/`v2` in `planefit.py`.

### Affine vs. metric depth

If `depth` came from `save_moge_raw.py` without `--metric_depth`, it is
affine — i.e., metric depth divided by an unknown per-frame `metric_scale`.
The fitted normals `(a,b,c)` are unaffected (scale-invariant), but `d` is
also scaled by `1/metric_scale`. See
`planamono/inference/planarity/compare_metric_depth.py` for a diagnostic
that detects this case, and the repo CLAUDE.md note on the `--metric_depth`
flag.

## Pixel-pixel filtering

Pixels with non-positive or non-finite depth are filtered automatically
inside `_gather_plane_data()` before fitting. If a plane drops below 3
valid pixels after filtering it is skipped silently (not reported in
the output dict).

## Integration

The module is auto-exposed via the dynamic `__getattr__` in
`planamono/shared/segmentation/__init__.py`, so:

```python
from planamono.shared.segmentation import compute_plane_params  # works
```

For full plane evaluation pipelines (RANSAC + LS refinement + multi-threshold
inlier counting + RMS) see `planamono.shared.plane_fitting.metrics`,
in particular `fit_planes_and_evaluate_multi_threshold()` (3× speedup vs.
running RANSAC at each threshold).

## See also

- `planamono/shared/plane_fitting/planefit.py` — `backproject_v1/v2`,
  `fit_planes_per_label_v1`, `refine_plane_least_squares`
- `planamono/shared/plane_fitting/metrics.py` — multi-threshold P/R + RMS
- `planamono/inference/planarity/compare_metric_depth.py` — affine-vs-metric
  diagnostic for saved inference H5s
- `docs/method.md` — overall pipeline context
