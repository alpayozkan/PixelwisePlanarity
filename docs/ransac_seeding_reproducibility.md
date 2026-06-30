# RANSAC Seeding & Evaluation Reproducibility

**Status:** Implemented (2026-06-29). Evaluation now seeds RANSAC by default
(`ransac_seed=0`). Sections 1–5 describe the original (pre-fix) state and the
investigation; section 6 documents the implemented fix and how to use it.
**Scope:** Whether random seeds are set in the inference / evaluation pipeline, with
particular focus on the RANSAC-based 3D plane metrics (precision / recall @ τ).

## TL;DR

- **Originally:** RANSAC plane fitting in evaluation was NOT seeded and was
  non-deterministic — no `seed=` was passed to Open3D's `segment_plane(...)`, and
  Open3D's global RNG (`open3d.utility.random`) was never seeded. 3D metrics
  (`prec@<τ>cm`, `plane_recall_*`, `offset_err_*`) wiggled run-to-run; 2D metrics
  (RI / VI / SC) were always deterministic (no RANSAC).
- **Now (fix implemented):** the evaluation entry points
  (`evaluate_single_frame*` in `eval_utils.py`) default to **`ransac_seed=0`** and
  seed Open3D's RNG per frame, so all ~33 scripts that call them are reproducible by
  default. `evaluate_gt_quality.py` and `evaluator.py` (their own RANSAC paths) are
  seeded too. Pass `--ransac-seed -1` (in `evaluate_all_baselines.py`) or
  `ransac_seed=None` to restore the legacy non-deterministic behaviour.
- A NumPy/Python seed does **not** control Open3D's C++ RNG — only
  `o3d.utility.random.seed()` (global) or the per-call `seed=` argument
  (open3d ≥ 0.18) does. Both are now used (see §6).

---

## 1. Where RANSAC happens

All RANSAC plane fitting funnels through Open3D's `PointCloud.segment_plane(...)`.
There are exactly three call sites, and **none** pass a `seed`:

| File:line | Function | Notes |
|-----------|----------|-------|
| `planamono/shared/plane_fitting/planefit.py:323` | `fit_planes_per_label_v1()` | Main per-label fitter (RANSAC + LS refinement) |
| `planamono/shared/plane_fitting/metrics.py:206` | `fit_planes_and_evaluate_multi_threshold()` | Multi-threshold fast path; single RANSAC, evaluated at several τ |
| `planamono/evaluation/quantitative/evaluator.py:226` | sequential multi-plane RANSAC | Iteratively peels planes from a residual cloud |

The canonical call (from `planefit.py:323`):

```python
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts[idx_global])
plane_model, inliers_local = pcd.segment_plane(
    distance_threshold=distance_threshold,
    ransac_n=ransac_n,
    num_iterations=num_iterations,
)   # <-- no seed= argument
```

These functions are what the evaluation scripts call to compute 3D precision/recall:
`eval_utils.py` → `evaluate_single_frame*()` → the fitters above. The
benchmark metric kernel (`metrics_benchmark.py`, used by
`evaluate_gt_moge_zeroplane_benchmark.py`) computes only RI / VI / SC plus a
segmentation-based recall and contains **no** RANSAC; the only RANSAC in that script
is the appended "planamono RANSAC prec/rec block" which routes back to the fitters above.

---

## 2. How Open3D seeds RANSAC internally

Open3D (≥0.16) centralizes randomness in a process-wide singleton,
`open3d::utility::random` — a global `std::mt19937` Mersenne-Twister engine.

- **Initial seed comes from `std::random_device()`** the first time the RNG is used,
  i.e. a fresh OS-entropy value per process. **There is no fixed default seed.**
- `segment_plane`'s RANSAC draws its minimal-sample point indices
  (`ransac_n` points × `num_iterations` rounds) from that global engine.
- The global engine can be pinned deterministically with
  `open3d.utility.random.seed(n)` — but this is **opt-in**, and our code never calls it.

### Version differences (matters — the two env files pin different versions)

| Env file | open3d | `segment_plane(seed=...)`? | Default behavior |
|----------|--------|---------------------------|------------------|
| `env/environment.yml` (primary → `planeseg`) | **0.19.0** | Yes, `seed=None` default | `None` ⇒ uses global RNG ⇒ **non-deterministic** |
| `env/environment_v1.yml` | **0.17.0** | **No `seed` param at all** | always global RNG ⇒ **non-deterministic** |

On 0.18+ you *could* pass `seed=` to make a single call deterministic; the code does
not, so it falls through to the unseeded global RNG either way.

---

## 3. Empirical confirmation

Tested with the Open3D available on the cluster (`/cluster/home/aoezkan/jup_env`,
which carries **open3d 0.17.0**, matching `environment_v1.yml`). Three back-to-back
unseeded fits on the **same** point cloud produced **different** planes and inlier
counts — proof there is no internal default seed:

```python
import open3d as o3d, numpy as np
np.random.seed(123)
pts = np.random.rand(500, 3); pts[:, 2] = 0.3 + 0.01 * np.random.randn(500)

def run(seed="UNSET"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    kw = dict(distance_threshold=0.01, ransac_n=3, num_iterations=200)
    if seed != "UNSET":
        kw["seed"] = seed
    m, inl = pcd.segment_plane(**kw)
    return tuple(np.round(m, 5)), len(inl)

print(run()); print(run()); print(run())
```

Output (open3d 0.17.0):

```
no-seed A: (..., d=-0.30476), 350 inliers
no-seed B: (..., d=-0.29958), 341 inliers
no-seed C: (..., d=-0.30430), 348 inliers
seed kwarg NOT supported in this version   # 0.17 cannot be seeded per-call
```

The plane offset `d` and the inlier count both change across identical inputs.

> **Note:** `inspect.signature(...).segment_plane)` raises
> `ValueError: no signature found for builtin` because `segment_plane` is a pybind11
> C++ method — introspect the docstring (`help(o3d.geometry.PointCloud.segment_plane)`)
> instead of `inspect.signature`.

---

## 4. Audit: every seed in the codebase

A full grep of `planamono/` for `seed` / `manual_seed` / `default_rng` /
`RandomState` / `np.random.seed` / `random.seed`. **None of these are in the RANSAC
metric path.** They fall into three harmless buckets:

### (a) Visualization — label coloring only
- `planamono/shared/plane_fitting/visualize_planes.py:82` — `np.random.default_rng(int(pid))`
- `planamono/evaluation/quantitative/visualize_scannetpp_all_baselines.py:563`,
  `..._v1.py:610`, `..._v2.py:581` — `np.random.seed(args.random_seed)`
- `planamono/evaluation/quantitative/visualize_and_generate_pdf.py:684,713`
- `planamono/evaluation/qualitative/visualize_predictions_pdf.py:103` (`seed 42`), `:232`
- `planamono/evaluation/qualitative/compare_signals_vs_zeroplane.py:329` (`0xC0FFEE`), `:335`, `:543`
- `planamono/evaluation/qualitative/compare_signals_vs_zeroplane_ablation.py:277,521`
- `planamono/evaluation/qualitative/visualize_metric3d.py:135,307`

### (b) Grid search — config / scene sampling only
- `planamono/evaluation/quantitative/grid_search_v10.py:273` — `random.seed(42)`
- `planamono/evaluation/quantitative/grid_search_v10_hypersim.py:277` — `random.seed(42)`
- `planamono/evaluation/quantitative/grid_search_segmentation.py:148` — `random.seed(42)`

### (c) Training / eval dataloader — not the metric RNG
- `planamono/evaluation/run_evaluation.py:90-97` — `torch.manual_seed(args.seed)`,
  `np.random.seed(...)`, `random.seed(...)`, plus per-worker seeds.

### (d) Inference helper scripts — local comparisons, not metric RANSAC
- `planamono/inference/planarity/segment_from_inference_h5.py:88` — `default_rng(seed=0)`
- `planamono/inference/planarity/compare_metric_depth.py:482`,
  `compare_metric3d_vs_moge.py:326`, `compare_plane_param_methods.py:207`,
  `visualize_plane_methods.py:210,311,367`

**Confirmed absent:**
- No `seed=` on any `segment_plane(...)` call.
- No `o3d.utility.random.seed(...)` / `open3d.utility.random` call anywhere
  (grep returns nothing).
- No global seed in `evaluate_all_baselines.py`, `eval_utils.py`,
  `evaluate_scannetpp*.py`, `evaluate_gt_moge_zeroplane*.py`.

---

## 5. Practical impact

- **2D metrics (RI, VI, SC):** deterministic — computed from label maps, no RANSAC.
- **3D metrics (`prec@<τ>cm`, `plane_recall_d/n_*`, `offset_err_m_*`, `normal_err_*`
  where they depend on fitted planes):** **non-deterministic**, fluctuating run-to-run.
- Variance is **small but nonzero**: `RANSAC_ITERATIONS = 200` (see
  `eval_utils.py` / per-script constants) plus the least-squares refinement on inliers
  (`refine_plane_least_squares` in `planefit.py`) damps it. In practice prec/recall@τ
  moves at roughly the 3rd–4th decimal — usually below the gap you report between
  methods, but **not** exactly reproducible.
- Tight thresholds (1 mm, 5 mm) are the most sensitive; 1 cm is the most stable.

Relevant constants (hardcoded per eval script, see CLAUDE.md "Evaluation Configuration"):
`THRESHOLDS = (0.001, 0.005, 0.01)` m, `RANSAC_ITERATIONS = 200`,
`INLIER_RATIO_GATE = 0.9`.

---

## 5b. Do the OTHER metrics need a seed? (depth / normal / 2D / params)

Audited every metric family. **Open3D RANSAC plane fitting is the ONLY randomness
source in the evaluation metrics.** Everything else is deterministic:

| Metric family | Where | Randomness? | Seed needed? |
|---------------|-------|-------------|--------------|
| **Depth** (REL, REL², log10, RMSE, RMSE-log, δ<1.25^n) | `metrics_benchmark.py::evaluate_depths` | None — pure numpy reductions over valid pixels | No |
| **Normal** (per-pixel/per-plane recall <5/11.25/22.5/30°, mean/median angle err) | `metrics_benchmark.py::eval_plane_recall_normal`, `eval_plane_bestmatch_normal_offset` | None — angle arithmetic + IoU greedy match + Hungarian `scipy.optimize.linear_sum_assignment` (deterministic) | No |
| **Offset / plane-param L2** | `metrics_benchmark.py::eval_plane_recall_offset`, `evaluate_masks` | None | No |
| **2D segmentation** (RI, VI, SC) | `metrics_benchmark.py::evaluate_masks`, `eval_utils::compute_clustering_metrics` | None — bincount / set ops | No |
| **Binary planarity** (bp_accuracy/precision/recall/f1/iou) | `eval_utils::compute_binary_planarity_metrics` | None | No |
| **Per-plane params for normal/offset metrics** | `compute_plane_params(method="svd")` | None — orthogonal regression via `np.linalg.svd` | No |
| **3D plane prec/recall @ τ** | `planefit.py::segment_plane` (Open3D RANSAC) | **Yes** | **Yes — fixed in §6** |

Notes:
- `compute_plane_params` *also* offers RANSAC variants (`fit_planes_ransac`,
  `fit_planes_ransac_normal`, `fit_planes_ransac_mestimator`) that use their **own**
  NumPy generator (`np.random.default_rng(rng)` with `rng=None` ⇒ non-deterministic).
  **These are not used by any evaluation script** — all eval paths set
  `MOGE_FIT_METHOD = "svd"` (and GT params via the deterministic
  `compute_gt_normals_from_depth_labels`). If you ever switch a metric to a RANSAC
  `method=`, pass an explicit `rng=np.random.default_rng(0)`.
- Hungarian matching ties: `linear_sum_assignment` is deterministic for a given cost
  matrix, so equal-cost ties don't introduce run-to-run variance.

**Conclusion:** no additional seeding is required for depth, normal, offset, param,
2D-segmentation, or binary-planarity metrics. Seeding the plane RANSAC (§6) makes the
entire metric suite reproducible.

---

## 6. Implemented fix

Both seeding mechanisms are wired, combined so they work on **both** open3d 0.17
(`environment_v1.yml`) and 0.19 (`environment.yml` / `planeseg`):

1. **Global RNG, per frame (primary).** `set_ransac_seed(seed)` (in `planefit.py`)
   calls `o3d.utility.random.seed(seed)`. It is called at the top of every
   `evaluate_single_frame*` (and `evaluate_gt_frame`, `pseudo_mono_infer`), i.e.
   **inside each loky worker, per frame** — so determinism is independent of worker
   scheduling and works on 0.17 (which has no per-call `seed`).
2. **Per-call seed (defense, open3d ≥ 0.18).** `_segment_plane(...)` (in `planefit.py`)
   forwards `seed=` to `segment_plane` *only when the installed open3d supports it*
   (detected once at import by `_detect_segment_plane_seed_support()` via a tiny
   throw-away fit). `fit_planes_per_label_v1` and `evaluator.py`'s candidate loop use it.

`seed=None` everywhere restores the legacy non-deterministic behaviour.

### Where the seed defaults live — `0` (reproducible) everywhere
- **`fit_planes_per_label_v1`** (`planefit.py`) is the chokepoint: default
  `ransac_seed=0`, and it **self-seeds the global RNG** (`set_ransac_seed`) at the top
  of every call. This means **every direct caller is reproducible by default without
  edits** — including the non-fast scripts (`evaluate_scannetpp.py`,
  `evaluate_scannetpp_merged.py`, `_profile`, `_gtseg`, `_gtplanarity_ourseg`,
  `_ourplanarity_gtseg`), `evaluator.py`, and the visualizers
  (`visualize_scannetpp_all_baselines{,_v1,_v2}.py`, `visualize_and_generate_pdf.py`).
- **`fit_planes_and_evaluate_multi_threshold`** (`metrics.py`),
  **`compute_plane_metrics{,_old,_multigates}`** and the six
  **`evaluate_single_frame*`** (`eval_utils.py`) → all default `ransac_seed=0` and
  thread it down. `evaluate_single_frame*` also call `set_ransac_seed()` at the top
  (belt-and-suspenders, documents intent).
- **`evaluate_all_baselines.py`** → `RANSAC_SEED=0` constant + `--ransac-seed` flag
  (loky-safe closure-local), printed in `[CONFIG]`.
- **`evaluate_scannetpp_fast.py`, `evaluate_hypersim_fast.py`** → `RANSAC_SEED=0`
  constant, threaded + printed.
- **`evaluate_gt_moge_zeroplane_benchmark.py`** → `--ransac_seed` flag (default 0),
  threaded through the `delayed()` args to `_eval_one_frame_benchmark`, which seeds
  per frame and passes it to its direct `compute_plane_metrics` call.
- **`evaluate_gt_quality.py`, `evaluate_scannet_gt_quality.py`, `evaluator.py`
  (`pseudo_mono_infer`)** → seed `0`.
- Pass `ransac_seed=None` (or `--ransac-seed -1`) anywhere to restore legacy
  non-deterministic behaviour.

### Usage
```bash
# Default: reproducible (seed = 0) — no flag needed
python evaluate_all_baselines.py --methods ours zeroplane

# Override the seed
python evaluate_all_baselines.py --ransac-seed 123

# Restore legacy NON-deterministic behaviour (variance studies)
python evaluate_all_baselines.py --ransac-seed -1
```
For the `*_fast.py` and other scripts (no CLI flag), edit the `RANSAC_SEED` constant
at the top of the file, or set `ransac_seed=` on the `evaluate_single_frame*` call.

### Implementation notes / caveats
- **loky workers re-import module globals.** A seed mutated in `main()` does **not**
  propagate to loky workers via a module global. `evaluate_all_baselines.py` therefore
  captures `RANSAC_SEED` into a **closure local** (`_ransac_seed`) inside
  `evaluate_method` so the value travels with the cloudpickled `eval_frame_wrapper`.
  The per-frame `set_ransac_seed()` call also runs inside the worker, which is what
  actually guarantees determinism regardless of scheduling.
- **OpenMP threads.** Open3D's RANSAC parallelizes iterations across OpenMP threads;
  for bit-exact numbers across machines also fix `OMP_NUM_THREADS=1`
  (`torch.set_num_threads(1)` in some workers does **not** constrain Open3D's pool).
- **One-time numeric shift.** Switching from an unseeded RNG to a fixed `seed=0`
  changes which RANSAC samples are drawn, so newly-computed 3D metrics differ from
  previously-reported numbers by the same ~3rd–4th-decimal margin as the old
  run-to-run variance. This is expected; re-baseline once.

### Verified
On open3d 0.17 (`jup_env`): `_SEGMENT_PLANE_HAS_SEED == False` (per-call seed correctly
skipped → global-seed fallback), and three seeded `fit_planes_per_label_v1` runs on the
same points produced **identical** plane models, while unseeded runs differed.

---

## 7. Scripts referenced in this report

**RANSAC / metric code**
- `planamono/shared/plane_fitting/planefit.py` (`fit_planes_per_label_v1`, `refine_plane_least_squares`)
- `planamono/shared/plane_fitting/metrics.py` (`fit_planes_and_evaluate_multi_threshold`, `compute_precision_recall_v1`)
- `planamono/evaluation/quantitative/evaluator.py` (sequential multi-plane RANSAC)
- `planamono/evaluation/quantitative/eval_utils.py` (`evaluate_single_frame*` entry points)
- `planamono/evaluation/quantitative/metrics_benchmark.py` (no RANSAC — RI/VI/SC only)

**Evaluation drivers (no seeding in metric path)**
- `evaluate_all_baselines.py`, `evaluate_scannetpp_fast.py`,
  `evaluate_hypersim_fast.py`, `evaluate_gt_moge_zeroplane{,_v1,_benchmark}.py`,
  `evaluate_gt_quality.py`

**Environment**
- `env/environment.yml` — `open3d==0.19.0` (primary `planeseg`)
- `env/environment_v1.yml` — `open3d==0.17.0`

## See also
- `planamono/evaluation/quantitative/METRIC_INCONSISTENCY_ANALYSIS.md` — the v1
  RANSAC-at-eval-threshold change (separate reproducibility concern: *which* threshold
  RANSAC runs at, not *seeding*).
- `docs/open3d_float32_segfault.md` — cast to float64 before `o3d.utility.Vector3dVector`.
- `docs/metrics_planes.md`, `planamono/evaluation/quantitative/METRICS.md` — metric definitions.
