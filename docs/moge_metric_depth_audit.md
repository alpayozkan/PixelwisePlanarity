# MoGe-2 Metric Depth — Audit of `moge_HIRES_4datasets_epoch1` Inference Dump

**TL;DR.** The saved H5 dump at `/cluster/scratch/ayavuz/inference/moge_HIRES_4datasets_epoch1/scannetpp/test/<scene>/inference.h5` stores **affine** depth, not metric. The model checkpoint itself can produce metric depth (its `scale_head` works correctly upstream — `model.infer()` recovers GT to ~3 %), but the inference code that produced this dump bypasses the `metric_scale` multiplication. Plane offsets fitted from this depth will be in arbitrary units; plane normals are unaffected.

---

## Affected Data

| Path | Notes |
|------|-------|
| `/cluster/scratch/ayavuz/inference/moge_HIRES_4datasets_epoch1/scannetpp/test/<scene>/inference.h5` | 42 ScanNet++ test scenes, ~436 frames each, 480 × 640 |
| `depth` field inside the H5 | **affine** (this audit) |
| `gt_depth` field inside the H5 | metric (ground truth) |
| Scenes also include: `gt_planes`, `normals`, `planarity`, `mask`, `intrinsics`, `gt_intrinsics`, `gt_pose`, `has_gt_pose`, `gt_sem`, `frame_ids` | |

The audit script and findings apply equally to any other dataset under the same `moge_HIRES_4datasets_epoch1/` parent (Hypersim / Synthia / VKITTI2) if those dumps used the same inference code path.

---

## Code-Level Evidence

In `planamono/moge/moge/model/v2.py`:

| Line | Behavior |
|------|----------|
| 65–66 | `if scale_head is not None: self.scale_head = MLP(**scale_head)` — `scale_head` is instantiated **only when** the loaded config provides `scale_head` kwargs. |
| 174 | `metric_scale = self.scale_head(cls_token) if hasattr(self, 'scale_head') else None` — predicted from CLS token in `forward()`. |
| 191 | `metric_scale = metric_scale.squeeze(1).exp()` — always positive. |
| 296–300 | In `infer()`: `if metric_scale is not None: points *= metric_scale[..., None, None, None]; depth *= metric_scale[..., None, None]` — **only `infer()` applies the scale**. |

Confirmed empirically:

```
$ python -c "from planamono.moge.moge.model.v2 import MoGeModel; \
    m = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal'); \
    print(hasattr(m, 'scale_head'))"
True
```

The upstream checkpoint (`Ruicheng/moge-2-vitl-normal`) carries `scale_head`. So does the fine-tuned `moge_HIRES_4datasets_epoch1` checkpoint (it inherits structure from the upstream). The capability is present; whether output depth is metric depends entirely on which code path the inference script used.

### Two code paths for getting depth out of MoGe-2

| Path | Output | Where |
|------|--------|-------|
| **`model.infer(image, ...)`** | metric depth (scale applied internally) | `v2.py:205–340` |
| **`out = model(image, ...); points = out['points']; depth = points[...,2]`** | **affine depth** (scale **NOT** applied) | raw forward |

`save_moge_raw.py`'s docstring explicitly notes this distinction:

> `--metric_depth` (uses `model.infer()` for absolute-scale depth instead of affine `points[:,:,2]`)

If `--metric_depth` is **not** passed, the dump is affine — exactly what this audit observes.

---

## Empirical Evidence

### How we tested

`planamono/inference/planarity/compare_metric_depth.py` (added in the same change set) loads the upstream `Ruicheng/moge-2-vitl-normal` checkpoint, picks N evenly-spaced frames from the H5, loads each frame's iPhone RGB at 480 × 640, and computes for each frame:

| Quantity | How |
|----------|-----|
| **`saved`** | `f['depth'][i]` from the inference H5 |
| **`gt`** | `f['gt_depth'][i]` from the inference H5 |
| **`infer`** | `model.infer()` depth — manually replicated via `forward()` + `recover_focal_shift()` + `× metric_scale` (the official `infer()` fails on the cluster's older `utils3d 0.1.1` which lacks `utils3d.torch.intrinsics_from_focal_center`) |
| **`affine`** | `points[..., 2] + shift` from `recover_focal_shift()`, **without** the scale multiplication |
| **`raw_z`** | `points[..., 2]` from raw forward, **without** the shift correction |

Median ratios per frame are reported, then aggregated over the sampled frames.

If the saved dump is metric:
- `saved / gt ≈ 1.0` with low variance
- `saved / infer ≈ 1.0` with low variance

If the saved dump is affine:
- `saved / gt` varies frame-to-frame by the (inverse of) `metric_scale`
- `saved / affine ≈ 1.0` with very low variance

### Result on `09c1414f1b` (5 frames)

```
 frame   scale | saved/gt        infer/gt       affine/gt       raw_z/gt       | saved/infer    saved/affine
     0   2.227 | med=0.435 σ=.03 med=0.952 σ=.06 med=0.427 σ=.03 med=0.448 σ=.03 | med=0.454 σ=.01 med=1.012 σ=.03
   108   1.333 | med=0.745 σ=.06 med=0.996 σ=.07 med=0.747 σ=.05 med=0.769 σ=.06 | med=0.747 σ=.02 med=0.996 σ=.03
   217   2.030 | med=0.483 σ=.03 med=0.981 σ=.06 med=0.483 σ=.03 med=0.496 σ=.03 | med=0.492 σ=.01 med=0.999 σ=.01
   326   1.033 | med=1.003 σ=.10 med=1.013 σ=.11 med=0.980 σ=.10 med=0.999 σ=.10 | med=0.992 σ=.03 med=1.025 σ=.03
   435   1.656 | med=0.631 σ=.04 med=1.045 σ=.06 med=0.631 σ=.04 med=0.645 σ=.04 | med=0.599 σ=.01 med=0.993 σ=.02
```

Across-frame aggregate of per-frame median ratios:

| Ratio | mean | std | range |
|-------|-----:|----:|-------|
| `saved / gt` | 0.659 | 0.204 | [0.435, 1.003] |
| **`infer / gt`** | **0.997** | **0.031** | [0.952, 1.045] |
| `affine / gt` | 0.654 | 0.198 | [0.427, 0.980] |
| `raw_z / gt` | 0.672 | 0.199 | [0.448, 0.999] |
| `saved / infer` | 0.657 | 0.196 | [0.454, 0.992] |
| **`saved / affine`** | **1.005** | **0.012** | [0.993, 1.025] |

Two ratios pin down the conclusion:

1. **`infer / gt ≈ 0.997 ± 0.031`** — `model.infer()` is genuinely metric to ~3 %. The `scale_head` works on this scene.
2. **`saved / affine ≈ 1.005 ± 0.012`** — across all 5 frames, the saved depth is virtually indistinguishable from the (no-scale) affine output. The 1.2 % residual std is consistent with `moge_HIRES_4datasets_epoch1` being a fine-tuned variant of the upstream checkpoint (small per-frame deviations from upstream `points[..., 2] + shift`), and the consistent ~0.5 % bias is consistent with that as well.

In the one frame (326) where the predicted `metric_scale` happens to be ≈ 1 (1.033), `saved/gt` becomes ≈ 1.00 — the saved depth incidentally matches GT scale. For frames with `metric_scale = 2.23` (frame 0) or 2.03 (frame 217), `saved/gt` drops to 0.43 / 0.48 — exactly `1 / metric_scale`.

### Cross-scene sanity check

A weaker version of the same test using just `saved / gt` (no model load) on a second scene:

| Scene | mean(`gt/saved`) | std | range |
|-------|-----:|----:|------|
| `09c1414f1b` | 1.61 | 0.43 | 1.01 – 2.45 |
| `0d2ee665be` | 1.02 | 0.22 | 0.65 – 1.39 |

Both scenes show per-frame variance much larger than the ~3 % we see for `infer / gt` — independent confirmation that the saved depth is not metric.

---

## What's Recoverable from the Existing Dump

`metric_scale` is a single scalar per image, predicted by a small MLP from the CLS token. Two options for retrofitting an existing dump without re-running the heavy heads:

### Option 1 — re-run Stage 1 with the metric-depth path

Cleanest. Either:
- run `model.infer()` end-to-end inside the dump script, or
- pass the existing `--metric_depth` flag in `save_moge_raw.py`.

Pays the cost of a second forward pass over every frame.

### Option 2 — recover `metric_scale` only

For each frame, run only the encoder + `scale_head` on RGB (skip the points / normal / planarity / mask heads). `scale_head` is a tiny MLP on the CLS token; the bulk of the cost is the encoder forward, but at 1024 tokens this is a fraction of full inference. Multiply the saved affine depth by the recovered scalar in place. Augment the H5 with a `metric_scale` field of shape `(N,)` and rewrite `depth = saved_depth * metric_scale` if you want the dataset on disk to be metric.

### Option 3 — accept affine and live with it

Plane *normals* `n̂` are scale-invariant — affine depth is fine for them. Plane *offset* `d` is in affine units; comparing `d` to GT in meters is meaningless without a per-frame rescaling. For ablation studies that don't compare offsets across the affine/metric boundary this is OK; for any 3D precision/recall metric at fixed distance thresholds (1 mm, 5 mm, 10 mm) this is **not** OK.

---

## How to Re-Run the Audit

```bash
conda activate planeseg
python planamono/inference/planarity/compare_metric_depth.py \
    --inference_h5 /cluster/scratch/ayavuz/inference/moge_HIRES_4datasets_epoch1/scannetpp/test/09c1414f1b/inference.h5 \
    --frames 5 --num_tokens 1024
```

Optional flags:
- `--rgb_native` — run MoGe at the native iPhone resolution and resample predictions back to 480 × 640 (default: resize RGB to 480 × 640 before MoGe).
- `--num_tokens` — match what the dump was produced with (1024 was the default in the older `inference_to_h5.py`; 1600 in `save_moge_raw.py`). Choice of `num_tokens` should not affect the *metric vs affine* conclusion; it only affects per-frame accuracy of the upstream prediction.
- `--scene_id` — override scene ID if the H5 path doesn't follow the `<scene>/inference.h5` layout.

The script prints both per-frame ratios and across-frame aggregates; only the aggregates matter for the metric-vs-affine determination.

---

## Caveats / Things That Don't Change the Conclusion

- **`moge_HIRES_4datasets_epoch1` is not the upstream checkpoint.** It's a fine-tuned variant. The audit uses upstream `Ruicheng/moge-2-vitl-normal` to provide an independent metric reference; the `saved / infer` ratio compares two different models so a small bias is expected. What matters is the per-frame *variance* of `saved / infer` (huge) vs `saved / affine` (tiny). The variance, not the absolute level, is the diagnostic.
- **`utils3d 0.1.1` mismatch.** The cluster has `utils3d 0.1.1`, which lacks `utils3d.torch.intrinsics_from_focal_center` (upstream MoGe expects a newer version). The audit script bypasses this by replicating the relevant lines of `infer()` manually (forward → `recover_focal_shift` → shift + scale). This does not affect the metric-depth determination; it just lets the script run on this cluster.
- **Resolution.** The H5 stores 480 × 640. The audit resizes RGB to 480 × 640 before running MoGe by default (`--rgb_native` is opt-in). Any resampling artifacts at boundaries cancel out in the per-frame *ratio* statistic.
- **`num_tokens`.** The audit defaults to 1024; the dump's actual `num_tokens` is unknown but doesn't affect whether `metric_scale` was applied — the `× metric_scale` step is independent of token count.

---

## Files Added/Touched

| File | Purpose |
|------|---------|
| `planamono/inference/planarity/compare_metric_depth.py` | Audit script. Compares saved-H5 depth, `model.infer()` depth (metric), affine depth, raw `points[..., 2]`, and GT. |
| `docs/moge_metric_depth_audit.md` | This document. |

No production code changed. Whether to retrofit existing dumps (Option 2) or re-run Stage 1 (Option 1) is a separate decision and not implemented here.
