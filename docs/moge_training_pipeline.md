# MoGe Training Pipeline: Depth and Surface Normal Supervision

This document explains how the MoGe (Monocular Geometry Estimation) model is trained to predict 3D geometry from a single RGB image. All references point to files under `planamono/moge/`.

## Table of Contents

- [What MoGe Predicts](#what-moge-predicts)
- [Training Data Pipeline](#training-data-pipeline)
- [Depth Supervision: Affine-Invariant Losses](#depth-supervision-affine-invariant-losses)
- [The Core Alignment Algorithm](#the-core-alignment-algorithm)
- [Normal Supervision: Edge Loss and Normal Loss](#normal-supervision-edge-loss-and-normal-loss)
- [Mask Supervision](#mask-supervision)
- [Metric Scale Supervision](#metric-scale-supervision)
- [Loss Configuration by Dataset Type](#loss-configuration-by-dataset-type)
- [Optimizer and Schedule](#optimizer-and-schedule)
- [Training Loop Details](#training-loop-details)

---

## What MoGe Predicts

MoGe does **not** predict depth directly. It predicts a **3D point map** in camera space:

| Output | Shape | Description |
|--------|-------|-------------|
| `points` | `(B, H, W, 3)` | Camera-space 3D coordinates (X, Y, Z). Depth is Z. |
| `normal` | `(B, H, W, 3)` | Unit surface normal vectors |
| `mask` | `(B, H, W)` | Validity confidence (after sigmoid) |
| `metric_scale` | `(B,)` | Global scale factor to convert to meters (after exp) |

Depth is derived at inference time as `depth = points[..., 2]`, and camera intrinsics are recovered from the point map via `recover_focal_shift()`.

Source: `moge/model/v2.py`, lines 147–202.

---

## Training Data Pipeline

Source: `moge/train/dataloader.py`

### Dataset Types

MoGe trains on 24 datasets simultaneously, each categorized into one of three label types based on the quality and format of their depth annotations:

| Type | Characteristics | Example Datasets |
|------|----------------|------------------|
| **A** (Synthetic) | Perfect metric depth, no holes | Hypersim, Structured3D, TartanAir, GTA-SfM, ObjaverseV1 |
| **B** (Real + metric) | Metric depth, may be sparse or completed | ScanNet++, ARKitScenes, Taskonomy, BlendedMVS |
| **C** (Real + relative) | Relative scale only, completed depth | MegaDepth, Argoverse2, A2D2 |

Each dataset has a sampling weight; within each training step, a dataset is chosen by weighted random sampling, then a random sample is drawn from it.

### Per-Instance Processing Pipeline

The dataloader is an async pipeline with parallel workers:

```
Sample batch (weighted random dataset + random sample)
  → Load (4 workers): read image.jpg, depth.png, meta.json
  → Process (8 workers): augment, warp, convert
  → Batch + Collate
  → Buffer (prefetch 8)
```

Each instance goes through the following processing (`_process_instance`, line 143):

**Step 1: Compute GT normals from depth**

```python
raw_normal, raw_normal_mask = utils3d.np.depth_map_to_normal_map(
    raw_depth, intrinsics=raw_intrinsics,
    mask=np.isfinite(raw_depth), edge_threshold=88
)
```

Surface normals are computed from depth using finite differences. Pixels at depth discontinuities (edge gradient exceeding threshold) are masked to avoid nonsensical normals at object boundaries.

**Step 2: Random perspective augmentation**

A random target viewpoint is sampled by perturbing the field of view and principal point:
- FoV is sampled relative to the original (`fov_range_relative`) and clamped to absolute bounds (`fov_range_absolute: [1°, 179°]`)
- Principal point is jittered by `center_augmentation`

This produces a homography `transform = K_tgt @ R @ K_src^{-1}` that warps the image to a virtual camera.

**Step 3: Warp depth in disparity space**

```python
# Bilinear interpolation in disparity maintains planar surfaces
warped_depth_bilinear = 1 / warp_perspective(1 / raw_depth, transform, ...)
# Nearest-neighbor at depth edges to avoid blending across discontinuities
warped_depth_nearest = warp_perspective(raw_depth, transform, ..., interpolation='nearest')
# Blend: bilinear where safe, nearest at edges
tgt_depth = np.where(bilinear_mask == 1.0, warped_depth_bilinear, warped_depth_nearest)
```

The depth is then converted from warped depth to z-depth in the target camera frame:

```python
tgt_depth = warped_depth / np.dot(tgt_uvhomo, np.linalg.inv(transform)[2, :])
```

**Step 4: Warp normals**

```python
warped_normal = warp_perspective(raw_normal, transform, ..., interpolation='bilinear')
tgt_normal = warped_normal @ R.T  # Rotate normals to target frame
```

**Step 5: Flip augmentation**

Random horizontal flip. Normal x-component is negated on flip.

**Step 6: Color augmentation**

Per-dataset augmentation: jittering, JPEG compression artifacts, Gaussian blur, depth-of-field blur, shot noise.

**Step 7: Depth clipping and masking**

```python
max_depth = np.nanquantile(tgt_depth, 0.01) * clamp_max_depth  # clamp_max_depth=1000
tgt_depth = np.clip(tgt_depth, 0, max_depth)
```

Removes the most extreme 1% of depth values. Two masks are produced:
- `depth_mask_fin`: finite (valid) pixels
- `depth_mask_inf`: infinite (sky/invalid) pixels

For some datasets (`finite_depth_mask: "only_known"`), only pixels with known depth contribute to training.

### Batch Output

```python
batch = {
    'image':          (B, 3, H, W),   # RGB [0, 1]
    'depth':          (B, H, W),       # z-depth in meters (if metric)
    'normal':         (B, H, W, 3),    # unit surface normals
    'depth_mask_fin': (B, H, W),       # bool: valid depth
    'depth_mask_inf': (B, H, W),       # bool: infinite/invalid
    'intrinsics':     (B, 3, 3),       # camera K matrix
    'label_type':     list[str],       # 'A', 'B', 'C', or 'invalid'
    'is_metric':      list[bool],      # whether depth is in metric units
}
```

---

## Depth Supervision: Affine-Invariant Losses

Source: `moge/train/losses.py`

The central challenge: training data comes from dozens of datasets with different depth scales, units, and completeness. MoGe addresses this with **affine-invariant losses** that align predictions to ground truth before computing the error.

### Global Loss (`affine_invariant_global_loss`)

**Purpose**: Supervise the overall 3D structure of the predicted point map with a single per-image alignment.

**Algorithm** (line 31):

1. **Downsample to alignment resolution** (default 64×64):
   ```python
   (pred_lr, gt_lr), lr_mask = mask_aware_nearest_resize(
       (pred_points, gt_points), mask=mask, size=(64, 64)
   )
   ```
   `mask_aware_nearest_resize` downsamples while respecting the validity mask — each low-res pixel takes the value of the nearest valid high-res pixel. This ensures the alignment is not biased by invalid regions.

2. **Solve for scale and z-shift**:
   ```python
   weight = lr_mask / gt_lr[..., 2].clamp_min(1e-2)    # inverse-depth weighting
   scale, shift = align_points_scale_z_shift(pred_lr, gt_lr, weight, trunc=1.0)
   ```
   This finds the scalar `s` and z-shift `t` that minimize:
   ```
   min_{s, t}  Σ_i  w_i · |s · pred_i + [0, 0, t] − gt_i|
   ```
   Only a z-shift is used (not full XYZ shift) because the camera is assumed roughly upright. The weight `w_i = 1/Z_i` gives higher importance to nearby geometry.

3. **Apply the alignment**:
   ```python
   pred_aligned = scale * pred_points + shift    # shift is [0, 0, t_z]
   ```

4. **Compute inverse-depth-weighted L1 loss**:
   ```python
   weight = mask.float() / gt_points[..., 2].clamp_min(1e-5)
   weight = weight.clamp_max(10 * weighted_mean(weight, mask))  # clip outliers
   loss = smooth(|pred_aligned − gt_points| * weight, beta).mean()
   ```

   The `_smooth` function is a smooth Huber loss:
   ```
   smooth(e, β) = { e²/(2β)   if e < β
                  { e − β/2   otherwise
   ```
   When `β=0` (default), this reduces to plain L1.

5. **Return metrics**:
   - `truncated_error`: mean of `min(||pred − gt|| / Z, 1.0)` over valid pixels
   - `delta`: fraction of pixels with relative error < 1

   The `scale` is also returned — it is passed to the local loss as a sanity check.

### Local Loss (`affine_invariant_local_loss`)

**Purpose**: Supervise fine-grained geometry at multiple scales using independently aligned patches.

Global alignment captures the overall structure, but a single scale+shift cannot model local surface details. The local loss independently aligns small patches, allowing the model to learn fine geometry even when the global alignment is imperfect.

**Three scales** are used simultaneously (from `configs/train/v2.json`):

| Level | Conceptual Patch Size | Num Patches | Alignment Resolution |
|-------|-----------------------|-------------|---------------------|
| 4 | 1/4 of image diagonal | 16 | 24×24 |
| 16 | 1/16 of image diagonal | 256 | 12×12 |
| 64 | 1/64 of image diagonal | 4096 | 6×6 |

**Algorithm** (line 111):

1. **Compute patch radius** in 2D and 3D:
   ```python
   radius_2d = ceil(0.5 / level * sqrt(H² + W²))           # pixels
   radius_3d = 0.5 / level / focal * gt_points[..., 2]     # meters (depth-dependent)
   ```
   Both pixel distance AND 3D distance must be within the radius for a point to belong to a patch. This prevents patches from spanning depth discontinuities.

2. **Importance sampling of patch anchors**:
   ```python
   weights = compute_anchor_sampling_weight(gt_points, mask, radius_2d, radius_3d)
   anchors = multinomial(weights, num_patches)
   ```
   `compute_anchor_sampling_weight` (line 77) estimates local point density by counting how many valid points fall within the patch radius of each pixel. Pixels in sparse/fine structures (e.g., thin objects, edges) get higher sampling weight, ensuring the model sees enough of these difficult regions.

   For each candidate anchor, 64 random offsets are tested. The weight is `1 / count_within_radius` — areas with fewer neighbors are sampled more often.

3. **Extract patch points**:
   ```python
   patch_mask = (within_radius_2d) & (gt_mask) & (within_radius_3d)
   ```
   Patches with fewer than 32 valid points are discarded.

4. **Independent per-patch alignment** using full 3D scale + XYZ shift:
   ```python
   local_scale, local_shift = align_points_scale_xyz_shift(
       pred_patch_lr, gt_patch_lr, weight, trunc=1.0
   )
   ```
   Unlike global alignment (z-shift only), local patches allow full 3D translation. This accommodates cases where the global alignment doesn't capture local geometry well.

   **Sanity check**: the local scale must be within 0.1× to 10× of the global scale:
   ```python
   patch_valid = (0.1 < local_scale / global_scale) & (local_scale / global_scale < 10)
   ```

5. **Compute patch loss** with harmonic-mean normalization:
   ```python
   gt_mean = harmonic_mean(gt_depth, gt_mask)                    # over entire image
   weight = patch_mask / gt_patch[..., 2].clamp_min(0.1 * gt_mean)
   loss = smooth(|aligned_pred − gt| * weight, beta).mean()
   ```
   The harmonic mean (reciprocal of mean reciprocal) is less sensitive to extreme depth values than arithmetic mean, providing more stable normalization.

6. **Aggregate**: patch losses are summed per image and divided by `num_patches`, then averaged across the batch.

---

## The Core Alignment Algorithm

Source: `moge/utils/alignment.py`

All affine-invariant losses depend on the `align()` function (line 52), which solves:

```
min_a  Σ_i  w_i · |a · x_i − y_i|
```

or with truncation:

```
min_a  Σ_i  min(τ, w_i · |a · x_i − y_i|)
```

### Without truncation (`trunc=None`)

This is a weighted L1 regression problem. The optimal `a` equals `y_j / x_j` for some data point `j`. The algorithm:

1. Sort by ratio `y_i / x_i`
2. Compute cumulative sum of weights `w_i · x_i`
3. The derivative of the objective changes sign at data points — find the zero crossing via `searchsorted`
4. Return `a = y_j / x_j` at the zero crossing

This runs in O(n log n) due to sorting.

### With truncation (`trunc=τ`)

Truncation makes the loss robust to outliers — errors above `τ` contribute a constant penalty instead of growing linearly. The algorithm is more involved:

1. For each data point, compute three critical values:
   - `A_i = y_i / x_i` (the breakpoint where error is zero)
   - `B_i = (w_i·y_i − τ) / (w_i·x_i)` (where truncation kicks in from below)
   - `C_i = (w_i·y_i + τ) / (w_i·x_i)` (where truncation kicks in from above)

2. Sort A, B, C independently and compute prefix sums of weights

3. For each candidate `a = A_i`, compute left and right derivatives using the prefix sums

4. Find extrema (local minima) where left derivative < 0 and right derivative ≥ 0

5. Evaluate the objective at all extrema, return the global minimum

This is an exact solver (not iterative) that finds the global optimum.

### Scale + Shift Variants

| Function | Solves | Used By |
|----------|--------|---------|
| `align_points_scale(pred, gt, w)` | `min \|\|s·pred − gt\|\|` | — |
| `align_points_scale_z_shift(pred, gt, w)` | `min \|\|s·pred + [0,0,t] − gt\|\|` | Global loss |
| `align_points_scale_xyz_shift(pred, gt, w)` | `min \|\|s·pred + t − gt\|\|` | Local loss |

The scale+shift variants work by anchoring: for each candidate anchor point, subtract its coordinates to remove the shift, then solve for scale using `align()`. The best anchor is selected by evaluating the loss at all candidates and taking the minimum (`scatter_min`). The final scale and shift are recomputed from the two selected data points for a shorter gradient graph.

---

## Normal Supervision: Edge Loss and Normal Loss

Source: `moge/train/losses.py`, lines 205–258

Surface normals are supervised **indirectly through geometry**: the losses compare directions derived from the predicted point map against directions from the GT point map. The model's `normal_head` output is not directly supervised in the current training config.

### Edge Loss (`edge_loss`) — Primary, weight=1.0

The simplest normal supervision. Compares gradient directions between neighboring pixels:

```python
dx = points[i, j, :] − points[i+1, j, :]    # vertical edge vectors
dy = points[i, j, :] − points[i, j+1, :]    # horizontal edge vectors
```

These edge vectors encode local surface orientation. The loss is:

```python
loss = angle_diff(dx_pred, dx_gt) + angle_diff(dy_pred, dy_gt)
```

where `angle_diff(v1, v2) = atan2(||v1 × v2||, v1 · v2)` — the angle between two 3D vectors.

The angle is clamped to [0.1°, 90°] and passed through a smooth Huber loss with β=3°:
```
smooth(θ, 3°) = { θ²/(6°)   if θ < 3°
                { θ − 1.5°  otherwise
```

Normalization: `loss / (2 · max(H, W))` scales the loss relative to image resolution.

**Why edge loss works for normals**: edge vectors span the local tangent plane. If pred and GT edges agree in direction, the surfaces have the same orientation. This is equivalent to comparing normals but more numerically stable because it avoids explicit cross-product normalization.

### Normal Loss (`normal_loss`) — Alternative

Computes surface normals explicitly from four triangles per pixel:

```
For each 2×2 quad of points (leftup, rightup, leftdown, rightdown):
  upxleft    = cross(rightup − rightdown,  leftdown − rightdown)
  leftxdown  = cross(leftup − rightup,     rightdown − rightup)
  downxright = cross(leftdown − leftup,    rightup − leftup)
  rightxup   = cross(rightdown − leftdown, leftup − leftdown)
```

Each cross product yields an (unnormalized) surface normal from a different triangle. The loss is the angular difference between pred and GT normals, averaged over all four triangles. Pixels where any of the three triangle vertices are invalid are masked.

Normalization: `loss / (4 · max(H, W))`.

The v2 config uses **`edge_loss`** (not `normal_loss`) as the primary normal supervision.

---

## Mask Supervision

Source: `moge/train/losses.py`, lines 261–270

### `mask_bce_loss` (used in v2 config, weight=0.1)

```python
loss = (gt_mask_pos | gt_mask_neg) * BCE(pred_mask, gt_mask_pos.float())
```

Where:
- `gt_mask_pos` = `depth_mask_fin` (pixels with valid finite depth)
- `gt_mask_neg` = `depth_mask_inf` (pixels with infinite depth, e.g., sky)
- Pixels that are neither (e.g., out-of-view after augmentation) are excluded

The mask head learns to predict which pixels have valid geometry.

### `mask_l2_loss` (alternative, not used in v2)

```python
loss = gt_mask_neg · pred² + gt_mask_pos · (1 − pred)²
```

Simple L2 regression toward 0 or 1.

---

## Metric Scale Supervision

The `scale_head` is an MLP that takes the CLS token and predicts a scalar `exp(output)` representing the conversion factor from the model's internal scale to metric meters.

In the training loop (line 316–318):

```python
if is_metric[i] and pred_metric_scale is not None:
    loss, misc = metric_scale_loss(pred_metric_scale[i], gt_metric_scale)
```

The `gt_metric_scale` is the `scale` value returned by `affine_invariant_global_loss` — it represents what factor was needed to align predictions to metric GT. The scale head learns to predict this factor directly.

**Note**: `metric_scale_loss` and `normal_map_loss` are imported in `train.py` but not defined in `losses.py` in our fork. These would need to be implemented to use those loss terms.

---

## Loss Configuration by Dataset Type

From `configs/train/v2.json`:

### Type A (Synthetic — full supervision)

| Loss | Function | Weight | Params |
|------|----------|--------|--------|
| `global` | `affine_invariant_global_loss` | 1.0 | `align_resolution=48` |
| `patch_4` | `affine_invariant_local_loss` | 1.0 | `level=4, num_patches=16, align_resolution=24` |
| `patch_16` | `affine_invariant_local_loss` | 1.0 | `level=16, num_patches=256, align_resolution=12` |
| `patch_64` | `affine_invariant_local_loss` | 1.0 | `level=64, num_patches=4096, align_resolution=6` |
| `normal` | `edge_loss` | 1.0 | — |
| `normal_map` | `normal_map_loss` | 0.1 | — |
| `metric_scale` | `metric_scale_loss` | 0.1 | — |
| `mask` | `mask_bce_loss` | 0.1 | — |

### Type B (Real + metric — no finest patches)

| Loss | Function | Weight | Params |
|------|----------|--------|--------|
| `global` | `affine_invariant_global_loss` | 1.0 | `align_resolution=48` |
| `patch_4` | `affine_invariant_local_loss` | 1.0 | `level=4, num_patches=16, align_resolution=24` |
| `patch_16` | `affine_invariant_local_loss` | 1.0 | `level=16, num_patches=256, align_resolution=12` |
| `normal` | `edge_loss` | 1.0 | — |
| `normal_map` | `normal_map_loss` | 0.1 | — |
| `metric_scale` | `metric_scale_loss` | 0.1 | — |
| `mask` | `mask_bce_loss` | 0.1 | — |

Type B drops `patch_64` (finest patches) because real-world depth maps don't have the detail needed at that resolution.

### Type C (Real + relative — minimal)

| Loss | Function | Weight | Params |
|------|----------|--------|--------|
| `global` | `affine_invariant_global_loss` | 1.0 | `align_resolution=48` |
| `patch_4` | `affine_invariant_local_loss` | 1.0 | `level=4, num_patches=16, align_resolution=24` |
| `metric_scale` | `metric_scale_loss` | 0.1 | — |
| `mask` | `mask_bce_loss` | 0.1 | — |

Type C drops normal supervision and fine patches entirely — relative-scale depth is too noisy for these.

### Invalid

No losses computed. The batch instance is effectively skipped.

---

## Optimizer and Schedule

Source: `configs/train/v2.json` and `moge/scripts/train.py`

### Optimizer: AdamW with two parameter groups

```json
{
    "type": "AdamW",
    "params": [
        {"params": {"include": ["*"], "exclude": ["*.backbone.*"]}, "lr": 1e-4},
        {"params": {"include": ["*.backbone.*"]}, "lr": 1e-5}
    ]
}
```

- **Heads, neck, projections**: lr = 1e-4
- **DINOv2 backbone**: lr = 1e-5 (10× lower to preserve pretrained features)

### Learning Rate Schedule: SequentialLR

```
Step 0–1000:    LR = 0 (frozen)
Step 1000–2000: Linear warmup from 0 to full LR
Step 2000+:     Halve LR every 25,000 steps (StepLR, γ=0.5)
```

### Low-Resolution Warm-Up

For the first 50,000 steps, `num_tokens` is fixed at the minimum (1200). After that, `num_tokens` is randomly sampled from [1200, 3600] per batch. This gives the model a stable training signal early on before introducing resolution variation.

---

## Training Loop Details

Source: `moge/scripts/train.py`

### Forward Pass (line 284–292)

```python
num_tokens = random.randint(1200, 3600)  # randomized per batch
with autocast(dtype=float16, enabled=enable_mixed_precision):
    output = model(image, num_tokens=num_tokens)
pred_points, pred_mask, pred_normal, pred_metric_scale = ...
```

GT 3D points are computed from depth + intrinsics:

```python
gt_points = utils3d.pt.depth_map_to_point_map(gt_depth, intrinsics=gt_intrinsics)
```

### Per-Instance Loss (line 296–334)

Losses are computed per-instance (not per-batch) because each instance may come from a different dataset type with different loss configurations.

For each instance `i`:
1. Look up the loss config for its `label_type` (A, B, C, or invalid)
2. Compute `affine_invariant_global_loss` first — this returns `gt_metric_scale` (the alignment scale)
3. Pass `gt_metric_scale` to `affine_invariant_local_loss` for the sanity check
4. Compute all other losses
5. Weighted sum: `total_loss = Σ (weight_k · loss_k)`

The per-instance losses are averaged to form the batch loss.

### Gradient Handling

- **Gradient clipping**: `clip_grad_norm_(model.parameters(), 1.0)` (line 346)
- **NaN detection**: if any gradient is NaN, the update is skipped entirely (line 341–345)
- **NaN loss**: logged but not skipped (the backward pass still runs)
- **Gradient accumulation**: configurable via `gradient_accumulation_steps`

### EMA (Exponential Moving Average)

```python
ema_avg = 0.999 * ema_param + 0.001 * model_param
```

Updated after every optimizer step on the main process only. The EMA model is saved separately and used for inference (more stable than raw training weights).

### Checkpointing

Three files are saved every `save_every` steps (default 10,000):
- `{step}.pt`: model weights + config
- `{step}_optimizer.pt`: optimizer state + LR scheduler state
- `{step}_ema.pt`: EMA model weights

Saving is done in a background thread to avoid blocking training.

---

## Summary: How It All Fits Together

```
24 datasets (A/B/C types, weighted sampling)
  ↓
Per-instance augmentation:
  - Random perspective warp (FoV, principal point, rotation)
  - Depth warped in disparity space (1/Z), nearest at edges
  - Normals rotated to new frame
  - Color jitter, JPEG, blur, DoF
  ↓
Forward: image → DINOv2 → neck → heads → (points, normal, mask, scale)
  ↓
Loss computation (per instance, per dataset type):
  - Global: align(scale + z-shift) → inverse-depth-weighted L1
  - Local @3 scales: importance-sampled patches → independent align → L1
  - Normals: edge direction angle error (smooth Huber, β=3°)
  - Mask: BCE on valid/invalid pixels
  - Scale: align model's internal scale to metric GT
  ↓
AdamW (backbone 1e-5, heads 1e-4) + gradient clipping + EMA
```

The affine-invariant design is what enables training across datasets with wildly different depth scales. The model learns **shape** (up to scale+shift) from all data, and learns **metric scale** only from datasets that provide it.
