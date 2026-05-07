# Method: Pixelwise Planarity Prediction and Plane Segmentation

## Overview

We predict per-pixel planarity from a single RGB image and segment planar surfaces using a two-stage pipeline:

1. **Planarity prediction**: A modified MoGe model with 4 output heads predicts planarity probability, depth, surface normals, and 3D points from a single image.
2. **Plane segmentation**: A GPU-accelerated region growing algorithm groups planar pixels into plane instances using normal consistency, depth continuity, and connected component analysis.

```
RGB Image → MoGe 4-Head Model → Planarity + Depth + Normals + Points
                                        ↓
                    Planarity thresholding → Binary planar mask
                                        ↓
                    Neighbor voting (normal + depth consistency)
                                        ↓
                    Connected components → Plane instance labels
```

## Stage 1: Planarity Prediction (MoGe 4-Head)

### Backbone

We build on **MoGe v2** (`Ruicheng/moge-2-vitl-normal`), which uses a **DINOv2 ViT-Large** backbone (`dinov2_vitl14`, patch size 14). Input images are resized so the number of patch tokens equals a target count (1024 at inference). The aspect ratio is preserved:

```
base_h = floor(sqrt(num_tokens / aspect_ratio))
base_w = floor(sqrt(num_tokens * aspect_ratio))
input_resolution = (14 * base_h, 14 * base_w)
```

Images are normalized with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).

### Architecture

The encoder extracts multi-scale features from DINOv2 intermediate layers, projects them to a shared dimension via 1x1 convolutions, and sums them. A shared **ConvStack neck** processes 5 feature pyramid levels. At each level, normalized UV coordinates encoding the image aspect ratio are concatenated to the features.

Four parallel **ConvStack decoder heads** produce the outputs:

| Head | Channels | Activation | Output |
|------|----------|------------|--------|
| `points_head` | 3 | none | Affine-invariant 3D point map `(H,W,3)` |
| `normal_head` | 3 | L2-normalize | Surface normals `(H,W,3)` |
| `mask_head` | 1 | sigmoid | Valid pixel confidence `(H,W)` |
| `planarity_head` | 1 | sigmoid | Planarity probability `(H,W)` |

Each ConvStack decoder uses an FPN-style architecture with residual blocks (GroupNorm + Conv3x3 + skip), transposed convolution upsampling stages, and UV coordinate concatenation at each scale. A separate `scale_head` MLP takes the CLS token and outputs a metric scale factor (exp-activated).

### Planarity Head Initialization

The planarity head is initialized by **deep-copying the pretrained normal head**, then replacing the final Conv2d layer (3 channels → 1 channel). The new layer's weights are initialized from the old layer's mean across output channels, scaled by 0.1 to prevent extreme initial logits:

```python
new_conv.weight = old_conv.weight.mean(dim=0, keepdim=True) * 0.1
new_conv.bias   = old_conv.bias.mean().unsqueeze(0) * 0.1
```

### Training

**Two-phase strategy:**

| Phase | Epochs | Unfrozen Parameters | Optimizer | LR Schedule |
|-------|--------|---------------------|-----------|-------------|
| 1 | 2 | Only the new final Conv2d in planarity head | AdamW (lr=1e-3, wd=1e-4) | OneCycleLR (pct_start=0.3) |
| 2 | 28 | Entire planarity head | AdamW (lr=1e-4, wd=1e-5) | ReduceLROnPlateau (factor=0.7, patience=5) |

All other model components (backbone, neck, points/normal/mask heads) remain frozen. BatchNorm layers in frozen modules are set to eval mode.

**Loss function**: Weighted sum of three components applied to planarity logits:

```
L = w_bce * L_bce + w_dice * L_dice + w_focal * L_focal
```

| Loss | Formula | Default Weight |
|------|---------|----------------|
| BCE | `F.binary_cross_entropy_with_logits(logits, gt)`, weighted by mask confidence | 0.34 |
| Soft Dice | `1 - (2 * Σ(p·y) + ε) / (Σp + Σy + ε)`, where `p = σ(logits)` | 0.33 |
| Focal | `(1 - p_t)^γ * BCE_per_pixel`, γ=2.0 | 0.33 |

The MoGe mask head's output serves as a soft confidence weight on BCE and Focal losses, downweighting uncertain pixels.

**Training data**: Mixed dataset combining ScanNet++ (real indoor scans, instance labels) and Hypersim (synthetic scenes, binary planarity). Binarization to `{0, 1}` happens at batch time via `(plane_labels > 0).float()`.

**Mixed precision**: bfloat16/float16 autocast with gradient scaling and gradient clipping (max_norm=1.0).

### Inference

At inference, the model processes an image resized to 512x768 with 1024 tokens:

- **Planarity**: sigmoid-activated probability map `(H,W)` in `[0,1]`
- **Depth**: recovered from affine point map via focal length estimation (`recover_focal_shift`)
- **Normals**: L2-normalized surface normals `(H,W,3)`
- **Mask**: binary valid-pixel mask; planarity is zeroed where mask is False

Batch inference is natively supported.

## Stage 2: Plane Segmentation

Given the predicted planarity, normals, and depth, we segment planar regions using GPU-accelerated neighbor voting followed by connected component labeling.

### Algorithm (v5 — Default)

**Step 1: Planarity thresholding**

Create a binary mask of planar pixels: `planar = (planarity_prob > τ)`, default τ=0.6.

**Step 2: Sobel edge detection on normals**

Compute the gradient magnitude of the 3-channel normal map using batched Sobel convolution on GPU:

```
∂n/∂x = conv2d(normal, sobel_x, groups=3)
∂n/∂y = conv2d(normal, sobel_y, groups=3)
grad_mag = sqrt(Σ_c (∂n_c/∂x)² + (∂n_c/∂y)²)
```

Pixels where the gradient magnitude is below a threshold derived from the angular threshold are marked as normal-consistent:

```
grad_threshold = sqrt(2 - 2·cos(θ_normal))     # θ_normal default = 10°
normal_similar = (grad_mag ≤ grad_threshold)
```

**Step 3: 5x5 neighbor voting**

Extract 5x5 neighborhoods using `F.unfold` (24 neighbors, excluding center pixel):

For each pixel, count neighbors satisfying all three conditions:
1. **Both planar**: center and neighbor are in the planarity mask
2. **Normal consistent**: both have similar normals (from Sobel check)
3. **Depth close**: `|depth_center - depth_neighbor| < δ_depth` (absolute, default 0.05m)

```
match_count = Σ (valid_pair AND normal_similar AND depth_close)
connected = (match_count ≥ threshold)     # default threshold = 8 of 24
```

**Step 4: Connected components**

Apply `cc3d.connected_components()` on the boolean connected-pixel map to produce instance labels.

### Algorithm Variants

| Version | Key Difference | Default Vote Threshold |
|---------|----------------|----------------------|
| **v5** | Sobel normal gradients, absolute depth threshold | 8/24 (33%) |
| **v6** | Pairwise normal dot-product per neighbor, relative depth threshold (`δ * depth_center`) | 18/24 (75%) |
| **v10** | Adaptive: requires `α` fraction of *valid* neighbors (preserves boundaries), small segment filter | 75% of valid neighbors |
| **v11** | Decoupled normal/depth voting with separate thresholds, optional post-merge of adjacent segments | 75% of valid neighbors |

**v10 adaptive threshold**: Instead of requiring a fixed count, require a fraction α (default 0.75) of valid neighbors (those that are planar and have positive depth):

```
adaptive_thresh = max(α · valid_count, min_valid_neighbors)
connected = (match_count ≥ adaptive_thresh) AND (valid_count ≥ min_valid_neighbors)
```

This preserves boundary pixels where fewer neighbors are available (e.g., at plane edges where only 8 of 24 neighbors are valid, the threshold becomes 6 instead of 18).

**v11 post-merge**: After connected components and small segment removal, iteratively merges adjacent segments whose mean normals and plane offsets are within thresholds (`merge_normal_deg`, `merge_offset_m`), with dilation-based gap bridging (`merge_gap_px`).

### Output

`labels (H,W) int32` — 0 = non-planar, positive integers = plane instance IDs.

## Ground Truth Generation

For training and evaluation, we generate plane ground truth from 3D meshes via a 7-stage pipeline:

1. **Label-strict region growing** on mesh faces — BFS growth with dihedral angle and distance gates, never crossing semantic label boundaries
2. **EM sweep** — iterative expansion of plane regions with 8 quality gates (p95 residual, inlier fraction, thickness, normal error, width, fill fraction)
3. **IRLS robust plane fitting** — iteratively reweighted least squares with Huber loss (k=1.345, 8 iterations), converging via MAD-based scale estimation
4. **Large label splitting** — recursive spatial bisection of oversized groups for memory safety
5. **Quality filtering** — 8-gate geometric test discards poorly-fit planes
6. **Parametric merging** — consolidates compatible planes using normal angle and offset thresholds with gap-based clustering
7. **Raycasting to 2D** — renders per-pixel plane labels and depth via Open3D raycasting

## Evaluation

### 2D Segmentation Metrics

| Metric | Definition | Better |
|--------|-----------|--------|
| **Segmentation Covering (SC)** | Area-weighted mean best-IoU: `SC = Σ_k (area_k · max_j IoU(k,j)) / total_area` | Higher |
| **Rand Index (RI)** | Fraction of pixel pairs with consistent labeling | Higher |
| **Variation of Information (VOI)** | Sum of conditional entropies: `H(S\|M) + H(M\|S)` | Lower |

### 3D Plane Geometry Metrics

For each predicted plane segment:

1. **Backproject** depth to 3D points (pinhole for ScanNet++, V-Ray rays for Hypersim)
2. **RANSAC plane fitting** (3-point, 200 iterations) at the evaluation threshold τ, followed by least-squares SVD refinement on inliers
3. **Inlier counting** at threshold τ: `inlier = (|ax + by + cz + d| < τ)`
4. **Inlier ratio gate** (default 0.9): segments with `n_inliers / n_points < gate` contribute 0 inliers (penalizes low-quality fits)

| Metric | Definition |
|--------|-----------|
| **Precision @ τ** | `total_inliers / total_predicted_planar_points` |
| **Recall @ τ** | `total_inliers / total_valid_scene_points` |

Default evaluation thresholds: τ ∈ {1mm, 5mm, 10mm}.

RANSAC is run separately at each evaluation threshold (current method, resolving earlier inconsistency where a fixed 2cm RANSAC was shared across all thresholds).
