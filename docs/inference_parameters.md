# Inference & Segmentation Parameters

This document explains every parameter used in the MoGe inference + region growing segmentation pipeline.

## Pipeline Overview

```
RGB Image
  │
  ▼
┌─────────────────────────────────┐
│  MoGe 4-Head Model              │  ← num_tokens, batch_size, target_height/width
│  (ViT-Large + DINOv2 backbone)  │
└──────┬──────┬──────┬──────┬─────┘
       │      │      │      │
  planarity  depth  normal  mask
  (sigmoid)  (z)    (unit)  (sigmoid)
       │      │      │
       ▼      ▼      ▼
┌─────────────────────────────────┐
│  Resize to original resolution  │  ← bilinear for continuous, NEAREST for binary
└──────┬──────┬──────┬────────────┘
       │      │      │
       ▼      ▼      ▼
┌─────────────────────────────────┐
│  Planarity thresholding         │  ← threshold_planarity
└──────┬──────┬──────┬────────────┘
       │      │      │
       ▼      ▼      ▼
┌─────────────────────────────────┐
│  Region growing (v5, GPU)       │  ← normal_threshold_deg, depth_threshold,
│  5×5 neighborhood matching      │    neighbor_match_count_thresh
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Connected components (cc3d)    │
│  + label remapping              │
└─────────────────────────────────┘
       │
       ▼
  plane_labels (H, W) int, 0 = non-planar
```

## MoGe Inference Parameters

### `num_tokens` (int, default: 1024)

Controls the spatial resolution of the ViT backbone's feature grid. The image is encoded into a grid of `base_h × base_w` tokens where:

```
base_h = floor(sqrt(num_tokens / aspect_ratio))
base_w = floor(sqrt(num_tokens * aspect_ratio))
```

For a 4:3 image (e.g. 1024×768), this gives roughly 29×35 = 1015 tokens. Higher values produce finer-grained features but cost more memory (quadratic in attention). The actual token count is `base_h * base_w` (may differ slightly from the requested value due to rounding).

All head outputs are bilinearly upsampled from this token grid back to the input image resolution.

**Typical values:** 1024 (default for all scripts). No script currently changes this.

### `batch_size` (int)

Number of images processed in a single GPU forward pass.

| Script | Default | Notes |
|--------|---------|-------|
| `inference_to_h5.py` (ScanNet++) | 16 | Higher because JPEG loading is fast |
| `inference_to_h5_hypersim.py` | 8 | Lower because HDF5/HDR loading + tonemapping is heavier |
| `moge_inference_only.py` | 1 | Processes one image at a time (no batching) |
| `run_inference.py` | 1 | Single-image predict loop |

Memory usage scales linearly with batch size. On a single GPU with 24 GB VRAM, batch_size=16 works for 512×768 inputs with ViT-Large.

### `target_height` / `target_width` (int, default: 512 / 768)

The input image is resized to this resolution before feeding to the model. Hardcoded in `preprocess_image()` / `preprocess_images()` (not exposed as CLI args).

The model was trained on 512×768 inputs. Changing these values is not recommended — the ViT backbone and DPT neck expect this aspect ratio for the positional embeddings.

After inference, outputs are resized back to the original image resolution.

### `device` (str, default: "cuda")

Torch device. Falls back to CPU if CUDA is unavailable, but inference is impractically slow on CPU.

### `model_path` (str)

Path to a `.pt` checkpoint containing `model_state_dict`. The checkpoint is loaded onto a freshly initialized `MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")` with an added planarity head (deep-copied from the normal head, last conv replaced with 1-channel output).

## Segmentation Parameters

These control `compute_vectorized_planar_segments_v5()` — the GPU-accelerated region growing that converts per-pixel predictions into instance plane labels.

### `threshold_planarity` (float, default: 0.6)

Threshold applied to the MoGe planarity probability map (output of sigmoid, range [0, 1]) to produce a binary mask:

```python
planarity_mask = (planarity_prob > threshold_planarity).astype(np.int32)
```

Pixels below this threshold are immediately labeled as non-planar (label 0) and excluded from all subsequent steps. This is the single most impactful parameter — it controls the recall/precision tradeoff of the entire pipeline.

| Value | Effect |
|-------|--------|
| 0.3–0.4 | High recall, includes uncertain pixels, more noisy segments |
| **0.5** | Neutral threshold (used in `run_inference.py` and internal binary output) |
| **0.6** | Default in H5 inference scripts. Slightly conservative, fewer false positives |
| 0.7–0.8 | High precision, misses uncertain planar regions |

**Note:** The inference class internally applies a hardcoded 0.5 threshold for `planarity_binary` output. The `threshold_planarity` parameter is applied *separately* in the pipeline scripts on the raw probability, *after* the model inference step.

### `normal_threshold_deg` (float, default: 10.0 degrees)

Maximum angular difference in surface normals between neighboring pixels for them to be considered part of the same planar region. Converted to radians before use:

```python
normal_threshold_rad = np.deg2rad(normal_threshold_deg)
```

Internally, instead of computing per-pair dot products, v5 uses a Sobel-filter approach: it computes the gradient magnitude of the normal map and compares it against a derived threshold:

```python
grad_mag_threshold = sqrt(2 - 2 * cos(normal_threshold_rad))
normal_similar = (normal_grad_mag <= grad_mag_threshold)
```

This produces a per-pixel boolean mask: `True` where the local neighborhood has consistent normals. The Sobel gradient captures normal variation across a 3×3 window, so the effective check is whether any neighbor within a 3×3 patch deviates by more than the threshold. This is then used as one of three conditions in the 5×5 matching step.

| Value | Effect |
|-------|--------|
| 5° | Very strict — only nearly-coplanar regions merge. Over-segments curved surfaces |
| **10°** | Default. Good balance for indoor scenes |
| 15–20° | Permissive — allows slightly curved surfaces into single segments |

### `depth_threshold` (float, default: 0.05 meters)

Maximum absolute depth difference (in meters) between a center pixel and a neighbor for them to be considered connected:

```python
depth_close = (|center_depth - neighbor_depth| < depth_threshold)
```

This prevents merging across depth discontinuities (e.g., a wall and a distant object at the same normal orientation).

| Value | Effect |
|-------|--------|
| 0.01–0.02 | Very strict. Breaks segments at small depth steps |
| **0.05** | Default. Tolerates MoGe depth noise while respecting major boundaries |
| 0.1+ | Permissive. Risk of merging foreground/background with similar normals |

**Important:** This operates on MoGe's predicted depth, which is affine-invariant (not metric). The absolute scale depends on the scene, so this threshold's effective strictness varies. For close-up scenes, 0.05m is strict; for large rooms, it's permissive.

### `neighbor_match_count_thresh` (int, default: 24)

Minimum number of matching neighbors (out of 24 in a 5×5 window, excluding center) required for a pixel to be considered "connected" to a planar region:

```python
connected = (neighbor_match_count >= neighbor_match_count_thresh)
```

A neighbor "matches" if **all three** conditions hold:
1. The neighbor is planar (passes `threshold_planarity`)
2. The local normal gradient is below `normal_threshold_deg`
3. The depth difference is below `depth_threshold`

After computing the connected mask, `cc3d.connected_components()` labels distinct connected regions.

| Value | Effect |
|-------|--------|
| 4–8 | Permissive. Only a few neighbors need to agree. Produces larger, noisier segments. Connects across partial boundaries |
| 12–16 | Moderate. Roughly half the neighborhood must agree |
| 20–22 | Strict. Most neighbors must agree. Clean boundaries but may fragment valid planes |
| **24** | Maximum strictness (all 24 neighbors must match). Only pixels fully surrounded by consistent planar neighbors survive. Very clean segments but can erode plane boundaries by ~2 pixels |

**Note:** The maximum possible value is 24 (all neighbors in the 5×5 window minus center). Setting this to 24 means every single neighbor must match — this is the most aggressive denoising but also the most lossy at segment boundaries.

## Parameter Defaults Across Scripts

| Parameter | `inference_to_h5.py` | `inference_to_h5_hypersim.py` | `run_inference.py` | `moge_inference_only.py` |
|-----------|---------------------|-------------------------------|---------------------|--------------------------|
| `num_tokens` | 1024 | 1024 | 1024 | 1024 |
| `batch_size` | 16 | 8 | 1 | 1 |
| `threshold_planarity` | 0.6 | 0.6 | 0.5 (`--threshold`) | N/A |
| `normal_threshold_deg` | 10.0 | 10.0 | N/A | N/A |
| `depth_threshold` | 0.05 | 0.05 | N/A | N/A |
| `neighbor_match_count_thresh` | 24 | 24 | N/A | N/A |
| Segmentation? | Yes (v5) | Yes (v5) | No | No |
| Split | val | val | N/A | val |

## Inference Class Variants

Two `MoGePlanarityInference` classes exist with a subtle difference:

| File | Normal output format | Used by |
|------|---------------------|---------|
| `moge_inference_v1.py` | `(H, W, 3)` — correct, no transpose | `inference_to_h5.py` (ScanNet++) |
| `moge_inference.py` | `(H, W, 3)` after `transpose(1,2,0)` from CHW | `inference_to_h5_hypersim.py`, `run_inference.py`, `moge_inference_only.py` |

Both produce the same final shape, but `moge_inference.py` transposes because it assumes CHW output, while `moge_inference_v1.py` correctly recognizes MoGe v2 already outputs HWC. See `BUG_NORMAL_TRANSPOSE.md` for history.

The downstream `inference_to_h5.py` (ScanNet++) then does `normal.transpose(2, 0, 1)` to get `(3, H, W)` for the segmentation function, which expects `(H, W, 3)` and transposes back — so the chain works out despite the different conventions.
