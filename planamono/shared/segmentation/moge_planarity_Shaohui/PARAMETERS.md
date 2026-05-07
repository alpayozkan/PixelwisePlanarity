# Shaohui's MoGe Planarity — Parameters & Design Choices

Source: `B1ueber2y/limap-structure-dev`, branch `tmp/improve_planes`

## Files

| File | Purpose |
|------|---------|
| `model.py` | `MogePlanarity` — limap `BasePlaneDetector` integration |
| `moge_planarity.py` | `MoGePlanarityInference` — MoGe 4-head inference wrapper |
| `plan2seg.py` | `compute_planar_segments()` — GPU segmentation + `remove_small_components()` |

## Segmentation Parameters (`model.py:_detect_plane_mask_impl`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold_planarity` | **0.6** | Binary threshold on planarity probability |
| `neighbor_match_count_thresh` | **18** | 75% of 24 neighbors (5x5 window minus center) |
| `normal_threshold_deg` | **10.0** | Angular threshold for normal similarity |
| `depth_threshold` | **0.02** | **Relative** threshold: 2% of center pixel depth |
| `min_size` (small component removal) | **500** | Minimum pixels per segment |

## Inference Parameters

| Parameter | Value |
|-----------|-------|
| `num_tokens` | 1024 |
| `target_height x target_width` | 512 x 768 (fixed resize) |
| Checkpoint | `final_planarity_4heads_model.pt` (hardcoded path) |

## Design Choices

### 1. Pairwise Normal Comparison (Direct Dot Product)

The segmentation checks each pixel against its 24 neighbors using **direct pairwise dot product**:

```
angle = arccos(clamp(dot(center_normal, neighbor_normal), -1, 1))
connected = angle < normal_threshold_rad
```

This compares the actual angle between each pixel pair, not a gradient magnitude.

### 2. Relative Depth Threshold

Depth proximity is **relative to center pixel depth**:

```
depth_close = |center_depth - neighbor_depth| < depth_threshold * center_depth
```

With `depth_threshold=0.02`, a pixel at 5m depth allows 10cm variation, while a pixel at 1m depth allows only 2cm. This adapts to scene scale.

### 3. High Neighbor Match Count (18/24 = 75%)

Requires 75% of neighbors to match before a pixel is considered connected. The comment says this "reduces boundary erosion" — i.e., pixels at region boundaries where half the neighbors belong to a different plane are more likely to be excluded.

### 4. Post-Segmentation Small Component Removal

After connected component labeling, segments with < 500 pixels are removed via `remove_small_components()`. This uses `scipy.ndimage.label` to split each label into spatially connected sub-components, then filters by size.

### 5. Depth from Points Head (Z-channel)

Depth is extracted from MoGe's `points` head: `depth = res["points"][:, :, 2]` (the Z coordinate of the 3D point map), not from a separate depth head.

### 6. Fixed Resolution Inference

All images are resized to 512x768 for inference, then results are resized back to original resolution. Planarity and depth use bilinear interpolation; binary masks use nearest-neighbor.
