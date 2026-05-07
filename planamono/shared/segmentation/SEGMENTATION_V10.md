# Segmentation v10: Adaptive Pairwise Planar Segmentation

## Overview

v10 segments a predicted planarity map into plane instances using GPU-accelerated neighborhood voting with connected components. It builds on v9 (which replaced v5's Sobel normal check with pairwise normal dot products and relative depth) by adding two improvements:

1. **Adaptive voting threshold** -- scales the required match count to the number of valid neighbors, preserving boundary pixels
2. **Minimum segment filter** -- removes tiny connected components that are too small for reliable plane fitting

## Algorithm Steps

### Input

| Input | Shape | Description |
|-------|-------|-------------|
| `planarity_mask` | (H, W) | Binary mask: 1 = planar pixel, 0 = non-planar |
| `normal` | (H, W, 3) | Per-pixel unit surface normals |
| `depth` | (H, W) | Depth map in meters |

All three come from the MoGe 4-head model inference.

### Step 1: Pairwise Normal Comparison

For each pixel, extract its 24 neighbors in a 5x5 window (excluding center). Compute the dot product between the center pixel's normal and each neighbor's normal:

```
dot(n_center, n_neighbor) > cos(normal_threshold_rad)
```

This is a **per-edge** check: two adjacent pixels are similar only if their normals point in nearly the same direction. This replaces v5's Sobel gradient, which computed a single scalar gradient magnitude per pixel and could miss boundaries between planes whose normals had similar magnitudes but different directions (e.g., two walls meeting at 90 degrees where both have smooth normal fields -- low Sobel gradient at the junction, but the pairwise dot product correctly detects the discontinuity).

Implementation: `F.unfold` extracts (3, 24, H, W) neighbor normal patches, then a batched dot product with the center normal gives (24, H, W) similarity scores.

### Step 2: Relative Depth Comparison

For each center-neighbor pair, check:

```
|d_center - d_neighbor| / avg(d_center, d_neighbor) < depth_threshold
```

This is a **relative** threshold (v5 used absolute meters). At close range (1m depth), a threshold of 0.025 allows 2.5cm difference. At far range (10m depth), it allows 25cm. This prevents over-segmentation of distant surfaces where small absolute depth noise would break v5's fixed threshold.

Both center and neighbor must have positive depth (`depth > 0`) for the pair to be valid.

### Step 3: Neighborhood Voting

A center-neighbor pair is a **match** if all three conditions hold:
- Both pixels are planar (`planarity_mask == 1`)
- Normals are similar (Step 1)
- Depths are close (Step 2)

Count how many of the 24 neighbors match (`match_count`) and how many are valid (planar + positive depth on both sides: `valid_count`).

### Step 4: Adaptive Threshold (v10 addition)

Instead of requiring a fixed number of matches (v5/v9 use `neighbor_match_count_thresh=24` or `18`), v10 computes a per-pixel threshold:

```
adaptive_thresh = max(adaptive_frac * valid_count, min_valid_neighbors)
```

A pixel is "connected" if:
```
match_count >= adaptive_thresh  AND  valid_count >= min_valid_neighbors
```

**Why this matters**: At a plane interior, all 24 neighbors are typically valid, so `0.75 * 24 = 18` matches are required (same strictness as v9). At a plane boundary where half the neighbors belong to a different plane, only ~12 are valid, so `0.75 * 12 = 9` matches suffice. Without this, boundary pixels get rejected because they can never reach the fixed threshold, causing systematic boundary erosion that hurts recall.

The `min_valid_neighbors` floor prevents isolated noisy pixels (surrounded by non-planar regions) from passing with just 1-2 matches.

### Step 5: Connected Components

The binary "connected" mask from Step 4 is passed to `scipy.ndimage.label` with 8-connectivity (3x3 structuring element). Each connected component becomes a plane instance (label 1, 2, 3, ...). Background is label 0.

### Step 6: Small Segment Filter (v10 addition)

Connected components with fewer than `min_segment_pixels` pixels are set to background (label 0). Uses `np.bincount` on the flat label array for O(N) counting.

**Why this matters**: Tiny fragments (< 50 pixels) produce unreliable RANSAC plane fits, inflating the predicted plane count while hurting precision. Removing them is a cheap post-processing step.

## Parameters

### Shared Parameters (inherited from v5/v9)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `threshold_planarity` | 0.6 | - | Planarity score threshold for the binary mask. Applied before segmentation: `planarity_mask = (planarity_score > threshold).astype(int32)`. Lower values include more pixels (higher recall, lower precision). |
| `normal_threshold_deg` | 10.0 | degrees | Maximum angular difference between adjacent normals. Converted to radians internally. Tighter values (5-7) produce cleaner boundaries but may over-segment curved surfaces. |
| `depth_threshold` | 0.025 | relative fraction | Maximum relative depth difference between adjacent pixels. 0.025 means 2.5% of the average depth. |
| `neighbor_match_count_thresh` | 18 | count | **Ignored by v10** when `adaptive_frac > 0`. Kept in the API for backward compatibility. In v5/v9 this is the fixed voting threshold over the 5x5 neighborhood (24 neighbors). |

### v10-Specific Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `adaptive_frac` | 0.75 | fraction | Fraction of valid neighbors that must match. At plane interiors (24 valid), requires 18 matches. At boundaries (8 valid), requires 6. Higher values are stricter and produce cleaner segments but erode boundaries. |
| `min_valid_neighbors` | 3 | count | Absolute floor for valid neighbor count. Pixels with fewer valid neighbors are rejected regardless of match ratio. Prevents noise in isolated planar pixels. |
| `min_segment_pixels` | 50 | pixels | Minimum connected component size. Smaller segments are removed (set to label 0). Set to 0 to disable. |

### Parameter Sensitivity (from sweep on 10 ScanNet++ test scenes)

Each parameter was swept independently while holding others at default. Best values maximize Segmentation Covering (SC):

| Parameter | Default | Best (SC) | SC at default | SC at best | Notes |
|-----------|---------|-----------|---------------|------------|-------|
| `threshold_planarity` | 0.6 | 0.6 | 0.671 | 0.671 | Default is optimal |
| `normal_threshold_deg` | 10.0 | 7.0 | 0.671 | 0.681 | Tighter normals help SC (+0.010) |
| `depth_threshold` | 0.025 | 0.05 | 0.671 | 0.672 | Marginal improvement |
| `adaptive_frac` | 0.75 | 0.85 | 0.671 | 0.678 | Stricter adaptive frac helps SC (+0.007) |
| `min_valid_neighbors` | 3 | 12 | 0.671 | 0.671 | No meaningful change |
| `min_segment_pixels` | 50 | 150 | 0.671 | 0.672 | Marginal improvement |

Note: these sweeps optimize for SC independently. Joint optimization may yield different results.

## v10 vs v5: Key Differences

| Aspect | v5 | v10 |
|--------|-----|------|
| Normal check | Sobel gradient magnitude (per-pixel scalar) | Pairwise dot product (per-edge, 24 comparisons) |
| Depth check | Absolute: `\|d_c - d_n\| < threshold` (meters) | Relative: `\|d_c - d_n\| / avg(d_c, d_n) < threshold` |
| Voting | Fixed: `match_count >= 24` | Adaptive: `match_count >= 0.75 * valid_count`, floor of 3 |
| Small segment removal | None | Components < 50 pixels removed |
| Boundary behavior | Erodes boundaries (fixed threshold too strict at edges) | Preserves boundaries (adaptive threshold relaxes at edges) |

### Parameters as Used in Evaluation

The `inference_to_h5.py` and `inference_to_h5_hypersim.py` scripts use v5 segmentation with these defaults:

| Parameter | v5 (inference scripts) | v10 (notebook) |
|-----------|----------------------|----------------|
| `threshold_planarity` | 0.6 | 0.3 |
| `normal_threshold_deg` | 10.0 | 5.0 |
| `depth_threshold` | 0.05 (absolute m) | 0.025 (relative) |
| `neighbor_match_count_thresh` | 24 | N/A (adaptive) |

Note: the v10 notebook used `threshold_planarity=0.3` (much lower than v5's 0.6), which substantially increases the number of planar pixels feeding into segmentation.

## Performance Comparison (10 ScanNet++ test scenes, mean)

| Method | SC | RI | VOI | P@1mm | R@1mm | P@5mm | R@5mm | P@1cm | R@1cm |
|--------|------|------|-------|-------|-------|-------|-------|-------|-------|
| GT | 1.000 | 1.000 | 0.000 | 0.558 | 0.404 | 0.852 | 0.609 | 0.876 | 0.625 |
| ZeroPlane | 0.676 | 0.799 | 1.584 | 0.543 | 0.406 | 0.868 | 0.661 | 0.916 | 0.696 |
| MoGe+v9 | 0.665 | 0.796 | 1.578 | 0.693 | 0.484 | 0.888 | 0.608 | 0.902 | 0.617 |
| MoGe+v10 | 0.650 | 0.790 | 1.727 | 0.687 | 0.526 | 0.913 | 0.681 | 0.964 | 0.717 |

v10 trades ~2% SC/RI for significantly better 3D geometry:
- P@1cm: +6.2% over v9 (0.902 -> 0.964)
- R@1cm: +10.0% over v9 (0.617 -> 0.717)
- R@5mm: +7.3% over v9 (0.608 -> 0.681)

The SC/RI drop comes from the adaptive threshold producing slightly less clean instance boundaries. The 3D gains come from preserving boundary pixels (recall) and removing junk fragments (precision).

## Source

- Implementation: `planamono/exploration/scannetpp/compare_segmentation_algorithms.ipynb` (Cell 8)
- Not yet promoted to `planamono/shared/segmentation/plan2seg.py`
- Comparison notebook also contains v7, v8, v9 variants and parameter sensitivity sweeps
