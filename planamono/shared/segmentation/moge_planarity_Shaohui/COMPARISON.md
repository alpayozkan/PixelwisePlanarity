# Comparison: Our Segmentation (v5) vs Shaohui's

## Parameter Differences

| Parameter | Ours (v5, inference scripts) | Shaohui's |
|-----------|------------------------------|-----------|
| `threshold_planarity` | 0.6 | 0.6 |
| `normal_threshold_deg` | 10.0 | 10.0 |
| `neighbor_match_count_thresh` | **24** | **18** |
| `depth_threshold` | **0.05 (absolute, meters)** | **0.02 (relative, fraction of depth)** |
| Small component removal | None (just `remap_labels`) | **500 pixels** min |

### Key Differences

**1. `neighbor_match_count_thresh`: 24 vs 18**

- **Ours (24/24 = 100%)**: Every neighbor in the 5x5 window must match. This is extremely strict — only pixels deep inside a planar region pass. Boundary pixels almost always fail because some neighbors belong to the adjacent plane.
- **Shaohui (18/24 = 75%)**: More lenient. Pixels near boundaries where ~6 neighbors belong to another plane can still be included. This produces larger segments with less boundary erosion.

**2. Depth Threshold: Absolute vs Relative**

- **Ours (0.05m absolute)**: `|center_depth - neighbor_depth| < 0.05`. Fixed 5cm regardless of distance. At 10m depth, 5cm is very tight (0.5%); at 0.5m depth, 5cm is loose (10%).
- **Shaohui (0.02 relative)**: `|center_depth - neighbor_depth| < 0.02 * center_depth`. At 10m → 20cm threshold; at 0.5m → 1cm threshold. Scales with scene depth — tighter for nearby surfaces, more forgiving for distant ones.

**3. Small Component Removal**

- **Ours**: No explicit small component removal. Only `remap_labels()` which reassigns IDs contiguously.
- **Shaohui**: Removes segments < 500 pixels using `scipy.ndimage.label` per-label connected component analysis.

## Algorithm Differences

### Normal Similarity Check

| | Ours (v5) | Shaohui's |
|---|-----------|-----------|
| **Method** | Sobel gradient magnitude | Pairwise dot product |
| **What it measures** | Normal change rate (edge detection) | Direct angle between pixel normals |
| **Threshold semantics** | `gradient_mag ≤ sqrt(2 - 2cos(θ))` | `arccos(dot) < θ` |
| **Where it's computed** | Per-pixel (scalar), then unfolded to 5x5 | Per-pair (center vs each of 24 neighbors) |

**Ours** uses a Sobel filter to detect where normals change rapidly — it's an edge-detection approach. The `normal_similar` mask is a per-pixel boolean that is then shared across all 24 neighbor checks.

**Shaohui's** computes the actual angle between the center pixel's normal and each of the 24 neighbors independently. Each neighbor pair gets its own normal similarity decision.

**Practical difference**: The Sobel approach can miss cases where two pixels have similar normals but are separated by a gradient peak (e.g., gentle curvature). The pairwise approach is more local and direct — it only cares about the two pixels being compared.

### Connected Component Labeling

| | Ours (v5) | Shaohui's |
|---|-----------|-----------|
| **Library** | `cc3d` | `scipy.ndimage.label` |
| **Connectivity** | 6-connected (default for 2D in cc3d) | 8-connected (3x3 structuring element) |

### GPU Implementation

| | Ours (v5) | Shaohui's |
|---|-----------|-----------|
| **Neighbor extraction** | `F.unfold` (batched) | Manual padding + slicing |
| **Optimization** | Grouped Sobel convolution, pre-allocated tensors | Standard tensor operations |
| **Speed** | Faster (unfold is more memory-efficient) | Slightly slower (24 separate slice ops) |

## Summary

The most impactful differences in practice:

1. **`neighbor_match_count_thresh` 24 vs 18** — Our 100% requirement causes significant boundary erosion. Shaohui's 75% preserves more boundary pixels. This is likely the biggest visual difference.
2. **Absolute vs relative depth threshold** — Shaohui's relative threshold adapts to scene scale; ours uses a fixed 5cm which may be too loose for nearby surfaces and too tight for distant ones.
3. **No small component removal in ours** — We keep all segments regardless of size; Shaohui discards fragments < 500 pixels.
