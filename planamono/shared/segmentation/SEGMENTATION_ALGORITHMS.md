# Planar Segmentation Algorithms — Detailed Comparison

All versions live in `planamono/shared/segmentation/plan2seg.py` (v7 is notebook-only for now).

Every algorithm follows the same 4-stage pipeline:

```
Input: planarity_mask (H,W), normal (H,W,3), depth (H,W)
  1. Normal similarity check   → per-pixel or per-edge boolean
  2. Depth proximity check     → per-pixel or per-edge boolean
  3. Neighbor match counting   → (H,W) int, thresholded to binary "connected" mask
  4. Connected components      → final segment labels
```

The versions differ in **how stages 1-2 are computed** and **what "similar" means**.

---

## Version Summary

| | v1 | v4 | v5 | v6 (Shaohui) | v7 (notebook) |
|---|---|---|---|---|---|
| **Backend** | NumPy (CPU) | PyTorch (GPU) | PyTorch (GPU) | PyTorch (GPU) | PyTorch (GPU) |
| **Neighborhood** | 3×3 (8 neighbors) | 5×5 (24 neighbors) | 5×5 (24 neighbors) | 5×5 (24 neighbors) | 5×5 (24 neighbors) |
| **Normal check** | Per-edge arccos | Per-pixel Sobel gradient | Per-pixel Sobel gradient | Per-edge dot → acos | Per-edge dot → cos |
| **Depth check** | Absolute (meters) | Absolute (meters) | Absolute (meters) | Relative (fraction) | Relative (fraction) |
| **Depth clamp** | N/A | N/A | N/A | None | `clamp(min=0.1)` |
| **Default thresh** | 1 | 8 | 8 | 18 | 8 |
| **CC library** | `scipy.ndimage.label` | `cc3d` | `cc3d` | `cc3d` | `scipy.ndimage.label` |
| **Speed** | Slow (~1s) | ~50ms | ~15ms | ~15ms | ~15ms |
| **Status** | Legacy | Debug only | **Default for inference** | Shaohui baseline | Experimental |

---

## Stage-by-Stage Comparison

### Stage 1: Normal Similarity

This is the most significant difference between versions.

#### v1: Per-edge arccos (CPU)

```python
# For each of 8 neighbors independently:
dot = sum(center_normal * neighbor_normal, axis=channel)
cos_angle = clip(dot / (|center| * |neighbor| + eps), -1, 1)
angle = arccos(cos_angle)
normal_similar = angle < normal_threshold_rad          # (H, W, 8)
```

Compares the center pixel's normal to each neighbor's normal individually. Correct per-edge check, but slow due to NumPy loops and small 3×3 window.

#### v4 & v5: Per-pixel Sobel gradient

```python
# Sobel filter on each normal channel (nx, ny, nz)
normal_dx = conv2d(normal, sobel_x)     # (3, H, W)
normal_dy = conv2d(normal, sobel_y)     # (3, H, W)
grad_mag = sqrt(sum(dx² + dy²))         # (H, W) scalar
threshold = sqrt(2 - 2·cos(normal_threshold_rad))
normal_similar = (grad_mag <= threshold) # (H, W) single boolean per pixel
```

Computes a **single gradient magnitude per pixel** via Sobel filters. This is a proxy: if the normal field is smooth at a pixel, all its neighbors are assumed similar. Fast but **misses sharp edges between two flat regions** — the gradient at a pixel can be low even though one neighbor across a boundary has a very different normal.

v5 is 3-5× faster than v4 through batched grouped convolution (`groups=3`) and `F.unfold` instead of manual shift-stacking.

**The `normal_similar` mask is (H,W)** — it's a per-pixel property, then unfolded and broadcast to all 24 neighbors. Every neighbor of a "smooth" pixel passes the normal check.

#### v6 (Shaohui): Per-edge acos

```python
# Unfold normals to get each neighbor's normal vector
dot = sum(center_normal * neighbor_normal, dim=channel)  # (24, H, W)
dot = clamp(dot, -1, 1)
angle = acos(dot)                                        # (24, H, W)
normal_similar = angle < normal_threshold_rad             # (24, H, W) per-edge
```

Compares center normal to **each neighbor independently** via `acos(dot_product)`. This gives a **per-edge** decision — the same center pixel can be "similar" to some neighbors and "dissimilar" to others.

#### v7 (notebook): Per-edge cosine comparison

```python
dot = sum(center_normal * neighbor_normal, dim=channel)  # (24, H, W)
cos_thresh = cos(normal_threshold_rad)
normal_similar = clamp(dot, -1, 1) > cos_thresh          # (24, H, W) per-edge
```

Mathematically equivalent to v6 (both check if the angle between normals is below threshold), but **avoids `acos`** by comparing the dot product directly against `cos(threshold)`. This is:
- Numerically more stable (acos has infinite derivative at ±1)
- Slightly faster (no transcendental function per element)
- Equivalent decision boundary: `acos(dot) < θ` ⟺ `dot > cos(θ)` for θ ∈ [0, π]

---

### Stage 2: Depth Proximity

#### v1, v4, v5: Absolute threshold

```python
depth_diff = |center_depth - neighbor_depth|
depth_close = depth_diff < depth_threshold    # threshold in meters (e.g., 0.05m)
```

Fixed threshold regardless of distance. A 5cm threshold works well at 1-3m depth, but at 10m depth it becomes overly strict (0.5% of depth), and at 0.2m it becomes overly permissive (25% of depth).

#### v6 (Shaohui): Relative threshold, no clamp

```python
depth_diff = |center_depth - neighbor_depth|
depth_close = depth_diff < (depth_threshold * center_depth)
```

Threshold scales linearly with depth. At 2m depth with `depth_threshold=0.02`, the effective threshold is 4cm; at 10m it's 20cm.

**Edge case**: When `center_depth ≈ 0` (e.g., invalid pixels), the threshold approaches 0, which is correct (no matches for invalid depth). However, for very close objects (depth < 0.1m), the threshold becomes very tight (< 2mm at 0.1m), which may reject valid neighbors on close surfaces.

#### v7 (notebook): Relative threshold with clamp

```python
depth_diff = |center_depth - neighbor_depth|
depth_close = depth_diff < (depth_threshold * clamp(center_depth, min=0.1))
```

Same as v6 but **clamps the depth denominator at 0.1m**. This ensures the effective threshold is at least `depth_threshold × 0.1` (e.g., 2mm for `depth_threshold=0.02`), preventing overly tight thresholds for close objects.

---

### Stage 3: Neighbor Match Counting

All versions count how many neighbors pass **all three checks** (valid pair + normal similar + depth close):

```python
matches = valid_pair & normal_similar & depth_close
neighbor_match_count = matches.sum(dim=neighbor_axis)
connected = neighbor_match_count >= neighbor_match_count_thresh
```

The threshold controls boundary erosion:

| Default | Max possible | Fraction | Effect |
|---------|-------------|----------|--------|
| v1: 1 | 8 | 12.5% | Very permissive, noisy boundaries |
| v4: 8 | 24 | 33% | Moderate erosion |
| v5: 8 | 24 | 33% | Moderate erosion |
| v6: 18 | 24 | 75% | Conservative, clean boundaries, erodes thin structures |
| v7: 8 | 24 | 33% | Moderate erosion |

Higher thresholds produce cleaner boundaries but shrink plane regions (especially thin or elongated ones). v6's default of 18 means a pixel needs 75% of its neighbors to agree — this aggressively erodes boundaries and can fragment narrow planes.

**Important**: For v5, the normal check is per-pixel (broadcast to all 24 neighbors), so the match count is dominated by the depth check and planarity mask. For v6/v7, each neighbor is independently checked, so the match count genuinely reflects local geometric consensus.

---

### Stage 4: Connected Components

| Version | Library | Connectivity |
|---------|---------|-------------|
| v1 | `scipy.ndimage.label` | 8-connected (3×3 struct) |
| v4, v5, v6 | `cc3d` | 26-connected (default for 2D = 8-connected) |
| v7 | `scipy.ndimage.label` | 8-connected (3×3 struct) |

Both produce equivalent results for 2D binary masks. `cc3d` is generally faster for large images; `scipy.ndimage.label` is more commonly available.

---

## Key Design Differences at a Glance

### Per-pixel vs Per-edge Normal Check

This is the **fundamental architectural split**:

```
v4/v5 (Sobel, per-pixel):
  pixel A has low gradient → ALL neighbors of A pass normal check
  Even if neighbor B across a plane boundary has a completely different normal!

v1/v6/v7 (pairwise, per-edge):
  pixel A checks each neighbor individually
  Neighbor B across boundary fails, neighbor C on same plane passes
```

**When Sobel fails**: Two large flat regions meeting at a sharp edge. Pixels right at the boundary have high Sobel gradient (correctly), but pixels one step away from the boundary have low gradient (their immediate local field is smooth). Since v5 uses a 5×5 window, these "one step away" pixels include the boundary pixel as a neighbor and incorrectly merge across the boundary.

**When pairwise wins**: Any scenario with adjacent planes at different orientations — walls meeting floors, ceiling/wall junctions, adjacent facade panels.

### Absolute vs Relative Depth

```
Absolute (v1/v4/v5): depth_threshold = 0.05m
  At 1m depth: 5% tolerance  ✓
  At 5m depth: 1% tolerance  (too strict, rejects valid neighbors)
  At 0.2m depth: 25% tolerance  (too loose, merges across gaps)

Relative (v6/v7): depth_threshold = 0.02 (2%)
  At 1m depth: 2cm tolerance  ✓
  At 5m depth: 10cm tolerance  ✓ (scales with distance)
  At 0.2m depth: 4mm tolerance  ✓ (tight for close objects)
```

---

## Parameter Defaults Comparison

| Parameter | v1 | v4 | v5 | v6 | v7 |
|-----------|-----|-----|-----|-----|-----|
| `normal_threshold_deg` | — | — | 10° | 10° | 10° |
| `depth_threshold` | — | — | 0.05 (m) | 0.02 (relative) | 0.02 (relative) |
| `neighbor_match_count_thresh` | 1 | 8 | 8 | 18 | 8 |
| Neighborhood size | 3×3 | 5×5 | 5×5 | 5×5 | 5×5 |

Note: `inference_to_h5.py` uses v5 with `threshold_planarity=0.6, normal_threshold_deg=10.0, depth_threshold=0.05, neighbor_match_count_thresh=24`.

---

## When to Use Which

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Production inference** | v5 | Current default, well-tested, fast |
| **Best boundary quality** | v6 or v7 | Per-edge normal check correctly handles plane boundaries |
| **Debugging** | v4 | Same algorithm as v5 but more readable, easier to instrument |
| **CPU-only** | v1 | No CUDA required |
| **Experimentation** | v7 (inline) | Editable in notebook, combines best ideas from v5+v6 |

---

## Evolution History

```
v1 (original)
│   CPU, 3×3, per-edge arccos, absolute depth
│   Correct but slow
│
├─→ v4 (GPU port)
│     GPU, 5×5, Sobel normals, absolute depth
│     Fast but Sobel is a proxy (per-pixel, not per-edge)
│
├─→ v5 (optimized v4)
│     Same algorithm as v4, 3-5× faster via batched ops + F.unfold
│     Current production default
│
├─→ v6 (Shaohui's design)
│     GPU, 5×5, per-edge acos, relative depth
│     Fixes both Sobel weakness AND absolute depth issue
│     High default thresh (18) — conservative boundaries
│
└─→ v7 (v5 + v6 ideas, notebook-only)
      GPU, 5×5, per-edge cosine, relative depth + clamp
      Like v6 but: cos comparison (no acos), clamp(min=0.1), thresh=8
      Not yet in plan2seg.py
```
