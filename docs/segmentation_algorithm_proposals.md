# Plane Segmentation Algorithm Proposals

Analysis of current segmentation algorithms and proposals for a stronger, fast algorithm to beat ZeroPlane.

## Current State

### Performance vs ZeroPlane

MoGe already dominates ZeroPlane on 3D metrics (2-3x precision @1mm on ScanNet++). The gap is in 2D segmentation metrics:

| Metric | Ours (v10) | ZeroPlane | Gap |
|--------|-----------|-----------|-----|
| SC (ScanNet++) | 0.651 | 0.591 | **+0.060 (winning)** |
| RI (ScanNet++) | 0.810 | 0.814 | -0.004 |
| SC (Hypersim) | 0.506 | 0.618 | **-0.112 (losing)** |
| P@1mm (ScanNet++) | 0.639 | 0.216 | **+0.423 (2.96x)** |
| P@1cm (ScanNet++) | 0.933 | 0.549 | **+0.384 (1.70x)** |

### Runtime

| Method | Total (ms) | FPS |
|--------|-----------|-----|
| MoGe + v5 seg | 180 | 5.6 |
| MoGe + v6 seg | 270 | 3.7 |
| ZeroPlane | 366 | 2.7 |

MoGe inference: ~120ms. Segmentation budget: <50-100ms.

## Diagnosis: Root Cause of 2D Metric Gap

**Over-segmentation.** v10 fragments large planes at subtle geometric boundaries, producing many small segments. This hurts SC (IoU drops when GT segments are split) and RI (pixel pairs in same GT segment end up in different predicted segments).

**The fundamental bottleneck in all current versions (v5-v11):**
1. Local pixel-by-pixel voting → binary connectivity → one-shot connected components
2. Once CC is done, over-segmentation is baked in
3. Merge as post-processing (v11) is expensive (~20-50ms for iterative dilation) and doesn't refine boundaries

### Existing Version Summary

| Version | Normal Check | Depth Threshold | Voting | Post-processing | Runtime |
|---------|-------------|-----------------|--------|-----------------|---------|
| v5 | Per-pixel Sobel (proxy) | Absolute (m) | Fixed 8/24 | None | ~15ms |
| v6 | Per-edge dot product | Relative (frac) | Fixed 18/24 | None | ~20ms |
| v10 | Per-edge cosine | Relative (symmetric avg) | Adaptive (75% valid) | Small segment filter | ~18ms |
| v11 | Per-edge cosine | Relative (symmetric avg) | Adaptive + decoupled | Small filter + iterative dilation merge | ~18ms (no merge) / ~50ms (merge) |

### What Drives Each Metric

| Metric | Rewards | Penalizes |
|--------|---------|-----------|
| **SC** | High IoU between predicted and GT segments | Over-segmentation (split GT), under-segmentation (merge GT), boundary shift |
| **RI** | Pixel pairs with consistent labels in GT and pred | Any label disagreement (symmetric) |
| **P@τ** | Clean plane fits (high inlier ratio above 0.9 gate) | Mixed planes in one segment (inlier ratio drops, fails gate → 0 contribution) |
| **R@τ** | All planar points in segments that pass inlier gate | Boundary erosion removes true planar pixels; fragmentation creates tiny segments that fail RANSAC |

**Key insight**: SC and 3D precision have partially conflicting drivers. SC wants fewer, larger segments; precision wants geometrically pure segments. The optimal algorithm balances both.

## Proposed Algorithms

### Approach A: v10 + Fast Union-Find Merge (~25ms)

Evolutionary improvement. Keep v10's strong GPU core, replace v11's slow dilation-merge with O(HW) single-pass.

```
Phase 1: v10 GPU core                     (~15ms)
  Per-edge cosine + symmetric relative depth + adaptive voting + CC

Phase 2: Single-pass adjacency + UF merge  (~5ms CPU)
  - Scan image once: for each pixel, if label[y,x] != label[y,x+1], record pair
  - Per-segment stats: mean normal, mean depth (O(HW) via bincount)
  - Union-Find: merge adjacent segments with cos(n_a, n_b) > thresh AND
    |d_a - d_b| < relative_thresh * max(d_a, d_b)

Phase 3: Small segment absorption          (~3ms CPU)
  - Remaining small segments -> absorbed into adjacent compatible large segment
  - LUT relabeling
```

**Why faster than v11 merge**: Single-pass adjacency detection (no iterative dilation), Union-Find is O(α(n)) per merge, no repeated `np.unique(labels)` scans.

**Expected gains**: +0.02-0.04 SC, +0.01-0.02 RI from reduced fragmentation, 3D metrics unchanged.

**Risk**: Low. This is a drop-in improvement over v10.

### Approach B: Felzenszwalb Graph-Based Segmentation (~20ms)

More novel. Instead of local voting → CC, build an edge-weighted graph and use a global criterion that naturally balances over/under-segmentation.

```
Phase 1: GPU edge weight computation                    (~5ms)
  For each 4-connected pixel pair (i,j):
    w_normal = 1 - dot(n_i, n_j)                      # normal dissimilarity
    w_depth  = |d_i - d_j| / max(d_i, d_j, 0.1)      # relative depth gap
    w_planar = 1 - min(p_i, p_j)                      # planarity boundary
    w(i,j) = α * w_normal + β * w_depth + γ * w_planar

Phase 2: Sort edges by weight                           (~8ms CPU)
  ~800K edges for 768x512, timsort

Phase 3: Felzenszwalb merge                             (~3ms CPU)
  - Process edges low -> high
  - Merge components A,B if: w(i,j) <= min(τ(A), τ(B))
    where τ(C) = k / |C| is the internal difference threshold
  - k controls segmentation granularity (tune per dataset)
  - Union-Find with path compression

Phase 4: Planarity filter + small segment removal       (~2ms CPU)
  - Reject segments where mean planarity < threshold
  - Remove segments < min_pixels
```

**Why this could be stronger**: The Felzenszwalb criterion naturally produces larger segments in uniform regions and smaller segments at real boundaries. It's a *global* decision based on minimum spanning tree, not pixel-by-pixel.

**Key advantage**: The `k/|C|` threshold means large planes need a BIGGER contrast to split → naturally prevents over-segmentation of large surfaces (exactly the current weakness).

**Risk**: Purely CPU for sort+merge (no GPU). Empirically ~15ms for 768x512 in optimized C implementations. The edge weight design (α, β, γ) needs tuning.

**Reference**: Felzenszwalb & Huttenlocher, "Efficient Graph-Based Image Segmentation", IJCV 2004.

### Approach C: Superpixel-then-Merge (PEAC-inspired, ~20-30ms)

Divide-and-conquer. Coarse patches first, then refine.

```
Phase 1: Coarse grid patches                            (~2ms GPU)
  Divide image into PxP patches (e.g., 16x16 -> ~3000 patches)
  Per patch: mean normal, mean depth, mean planarity

Phase 2: Patch similarity graph + merge                 (~3ms CPU)
  4-connected patch adjacency
  Merge patches with similar plane equations
  Union-Find -> coarse segments

Phase 3: Boundary refinement at pixel level             (~10ms GPU)
  For pixels at patch boundaries:
    Reassign to the adjacent segment with best normal+depth match
  Run 2-3 iterations of boundary pixel refinement

Phase 4: Quality filtering                              (~2ms CPU)
  Planarity threshold on segments, small segment removal
```

**Advantage**: Very fast coarse pass, boundary refinement only touches ~10% of pixels.

**Disadvantage**: Patch boundaries don't respect true plane boundaries → refinement phase is critical and may need more iterations.

**Reference**: Feng et al., "Fast Plane Extraction in Organized Point Clouds" (PEAC), ICRA 2014.

## Recommendation

**Start with Approach B (Felzenszwalb)**, for these reasons:

1. **Addresses the root cause**: Global merge criterion naturally prevents over-segmentation. The `k/|C|` mechanism is exactly what's needed — large planes stay whole, real boundaries are preserved.

2. **Leverages all three inputs**: The edge weight `α * w_normal + β * w_depth + γ * w_planar` combines normal, depth, AND planarity. Current methods use planarity only as a binary mask, wasting the soft probability information.

3. **Fast enough**: ~20ms total fits well within the 50-100ms budget. GPU preprocessing is trivial, Felzenszwalb on ~800K edges is fast.

4. **Simple to implement**: ~100 lines of code. Union-Find already exists in `postprocess.py:53-63`.

5. **Well-understood tuning**: One main parameter `k` (granularity) plus edge weight coefficients `α, β, γ`. Grid search is cheap.

6. **Hybrid fallback**: If Felzenszwalb alone doesn't reach v10's 3D precision, use v10 as Phase 1 (initial segments) and Felzenszwalb as Phase 2 (merge pass), getting the best of both.

### Tuning Strategy

| Parameter | Starting Value | Sweep Range | Effect |
|-----------|---------------|-------------|--------|
| `k` (granularity) | 300 | [50, 100, 200, 300, 500, 1000] | Higher = fewer segments (less over-seg) |
| `α` (normal weight) | 0.5 | [0.3, 0.5, 0.7] | Higher = stricter normal boundaries |
| `β` (depth weight) | 0.3 | [0.1, 0.3, 0.5] | Higher = stricter depth boundaries |
| `γ` (planarity weight) | 0.2 | [0.0, 0.1, 0.2, 0.3] | Higher = more separation at planarity dips |
| `min_segment_pixels` | 50 | [20, 50, 100] | Minimum segment size |
| `planarity_threshold` | 0.3 | [0.2, 0.3, 0.4, 0.5] | Per-segment mean planarity gate |

### Implementation Plan

1. Implement as `compute_vectorized_planar_segments_v12()` in `plan2seg.py` with the same interface as v10
2. Add `--seg_version v12` support to `inference_to_h5.py` and `segment_from_raw.py`
3. Run grid search over `(k, α, β, γ)` on ScanNet++ val split
4. Evaluate on test split vs v10 and ZeroPlane
5. If SC improves but 3D precision drops, implement hybrid: v10 initial segments → Felzenszwalb merge (Approach A+B)
