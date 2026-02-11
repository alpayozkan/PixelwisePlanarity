# Ground Truth Generation Pipeline

This document describes the ground truth (GT) plane extraction pipeline for ScanNet++ and Hypersim datasets. The pipeline extracts planar surfaces from 3D semantic meshes and renders them to 2D image space.

## Overview

The GT generation is a **label-strict** algorithm that extracts planes from 3D meshes while respecting semantic boundaries. A plane belonging to "wall" will never include faces labeled as "floor", even if they are geometrically coplanar.

**Source code**: `planamono/gt_creation/scannetpp/plane_extraction.py` (~2,800 lines)

## Quick Start

```bash
# ScanNet++ - single scene
python planamono/gt_creation/scannetpp/scene_runner.py <scene_id> \
    --config planamono/gt_creation/configs/scannetpp_default.yml \
    --input_root /path/to/scannetpp/data \
    --output_root /path/to/output

# Hypersim - single scene
python planamono/gt_creation/hypersim/scene_runner.py <scene_id> \
    --config planamono/gt_creation/configs/hypersim_default.yml
```

## Input Data

### ScanNet++
| File | Description |
|------|-------------|
| `mesh_aligned_0.05_semantic.ply` | Semantic mesh with per-vertex colors |
| `segments.json` | Per-vertex/face segment indices |
| `segments_anno.json` | Segment ID → semantic label mapping |

### Hypersim
| File | Description |
|------|-------------|
| `mesh.obj` | Scene geometry |
| `semantic_label.hdf5` | Per-face semantic labels |

## Output Files

| File | Description |
|------|-------------|
| `planes.json` | Plane metadata: parameters `(n, d)`, area, color, semantic label |
| `planes.ply` | Original mesh with per-face `plane_id` and `label_int` attributes |

### planes.json Format

```json
[
  {
    "plane_id": 0,
    "label_int": 3,
    "label_raw": "wall",
    "faces": 12534,
    "area": 4.82,
    "n": [0.0, 0.0, 1.0],
    "d": 2.45,
    "color_rgb": [128, 64, 192],
    "alg": "RG+EM+SAT+LAST-RG"
  }
]
```

The plane equation is `n·x + d = 0` where `n = (a, b, c)` is the unit normal.

---

## Pipeline Stages

The pipeline consists of 7 stages, each refining the plane extraction:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: Region Growing                       │
│         BFS growth within semantic labels, strict gates          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: EM Sweep Expansion                   │
│         Iterative inlier sweep + cluster splitting               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 3: Quality Filtering                    │
│         8 geometric checks per plane                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 4: IRLS Plane Fitting                   │
│         Robust least-squares with Huber loss                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 5: Large Label Splitting                │
│         Memory-safe recursive partitioning                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 6: Parametric Merging                   │
│         Merge compatible planes within same label                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 7: Final Relaxed RG                     │
│         Last-stage cleanup with relaxed thresholds               │
└─────────────────────────────────────────────────────────────────┘
```

---

### Stage 1: Label-Strict Region Growing

For each semantic label (wall, floor, table, etc.), grow plane regions using BFS:

**Algorithm:**
1. Sort faces by area (largest first)
2. For each unvisited face as seed:
   - Fit initial plane via SVD on the 3 triangle vertices
   - BFS grow to adjacent faces passing all gates
   - Refit plane via IRLS every N faces (default: 15)
3. Keep patches with sufficient faces and area

**Gates (all must pass):**

| Gate | Formula | Default |
|------|---------|---------|
| Dihedral | `\|N_u · N_v\| >= cos(rg_dihedral_deg)` | 55° |
| Normal | `\|N_face · n_plane\| >= cos(rg_theta_deg)` | 8° |
| Distance | k-of-3 vertices within `rg_dist_m` | 1.5cm, k=2 |

**Distance gate modes:**
- `kof3` (default): At least k of 3 triangle vertices within threshold
- `centroid`: Face centroid within threshold
- `none`: No distance check (normal gate only)

**Minimum patch requirements:**
- `min_faces_patch`: 60 faces
- `min_area_patch`: 0.1 m²

---

### Stage 2: EM Sweep Expansion

Iteratively expand each initial patch by sweeping for inliers:

**Algorithm:**
```
for iteration in range(em_max_iters):  # default: 4
    1. Sweep: Find all faces in label satisfying normal + distance gates
    2. Consensus filter: Keep faces with ≥2 neighbors also in sweep
    3. Cluster split: If gaps >1.2cm along plane offset, split into clusters
    4. Select cluster overlapping most with original seed
    5. Refit plane, verify quality gates
    6. If growth < em_min_growth (0.5%), stop
```

**Sweep parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sweep_normal_deg` | 9° | Normal alignment threshold |
| `sweep_dist_m` | 1.2cm | Distance threshold |
| `sweep_frac_vertices` | 1.0 | Fraction for k-of-3 (1.0 → all 3) |

**Local consensus filter:**
Uses CSR adjacency + optional Numba JIT for speed. Removes isolated faces that don't have neighbors also in the sweep set.

---

### Stage 3: Quality Filtering

Each plane must pass 8 geometric quality checks:

| Check | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| p95 residual | `p95_final_max` | 3cm | 95th percentile point-to-plane distance |
| Inlier fraction | `inlier_frac_min` | 80% | Fraction of points within `dist_thr` |
| Normal p95 | `normal_p95_deg_max` | 8° | 95th percentile face normal deviation |
| Thickness | `thickness_max_mul` | 1.6× | Max extent along normal (× sweep_dist) |
| Min width | `min_width_m` | 6cm | Minimum extent in tangent directions |
| Fill fraction | `fill_frac_min` | 18% | area / bounding_box_area |
| Min faces | `min_faces_patch` | 60 | Minimum triangle count |
| Min area | `min_area_patch` | 0.1m² | Minimum surface area |

Planes failing any check are discarded.

---

### Stage 4: IRLS Robust Plane Fitting

Iteratively Reweighted Least Squares with Huber loss for robust plane estimation:

```python
def fit_plane_irls(points, max_iters=8, huber_k=1.345):
    # Initial fit via SVD
    n, d = fit_plane_svd(points)

    for iteration in range(max_iters):
        # Compute residuals
        r = points @ n + d

        # Robust scale estimate (MAD)
        sigma = 1.4826 * median(|r|)

        # Huber weights
        c = huber_k * sigma
        w = where(|r| <= c, 1.0, c / |r|)

        # Weighted least-squares refit
        mu = weighted_mean(points, w)
        X = (points - mu) * sqrt(w)
        _, _, Vt = svd(X)
        n_new = Vt[-1]  # smallest singular value
        d_new = -n_new @ mu

        # Convergence check
        if ||n_new - n|| < eps and |d_new - d| < eps:
            break
        n, d = n_new, d_new

    return n, d
```

**Why IRLS?**
- Downweights outliers (noisy vertices, mesh artifacts)
- More stable than vanilla SVD on noisy real-world meshes
- Huber loss provides smooth transition between L2 and L1

---

### Stage 5: Large Label Splitting

Memory safeguard for labels with many vertices (e.g., large walls):

**Trigger:** Labels with ≥ `large_split_verts` (default: 700,000) unique vertices

**Algorithm:**
```
def recursive_partition(faces, threshold, max_parts=8):
    queue = [faces]
    parts = []

    while queue and len(parts) + len(queue) < max_parts:
        current = queue.pop()

        if unique_vertices(current) < threshold:
            parts.append(current)
            continue

        # Split by spatial median (axis-variance or PCA)
        A, B = split_by_median(current)
        queue.extend([A, B])

    return parts
```

**Split modes:**
- `axis`: Split along axis with maximum variance
- `pca`: Split along first principal component

After processing all parts independently, a **cross-split merge** reunifies compatible planes that were separated by the spatial partitioning.

---

### Stage 6: Parametric Merging

Merge planes within the same semantic label if geometrically compatible:

**Merge criteria:**
1. Normal similarity: `|n₁ · n₂| >= cos(merge_theta_deg)` (default: 10°)
2. Distance: 85th percentile point-to-plane distance ≤ `merge_dist_m` (default: 2cm)
3. Merged plane passes all quality gates

**Algorithm:**
```
while changed:
    for each pair (plane_i, plane_j) in same label:
        if compatible(plane_i, plane_j):
            merged = union(plane_i.faces, plane_j.faces)
            if quality_check(merged):
                replace plane_i with merged
                remove plane_j
                changed = True
```

---

### Stage 7: Final Relaxed Region Growing

Last-stage cleanup with relaxed thresholds to fill small gaps:

**Relaxed parameters:**
| Parameter | Strict | Relaxed |
|-----------|--------|---------|
| Normal threshold | 8° | 18° (`last_normal_deg`) |
| Distance threshold | 1.5cm | 2cm (`last_dist_m`) |

**Additional behaviors:**
- **Unlabeled absorption**: Can absorb nearby unlabeled faces if their total area ≤ `last_unlabeled_ratio` × base_area
- **Plane stealing**: Larger planes can absorb faces from smaller planes if area ratio ≥ `last_steal_factor` (default: 5×)
- Only operates **within same semantic label**

---

## Saturation Pass

Between main passes, a **saturation pass** greedily expands planes:

1. Sort planes by area (largest first)
2. For each plane, sweep for new inliers
3. Resolve overlaps: larger planes take precedence
4. Light intra-label merge

Parameters: `sat_rounds` (default: 2), `sat_normal_deg` (12°), `sat_dist_m` (2cm)

---

## Recovery Pass

After the main pass, attempt to extract planes from leftover (unlabeled) faces:

- Uses relaxed thresholds (`*_small` parameters)
- Only processes faces not already assigned to a plane
- Helps capture small planes missed by strict pass

---

## Configuration Reference

### scannetpp_default.yml

```yaml
# I/O
input_root: "/path/to/scannetpp/data"
output_root: "/path/to/output"
progress: 1              # Show progress bars
jobs: 8                  # Parallel workers
backend: "threads"       # threads or processes

# Region Growing (Stage 1)
rg_theta_deg: 8.0        # Normal angle threshold
rg_dist_m: 0.015         # Distance threshold (1.5cm)
rg_dihedral_deg: 55.0    # Adjacent face normal threshold
rg_refit_every: 15       # Refit plane every N faces
rg_gate_mode: "kof3"     # Distance gate mode

# EM Sweep (Stage 2)
sweep_normal_deg: 9.0    # Sweep normal threshold
sweep_dist_m: 0.012      # Sweep distance threshold
sweep_frac_vertices: 1.0 # k-of-3 fraction
em_max_iters: 4          # Max EM iterations
em_min_growth: 0.005     # Stop if growth < 0.5%

# Quality Gates (Stage 3)
p95_final_max: 0.03      # Max p95 residual (3cm)
inlier_frac_min: 0.80    # Min inlier fraction
dist_thr: 0.012          # Inlier distance threshold
normal_p95_deg_max: 8.0  # Max p95 normal angle
thickness_max_mul: 1.6   # Max thickness multiplier
min_width_m: 0.06        # Min plane width
fill_frac_min: 0.18      # Min fill fraction
min_faces_patch: 60      # Min faces per plane
min_area_patch: 0.10     # Min area (m²)

# IRLS (Stage 4)
irls_max_iters: 8        # IRLS iterations
irls_eps: 1e-6           # Convergence epsilon

# Large Label Splitting (Stage 5)
large_split_enable: 1
large_split_verts: 700000
large_split_max_parts: 8
large_split_min_faces: 50000
large_split_mode: "axis"

# Merging (Stage 6)
merge_theta_deg: 10.0    # Normal similarity threshold
merge_dist_m: 0.02       # Distance threshold

# Final RG (Stage 7)
last_enable: 1
last_dist_m: 0.020       # Relaxed distance
last_normal_deg: 18.0    # Relaxed normal
last_unlabeled_ratio: 1.0
last_steal_factor: 5.0
last_rg_iters: 1

# Policies
policy_single_plane_labels: "floor"  # Fit single plane to these
policy_skip_labels: []               # Skip these labels entirely
```

---

## Policies

### Single-Plane Labels
Some semantic classes are expected to be single planes (e.g., floor):

```yaml
policy_single_plane_labels: "floor,ceiling"
```

These labels skip region growing entirely; a single plane is fit to all faces.

### Skip Labels
Some labels should be excluded from plane extraction:

```yaml
policy_skip_labels: ["curtain", "plant", "person"]
```

---

## Rendering to 2D

After 3D extraction, planes are rendered to 2D image space:

```bash
# ScanNet++
python planamono/gt_creation/scannetpp/render_planes.py <scene_id> \
    --input_root /path/to/scannetpp \
    --plane_root /path/to/planes \
    --output_root /path/to/rendered

# Hypersim
python planamono/gt_creation/hypersim/rendering.py <scene_id>
```

**Rendering process:**
1. Load mesh with per-face plane IDs
2. Propagate face labels to vertices (for Open3D raycasting)
3. For each camera pose:
   - Cast rays from image pixels
   - Record hit plane ID per pixel
4. Save as HDF5 with plane labels, depth, semantics

**Output HDF5 structure:**
```
scene.h5
├── plane_labels/     # (N_frames, H, W) int32
├── depth/            # (N_frames, H, W) float32
├── semantics/        # (N_frames, H, W) int32
└── intrinsics/       # (N_frames, 3, 3) float32
```

---

## Performance Optimizations

The pipeline includes several optimizations for large meshes:

| Optimization | Description |
|--------------|-------------|
| CSR adjacency | Compressed sparse row format for fast neighbor lookup |
| Numba JIT | Optional JIT compilation for consensus filter |
| Float32 geometry | Reduced memory footprint |
| Parallel workers | ThreadPoolExecutor/ProcessPoolExecutor per label |
| Large label splitting | Memory-safe processing of huge labels |
| Largest-first scheduling | Process big labels first for better parallelism |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No planes extracted | Relax `p95_final_max`, `inlier_frac_min`, or `min_area_patch` |
| Too many small planes | Increase `min_faces_patch` and `min_area_patch` |
| OOM on large labels | Enable `large_split_enable`, reduce `large_split_verts` |
| Planes crossing labels | This shouldn't happen; check `segments.json` consistency |
| Slow processing | Increase `jobs`, use `backend: "processes"` |
| Missing floor/ceiling | Check `policy_single_plane_labels` configuration |

---

## References

- **ScanNet++**: Yeshwanth et al., "ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes", ICCV 2023
- **Hypersim**: Roberts et al., "Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding", ICCV 2021
- **IRLS**: Holland & Welsch, "Robust Regression Using Iteratively Reweighted Least-Squares", Communications in Statistics, 1977
