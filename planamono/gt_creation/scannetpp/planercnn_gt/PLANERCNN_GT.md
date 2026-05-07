# PlaneRCNN GT Plane Extraction Algorithm

Ground truth plane segmentation adapted from **PlaneRCNN** (CVPR 2019, Liu et al.) for ScanNet++ meshes. The algorithm extracts planar regions from annotated 3D meshes using per-object RANSAC plane fitting, then raycasts the result to 2D images.

**Source**: `original_planercnn_for_scannetpp.py` (adapted from NVIDIA's PlaneRCNN codebase)
**Notebook**: `exploration/scannetpp/planercnn_gt_pipeline.ipynb`

## Algorithm Overview

```
For each annotated object in the scene:
  1. Look up semantic label → get [min_planes, max_planes] bounds
  2. If max_planes == 0 or too few vertices → mark as non-planar
  3. Try single least-squares plane fit (Ax + By + Cz = 1)
  4. If single fit error < fittingErrorThreshold → accept single plane
  5. Else → RANSAC to extract up to numPlanesPerSegment planes:
     a. Sample 3 points, fit plane
     b. Count inliers (distance < planeDiffThreshold)
     c. Keep best, refit with all inliers
     d. Remove inliers, repeat
  6. If < 50% of object points explained → mark entire object as non-planar
  7. Enforce min_planes constraint (e.g., floor must have >= 1 plane)
  8. Enforce max_planes constraint (e.g., floor must have exactly 1 plane)
```

After all objects are processed, vertex-level plane IDs are raycasted to 2D images using Open3D.

## Parameters

### Plane Fitting Parameters

| Parameter | Value | Variable Name | Description |
|-----------|-------|---------------|-------------|
| **numPlanesPerSegment** | `2` | `NUM_PLANES_PER_SEGMENT` | Maximum planes extracted per object via RANSAC. After the single-plane fit fails, RANSAC runs up to this many iterations of extract-and-remove. |
| **planeAreaThreshold** | `10` | `PLANE_AREA_THRESHOLD` | Minimum number of vertices for a valid plane. Objects with fewer points are marked non-planar. Also used as the minimum inlier count during RANSAC. |
| **numIterations** | `100` | `NUM_ITERATIONS` | Maximum RANSAC iterations per plane extraction. Actual iterations are `min(numIterations, num_remaining_points)`. |
| **planeDiffThreshold** | `0.05` (5cm) | `PLANE_DIFF_THRESHOLD` | RANSAC inlier distance threshold. A point is an inlier if its distance to the plane `|Ax+By+Cz-1| / ||(A,B,C)||` is below this value. |
| **fittingErrorThreshold** | `0.05` (5cm) | `FITTING_ERROR_THRESHOLD` | Maximum mean fitting error to accept a single-plane fit without resorting to RANSAC. Set equal to `planeDiffThreshold`. |
| **Coverage gate** | `0.5` (50%) | hardcoded | If RANSAC explains fewer than 50% of an object's vertices, the entire object is marked non-planar instead. |

### Merging Parameters (original only, skipped for ScanNet++)

These parameters are used in `mergePlanes()` in the original script but **skipped for ScanNet++** because objects are processed individually (not per-segment):

| Parameter | Value | Description |
|-----------|-------|-------------|
| **orthogonalThreshold** | `cos(60°) ≈ 0.5` | Minimum dot product between plane normals to consider merging. Planes more orthogonal than 60° are never merged. |
| **parallelThreshold** | `cos(30°) ≈ 0.866` | If normal dot product exceeds this AND the neighbor has > 50% as many points, refit a new joint plane. Otherwise, keep the original plane and just absorb the neighbor's points. |
| **numIterationsPair** | `1000` | RANSAC iterations for pair-wise plane fitting (unused in current pipeline). |

### Rendering Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **W_TARGET, H_TARGET** | `640 x 480` | Output image resolution (downscaled from 1920x1440). Matches our GT pipeline. |
| **FRAME_SKIP** | `25` | Render every 25th frame from the iPhone trajectory. Matches our GT pipeline. |
| **Label shift** | `+1` | Vertex plane IDs (-1=non-planar, 0+=planes) are shifted to (0=non-planar, 1+=planes) in the H5 output. |

## Plane Model

The algorithm uses the **implicit plane form** `Ax + By + Cz = 1`, where `[A, B, C]` are the plane parameters.

- **`fitPlane(XYZ)`** / **`fit_plane(XYZ)`**: Solves the least-squares system `XYZ @ [A,B,C]^T = 1` via `np.linalg.lstsq`.
- The plane normal direction is `(A, B, C) / ||(A, B, C)||`.
- The distance from origin to the plane is `1 / ||(A, B, C)||`.
- Point-to-plane distance: `|Ax + By + Cz - 1| / ||(A, B, C)||`.

**Note**: This representation cannot express planes passing through the origin (`d = 0`). In practice this is not an issue for indoor scenes.

## Per-Label Plane Count Bounds

The `labelNumPlanes` dictionary maps ScanNet++ semantic labels to `[min_planes, max_planes]`:

### Always planar (min >= 1)

| Label | [min, max] | Notes |
|-------|-----------|-------|
| `wall` | [1, 3] | At least 1, up to 3 (L-shaped walls) |
| `floor` | [1, 1] | Exactly 1 plane; relaxed fitting error (3x) for floor due to mesh alignment |
| `door` | [1, 2] | |
| `picture` | [1, 1] | Exactly 1 plane |
| `entrance` | [1, 1] | |
| `floor mat` | [1, 1] | |
| `whiteboard` | [1, 5] | |
| `night stand` | [1, 5] | |
| `television` | [1, 1] | |

### Optionally planar (min = 0)

| Label | [min, max] | Label | [min, max] |
|-------|-----------|-------|-----------|
| `ceiling` | [0, 5] | `cabinet` | [0, 5] |
| `bed` | [0, 5] | `chair` | [0, 5] |
| `sofa` | [0, 10] | `table` | [0, 5] |
| `window` | [0, 2] | `bookshelf` | [0, 5] |
| `counter` | [0, 10] | `desk` | [0, 10] |
| `shelf` / `shelves` | [0, 5] | `dresser` | [0, 5] |
| `box` | [0, 5] | `toilet` | [0, 5] |
| `sink` | [0, 5] | `bathtub` | [0, 5] |
| `refridgerator` | [0, 5] | `book` / `books` | [0, 1] |
| `paper` | [0, 1] | `towel` | [0, 1] |
| `shower curtain` | [0, 1] | `bag` | [0, 1] |
| `lamp` | [0, 1] | `otherprop/structure/furniture` | [0, 5] |
| `unannotated` | [0, 5] | | |

### Never planar (max = 0)

`mirror`, `curtain`, `pillow`, `blinds`, `clothes`, `person`, `''` (empty label)

### Explicitly non-planar labels

`bicycle`, `bottle`, `water bottle` — hardcoded in `nonPlanarGroupLabels`.

### Unknown labels

Labels not in the dictionary default to `[0, 5]`.

## Key Differences: ScanNet vs ScanNet++

| Aspect | Original (ScanNet) | Adapted (ScanNet++) |
|--------|-------------------|---------------------|
| **Segmentation** | Over-segmentation (few hundred segments per scene) | Identity segmentation (~590K segments = vertices) |
| **Processing unit** | Per-segment, then merge across segments | Per-object (no merging needed) |
| **Merging** | `mergePlanes()` merges neighboring segment planes | Skipped — objects are atomic |
| **Unannotated segments** | Added as individual objects | Skipped (too many: ~55K vs ~100 annotated groups) |
| **Mesh file** | `*_vh_clean_2.labels.ply` | `mesh_aligned_0.05_semantic.ply` |
| **Annotations** | `.aggregation.json` + `.segs.json` | `segments_anno.json` + `segments.json` |
| **Output** | `planes.npy`, `plane_info.npy`, `planes.ply` | `rendered_planercnn.h5` (same format as `rendered.h5`) |

## Output Format

### H5 file (`rendered_planercnn.h5`)

```python
with h5py.File("rendered_planercnn.h5", "r") as f:
    planes = f['planes'][:]      # (N_frames, 480, 640) uint16
    frame_ids = f['frame_ids'][:]  # (N_frames,) bytes, e.g. b'frame_000000'
```

- Label `0` = non-planar
- Labels `1, 2, ...` = plane IDs (globally consistent across frames)
- Format is identical to our GT `rendered.h5`

### Inference H5 (`planes.h5`)

For evaluation compatibility, the same data can be saved as:
```
H5_ROOT/planercnn_gt/<scene_id>/planes.h5
```
with the same keys and format, allowing direct use with `evaluate_all_baselines.py`.

## Typical Results (ScanNet++ test scenes)

| Scene | Planes | Vertex Coverage | Avg Planes/Frame | Avg Frame Coverage |
|-------|--------|----------------|-------------------|--------------------|
| `0a5c013435` | 33 | 80.1% | 10.3 | 84.5% |
| `c50d2d1d42` | 112 | 84.1% | 22.4 | 90.8% |
| `fb5a96b1a2` | 123 | 83.8% | 28.6 | 82.2% |
| `a24f64f7fb` | 60 | 68.2% | 19.7 | 84.6% |

PlaneRCNN GT consistently produces **more planes** and **higher coverage** than our GT (which uses stricter quality filtering). The trade-off is that PlaneRCNN planes may include lower-quality fits — see the inlier/outlier analysis in the notebook.
