# HypersimPlaneDataset — Known Issues

Analysis of depth, intrinsics (K), and camera-to-world (c2w) correctness in
`hypersim_plane_dataset.py`.

---

## 1. Depth: Ray Distance Not Converted to Z-Depth

**Severity: High**

Hypersim's `frame.XXXX.depth_meters.hdf5` stores **Euclidean ray distance**
(distance from camera center to the surface point along the viewing ray), not
**Z-depth** (perpendicular distance to the image plane).

The dataset loads it raw without conversion (`hypersim_plane_dataset.py:246`):

```python
depth = f[key][:].astype(np.float32)  # treated as Z-depth, but it's ray distance
```

The evaluation pipeline calls `backproject_v1` (`planefit.py:57-58`), which
assumes Z-depth:

```python
rays = (Kinv @ uv1.T).T          # unit rays through each pixel
pts_cam = rays[valid] * z[valid, None]  # scale by depth
```

When `z` is ray distance instead of Z-depth, the resulting 3D points are
pushed outward — the error grows with distance from the principal point.
Corner pixels can be off by several percent.

### Correct conversion

The codebase already has `extract_zdepth()` in `shared/utils/depth_normal.py:92`:

```python
x_n = (u - cx) / fx
y_n = (v - cy) / fy
ray_length = np.sqrt(x_n**2 + y_n**2 + 1)
Z = depth_map / ray_length
```

This is never called in the dataset class or evaluation scripts.

### Impact

- Plane fitting (RANSAC) receives distorted 3D point clouds.
- Precision/recall at tight thresholds (1 mm, 5 mm) are affected most.
- The error is systematic: planes near image borders appear curved.

---

## 2. c2w: Always Set to Identity

**Severity: Low (for current single-frame evaluation)**

`hypersim_plane_dataset.py:257` always returns identity:

```python
c2w = np.eye(4, dtype=np.float32)
```

The `params_root` is discovered during `__init__` (line 100) and
`camera_keyframe_positions.hdf5` / `camera_keyframe_orientations.hdf5` exist
on disk, but they are **never read** in `__getitem__`.

The GT rendering script (`gt_creation/hypersim/rendering.py:139-161`) shows
the correct loading procedure:

```python
with h5py.File(cam_pos_path, "r") as f:
    cam_positions = f["dataset"][:]       # (N_frames, 3)
with h5py.File(cam_rot_path, "r") as f:
    cam_orientations = f["dataset"][:]    # (N_frames, 3, 3)

c2w = np.eye(4)
c2w[:3, :3] = cam_orientations[frame_id]
c2w[:3, 3]  = cam_positions[frame_id]
c2w = c2w @ np.diag([1, -1, -1, 1])      # flip Y,Z for Open3D convention
```

### Why severity is low for now

`backproject_v1` transforms camera-space points by c2w. With identity, the
"world" points are really camera-space points. Since both GT segmentation and
predicted segmentation go through the same identity transform, RANSAC plane
fitting and precision/recall are self-consistent within a single frame.

### When it would matter

- Any cross-frame aggregation (multi-view consistency checks).
- Any comparison requiring true metric world coordinates.
- Visualization overlaid on a 3D scene reconstruction.

---

## 3. K: Default Fallback Always Used — metadata_csv Never Passed

**Severity: Medium**

`_get_intrinsics()` (`hypersim_plane_dataset.py:173-196`) has two branches:

1. **Metadata CSV branch** — computes per-scene intrinsics from `M_proj`.
2. **Default fallback** — uses `fx = fy = 886.81`, `cx = W/2`, `cy = H/2`.

The evaluation script (`evaluate_hypersim_all_baselines.py:199-208`)
instantiates the dataset **without** `metadata_csv`:

```python
dataset = HypersimPlaneDataset(
    hypersim_root=HYPERSIM_ROOT,
    plane_label_root=PLANE_LABEL_ROOT,
    params_root=PARAMS_ROOT,
    split_txt_dir=...,
    split=split,
    image_height=512,
    image_width=768,
    max_scenes=None,
    # metadata_csv is never passed → defaults to None
)
```

Since `metadata_csv=None`, `_get_intrinsics` always takes the default branch.
The default `886.81` is approximately correct for the majority of Hypersim
scenes (which share a common FoV), but scenes with different field-of-view
settings will have wrong intrinsics.

Additionally, the default is calibrated for 1024×768 resolution
(`cx=512, cy=384`). When `image_height=512, image_width=768` is used (as in
the evaluation script), the principal point shifts to `cx=384, cy=256`, but
`fx=fy` should also be rescaled proportionally. The current code uses
`cx = self.image_width / 2.0` which is correct for the principal point, but
the focal length `886.81` was measured at 1024×768 and is not rescaled to the
target resolution.

### Correct approach

Either:
- Always pass `metadata_csv` to the dataset, or
- Compute intrinsics from the per-scene HDF5 camera parameters in `params_root`.

When rescaling images, scale `fx, fy, cx, cy` by `target_width / original_width`.

---

## Summary

| Field | Problem | File:Line | Severity |
|-------|---------|-----------|----------|
| **depth** | Raw ray distance used as Z-depth | `hypersim_plane_dataset.py:246` | High |
| **c2w** | Always identity, poses never loaded | `hypersim_plane_dataset.py:257` | Low (current use) |
| **K** | `metadata_csv` never passed in eval; default `fx=886.81` not rescaled | `hypersim_plane_dataset.py:192`, `evaluate_hypersim_all_baselines.py:199` | Medium |

### Affected downstream code

- `evaluate_hypersim_all_baselines.py` — all 3D plane metrics (precision/recall)
- `evaluate_hypersim_fast.py` — same
- `eval_utils.py:evaluate_single_frame` → `backproject_v1` — receives wrong depth + approximate K
