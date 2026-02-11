# Hypersim Plane Rendering Pipeline

How GT plane labels from the 3D mesh get projected to 2D per-camera images.

## Overview

```
planes.ply (Stage 1 output)
    │
    ├── V: (N,3) vertices in asset units
    ├── F: (M,3) face indices
    └── plane_id_face: (M,) per-face plane ID
         │
         ▼
  ┌──────────────────┐
  │  rendering.py     │  For each camera, for each frame:
  │                    │
  │  1. Load mesh      │  read_ply_faces_with_plane_ids()
  │  2. Load K         │  metadata_camera_parameters.csv → M_proj → K
  │  3. Load c2w       │  camera_keyframe_positions/orientations.hdf5
  │  4. Flip c2w       │  c2w = c2w @ diag(1,-1,-1,1)
  │  5. Raycast        │  raycast_semantic_face_labels()
  │  6. Flip Y         │  np.flipud(semantic_img)
  │  7. Remap          │  -1 → 0
  └──────────────────┘
         │
         ▼
  rendered_planes_cam_XX.h5
    ├── planes: (num_frames, H, W) uint16
    └── frame_ids: (num_frames,) string
```

## Step-by-Step

### 1. Load Mesh (`read_ply_faces_with_plane_ids`)

**File:** `planamono/shared/rendering/mesh_io.py:123`

Reads `planes.ply` produced by Stage 1 (plane extraction). The PLY has per-face properties:

```
vertex: x y z (float32)
face: 3 v0 v1 v2 (uchar + 3×int32) + plane_id (int32) + label_int (int32)
```

Returns:
- `V (N,3)` — vertex positions in **Hypersim asset units** (not meters)
- `F (M,3)` — triangle vertex indices
- `plane_id_face (M,)` — plane ID per face (0 = non-planar)
- `label_int_face (M,)` — semantic label per face

The mesh is loaded into an Open3D `TriangleMesh` with computed vertex normals:

```python
sem_mesh = o3d.geometry.TriangleMesh()
sem_mesh.vertices = o3d.utility.Vector3dVector(V)
sem_mesh.triangles = o3d.utility.Vector3iVector(F)
sem_mesh.compute_vertex_normals()
```

### 2. Compute Intrinsics from Projection Matrix

**File:** `planamono/gt_creation/hypersim/rendering.py:35`

Hypersim stores a 4×4 OpenGL projection matrix `M_proj` per scene in `metadata_camera_parameters.csv`. This is converted to a 3×3 OpenCV intrinsics matrix:

```python
def compute_intrinsics_from_proj(M_proj, width, height):
    fx =  M_proj[0,0] * 0.5 * width
    fy = -M_proj[1,1] * 0.5 * height   # Y-axis flip (OpenGL → OpenCV)
    cx =  M_proj[0,2] * 0.5 * width  + 0.5 * width
    cy = -M_proj[1,2] * 0.5 * height + 0.5 * height
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])
```

**Why the negation on fy and cy?** OpenGL's projection matrix has Y pointing up, while OpenCV has Y pointing down. The negation converts the Y component from OpenGL to OpenCV convention.

**Resolution:** `width` and `height` come from `settings_output_img_width/height` in the metadata CSV. Typically 1024×768.

**Note:** The same K is used for all cameras within a scene — Hypersim uses the same virtual lens for all cameras.

### 3. Load Camera Poses

**File:** `planamono/gt_creation/hypersim/rendering.py:138-142`

Per-camera HDF5 files store keyframe poses:

```
{input_root}/{scene_id}/_detail/{cam_name}/
├── camera_keyframe_positions.hdf5     → dataset: (num_frames, 3)
└── camera_keyframe_orientations.hdf5  → dataset: (num_frames, 3, 3)
```

For each frame, a 4×4 camera-to-world matrix is assembled:

```python
c2w = np.eye(4)
c2w[:3, :3] = cam_orientations[frame_id]   # 3×3 rotation
c2w[:3,  3] = cam_positions[frame_id]       # 3D translation
```

### 4. Coordinate Convention Flip

```python
c2w = c2w @ np.diag([1, -1, -1, 1])
```

Hypersim camera poses follow OpenCV convention (X-right, Y-down, Z-forward). The raycasting function `raycast_semantic_face_labels` works in OpenGL convention (X-right, Y-up, Z-backward). This flip converts between them by negating Y and Z axes of the camera frame.

### 5. Raycasting (`raycast_semantic_face_labels`)

**File:** `planamono/shared/rendering/render.py:154`

This is the core function that projects the 3D mesh plane IDs to a 2D image.

#### 5a. Build Raycasting Scene

```python
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sem_mesh))
```

Uses Open3D's tensor-based `RaycastingScene` which builds a BVH (bounding volume hierarchy) for fast ray-triangle intersection.

#### 5b. Apply OpenGL Coordinate Flip

```python
flip_yz = np.diag([1, -1, -1, 1])
c2w_gl = c2w @ flip_yz
R = c2w_gl[:3, :3]
t = c2w_gl[:3, 3]      # = camera origin in world coordinates
```

The input `c2w` was already flipped in Step 4, so after this second flip it's back to the original Hypersim convention. **However**, this function is generic — when called from other code paths where c2w hasn't been pre-flipped, this internal flip ensures OpenGL convention is used for ray generation.

In the Hypersim rendering pipeline, the two flips cancel out:
```
c2w_original → flip (Step 4) → flip (Step 5b) → c2w_original
```

#### 5c. Generate Ray Directions

```python
u, v = np.meshgrid(np.arange(W), np.arange(H))
v = H - 1 - v                          # Flip Y: OpenCV (top-left origin) → OpenGL (bottom-left origin)
dirs = np.stack([
    (u - cx) / fx,                      # X: normalized image coordinates
    (v - cy) / fy,                      # Y: normalized (flipped)
    -np.ones_like(u)                    # Z: negative (OpenGL looks along -Z)
], axis=-1)
dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)   # unit vectors
dirs_world = dirs @ R.T                 # rotate to world frame
```

For each pixel `(u, v)`:
1. Compute the direction in camera space using the pinhole model
2. Z is -1 because OpenGL camera looks along negative Z
3. Normalize to unit length
4. Rotate from camera to world frame using `R.T` (equivalent to `R^{-1}` since R is orthogonal)

All ray origins are the camera center `t`:
```python
rays_o = np.tile(cam_origin, (H, W, 1))
```

#### 5d. Cast Rays

```python
rays = np.concatenate([rays_o, dirs_world], axis=-1)   # (H, W, 6): [origin_xyz, direction_xyz]
ans = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))
```

Open3D returns:
- `primitive_ids`: which triangle was hit per ray (INVALID_ID if miss)
- `t_hit`: ray parameter at intersection (not used here)

#### 5e. Map Triangle ID → Plane ID

```python
triangle_ids = ans['primitive_ids'].numpy().reshape(H, W)
hit_mask = triangle_ids != o3d.t.geometry.RaycastingScene.INVALID_ID

semantic_img = np.full((H, W), fill_value=-1, dtype=np.int32)
semantic_img[hit_mask] = face_labels[triangle_ids[hit_mask]]
```

Direct lookup: `triangle_id → face_labels[triangle_id] → plane_id`. No interpolation, no vertex voting — each pixel gets the exact plane ID of the triangle it hits.

### 6. Post-Processing (back in `rendering.py`)

```python
semantic_img = remap_semantic(semantic_img_face)    # -1 → 0
semantic_img = np.flipud(semantic_img)              # OpenGL (bottom-up) → image (top-down)
semantic_img = np.clip(semantic_img, 0, 65535).astype(np.uint16)
```

The `flipud` undoes the Y-flip from Step 5c so the output image has standard top-left origin.

### 7. Save to HDF5

```python
with h5py.File(f"rendered_planes_{cam_name}.h5", "w") as f:
    f.create_dataset("planes", data=np.stack(planes_list), compression="gzip")
    f.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))
```

Output shape: `(num_rendered_frames, H, W)` as uint16. Frame IDs are zero-padded strings like `"0000"`, `"0025"`, etc. (depends on `--frame_skip`).

## Coordinate System Summary

```
Hypersim Camera Poses     OpenCV convention    (X-right, Y-down,  Z-forward)
         │
    diag(1,-1,-1,1)       Step 4 flip
         │
         ▼
raycast_semantic_face_labels() input
         │
    diag(1,-1,-1,1)       Step 5b internal flip (cancels Step 4)
         │
         ▼
Ray Generation            OpenGL convention    (X-right, Y-up,    Z-backward)
  - v = H-1-v             Y pixel flip
  - Z direction = -1      Camera looks along -Z
         │
    R.T (rotation)        Camera → World
         │
         ▼
World Space Rays → Open3D RaycastingScene → triangle_ids → plane_ids
         │
    np.flipud             Step 6: OpenGL image → OpenCV image
         │
         ▼
Final Image               OpenCV convention    (top-left origin)
```

## Two Raycasting Functions

`render.py` has two raycasting functions:

| Function | Label Source | Used By |
|----------|-------------|---------|
| `raycast_semantic(sem_mesh, vertex_labels, ...)` | Per-vertex labels, nearest-vertex voting at hit point | ScanNet++ (vertex-labeled meshes) |
| `raycast_semantic_face_labels(sem_mesh, face_labels, ...)` | Per-face labels, direct triangle ID lookup | Hypersim (face-labeled PLY from plane extraction) |

The face-label version is simpler and more accurate — no interpolation or voting needed since each triangle belongs to exactly one plane.

## Units

- Mesh vertices are in **Hypersim asset units** (not meters)
- MPAU (meters per asset unit) is NOT applied during rendering
- The raycasting operates purely on geometry — units don't matter for which triangle is hit
- Units only matter later during evaluation when depth (in meters) is backprojected to 3D
