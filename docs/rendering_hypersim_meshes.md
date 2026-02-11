# Rendering Hypersim Meshes and Using Camera Parameters

A detailed guide for loading, projecting, raycasting, and rendering Hypersim scene geometry with correct camera parameters.

---

## Table of Contents

1. [Data Layout and File Formats](#1-data-layout-and-file-formats)
2. [Loading HDF5 Image Data](#2-loading-hdf5-image-data)
3. [Camera Extrinsics (Poses)](#3-camera-extrinsics-poses)
4. [Camera Intrinsics (Per-Scene Projection)](#4-camera-intrinsics-per-scene-projection)
5. [Projecting 3D Points into Images](#5-projecting-3d-points-into-images)
6. [Casting Rays from Pixels](#6-casting-rays-from-pixels)
7. [Rendering Meshes with PyTorch3D](#7-rendering-meshes-with-pytorch3d)
8. [Tone Mapping HDR Images](#8-tone-mapping-hdr-images)
9. [Converting Euclidean Depth to Planar Depth](#9-converting-euclidean-depth-to-planar-depth)
10. [Coordinate Conventions Summary](#10-coordinate-conventions-summary)

---

## 1. Data Layout and File Formats

Each scene `ai_VVV_NNN` has the following structure relevant to rendering:

```
ai_VVV_NNN/
├── _detail/
│   ├── metadata_scene.csv                        # meters_per_asset_unit scale factor
│   ├── cam_XX/
│   │   ├── camera_keyframe_positions.hdf5        # (N, 3) camera positions in asset coords
│   │   └── camera_keyframe_orientations.hdf5     # (N, 3, 3) rotation matrices
│   └── mesh/
│       ├── mesh_vertices.hdf5                    # (V, 3) vertex positions in asset coords
│       ├── mesh_faces_vi.hdf5                    # (F, 3) face vertex indices
│       ├── mesh_faces_oi.hdf5                    # (F,) face-to-object-ID mapping
│       ├── mesh_objects_si.hdf5                  # (O,) object-ID → NYU40 semantic label
│       └── mesh_objects_sii.hdf5                 # (O,) object-ID → semantic instance ID
└── images/
    ├── scene_cam_XX_final_hdf5/
    │   ├── frame.IIII.color.hdf5                 # (H, W, 3) HDR RGB, no tone mapping
    │   ├── frame.IIII.diffuse_reflectance.hdf5   # (H, W, 3) albedo
    │   ├── frame.IIII.diffuse_illumination.hdf5  # (H, W, 3) diffuse shading
    │   └── frame.IIII.residual.hdf5              # (H, W, 3) specular + view-dependent
    └── scene_cam_XX_geometry_hdf5/
        ├── frame.IIII.depth_meters.hdf5          # (H, W) Euclidean distance to camera
        ├── frame.IIII.normal_cam.hdf5            # (H, W, 3) normals in camera space
        ├── frame.IIII.normal_world.hdf5          # (H, W, 3) normals in world space
        ├── frame.IIII.position.hdf5              # (H, W, 3) world positions in asset coords
        ├── frame.IIII.render_entity_id.hdf5      # (H, W) V-Ray node IDs
        ├── frame.IIII.semantic.hdf5              # (H, W) NYU40 semantic labels
        └── frame.IIII.semantic_instance.hdf5     # (H, W) instance IDs
```

All binary data is stored as HDF5 with key `"dataset"`.

---

## 2. Loading HDF5 Image Data

```python
import h5py
import numpy as np

scene_name  = "ai_001_001"
camera_name = "cam_00"
frame_id    = 0

base = f"{scene_name}/images"

# HDR color (float, no tone mapping applied)
with h5py.File(f"{base}/scene_{camera_name}_final_hdf5/frame.{frame_id:04d}.color.hdf5", "r") as f:
    rgb_hdr = f["dataset"][:].astype(np.float32)  # (H, W, 3)

# Depth (Euclidean distance in meters to camera optical center)
with h5py.File(f"{base}/scene_{camera_name}_geometry_hdf5/frame.{frame_id:04d}.depth_meters.hdf5", "r") as f:
    depth_meters = f["dataset"][:].astype(np.float32)  # (H, W)

# Semantic labels (NYU40)
with h5py.File(f"{base}/scene_{camera_name}_geometry_hdf5/frame.{frame_id:04d}.semantic.hdf5", "r") as f:
    semantic = f["dataset"][:].astype(np.int32)  # (H, W)

# World-space positions (in asset coordinates, NOT meters)
with h5py.File(f"{base}/scene_{camera_name}_geometry_hdf5/frame.{frame_id:04d}.position.hdf5", "r") as f:
    positions_world = f["dataset"][:].astype(np.float32)  # (H, W, 3)
```

**Important**: The `color` image satisfies `color ≈ diffuse_reflectance × diffuse_illumination + residual`. These are raw HDR values — you must apply tone mapping before use in learning tasks (see [Section 8](#8-tone-mapping-hdr-images)).

---

## 3. Camera Extrinsics (Poses)

Camera trajectories are stored per-camera as dense keyframe sequences:

```python
camera_dir = f"{scene_name}/_detail/{camera_name}"

with h5py.File(f"{camera_dir}/camera_keyframe_positions.hdf5", "r") as f:
    camera_positions = f["dataset"][:]     # (N, 3) in asset coordinates

with h5py.File(f"{camera_dir}/camera_keyframe_orientations.hdf5", "r") as f:
    camera_orientations = f["dataset"][:]  # (N, 3, 3) rotation matrices
```

For a given frame:

```python
# Camera position in asset coordinates
camera_position_world = camera_positions[frame_id]        # (3,)

# Rotation: maps camera-space → world-space
R_world_from_cam = camera_orientations[frame_id]           # (3, 3)
```

**Camera-space convention**: +x right, +y up, +z away from look direction (i.e., the camera looks along -z in its own coordinate system).

To build the full 4×4 camera-from-world (view) matrix:

```python
t_world_from_cam = np.matrix(camera_position_world).T           # (3, 1)
R_cam_from_world = np.matrix(R_world_from_cam).T                # inverse = transpose for rotation
t_cam_from_world = -R_cam_from_world * t_world_from_cam         # (3, 1)

M_cam_from_world = np.matrix(np.block([
    [R_cam_from_world, t_cam_from_world],
    [np.zeros((1, 3)), 1.0]
]))  # (4, 4)
```

---

## 4. Camera Intrinsics (Per-Scene Projection)

**Each scene uses different camera intrinsics** because some scenes use non-standard tilt-shift photography parameters. The per-scene parameters are provided in:

```
contrib/mikeroberts3000/metadata_camera_parameters.csv
```

Load intrinsics for a scene:

```python
import pandas as pd

df = pd.read_csv("contrib/mikeroberts3000/metadata_camera_parameters.csv", index_col="scene_name")
params = df.loc[scene_name]

width_pixels  = int(params["settings_output_img_width"])
height_pixels = int(params["settings_output_img_height"])
meters_per_asset_unit = params["settings_units_info_meters_scale"]
```

### Projection Matrix (M_proj)

A 4×4 perspective projection matrix (drop-in replacement for the standard OpenGL projection matrix) that accounts for tilt-shift:

```python
M_proj = np.matrix([
    [params["M_proj_00"], params["M_proj_01"], params["M_proj_02"], params["M_proj_03"]],
    [params["M_proj_10"], params["M_proj_11"], params["M_proj_12"], params["M_proj_13"]],
    [params["M_proj_20"], params["M_proj_21"], params["M_proj_22"], params["M_proj_23"]],
    [params["M_proj_30"], params["M_proj_31"], params["M_proj_32"], params["M_proj_33"]],
])
```

### UV-to-Camera-Ray Matrix (M_cam_from_uv)

A 3×3 matrix that maps normalized pixel coordinates (u, v ∈ [-1, 1]) to camera-space ray directions:

```python
M_cam_from_uv = np.matrix([
    [params["M_cam_from_uv_00"], params["M_cam_from_uv_01"], params["M_cam_from_uv_02"]],
    [params["M_cam_from_uv_10"], params["M_cam_from_uv_11"], params["M_cam_from_uv_12"]],
    [params["M_cam_from_uv_20"], params["M_cam_from_uv_21"], params["M_cam_from_uv_22"]],
])
```

### Field of View

```python
if params["use_camera_physical"]:
    fov_x = params["camera_physical_fov"]
else:
    fov_x = params["settings_camera_fov"]

fov_y = 2.0 * np.arctan(height_pixels * np.tan(fov_x / 2.0) / width_pixels)
```

---

## 5. Projecting 3D Points into Images

To project world-space 3D points (in asset coordinates) to pixel coordinates:

```python
# NDC-to-screen matrix (maps [-1,1] NDC to pixel coordinates)
M_screen_from_ndc = np.matrix([
    [0.5 * (width_pixels - 1), 0,                         0,   0.5 * (width_pixels - 1)],
    [0,                        -0.5 * (height_pixels - 1), 0,   0.5 * (height_pixels - 1)],
    [0,                        0,                          0.5, 0.5],
    [0,                        0,                          0,   1.0],
])

# points_world: (N, 3) array of 3D points in asset coordinates
num_points = points_world.shape[0]
P_world = np.matrix(np.c_[points_world, np.ones(num_points)]).T   # (4, N) homogeneous

P_cam    = M_cam_from_world @ P_world       # world → camera space
P_clip   = M_proj @ P_cam                   # camera → clip space
P_ndc    = np.array(P_clip) / np.array(P_clip[3])  # perspective divide → NDC
P_screen = M_screen_from_ndc @ np.matrix(P_ndc)    # NDC → pixel coordinates

pixel_x = np.asarray(P_screen[0]).flatten()  # horizontal pixel coords
pixel_y = np.asarray(P_screen[1]).flatten()  # vertical pixel coords
```

### Projecting Mesh Vertices of a Specific Object

To project only vertices belonging to a specific semantic class (e.g., NYU40 class 6 = "sofa"):

```python
mesh_dir = f"{scene_name}/_detail/mesh"

with h5py.File(f"{mesh_dir}/mesh_vertices.hdf5", "r") as f:
    mesh_vertices = f["dataset"][:]
with h5py.File(f"{mesh_dir}/mesh_faces_vi.hdf5", "r") as f:
    mesh_faces_vi = f["dataset"][:]
with h5py.File(f"{mesh_dir}/mesh_faces_oi.hdf5", "r") as f:
    mesh_faces_oi = f["dataset"][:]
with h5py.File(f"{mesh_dir}/mesh_objects_si.hdf5", "r") as f:
    mesh_objects_si = f["dataset"][:]

# Get faces belonging to semantic class si=6
target_si = 6
face_indices = np.where(np.isin(mesh_faces_oi, np.where(mesh_objects_si == target_si)[0]))[0]
vertex_indices = mesh_faces_vi[face_indices].ravel()
points_world = mesh_vertices[vertex_indices]  # (N, 3) in asset coordinates

# Then project using the pipeline above
```

---

## 6. Casting Rays from Pixels

To generate camera rays that correspond exactly to each pixel in a Hypersim image, use `M_cam_from_uv`:

```python
# Create a grid of normalized UV coordinates in [-1, 1]
u_min, u_max = -1.0, 1.0
v_min, v_max = -1.0, 1.0
half_du = 0.5 * (u_max - u_min) / width_pixels
half_dv = 0.5 * (v_max - v_min) / height_pixels

u, v = np.meshgrid(
    np.linspace(u_min + half_du, u_max - half_du, width_pixels),
    np.linspace(v_min + half_dv, v_max - half_dv, height_pixels)[::-1]  # note: reversed
)

# Stack into (H*W, 3) homogeneous UV coordinates
uvs = np.dstack((u, v, np.ones_like(u)))
P_uv = np.matrix(uvs.reshape(-1, 3)).T  # (3, H*W)

# Transform UV → camera-space ray directions, then → world-space
R_world_from_cam_mat = np.matrix(R_world_from_cam)
P_world = R_world_from_cam_mat @ M_cam_from_uv @ P_uv  # (3, H*W)

ray_directions_world = np.array(P_world.T)                           # (H*W, 3)
ray_positions_world  = np.ones_like(ray_directions_world) * camera_position_world  # (H*W, 3)
```

### Raycasting Against the Mesh

Using the Embree-based raycaster provided by the toolkit:

```python
import sys, inspect
sys.path.insert(0, "code/python/lib")
import embree_utils

intersection_distances, intersection_normals, prim_ids = \
    embree_utils.generate_ray_intersections(
        mesh_vertices, mesh_faces_vi,
        ray_positions_world, ray_directions_world,
        tmp_dir="_tmp"
    )

# Reshape to image and convert from asset units to meters
depth_from_raycasting = intersection_distances.reshape(height_pixels, width_pixels)
depth_from_raycasting *= meters_per_asset_unit
```

The resulting depth image should closely match the Hypersim `depth_meters` ground truth.

---

## 7. Rendering Meshes with PyTorch3D

PyTorch3D uses different camera-space conventions than Hypersim. You must account for this to get aligned renders.

### Convention Differences

| Axis | Hypersim | PyTorch3D |
|------|----------|-----------|
| +x   | Right    | **Left**  |
| +y   | Up       | Up        |
| +z   | Away from view | **Towards view** |

### Handling Non-Standard Intrinsics

Hypersim scenes may have non-standard projection matrices (due to tilt-shift parameters). PyTorch3D's `FoVPerspectiveCameras` only supports standard perspective, so we warp the vertices in camera-space to compensate:

```python
import torch
import scipy.linalg
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
)

# Build the canonical (no tilt-shift) camera-from-UV matrix
M_cam_from_uv_canonical = np.matrix([
    [np.tan(fov_x / 2.0), 0.0,                0.0],
    [0.0,                  np.tan(fov_y / 2.0), 0.0],
    [0.0,                  0.0,                 -1.0],
])

# Warp matrix: transforms camera-space points to account for non-standard intrinsics
M_warp_cam_pts_ = M_cam_from_uv_canonical @ np.linalg.inv(M_cam_from_uv)
M_warp_cam_pts  = scipy.linalg.block_diag(M_warp_cam_pts_, 1)  # (4, 4)

# Convention conversion: Hypersim camera-space → PyTorch3D camera-space
M_p3dcam_from_cam = np.eye(4)
M_p3dcam_from_cam[0, 0] = -1  # flip x
M_p3dcam_from_cam[2, 2] = -1  # flip z

# Transform all mesh vertices: world → Hypersim cam → warped cam → PyTorch3D cam
num_verts = mesh_vertices.shape[0]
P_world = np.matrix(np.c_[mesh_vertices, np.ones(num_verts)]).T
P_p3dcam = M_p3dcam_from_cam @ M_warp_cam_pts @ M_cam_from_world @ P_world

# Build PyTorch3D mesh and renderer
verts = torch.tensor(np.asarray(P_p3dcam.T[:, 0:3]), dtype=torch.float32)
faces = torch.tensor(mesh_faces_vi)
mesh  = Meshes(verts=[verts], faces=[faces])

device = torch.device("cpu")  # or "cuda:0"

# aspect_ratio=1.0 refers to pixel aspect ratio, NOT image aspect ratio
cameras = FoVPerspectiveCameras(
    device=device, fov=fov_y, degrees=False,
    aspect_ratio=1.0, znear=1.0, zfar=400.0
)

raster_settings = RasterizationSettings(
    image_size=[height_pixels, width_pixels],
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardDepthShader(device=device, cameras=cameras),  # see custom shader below
)

images = renderer(mesh)
rendered_depth = images[0, ..., 0].cpu().numpy()
```

### Custom Depth Shader

PyTorch3D doesn't ship a depth shader by default. Use this:

```python
from pytorch3d.renderer.mesh.shader import ShaderBase

class HardDepthShader(ShaderBase):
    """Renders Z distances of the closest face per pixel."""
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = self.cameras
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0
        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = zfar
        return zbuf
```

---

## 8. Tone Mapping HDR Images

Hypersim `color` images are raw HDR. The toolkit implements a tone mapping operator based on the CCIR601 YIQ brightness method (following the CGIntrinsics / Snavely strategy):

```python
def tonemap_hypersim(rgb_hdr, render_entity_id=None):
    """
    Tone-map a Hypersim HDR color image to [0, 1] LDR.

    Args:
        rgb_hdr: (H, W, 3) float32, raw HDR color
        render_entity_id: (H, W) int32, optional. If provided, only valid pixels
                          (render_entity_id != -1) are used for brightness estimation.
                          If None, all pixels are used.
    Returns:
        (H, W, 3) float32 in [0, 1]
    """
    gamma = 1.0 / 2.2
    inv_gamma = 1.0 / gamma
    percentile = 90
    brightness_nth_percentile_desired = 0.8

    if render_entity_id is not None:
        valid_mask = render_entity_id != -1
    else:
        valid_mask = np.ones(rgb_hdr.shape[:2], dtype=bool)

    if np.count_nonzero(valid_mask) == 0:
        scale = 1.0
    else:
        brightness = 0.3 * rgb_hdr[:, :, 0] + 0.59 * rgb_hdr[:, :, 1] + 0.11 * rgb_hdr[:, :, 2]
        brightness_valid = brightness[valid_mask]

        eps = 0.0001
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:
            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    rgb_tm = np.power(np.maximum(scale * rgb_hdr, 0), gamma)
    return np.clip(rgb_tm, 0, 1)
```

The `render_entity_id` image is used to mask out background pixels (where `render_entity_id == -1`) so they don't affect the brightness percentile computation.

---

## 9. Converting Euclidean Depth to Planar Depth

Hypersim `depth_meters` stores **Euclidean distance** from the pixel to the camera optical center, not planar depth (negative z in camera-space). To convert:

```python
def euclidean_to_planar_depth(depth_meters, M_cam_from_uv, width_pixels, height_pixels):
    """
    Convert Hypersim Euclidean depth to planar depth (z-buffer style).

    The idea: each pixel has a ray direction in camera space. The planar depth
    is the z-component of the 3D point along that ray at the given Euclidean distance.
    """
    u_min, u_max = -1.0, 1.0
    v_min, v_max = -1.0, 1.0
    half_du = 0.5 * (u_max - u_min) / width_pixels
    half_dv = 0.5 * (v_max - v_min) / height_pixels

    u, v = np.meshgrid(
        np.linspace(u_min + half_du, u_max - half_du, width_pixels),
        np.linspace(v_min + half_dv, v_max - half_dv, height_pixels)[::-1]
    )

    uvs = np.dstack((u, v, np.ones_like(u)))
    # Ray directions in camera space
    ray_dirs_cam = (M_cam_from_uv @ uvs.reshape(-1, 3).T).T.reshape(height_pixels, width_pixels, 3)
    # Normalize ray directions
    ray_dirs_cam = np.array(ray_dirs_cam)
    ray_lengths = np.linalg.norm(ray_dirs_cam, axis=2)

    # planar depth = euclidean_depth * |z_component_of_normalized_ray|
    # Since z points away from view in Hypersim, planar depth = depth_meters * (-z / ||ray||)
    planar_depth = depth_meters * np.abs(ray_dirs_cam[:, :, 2]) / ray_lengths

    return planar_depth
```

---

## 10. Coordinate Conventions Summary

| Property | Convention |
|----------|-----------|
| **World coordinates** | "Asset coordinates" — multiply by `meters_per_asset_unit` from `metadata_scene.csv` to get meters |
| **Camera +x** | Right |
| **Camera +y** | Up |
| **Camera +z** | Away from look direction (camera looks along -z) |
| **Camera orientations** | 3×3 rotation matrix mapping camera-space → world-space |
| **Camera positions** | Stored in asset coordinates (not meters) |
| **depth_meters** | Euclidean distance to optical center, **already in meters** (exception to the asset-coordinate convention) |
| **position images** | World-space, in asset coordinates |
| **Normals (normal_cam)** | In camera-space |
| **Normals (normal_world)** | In world-space |
| **HDF5 key** | Always `"dataset"` |
| **Image resolution** | Varies per scene; typically 1024×768; check `settings_output_img_width/height` |
| **Label interpolation** | Always use `cv2.INTER_NEAREST` for semantic/instance labels — never linear |

### Unit Conversion

```python
import pandas as pd

# From metadata_scene.csv
scene_metadata = pd.read_csv(f"{scene_name}/_detail/metadata_scene.csv")
meters_per_asset_unit = scene_metadata["meters_per_asset_unit"].iloc[0]

# Or from the camera parameters CSV
df = pd.read_csv("contrib/mikeroberts3000/metadata_camera_parameters.csv", index_col="scene_name")
meters_per_asset_unit = df.loc[scene_name]["settings_units_info_meters_scale"]

# Convert positions from asset coords to meters
positions_meters = positions_asset * meters_per_asset_unit
```

**Important caveat**: `depth_meters` is already in meters. Camera positions and mesh vertices are in asset coordinates. When mixing them (e.g., comparing raycasted distances to `depth_meters`), multiply raycasted distances by `meters_per_asset_unit`.

---

## References

- [Hypersim paper (ICCV 2021)](https://arxiv.org/abs/2011.02523)
- Jupyter notebooks with runnable examples: `contrib/mikeroberts3000/jupyter/`
  - `00_projecting_points_into_hypersim_images.ipynb` — 3D→2D point projection
  - `01_casting_rays_that_match_hypersim_images.ipynb` — pixel→ray generation with Embree raycasting
  - `02_rendering_hypersim_meshes_with_pytorch3d.ipynb` — PyTorch3D depth rendering
- Tone mapping implementation: `code/python/tools/scene_generate_images_tonemap.py`
- Depth conversion discussion: [GitHub issue #9](https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697)
