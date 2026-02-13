# Hypersim Depth Bugs

---

# Bug 1: Missing MPAU (meters_per_asset_unit) Conversion

## Problem

When raycasting the planes mesh (`planes.ply`) to obtain depth, the returned `t_hit` values were **orders of magnitude too large** compared to the ground truth depth from `depth_meters.hdf5`.

| | Raycasted (buggy) | Original V-Ray | Ratio |
|---|---|---|---|
| `ai_001_004` frame 0 | 117.8 – 216.5 | 2.99 – 5.50 m | ~39x |

This caused precision/recall to drop significantly:

| Threshold | Original Depth P/R | Raycasted (buggy) P/R |
|---|---|---|
| 1mm | 0.81 / 0.70 | 0.28 / 0.24 |
| 5mm | 1.00 / 0.86 | 0.88 / 0.76 |
| 10mm | 1.00 / 0.86 | 0.98 / 0.84 |

## Root Cause

Hypersim scenes are authored in **asset units** (varies per scene — inches, centimeters, or arbitrary units). All spatial data in the scene — mesh vertex positions, camera positions, camera orientations — are in these asset units. Open3D raycasting therefore returns `t_hit` in asset units.

However, the `depth_meters.hdf5` files shipped with Hypersim have already been converted to **meters** by V-Ray during rendering.

The conversion factor is **MPAU** (meters per asset unit), stored per-scene in the metadata:
- Primary location: `metadata_camera_parameters.csv` column `settings_units_info_meters_scale`
- Fallback: `<scene_id>/_detail/metadata_scene.csv` row `meters_per_asset_unit`

Example MPAU values:
- `ai_021_002`: 0.01 (1 asset unit = 1 cm)
- `ai_001_004`: ~0.0254 (1 asset unit = 1 inch)

## Buggy Code

```python
# raycast_mesh_zdepth() — BEFORE fix
t_hit = ans["t_hit"].numpy().reshape(H, W)
hit_mask = np.isfinite(t_hit) & (t_hit > 0)
t_hit[~hit_mask] = 0.0

# BUG: t_hit is in asset units, not meters!
z_depth = euclidean_to_zdepth(t_hit, K)
```

## Fix

Multiply `t_hit` by MPAU to convert from asset units to meters before any further processing:

```python
# raycast_mesh_zdepth() — AFTER fix
t_hit = ans["t_hit"].numpy().reshape(H, W)
hit_mask = np.isfinite(t_hit) & (t_hit > 0)

# Convert asset units -> meters
mpau = get_mpau(scene_id)  # e.g. 0.0254 for inch-based scenes
t_hit_meters = np.where(hit_mask, t_hit * mpau, 0.0)

# Now convert Euclidean distance (meters) to z-depth
z_depth = euclidean_to_zdepth(t_hit_meters, K)
z_depth[~hit_mask] = 0.0
```

Where `get_mpau()` reads from the metadata CSV:
```python
def get_mpau(scene_id):
    row = df_meta.loc[scene_id]
    mpau = float(row.get("settings_units_info_meters_scale", 0.0))
    if mpau > 0 and not np.isnan(mpau):
        return mpau
    # fallback: read from per-scene metadata_scene.csv
    ...
```

## Verification

The fix was confirmed by `debug_hypersim_rendering_fixed_v3.ipynb`, which already uses this conversion correctly:
```python
depth_raycast = np.where(hit_mask, t_hit * mpau, 0.0)
```
That notebook reports depth correlation of 0.999999 between raycasted and V-Ray depth after applying MPAU.

## Affected Files

- `evaluate_gt_inliers_hypersim_raycasted_depth.ipynb` — fixed by adding `get_mpau()` and `t_hit * mpau`

## Key Takeaway

**Any code that raycasts the Hypersim mesh must multiply `t_hit` by MPAU to get meters.** The mesh, camera positions, and camera orientations are all in asset units. Only V-Ray's output files (`depth_meters.hdf5`) are pre-converted to meters.

---

# Bug 2: Euclidean-to-Z-Depth Conversion Uses Pinhole K Instead of M_cam_from_uv

## Problem

`HypersimPlaneDataset._euclidean_to_zdepth()` converts V-Ray's Euclidean ray distance to z-depth using a **pinhole camera model (K)**, but Hypersim renders using V-Ray's camera defined by **`M_cam_from_uv`** — a 3x3 matrix that is NOT a standard pinhole. The pinhole K is only an approximation derived from `M_proj`.

This means the z-depth values stored in `sample["depth"]` are **approximate**, not exact.

## How the Two Camera Models Differ

### V-Ray camera (M_cam_from_uv) — how depth_meters.hdf5 is actually generated

For each pixel `(u_px, v_px)`, V-Ray generates a ray in camera space:

```
u_ndc = (2 * u_px + 1) / W - 1
v_ndc = 1 - (2 * v_px + 1) / H
d_cam = M_cam_from_uv @ [u_ndc, v_ndc, 1]     ← actual ray direction
```

`depth_meters.hdf5` stores `t_hit` = Euclidean distance along this ray. The **correct** z-depth is:

```
z_correct = t_hit * d_cam.z / |d_cam|
```

where `d_cam.z` is the z-component of the camera-space ray direction (varies per pixel).

### Pinhole camera (K) — what the dataset code does

```python
# HypersimPlaneDataset._euclidean_to_zdepth()
x_n = (u_px - cx) / fx
y_n = (v_px - cy) / fy
z_approx = depth_euc / sqrt(x_n² + y_n² + 1)
```

This assumes the ray direction is `[x_n, y_n, 1]`, where the **z-component is always 1**. But M_cam_from_uv has cross-terms that make the z-component vary per pixel.

### Concrete example: M_cam_from_uv for ai_021_002

```
[[ 0.592   0.000  -0.036]
 [ 0.000   0.443   0.354]
 [ 0.000   0.032  -1.026]]
```

Key observations:
- **(2,1) = 0.032**: The z-component of the ray depends on `v_ndc`, not just the constant column `[0, 0, 1]`
- **(2,2) = -1.026**: The z-component is -1.026, not -1.0 as pinhole would assume
- **(0,2) = -0.036** and **(1,2) = 0.354**: The x and y components have offsets from `[u_ndc, v_ndc, 1]`

These terms mean M_cam_from_uv encodes a more complex camera model than a simple pinhole.

## What Goes Wrong

The error shows up in the **z-component ratio** used for the conversion:

| | z-component / ray_length |
|---|---|
| **Pinhole K** | `1 / sqrt(x_n² + y_n² + 1)` — always uses z=1 |
| **M_cam_from_uv** | `d_cam.z / |d_cam|` — z varies per pixel |

For center pixels this difference is tiny. For edge/corner pixels the error grows because the cross-terms in M_cam_from_uv have more effect.

## Why It Partially Cancels (But Not Fully)

The same pinhole K is used for both the conversion and backprojection:

```python
# Step 1: euclidean_to_zdepth (conversion)
z_depth = depth_euc / sqrt(x_n² + y_n² + 1)

# Step 2: backproject_v1 (reconstruction)
X = z_depth * (u - cx) / fx = depth_euc * x_n / sqrt(x_n² + y_n² + 1)
Y = z_depth * (v - cy) / fy = depth_euc * y_n / sqrt(x_n² + y_n² + 1)
Z = z_depth                  = depth_euc * 1   / sqrt(x_n² + y_n² + 1)
```

Combined: `P = depth_euc * [x_n, y_n, 1] / |[x_n, y_n, 1]|`

This places the 3D point at the correct **Euclidean distance** from the camera, but along the **pinhole ray direction** instead of the true M_cam_from_uv ray direction. So:

- **Distance from camera**: correct (depth_euc is preserved)
- **Direction from camera**: slightly wrong (pinhole vs M_cam_from_uv)
- **3D position**: slightly wrong, error grows toward image edges

For plane fitting, this means points that should lie on a perfect plane will have small systematic offsets. At tight thresholds (1mm), this can cause real inlier count differences.

## Correct Solution

Replace the pinhole conversion with M_cam_from_uv-based conversion:

```python
def euclidean_to_zdepth_mcam(depth_euc, M_cam_from_uv, W, H):
    """Convert Euclidean ray distance to z-depth using actual V-Ray ray model.

    Uses M_cam_from_uv to compute per-pixel ray directions, then extracts
    the z-component ratio for exact conversion.
    """
    # Build NDC grid (same as rendering.py)
    half_du = 1.0 / W
    half_dv = 1.0 / H
    u_ndc = np.linspace(-1 + half_du, 1 - half_du, W)
    v_ndc = np.linspace(-1 + half_dv, 1 - half_dv, H)[::-1]
    uu, vv = np.meshgrid(u_ndc, v_ndc)

    # Camera-space ray directions
    uvs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)
    dirs_cam = uvs @ M_cam_from_uv.T                      # (H, W, 3)

    # Per-pixel conversion factor: z_component / ray_magnitude
    ray_lengths = np.linalg.norm(dirs_cam, axis=-1)        # (H, W)
    z_components = dirs_cam[:, :, 2]                       # (H, W)
    cos_theta = z_components / ray_lengths                 # (H, W)

    # z_depth = depth_euc * cos(theta)
    # Note: cos_theta can be negative (V-Ray z-axis points backward)
    z_depth = depth_euc * np.abs(cos_theta)
    return z_depth
```

And for backprojection, instead of using pinhole K, use M_cam_from_uv ray directions directly:

```python
def backproject_mcam(depth_euc, M_cam_from_uv, R_world_from_cam, cam_position, W, H):
    """Backproject using M_cam_from_uv ray directions (exact).

    Returns 3D points in world coordinates.
    """
    # NDC grid
    half_du = 1.0 / W
    half_dv = 1.0 / H
    u_ndc = np.linspace(-1 + half_du, 1 - half_du, W)
    v_ndc = np.linspace(-1 + half_dv, 1 - half_dv, H)[::-1]
    uu, vv = np.meshgrid(u_ndc, v_ndc)

    # Camera-space ray directions
    uvs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    dirs_cam = uvs @ M_cam_from_uv.T
    dirs_world = dirs_cam @ R_world_from_cam.T
    dirs_normed = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    # 3D points: origin + depth_euc * ray_direction
    pts = cam_position + depth_euc[..., None] * dirs_normed
    return pts  # (H, W, 3) in world coordinates
```

## Impact Assessment

| Aspect | Severity |
|---|---|
| Plane fitting at 10mm threshold | Low — error is much smaller than threshold |
| Plane fitting at 1mm threshold | Medium — systematic ray direction error at edges can cause ~0.5-1mm position error |
| Absolute 3D positions | Medium — points are shifted from true positions, grows toward image edges |
| Comparison between methods | Low — all methods use the same approximate K, so bias cancels in relative comparison |
| Original vs raycasted depth consistency | Low — both use same pinhole K, so z-depth values are consistent |

## Affected Code

- `HypersimPlaneDataset._euclidean_to_zdepth()` — uses pinhole K approximation
- `HypersimPlaneDataset._get_intrinsics()` — derives pinhole K from M_proj
- `backproject_v1()` / `backproject_v2()` — backproject using pinhole K
- `evaluate_gt_inliers_hypersim_raycasted_depth.ipynb` — `euclidean_to_zdepth()` helper
- Any evaluation code that backprojects Hypersim depth to 3D

## Key Takeaway

The pinhole K is a **convenience approximation** of the V-Ray camera. For exact 3D reconstruction from Hypersim, use `M_cam_from_uv` for both ray generation and depth conversion. The pinhole approximation is acceptable for evaluation where all methods share the same bias, but not for absolute geometric accuracy at sub-millimeter thresholds.
