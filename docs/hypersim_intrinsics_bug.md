# Hypersim Camera Ray Bugs

Three bugs exist in the code that converts Hypersim's camera parameters to ray directions for raycasting plane labels onto 2D images.

## Affected Code

| File | Function | Lines |
|------|----------|-------|
| `PixelwisePlanarity/planamono/gt_creation/hypersim/rendering.py` | `compute_intrinsics_from_proj()` | 35–43 |
| `PixelwisePlanarity/planamono/shared/datasets/hypersim_plane_dataset.py` | `_get_intrinsics()` | 191–194 |
| `PixelwisePlanarity/planamono/shared/rendering/render.py` | `raycast_semantic_face_labels()` | 154–194 |

---

## Summary of All Three Bugs

| Bug | Severity | Affected Scenes | Error Magnitude | Status |
|-----|----------|----------------|-----------------|--------|
| 1. Sign error on cx/cy | **Large** | 22 of 482 (tilt-shift) | Up to 584.6 px | Fixed (sign flip) |
| 2. W vs (W-1) off-by-one | Small | All 482 | ~0.5 px at center | Not yet fixed |
| 3. Pixel convention mismatch | Small | All 482 | ~0.5 px directional | Not yet fixed |

**Bugs 2 and 3 are both sub-pixel, but they compound.** The proper fix for all three is to abandon the pinhole-from-`M_proj` approach entirely and use Hypersim's official `M_cam_from_uv` matrix directly (see "The Correct Approach" below).

---

## Bug 1: Sign Error on cx/cy Offsets (Large — Tilt-Shift Scenes)

### The Bug

The original code has the sign of `M_proj[0,2]` and `M_proj[1,2]` **flipped**:

```python
# ORIGINAL CODE (WRONG):
cx =  M_proj[0, 2] * 0.5 * width  + 0.5 * width   # = 0.5*W*(1 + M_proj[0,2])
cy = -M_proj[1, 2] * 0.5 * height + 0.5 * height   # = 0.5*H*(1 - M_proj[1,2])
```

The correct derivation (from `M_screen_from_ndc`) gives:
```
cx = 0.5*(W-1)*(1 - M_proj[0,2])    ← MINUS sign on M_proj[0,2]
cy = 0.5*(H-1)*(1 + M_proj[1,2])    ← PLUS sign on M_proj[1,2]
```

This **mirrors the principal point across the image center**.

### Impact

- **22 of 482 scenes** have non-zero tilt-shift offsets and are affected.
- The remaining 460 standard scenes are unaffected (`M_proj[0,2] = M_proj[1,2] = 0`).
- Worst case: **ai_021_002** (`M_proj[1,2] = 0.761`), cy shifts by **584.6 pixels**.

### Example: ai_021_002 (1024×768)

```
cy_buggy   = 0.5 * 768 * (1 - 0.761) =  91.7   ← near TOP of image (wrong)
cy_correct = 0.5 * 767 * (1 + 0.761) = 675.4   ← near BOTTOM of image (correct)
```

### Why it was hidden

The debug notebook compared re-rendered planes against stored H5 — both rendered with the same wrong K, so the comparison was self-consistent. Depth correlation: 0.44 (buggy) vs 0.93 (fixed), confirming rays were aimed at wrong scene regions.

### Status

**Fixed** in `rendering.py` and `hypersim_plane_dataset.py`. Sign of `M_proj[0,2]` negated in cx, sign of `M_proj[1,2]` negated in cy.

---

## Bug 2: `W` vs `(W-1)` Off-by-One (Small — All Scenes)

### The Bug

The code uses **`W`** and **`H`** where the official `M_screen_from_ndc` uses **`(W-1)`** and **`(H-1)`**:

```python
# AFTER SIGN FIX (STILL WRONG):
fx =  M_proj[0, 0] * 0.5 * width                       # should be 0.5 * (width - 1)
fy = -M_proj[1, 1] * 0.5 * height                      # should be 0.5 * (height - 1)
cx = -M_proj[0, 2] * 0.5 * width  + 0.5 * width       # should use (width - 1)
cy =  M_proj[1, 2] * 0.5 * height + 0.5 * height      # should use (height - 1)
```

### Why `(W-1)` is correct

`M_screen_from_ndc` maps `NDC = -1 → pixel 0`, `NDC = +1 → pixel W-1`. The scaling factor is `0.5*(W-1)`, not `0.5*W`. Using `W` places the optical center 0.5 pixels off.

### Impact (1024×768)

| Parameter | Code (`W`) | Correct (`W-1`) | Error |
|-----------|-----------|-----------------|-------|
| fx | 864.7 | 863.9 | +0.84 px |
| cx (standard) | 512.0 | 511.5 | +0.5 px |

### Status

**Not yet fixed** — superseded by the M_cam_from_uv approach (Bug 3 fix resolves this too).

---

## Bug 3: Pixel Convention Mismatch (Small — All Scenes)

### The Bug

Even after fixing Bugs 1 and 2, **there remains a ~0.5 pixel directional misalignment** visible in plane-edge overlays. The camera appears to be looking slightly differently than V-Ray's actual rendering.

### Root Cause: Two Different Pixel Sampling Conventions

The pinhole approach (Bugs 1–2) reconstructs rays via `M_proj → M_screen_from_ndc → pinhole K`, then generates rays using an **integer pixel grid** (`np.arange(W)`). But Hypersim's V-Ray renderer uses a **different pixel sampling convention** defined by `M_cam_from_uv`.

**NDC convention** (used by `M_screen_from_ndc`):
```
pixel 0     → NDC = -1        (first pixel center)
pixel W-1   → NDC = +1        (last pixel center)
pixel i     → NDC = 2*i/(W-1) - 1
```

**UV convention** (used by V-Ray / `M_cam_from_uv`):
```
pixel 0     → UV = -1 + 1/W   (half-pixel inset from -1)
pixel W-1   → UV = +1 - 1/W   (half-pixel inset from +1)
pixel i     → UV = -1 + (2*i + 1)/W
```

The key difference: UV coordinates have **half-pixel insets** at the boundaries. NDC maps pixel centers exactly to [-1, +1], while UV maps pixel centers to [-1 + 1/W, +1 - 1/W]. For W=1024:

```
Pixel 0:     NDC = -1.000000    UV = -0.999023    Δ = 0.000977 (~0.5 px)
Pixel 511:   NDC = -0.000978    UV = -0.000977    Δ ≈ 0
Pixel 1023:  NDC = +1.000000    UV = +0.999023    Δ = 0.000977 (~0.5 px)
```

The error is zero at the image center but grows to ~0.5 pixels at the edges. This is a **directional** offset — the rays point in slightly wrong directions, especially near image boundaries.

### Why the Pinhole Approach Cannot Match V-Ray Exactly

The pinhole model (`u = fx·X/d + cx`) generates rays via:
```python
dirs = [(pixel_x - cx) / fx, (pixel_y - cy) / fy, -1]
```

This assumes a specific relationship between pixel coordinates and ray angles. But V-Ray doesn't use this convention — it uses `M_cam_from_uv` applied to a UV grid with half-pixel offsets. The two ray generation methods are **algebraically different** and cannot be made identical by adjusting `fx/fy/cx/cy` alone.

### Observable Symptom

After fixing Bugs 1 and 2, overlaying rendered plane edges on RGB shows very slight misalignment: "camera looking slightly up but rendered slightly down." Depth correlation improves from 0.93 (pinhole fixed) to higher values with `M_cam_from_uv`.

### Status

**Not yet fixed in scripts.** The proper fix is to replace the pinhole approach with `M_cam_from_uv` ray generation (see below).

---

## The Correct Approach: `M_cam_from_uv`

### Source

From `ml-hypersim/contrib/mikeroberts3000/jupyter/01_casting_rays_that_match_hypersim_images.ipynb`:

```python
# M_cam_from_uv is a 3×3 matrix stored per-scene in metadata_camera_parameters.csv
# It maps UV coordinates directly to camera-space ray directions.

# Official Hypersim UV grid with half-pixel offsets
half_du = 1.0 / W
half_dv = 1.0 / H
u = np.linspace(-1 + half_du, 1 - half_du, W)
v = np.linspace(-1 + half_dv, 1 - half_dv, H)[::-1]  # top row = largest v (y-flip)
uu, vv = np.meshgrid(u, v)

# Ray direction in camera space
P_uv = np.stack([uu, vv, np.ones_like(uu)], axis=-1)   # (H, W, 3)
dirs_cam = P_uv @ M_cam_from_uv.T                       # (H, W, 3)

# Transform to world space
dirs_world = dirs_cam @ R_world_from_cam.T
dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)
```

### Key Differences from Pinhole Approach

| Aspect | Pinhole (`M_proj` → K) | M_cam_from_uv |
|--------|------------------------|---------------|
| Pixel grid | `np.arange(W)` (integers 0..W-1) | `linspace(-1+1/W, 1-1/W, W)` (UV with half-pixel insets) |
| Ray generation | `(pixel - cx) / fx` | `M_cam_from_uv @ [u, v, 1]` |
| Y-flip | `v = H - 1 - v` + `np.flipud()` after raycast | `v[::-1]` in UV grid (built into the grid) |
| Tilt-shift | Encoded in cx/cy via `M_proj[0,2]`, `M_proj[1,2]` | Encoded directly in `M_cam_from_uv` matrix |
| Accuracy | Approximate (~0.5 px error at edges) | **Exact** (matches V-Ray renderer) |

### Raycasting Function Using `M_cam_from_uv`

```python
def raycast_planes_mcam(sem_mesh, face_labels, M_cam_from_uv, R_world_from_cam,
                        cam_position, width, height):
    """Raycast plane labels using Hypersim's official M_cam_from_uv convention.

    This matches V-Ray's pixel sampling exactly, avoiding the ~0.5px error
    of the pinhole-from-M_proj approach.

    Args:
        sem_mesh: Open3D TriangleMesh with plane geometry
        face_labels: (N_faces,) int array of plane IDs per face
        M_cam_from_uv: (3, 3) matrix from metadata_camera_parameters.csv
        R_world_from_cam: (3, 3) rotation matrix (camera-to-world)
        cam_position: (3,) camera position in world space
        width, height: Image dimensions in pixels

    Returns:
        (H, W) int32 array of plane IDs (0 = no hit)
    """
    W, H = width, height
    scene_rc = o3d.t.geometry.RaycastingScene()
    scene_rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sem_mesh))

    # Official Hypersim UV grid: half-pixel offsets within [-1, 1]
    half_du = 1.0 / W
    half_dv = 1.0 / H
    u = np.linspace(-1 + half_du, 1 - half_du, W)
    v = np.linspace(-1 + half_dv, 1 - half_dv, H)[::-1]  # top row = largest v
    uu, vv = np.meshgrid(u, v)

    # Camera-space ray directions via M_cam_from_uv
    uvs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)   # (H, W, 3)
    dirs_cam = uvs @ M_cam_from_uv.T                       # (H, W, 3)

    # World-space ray directions
    dirs_world = dirs_cam @ R_world_from_cam.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    # Raycast
    origins = np.broadcast_to(cam_position, dirs_world.shape).copy()
    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    ans = scene_rc.cast_rays(
        o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)
    )

    triangle_ids = ans['primitive_ids'].numpy().reshape(H, W)
    hit_mask = triangle_ids != o3d.t.geometry.RaycastingScene.INVALID_ID

    semantic_img = np.zeros((H, W), dtype=np.int32)
    semantic_img[hit_mask] = face_labels[triangle_ids[hit_mask]]
    return semantic_img   # NO flipud needed — y-flip built into UV grid
```

### Equivalent Pinhole K (for depth conversion only)

For code that needs a pinhole K matrix (e.g., `_euclidean_to_zdepth()`), the closest approximation using `(W-1)`:

```python
W1 = width - 1
H1 = height - 1
fx =  M_proj[0, 0] * 0.5 * W1
fy = -M_proj[1, 1] * 0.5 * H1
cx = -M_proj[0, 2] * 0.5 * W1 + 0.5 * W1
cy =  M_proj[1, 2] * 0.5 * H1 + 0.5 * H1
```

This is accurate to ~0.5 px — sufficient for `_euclidean_to_zdepth()` where the error is negligible.

---

## Affected Scenes

### Bug 1 (sign error): 22 of 482 scenes

Scenes with non-zero `M_proj[0,2]` or `M_proj[1,2]`:

| Scene | M_proj[0,2] | M_proj[1,2] | Shift (px) |
|-------|-------------|-------------|------------|
| ai_021_002/003/004 | -0.057 | 0.761 | 584.6 (cy) |
| ai_036_001–010 | 0.341 | 0.000 | 349.7 (cx) |
| ai_028_005–008 | 0.000 | 0.354 | 271.6 (cy) |
| ai_035_002/003/010 | 0.000 | 0.344 | 264.5 (cy) |

### Bugs 2 & 3 (pixel convention): All 482 scenes

~0.5 pixel shift at center, ~1 pixel at edges. Sub-pixel for most applications.

---

## Scripts That Need Fixing

### 1. `rendering.py` — Plane label raycasting (rendering pipeline)

**Current**: Derives pinhole K from `M_proj`, passes to `raycast_semantic_face_labels()`, applies `np.flipud()`.

**Fix**: Load `M_cam_from_uv` from metadata CSV. Use `raycast_planes_mcam()`. Remove `np.flipud()`. Pass `R_world_from_cam` (= `cam_orientations[frame_id]`) and `cam_position` (= `cam_positions[frame_id]`) directly — no `diag(1,-1,-1,1)` flip needed.

### 2. `hypersim_plane_dataset.py` — Intrinsics for training/evaluation

**Current**: Derives pinhole K from `M_proj` in `_get_intrinsics()`.

**Fix**: Use the corrected `(W-1)` pinhole formula. The K matrix here is used for `_euclidean_to_zdepth()` and backprojection, where ~0.5px error is negligible.  Alternatively, load `M_cam_from_uv` columns and provide them alongside K.

### 3. `render.py` — Shared raycast functions

**Current**: `raycast_semantic_face_labels()` uses pinhole K with integer pixel grid + `diag(1,-1,-1,1)` flip + `v = H-1-v`.

**Fix**: Add a new `raycast_semantic_face_labels_mcam()` function that takes `M_cam_from_uv`, `R_world_from_cam`, and `cam_position` instead of K and c2w. Uses UV grid with half-pixel offsets. No flipud. The existing pinhole functions should be kept for ScanNet++ (which uses standard pinhole cameras).

---

## Re-rendering Required

All stored plane-label H5 files **must** be re-rendered with the corrected ray generation:
- **22 tilt-shift scenes**: Bug 1 made rays completely wrong (up to 584 px shift)
- **All 482 scenes**: Bugs 2+3 cause ~0.5–1 px misalignment (may matter for edge-level accuracy)

---

## Verification

The fix was verified in `debug_hypersim_rendering_fixed_v2.ipynb` on scene **ai_021_002** (tilt-shift):

| Method | Depth Correlation | Edge Alignment |
|--------|------------------|----------------|
| Buggy (original) | 0.44 | Completely wrong |
| Pinhole sign-fixed | 0.93 | Good but ~0.5px off |
| M_cam_from_uv | Best | Pixel-perfect |

---

## Note on Negative fy

The pinhole K has `fy < 0`. This is intentional: it encodes the y-flip from camera +y (up) to image +v (down). It works correctly in `_euclidean_to_zdepth()` (squares `y_n`, sign cancels). Code assuming OpenCV convention (`fy > 0`) will produce Y-flipped 3D coordinates.
