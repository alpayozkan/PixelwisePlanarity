# Hypersim Intrinsics Bug: Wrong Principal Point for Tilt-Shift Scenes

## Summary

`compute_intrinsics_from_proj()` in `rendering.py` derives a pinhole intrinsics matrix K from Hypersim's 4x4 projection matrix `M_proj`. The formula for `cx` and `cy` has a **sign error on the offset terms** (`M_proj[0,2]` and `M_proj[1,2]`). For scenes without tilt-shift these terms are zero and the bug is invisible. For tilt-shift scenes, the rendered plane labels view a completely different part of the scene than the V-Ray-rendered RGB/depth images.

## Affected files

| File | Function/Lines |
|------|---------------|
| `planamono/gt_creation/hypersim/rendering.py` | `compute_intrinsics_from_proj()` (lines 35-43) |
| `planamono/shared/datasets/hypersim_plane_dataset.py` | `_get_intrinsics()` (lines 191-194) |

Both contain the same formula. All stored plane label H5 files for tilt-shift scenes were rendered with the wrong K.

## Derivation of the correct formula

### Hypersim's official projection pipeline

From `ml-hypersim/contrib/mikeroberts3000/jupyter/00_projecting_points_into_hypersim_images.ipynb`, the full pipeline for projecting a 3D point to a screen pixel is:

```
P_world → M_cam_from_world → P_cam → M_proj → P_clip → perspective_divide → P_ndc → M_screen_from_ndc → P_screen
```

Where `M_screen_from_ndc` maps NDC ([-1, +1]) to pixel coordinates:

```python
M_screen_from_ndc = [[0.5*(W-1),  0,             0,    0.5*(W-1)],
                     [0,          -0.5*(H-1),    0,    0.5*(H-1)],
                     [0,          0,             0.5,  0.5       ],
                     [0,          0,             0,    1.0       ]]
```

### Step-by-step for a camera-space point

Given a point `(X, Y, Z, 1)` in Hypersim camera space (+x right, +y up, +z away from look direction), with `M_proj[3,2] = -1` (standard OpenGL):

**Clip space:**
```
P_clip[0] = M_proj[0,0]*X + M_proj[0,2]*Z
P_clip[1] = M_proj[1,1]*Y + M_proj[1,2]*Z
P_clip[3] = -Z
```

**NDC (perspective divide):**
```
P_ndc[0] = P_clip[0] / P_clip[3] = -M_proj[0,0]*X/Z - M_proj[0,2]
P_ndc[1] = P_clip[1] / P_clip[3] = -M_proj[1,1]*Y/Z - M_proj[1,2]
```

Substituting `d = -Z` (positive depth in front of camera):
```
P_ndc[0] = M_proj[0,0]*X/d - M_proj[0,2]
P_ndc[1] = M_proj[1,1]*Y/d - M_proj[1,2]
```

**Screen coordinates (applying M_screen_from_ndc):**
```
screen_x =  0.5*(W-1) * P_ndc[0] + 0.5*(W-1)
screen_y = -0.5*(H-1) * P_ndc[1] + 0.5*(H-1)
```

Expanding:
```
screen_x = [0.5*(W-1)*M_proj[0,0]] * X/d  +  [0.5*(W-1)*(1 - M_proj[0,2])]
screen_y = [-0.5*(H-1)*M_proj[1,1]] * Y/d  +  [0.5*(H-1)*(1 + M_proj[1,2])]
```

This gives the pinhole form `u = fx*X/d + cx`, `v = fy*Y/d + cy`:

```
fx =  0.5*(W-1) * M_proj[0,0]
fy = -0.5*(H-1) * M_proj[1,1]          (negative: Y up → v down)
cx =  0.5*(W-1) * (1 - M_proj[0,2])    ← MINUS sign
cy =  0.5*(H-1) * (1 + M_proj[1,2])    ← PLUS sign
```

### Verification: optical axis projection

For a point on the optical axis `(X=0, Y=0, Z=-d)`:
```
screen_x = cx = 0.5*(W-1)*(1 - M_proj[0,2])
screen_y = cy = 0.5*(H-1)*(1 + M_proj[1,2])
```

For a standard scene (M_proj[0,2] = M_proj[1,2] = 0):
- `cx = 0.5*(W-1) ≈ 512` (image center) ✓
- `cy = 0.5*(H-1) ≈ 384` (image center) ✓

For scene ai_021_002 (M_proj[1,2] ≈ 0.761):
- `cy = 0.5*767*(1 + 0.761) = 675.4` (near bottom of 768-pixel image) ✓

This matches the tilt-shift behavior: the optical axis projects near the bottom of the image because the lens shift captures mostly what's above the optical axis (used for tall interior photography without converging verticals).

## The bug

### Current code (`rendering.py:35-43`)

```python
def compute_intrinsics_from_proj(M_proj, width, height):
    fx =  M_proj[0, 0] * 0.5 * width
    fy = -M_proj[1, 1] * 0.5 * height
    cx =  M_proj[0, 2] * 0.5 * width  + 0.5 * width   # 0.5*W*(1 + M_proj[0,2])  ← WRONG
    cy = -M_proj[1, 2] * 0.5 * height + 0.5 * height   # 0.5*H*(1 - M_proj[1,2])  ← WRONG
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
```

### The sign error

| Parameter | Code | Correct | Error |
|-----------|------|---------|-------|
| cx | `0.5*W*(1 + M_proj[0,2])` | `0.5*W*(1 - M_proj[0,2])` | Sign of `M_proj[0,2]` flipped |
| cy | `0.5*H*(1 - M_proj[1,2])` | `0.5*H*(1 + M_proj[1,2])` | Sign of `M_proj[1,2]` flipped |

### Why it's invisible for most scenes

Most Hypersim scenes have symmetric frustums (no tilt-shift), so `M_proj[0,2] = 0` and `M_proj[1,2] = 0`. Both formulas reduce to `cx = 0.5*W` and `cy = 0.5*H`, which are correct.

### Impact on tilt-shift scenes (e.g., ai_021_002)

With `M_proj[1,2] ≈ 0.761`:

```
cy_code    = 0.5 * 768 * (1 - 0.761) =  91.7   ← near TOP of image
cy_correct = 0.5 * 768 * (1 + 0.761) = 676.2   ← near BOTTOM of image
```

The principal point is **mirrored** across the image center. The raycasting pipeline generates rays aimed at a completely different part of the scene than what V-Ray rendered. The angular error at the top pixel row:

```
Code ray angle:    arctan((0 -  91.7) / 846) =  6.2° above optical axis
Correct ray angle: arctan((0 - 676.2) / 846) = 38.6° above optical axis
```

At the image center (r=384), the rays even point in **opposite vertical directions**:
```
Code:    (384 -  91.7) / (-846) = -0.345   (downward in camera space)
Correct: (384 - 676.2) / (-846) = +0.345   (upward in camera space)
```

### Why `debug_hypersim_rendering.ipynb` shows 100% plane match

The stored H5 files were rendered by `rendering.py` using the same wrong K. Re-rendering with the same K reproduces the same wrong result. The comparison is self-consistent but both copies are wrong relative to the actual RGB/depth images.

### Why depth correlation is 0.44

The notebook's Section 5 raycasts depth from the mesh using the same wrong K. The rays hit different mesh surfaces than V-Ray saw for that frame, so the depth patterns don't correlate — and no neighboring frame index helps because the error is angular, not temporal.

## The fix

### `rendering.py`

```python
def compute_intrinsics_from_proj(M_proj, width, height):
    """Convert Hypersim 4x4 OpenGL projection matrix to pinhole intrinsics.

    Derivation follows the official Hypersim NDC-to-screen mapping from
    ml-hypersim/contrib/mikeroberts3000/jupyter/00_projecting_points_into_hypersim_images.ipynb
    """
    fx =  M_proj[0, 0] * 0.5 * width
    fy = -M_proj[1, 1] * 0.5 * height
    cx = -M_proj[0, 2] * 0.5 * width  + 0.5 * width   # 0.5*W*(1 - M_proj[0,2])
    cy =  M_proj[1, 2] * 0.5 * height + 0.5 * height   # 0.5*H*(1 + M_proj[1,2])
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
```

### `hypersim_plane_dataset.py` (`_get_intrinsics`, lines 191-194)

```python
fx =  M_proj[0, 0] * 0.5 * native_w
fy = -M_proj[1, 1] * 0.5 * native_h
cx = -M_proj[0, 2] * 0.5 * native_w + 0.5 * native_w
cy =  M_proj[1, 2] * 0.5 * native_h + 0.5 * native_h
```

### Verification after fix

For ai_021_002:
```
K_fixed = [[864.7,    0,  541.2],
           [   0, -846.0, 676.2],
           [   0,    0,     1  ]]
```

Optical axis pixel: (541, 676) — near bottom-center of 1024x768 image, consistent with upward tilt-shift.

For ai_001_001 (no tilt-shift, M_proj offsets = 0):
```
K_fixed = [[886.8,    0,  512],
           [   0, -886.8, 384],
           [   0,    0,     1]]
```

Unchanged from the buggy version — symmetric scenes are unaffected.

## Re-rendering required

All stored plane label H5 files for scenes with non-zero `M_proj[0,2]` or `M_proj[1,2]` need to be re-rendered with the corrected K. To identify affected scenes:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('metadata_camera_parameters.csv', index_col='scene_name')
for scene in df.index:
    row = df.loc[scene]
    M_proj_02 = row['M_proj_02']
    M_proj_12 = row['M_proj_12']
    if abs(M_proj_02) > 1e-6 or abs(M_proj_12) > 1e-6:
        print(f"{scene}: M_proj[0,2]={M_proj_02:.6f}, M_proj[1,2]={M_proj_12:.6f}")
```

## Note on negative fy

The corrected K still has negative fy. This is an intentional encoding: Hypersim's camera +y is up while image +v is down, and the negative fy absorbs this flip. It works correctly within the raycasting pipeline (v-flip + flipud cancel, leaving `y_dir = (r - cy) / fy`). It also works in `_euclidean_to_zdepth` because `y_n` is squared.

However, any code that assumes standard OpenCV convention (positive fy) will silently produce Y-flipped 3D coordinates when using this K for backprojection. This is a separate concern from the sign bug and may warrant a convention note in the dataset class.
