# Paper-Quality Plane Segmentation Visualization Guide

This document describes how to produce the **RGB | Segmentation Mask | 3D Mesh** triplet visualizations used in ZeroPlane (Figures 3–5) and how to adapt the same style for our MoGe-based region-growing pipeline.

---

## 1. Overview of the Triplet

Every row in the visualization grid contains three images:

| Column | Content | Purpose |
|--------|---------|---------|
| **RGB** | Original input image | Reference |
| **Segmentation** | Per-plane colored overlay on the image | Shows 2D detection quality |
| **Mesh** | Textured or colored 3D mesh rendered from a rotated viewpoint | Shows 3D reconstruction quality |

The segmentation and mesh images **must share the same color mapping** — plane _k_ gets the same color in both views. Inconsistent colors between 2D and 3D was one of the bugs we previously fixed.

---

## 2. Segmentation Mask Image

This step is identical for both ZeroPlane and our pipeline.

### Inputs

- `rgb`: H × W × 3 input image (uint8, 0–255)
- `seg_map`: H × W integer array where each pixel holds a plane label (0 = non-planar / background)

### Procedure

```python
import numpy as np
from matplotlib import cm

def generate_color_map(num_planes, cmap_name="tab20"):
    """
    Returns a dict  {label: (R, G, B)}  with values in [0, 1].
    Use a qualitative colormap so adjacent planes are visually distinct.
    """
    cmap = cm.get_cmap(cmap_name, max(num_planes, 20))
    colors = {}
    for i, label in enumerate(range(1, num_planes + 1)):
        colors[label] = np.array(cmap(i % cmap.N)[:3])
    return colors

def render_segmentation_overlay(rgb, seg_map, color_map, alpha=0.55):
    """
    Blend per-plane colors onto the RGB image.
    
    Parameters
    ----------
    rgb       : H x W x 3, uint8
    seg_map   : H x W, int  (0 = background)
    color_map : dict {label: (R,G,B)} in [0,1]
    alpha     : blending weight for the color overlay
    
    Returns
    -------
    overlay : H x W x 3, uint8
    """
    overlay = rgb.astype(np.float64) / 255.0
    for label, color in color_map.items():
        mask = seg_map == label
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color
    return (overlay * 255).clip(0, 255).astype(np.uint8)
```

### Style notes

- ZeroPlane uses **opaque flat colors** (alpha ≈ 1.0) in most figures — no underlying RGB visible.
- For our paper, a **semi-transparent overlay** (alpha ≈ 0.5–0.6) can look more informative since it shows texture under the mask.
- Background / non-planar pixels: leave as original RGB, or darken them slightly to make planes pop.

---

## 3. 3D Mesh — ZeroPlane's Approach (Parametric)

ZeroPlane predicts per-plane parameters `(n, d)` and a binary mask directly from its Transformer queries.

### 3.1 Back-project pixels to 3D via the plane equation

For each plane with normal **n** and offset _d_, the depth at pixel (u, v) is:

```
ray = K_inv @ [u, v, 1]^T          # 3D ray direction
depth = d / (n^T · ray)            # analytic depth from plane eq.
P_3d  = depth * ray                # 3D point
```

This is exact by construction — every point lies perfectly on the plane.

### 3.2 Triangulate

Use the **quad-grid** method: for every 2×2 pixel block where all four corners belong to the same plane mask, emit two triangles. This naturally produces a watertight mesh per segment.

```python
def triangulate_mask(mask, H, W):
    idx = np.full((H, W), -1, dtype=np.int64)
    flat_idx = np.where(mask.reshape(-1))[0]
    idx.reshape(-1)[flat_idx] = np.arange(len(flat_idx))
    
    faces = []
    for r in range(H - 1):
        for c in range(W - 1):
            tl, tr = idx[r, c], idx[r, c+1]
            bl, br = idx[r+1, c], idx[r+1, c+1]
            if tl >= 0 and tr >= 0 and bl >= 0:
                faces.append([tl, tr, bl])
            if tr >= 0 and br >= 0 and bl >= 0:
                faces.append([tr, br, bl])
    return np.array(faces)
```

### 3.3 Color the mesh

Assign per-vertex color from either:
- The **input RGB** image → textured look (ZeroPlane Figure 5 style)
- The **per-plane color map** → colored look matching the segmentation column

---

## 4. 3D Mesh — Our MoGe-Based Pipeline

The fundamental difference: we do **not** compute depth from `(n, d)`. MoGe V2 gives us a dense 3D point map, and our plane parameters come from RANSAC on that point map.

### 4.1 The three render modes (important for paper honesty)

| Render | Depth source | Plane params source | Use in paper? |
|--------|-------------|---------------------|---------------|
| **A** | GT depth | — (mesh colored by GT seg) | GT reference only |
| **B1** | GT depth | RANSAC on GT depth | ❌ Misleading hybrid — don't use |
| **B2** | MoGe predicted | RANSAC on MoGe predicted | ✅ Correct — this is our method's output |

**Always use Render B2** for the "Ours-Mesh" column. B1 mixes GT geometry with predicted planes, which overstates the method's actual 3D quality.

### 4.2 Step-by-step mesh generation

```python
import numpy as np
import open3d as o3d

def build_plane_mesh(pts_map, seg_map, plane_params, rgb, 
                     color_map, max_edge_len=0.3, 
                     project_to_plane=True, use_plane_colors=True):
    """
    Build a combined Open3D mesh from MoGe points + our segmentation.
    
    Parameters
    ----------
    pts_map       : H x W x 3, MoGe predicted 3D coordinates
    seg_map       : H x W, integer plane labels (0 = background)
    plane_params  : dict {label: (n, d)} from RANSAC on MoGe points
    rgb           : H x W x 3, uint8 input image
    color_map     : dict {label: (R,G,B)} in [0,1]
    max_edge_len  : float, max 3D edge length for triangle filtering
    project_to_plane : if True, snap points to RANSAC plane
    use_plane_colors : if True, per-plane colors; if False, RGB texture
    
    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
    """
    H, W = seg_map.shape
    all_verts, all_faces, all_colors = [], [], []
    vert_offset = 0
    
    for label in sorted(np.unique(seg_map)):
        if label == 0:
            continue
        
        mask = (seg_map == label)
        n, d = plane_params[label]
        
        # --- vertices ---
        verts = pts_map[mask].copy()  # N x 3
        
        if project_to_plane:
            # project each point onto its fitted plane
            # signed distance: s = n·p - d
            signed_dist = verts @ n - d
            verts = verts - np.outer(signed_dist, n)
        
        # --- colors ---
        if use_plane_colors:
            c = color_map[label]
            colors = np.tile(c, (len(verts), 1))
        else:
            colors = rgb[mask].astype(np.float64) / 255.0
        
        # --- triangulation with edge filtering ---
        idx = np.full((H, W), -1, dtype=np.int64)
        rows, cols = np.where(mask)
        for i, (r, c) in enumerate(zip(rows, cols)):
            idx[r, c] = i + vert_offset
        
        faces = []
        for r in range(H - 1):
            for c in range(W - 1):
                tl, tr = idx[r, c], idx[r, c+1]
                bl, br = idx[r+1, c], idx[r+1, c+1]
                
                # triangle 1: tl-tr-bl
                if tl >= 0 and tr >= 0 and bl >= 0:
                    p = [pts_map[r,c], pts_map[r,c+1], pts_map[r+1,c]]
                    edge = max(np.linalg.norm(p[0]-p[1]),
                               np.linalg.norm(p[0]-p[2]),
                               np.linalg.norm(p[1]-p[2]))
                    if edge < max_edge_len:
                        faces.append([tl, tr, bl])
                
                # triangle 2: tr-br-bl
                if tr >= 0 and br >= 0 and bl >= 0:
                    p = [pts_map[r,c+1], pts_map[r+1,c+1], pts_map[r+1,c]]
                    edge = max(np.linalg.norm(p[0]-p[1]),
                               np.linalg.norm(p[0]-p[2]),
                               np.linalg.norm(p[1]-p[2]))
                    if edge < max_edge_len:
                        faces.append([tr, br, bl])
        
        all_verts.append(verts)
        all_faces.extend(faces)
        all_colors.append(colors)
        vert_offset += len(verts)
    
    # assemble Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    
    return mesh
```

### 4.3 Plane projection: why and when

Setting `project_to_plane=True` snaps every point onto the RANSAC-fitted plane. This:

- Makes each segment visually flat (like ZeroPlane's analytic planes)
- Is the honest representation of what our method predicts — a plane `(n, d)` per segment
- Without projection, the mesh shows MoGe's raw point cloud colored by segment, which doesn't demonstrate planarity

**Formula:** For a point **p**, normal **n**, offset _d_ (plane equation: **n**·**x** = _d_):

```
p_projected = p - ((n · p - d) · n)
```

### 4.4 Edge length threshold tuning

The `max_edge_len` parameter prevents triangles from spanning across depth discontinuities (e.g., a wall meeting a far-away floor). Guidelines:

| Scene type | Typical `max_edge_len` | Rationale |
|-----------|----------------------|-----------|
| Indoor (ScanNet++) | 0.05–0.15 m | Tight spaces, small depth jumps |
| Indoor (NYUv2) | 0.10–0.30 m | Moderate depth range |
| Outdoor | 0.50–2.0 m | Large depth range, faraway surfaces |

If using MoGe's affine/relative coordinates (not metric), you'll need to calibrate this threshold relative to the scene scale.

---

## 5. Rendering the 3D Mesh with Open3D

### 5.1 Offscreen rendering (recommended for paper figures)

```python
import open3d as o3d
import numpy as np

def render_mesh_offscreen(mesh, width, height, intrinsic_params, 
                           extrinsic, bg_color=[0, 0, 0, 1]):
    """
    Render a mesh to an image from a specified camera pose.
    
    Parameters
    ----------
    mesh             : o3d.geometry.TriangleMesh
    width, height    : int, output image size
    intrinsic_params : (fx, fy, cx, cy)
    extrinsic        : 4x4 numpy array (world-to-camera transform)
    bg_color         : RGBA background color
    
    Returns
    -------
    img : H x W x 3 numpy uint8 array
    """
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(bg_color)
    
    # material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"  # no lighting = flat per-vertex color
    # use "defaultLit" if you want shading to reveal surface geometry
    
    renderer.scene.add_geometry("mesh", mesh, mat)
    
    # set up camera
    fx, fy, cx, cy = intrinsic_params
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    renderer.setup_camera(intrinsic, extrinsic)
    
    img = np.asarray(renderer.render_to_image())
    return img
```

### 5.2 Choosing the camera viewpoint

ZeroPlane renders from a **rotated viewpoint** (~20–40° from the original camera) to reveal depth layering. Two strategies:

**Strategy A — Rotate around the scene centroid:**

```python
def rotated_extrinsic(original_extrinsic, angle_deg=25, axis='y'):
    """
    Rotate the camera around the scene centroid.
    """
    R_orig = original_extrinsic[:3, :3]
    t_orig = original_extrinsic[:3, 3]
    
    angle = np.radians(angle_deg)
    if axis == 'y':
        R_delta = np.array([
            [ np.cos(angle), 0, np.sin(angle)],
            [ 0,             1, 0            ],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'x':
        R_delta = np.array([
            [1, 0,              0             ],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle),  np.cos(angle)]
        ])
    
    new_ext = np.eye(4)
    new_ext[:3, :3] = R_delta @ R_orig
    new_ext[:3, 3]  = R_delta @ t_orig
    return new_ext
```

**Strategy B — Look-at a specific point:**

```python
center = mesh.get_center()
eye    = center + np.array([0.5, -0.3, -1.5])  # offset from center
up     = np.array([0.0, -1.0, 0.0])             # y-down convention

renderer.setup_camera(60.0, center, eye, up)  # FOV-based setup
```

Strategy B is often easier to tune interactively. Use `"defaultLit"` shader with a directional light to add subtle shading that reveals the plane geometry.

### 5.3 Open3D version gotcha

The camera API changed between Open3D versions:

| Version | Method |
|---------|--------|
| ≤ 0.16 | `setup_camera(intrinsic, extrinsic)` where extrinsic is 4×4 |
| 0.17+ | Same signature but `extrinsic` convention may differ (check if it's world-to-cam or cam-to-world) |

Check your version with `print(o3d.__version__)` and test with a known simple scene first.

---

## 6. Putting It All Together

### Full pipeline for one image row in the paper figure

```python
# 0. shared color map (CRITICAL: same for seg and mesh)
labels = sorted([l for l in np.unique(seg_map) if l > 0])
color_map = generate_color_map(len(labels))
# remap so actual labels map to colors
color_map = {label: color_map[i+1] for i, label in enumerate(labels)}

# 1. segmentation overlay
seg_img = render_segmentation_overlay(rgb, seg_map, color_map, alpha=0.55)

# 2. build mesh (Render B2: MoGe points + RANSAC planes)
mesh = build_plane_mesh(
    pts_map=moge_pts,           # MoGe V2 predicted point map
    seg_map=pred_seg,           # our v11 region growing output
    plane_params=ransac_params, # RANSAC on MoGe points per segment
    rgb=rgb,
    color_map=color_map,
    max_edge_len=0.1,
    project_to_plane=True,
    use_plane_colors=True       # set False for textured look
)

# 3. render mesh from rotated view
ext_rotated = rotated_extrinsic(original_extrinsic, angle_deg=25, axis='y')
mesh_img = render_mesh_offscreen(
    mesh, W, H,
    intrinsic_params=(fx, fy, cx, cy),
    extrinsic=ext_rotated,
    bg_color=[0, 0, 0, 1]
)

# 4. assemble the triplet
import cv2
triplet = np.concatenate([rgb, seg_img, mesh_img], axis=1)
cv2.imwrite("figure_row.png", cv2.cvtColor(triplet, cv2.COLOR_RGB2BGR))
```

### Checklist before generating final figures

- [ ] Color map is **identical** between segmentation overlay and mesh vertex colors
- [ ] Mesh uses **Render B2** (MoGe predicted depth, not GT depth)
- [ ] Points are **projected onto RANSAC planes** (not raw MoGe points)
- [ ] `max_edge_len` is tuned per scene type (indoor vs outdoor)
- [ ] Camera viewpoint reveals depth layering without occluding important planes
- [ ] Background is clean (black or white, consistent across all rows)
- [ ] Variable `t` is not shadowed in any loop (previously fixed bug)
- [ ] Open3D camera API matches your installed version

---

## 7. Common Issues and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| Colors differ between 2D and 3D | Two separate color map generations | Generate **one** `color_map` dict and pass to both functions |
| "Curtain" artifacts at depth edges | Triangles span depth discontinuities | Decrease `max_edge_len` |
| Mesh looks flat / no depth structure | Camera not rotated enough | Increase rotation angle to 30–40° |
| Holes in the mesh | `max_edge_len` too tight | Increase slightly, or fill small holes with `mesh.fill_holes()` |
| Mesh geometry doesn't match 2D masks | Using `project_to_plane` shifted vertices | Expected — projected points move slightly; the 2D mask is pixel-accurate, the 3D mesh is plane-accurate |
| Render is blank | Extrinsic convention mismatch (world-to-cam vs cam-to-world) | Try `np.linalg.inv(extrinsic)` |
| Over-segmented mesh (35+ tiny planes) | v11 merging not applied, or threshold too strict | Check coplanar merging step; relax angle/distance thresholds |

---

## 8. Adapting for GT Visualization (Render A)

For ground truth reference figures (e.g., showing GT annotation quality):

```python
# use GT depth to build point map
gt_pts_map = backproject_depth(gt_depth, K)  # standard K_inv @ [u,v,1]^T * depth

# use GT segmentation labels
mesh_gt = build_plane_mesh(
    pts_map=gt_pts_map,
    seg_map=gt_seg,
    plane_params=gt_plane_params,  # or skip projection, use raw GT points
    rgb=rgb,
    color_map=color_map,
    project_to_plane=False,        # GT points are already "on" the plane
    use_plane_colors=True
)
```

This produces the reference column if you want a side-by-side **GT vs Ours** comparison.
