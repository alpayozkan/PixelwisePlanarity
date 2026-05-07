# Plane Mesh Visualization: Two Rendering Modes Explained

This document explains the two ways to render 3D plane meshes for your plane
segmentation paper, what each one actually shows, and which one should go where.

---

## Background: What Your Pipeline Produces

After running your full method on a single image you have:

```
Input image
    │
    ▼
MoGe V2
    ├── moge_depth   (H, W)       — predicted metric depth in metres
    ├── moge_normals (H, W, 3)    — predicted surface normals
    └── moge_points  (H, W, 3)    — 3D point map (depth × ray directions)
    │
    ▼
Planarity head + region growing
    └── pred_seg     (H, W) int   — instance label per pixel (0 = non-planar)
    │
    ▼
RANSAC on MoGe point cloud
    └── results      {pid → {normal: [nx,ny,nz], offset: d}}
    │
    ▼
(Evaluation only) compare against GT
    ├── gt_depth     (H, W)       — sensor depth from ScanNet++
    └── gt_seg       (H, W) int   — ground-truth plane labels
```

The key point: **GT depth and GT segmentation are never used to generate your
method's output.** They are only used to compute evaluation metrics.

---

## Render A — GT Depth Mesh

### How it is built

```
For every pixel (u, v) where pred_seg[u,v] > 0:
    depth[u,v]  ←  gt_depth[u,v]         ← sensor measurement
    point_3d    ←  backproject(depth, K)
    colour      ←  colormap[pred_seg[u,v]]

Triangulate the coloured point cloud → mesh
```

### What it shows

The **real, measured 3D geometry of the scene**, coloured by your predicted
plane labels.

- The shape of every surface is exactly what the ScanNet++ sensor measured.
- The only thing being visualised is whether your **segmentation boundaries**
  make sense on the real surface.
- Plane geometry quality (are `n` and `d` correct?) is **not visible** here —
  a totally wrong normal would still look fine because the depth is from GT.

### What it is good for

- Debugging your region growing — do the boundaries land in the right places?
- Checking whether non-planar pixels (label 0, shown in grey) are correctly
  excluded.
- A sanity check that RANSAC inlier filtering is not removing too many points.

### What it is NOT good for

- Showing the geometric quality of your plane predictions.
- Paper figures that make a claim about 3D plane reconstruction accuracy.
- Fair comparison with methods like ZeroPlane, PlaneRCNN, PlaneTR — those
  papers do not use GT depth in their mesh renders.

### Why your current notebook does this

`backproject_v2(depth, K, c2w, pred_seg)` receives `sample["depth"]` which is
GT depth from the dataset. RANSAC then fits `(n, d)` on those GT-depth points.
The resulting plane parameters are partially GT-informed, not purely predicted.

---

## Render B — Planar Depth Mesh (Paper-Equivalent)

### How it is built

```
Step 1 — get plane parameters from YOUR method:
    (n, d) per segment  ←  RANSAC on MoGe point cloud   ← moge_depth only

Step 2 — compute planar depth for each pixel from those parameters:
    For pixel (u, v) belonging to plane pid:

        ray = K⁻¹ · [u, v, 1]ᵀ          (un-normalised ray direction)

        DM[u,v] = d / (nᵀ · ray)         (plane equation solved for depth)

    Pixels with label 0 (non-planar) → DM = 0 (no mesh there)

Step 3 — backproject planar depth → 3D:
    point_3d = DM[u,v] · ray             (in camera space)
    point_3d = R · point_3d + t          (to world space via c2w)

Step 4 — triangulate and render
```

The formula `DM = d / (nᵀ K⁻¹ q)` comes directly from the plane equation
`nᵀ x = d` where `x = DM · K⁻¹ q` is the 3D point.

### What it shows

The **3D geometry that your method actually claims** — the flat planes your
model says are there, at the positions and orientations it predicts.

- If your `(n, d)` are accurate → every plane is a perfectly flat polygon at
  the correct depth and tilt. The scene looks like a clean geometric model.
- If your normal `n` is wrong → the plane is tilted at the wrong angle in 3D,
  even if the 2D segmentation looks fine.
- If your offset `d` is wrong → the plane floats at the wrong depth, even if
  the normal direction is correct.
- Non-planar pixels contribute nothing (no mesh) — holes in the mesh are
  expected and correct.

### What it is good for

- **Paper teaser figures** (equivalent to ZeroPlane Fig 1, Fig 3).
- Showing that your method recovers correct 3D plane geometry, not just 2D
  segmentation.
- Fair comparison with ZeroPlane, PlaneTR, PlaneRCNN — this is exactly what
  those papers render.

### What it is NOT good for

- Debugging segmentation quality — a wrong boundary is hard to see when every
  plane is geometrically perfect.
- Frames where RANSAC fails on many segments — those will just be missing from
  the mesh.

---

## The Critical Dependency: Where Does `(n, d)` Come From?

This is what you caught and it is the most important question.

### Option B1 — RANSAC on GT depth (what the notebook currently does, wrong for paper)

```python
pts_world, labels, _ = backproject_v2(gt_depth, K, c2w, pred_seg)
results, _ = fit_planes_per_label_v1(pts_world, labels, ...)
```

The RANSAC inliers and the fitted plane come from GT geometry. The resulting
`(n, d)` is partially ground-truth-informed. This is appropriate for
**evaluation** (you want an accurate geometric reference for inlier ratio
computation) but **not for paper mesh figures** — you would be visualising
geometry that partially comes from the sensor, not your model.

### Option B2 — RANSAC on MoGe depth (correct for paper figures)

```python
pts_world, labels, _ = backproject_v2(moge_depth, K, c2w, pred_seg)
results, _ = fit_planes_per_label_v1(pts_world, labels, ...)
```

RANSAC runs on MoGe's predicted point cloud. The resulting `(n, d)` comes
entirely from your method's outputs. The mesh shows purely what your model
predicted — no GT information involved at any stage.

**This is the correct input for Render B figures in your paper.**

### Where to get `moge_depth`

Your H5 files store predicted segmentation masks. MoGe depth is likely stored
separately. There are three scenarios:

**Scenario 1 — MoGe depth is in the H5 file:**
```python
with h5py.File(h5_path, 'r') as f:
    moge_depth = f[scene_id][frame_idx]['depth'][:]   # check your H5 schema
```

**Scenario 2 — MoGe depth is in a separate inference folder:**
```python
# e.g. /cluster/scratch/aoezkan/planeseg/scannetpp/moge_depth/{scene_id}/{frame}.npy
moge_depth = np.load(moge_depth_path)
```

**Scenario 3 — Re-run MoGe on the fly (slow but always correct):**
```python
import moge
model = moge.MoGeModel.from_pretrained("Ruicheng/moge-vitl")
output = model.infer(rgb_tensor)
moge_depth   = output['depth'].numpy()    # (H, W)
moge_normals = output['normal'].numpy()   # (H, W, 3)
moge_points  = output['points'].numpy()   # (H, W, 3)
```

---

## Side-by-Side Comparison

| Property | Render A (GT depth) | Render B (planar depth, B2) |
|---|---|---|
| Depth source | ScanNet++ sensor | MoGe predicted depth |
| Plane params `(n,d)` source | RANSAC on GT points | RANSAC on MoGe points |
| Surface shape | Noisy / bumpy (real sensor) | Perfectly flat per plane |
| Wrong normal visible? | No | Yes — plane visibly tilted |
| Wrong offset visible? | No | Yes — plane at wrong depth |
| Non-planar pixels | Shown with GT geometry | Missing from mesh (correct) |
| Use for paper teaser | No | Yes |
| Use for evaluation | Yes (GT reference) | No |
| Use for debugging seg | Yes | Partial |
| What ZeroPlane renders | No | Yes |

---

## Recommended Figure Structure for Your Paper

### Figure 1 — Teaser (qualitative, multiple scenes)

Use **Render B2** throughout. Show diverse scenes from your test set.

```
[ RGB ] [ Pred segmentation ] [ Planar mesh (B2) ]
[ RGB ] [ Pred segmentation ] [ Planar mesh (B2) ]
...
```

Caption: *"Our method recovers geometrically accurate 3D planes from a single
image across diverse indoor scenes."*

### Figure 2 — Qualitative comparison vs baselines

For a fair comparison, every method should use its own predicted depth (or GT
depth if it does not predict depth). Do not mix.

```
[ RGB ] [ GT seg ] [ Ours B2 ] [ ZeroPlane B ] [ PlaneTR B ]
```

### Figure 3 — Ablation / failure cases

Render A (GT depth) is useful here to isolate segmentation errors from geometry
errors. If you show Render A and Render B side-by-side for a failure case, it
helps the reader understand whether the failure is in segmentation or in the
plane parameter estimation.

---

## Concrete Code Changes Needed

In `load_scene_data()`, add a `depth_source` parameter:

```python
def load_scene_data(sample, method_key, depth_source="gt"):
    """
    depth_source:
        "gt"   — use sensor depth (for evaluation and Render A)
        "moge" — use MoGe predicted depth (for Render B paper figures)
    """
    gt_depth   = sample["depth"].numpy().squeeze()
    moge_depth = load_moge_depth(sample["scene_id"], sample["frame_idx"])

    fit_depth = moge_depth if depth_source == "moge" else gt_depth

    pts_world, labels, valid_idx = backproject_v2(fit_depth, K, c2w, pred_seg)
    results, plane_df = fit_planes_per_label_v1(pts_world, labels, ...)

    return dict(
        ...
        depth_gt=gt_depth,
        depth_moge=moge_depth,
        depth_used_for_fit=fit_depth,
        results=results,   # (n,d) fitted on fit_depth
    )
```

Then in `render_planar_mesh()`:
```python
def render_planar_mesh(scene_data, use_planar_depth=True):
    if use_planar_depth:
        # Render B: plane equation depth — shows predicted geometry
        depth = compute_planar_depth(pred_seg, results, K)
    else:
        # Render A: whichever depth was used for fitting
        depth = scene_data["depth_used_for_fit"]
    ...
```

For paper figures always call:
```python
sd = load_scene_data(sample, method_key, depth_source="moge")
img = render_planar_mesh(sd, use_planar_depth=True)   # pure Render B2
```

---

## Summary in One Sentence Each

**Render A (GT depth mesh):** colours the real, sensor-measured 3D scene by
your predicted labels — useful for debugging segmentation, not for paper figures.

**Render B1 (planar depth, GT-RANSAC):** flat planes at positions derived from
a mix of your segmentation and GT depth — misleading for paper figures.

**Render B2 (planar depth, MoGe-RANSAC):** flat planes at positions and
orientations predicted entirely by your method — this is your method's actual
3D output and the correct thing to put in your paper.
