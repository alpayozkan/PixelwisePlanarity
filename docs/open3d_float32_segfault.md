# Open3D 0.17.0: Float32 Segfault with Vector3dVector

## Problem

Passing `float32` numpy arrays to `o3d.utility.Vector3dVector` causes a **silent segfault** (exit code 139) in Open3D 0.17.0. No Python exception is raised — the process (or Jupyter kernel) simply dies.

```python
import numpy as np
import open3d as o3d

V = np.random.rand(1000, 3).astype(np.float32)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(V)  # SEGFAULT
```

## Root Cause

`Vector3dVector` internally expects `float64` (`double`) arrays. When given `float32`, it reinterprets the memory incorrectly (reads 8 bytes per value from 4-byte storage), causing an out-of-bounds read and segfault.

Open3D does not validate the dtype or raise a `TypeError` — it crashes silently.

## Where This Hits Us

`read_ply_faces_with_plane_ids()` in `planamono/shared/rendering/mesh_io.py` returns vertices as `float32` (matching the PLY binary format). When these are passed directly to Open3D:

```python
V, F, plane_id_face, _ = read_ply_faces_with_plane_ids(mesh_path)
# V.dtype == float32

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(V)  # SEGFAULT
```

This crashes the Jupyter kernel or Python process with no traceback.

## Solution

Cast to `float64` before passing to Open3D:

```python
mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
```

## Affected Functions

Any code that creates Open3D meshes from our PLY reader:

| Function | File | Notes |
|----------|------|-------|
| `read_ply_faces_with_plane_ids()` | `shared/rendering/mesh_io.py` | Returns `float32` vertices |
| `read_ply_with_labels()` | `shared/rendering/mesh_io.py` | Same issue |
| Rendering notebooks | `exploration/hypersim/` | Build meshes for raycasting |
| `rendering.py` | `gt_creation/hypersim/rendering.py` | GT rendering pipeline |

Note: `Vector3iVector` (for face indices, `int32`) does **not** have this issue.

## Environment

- Open3D 0.17.0
- Python 3.10
- NumPy 1.x
- Linux (RHEL/Ubuntu)
