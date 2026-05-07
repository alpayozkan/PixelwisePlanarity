"""
Public helpers for visualizing planes from (a, b, c, d) plane parameters.

Two functions:
    render_planes_to_depth(plane_params, labels, K, H, W) -> (depth, normal)
        Render fitted-plane depth and normal as per-pixel maps. Plane equation:
        Z = -d / (a·xn + b·yn + c), with xn=(u-cx)/fx, yn=(v-cy)/fy.

    planes_to_mesh(plane_params, labels, K, H, W, ...) -> o3d.geometry.TriangleMesh
        Build a single triangle mesh containing all fitted planes (Render B from
        docs/plane_mesh_visualization_guide.md). Vertex-colored by plane id.

Compatible with the per-frame plane_params dicts from `compute_plane_params`
and the per-frame plane_params.h5 written by compare_plane_param_methods.py.
"""

from typing import Dict, Optional, Union

import numpy as np

# Open3D is only required for planes_to_mesh; render_planes_to_depth is pure numpy.
try:
    import open3d as o3d
    _O3D_AVAILABLE = True
except Exception:
    _O3D_AVAILABLE = False


__all__ = ["render_planes_to_depth", "planes_to_mesh"]


# --------------------------------------------------------------------------- #
# 1) Render plane params to per-pixel depth + normal maps
# --------------------------------------------------------------------------- #

def render_planes_to_depth(
    plane_params: Dict[int, np.ndarray],
    labels: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
):
    """For each (pid → [a,b,c,d]) writes Z = −d / (a·xn + b·yn + c) into
    depth[mask] and (a, b, c) into normal[mask].

    Pixels outside any fitted-plane mask are zero in both outputs. Pixels where
    the plane is parallel to the ray (denominator ≈ 0) or the resulting Z is
    non-positive / non-finite are also zero in depth (the corresponding normal
    is still written, since orientation is well-defined).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    xn = (us.astype(np.float64) - cx) / fx
    yn = (vs.astype(np.float64) - cy) / fy

    depth = np.zeros((H, W), dtype=np.float32)
    normal = np.zeros((H, W, 3), dtype=np.float32)

    for pid, p in plane_params.items():
        a, b, c, d = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        mask = (labels == pid)
        if not mask.any():
            continue
        denom = a * xn + b * yn + c
        valid = np.abs(denom) > 1e-9
        z = np.zeros_like(denom)
        z[valid] = -d / denom[valid]
        ok = mask & (z > 0) & np.isfinite(z)
        depth[ok] = z[ok].astype(np.float32)
        normal[mask] = (a, b, c)

    return depth, normal


# --------------------------------------------------------------------------- #
# 2) Build a TriangleMesh from plane params (Render B recipe)
# --------------------------------------------------------------------------- #

def _color_for_plane_id(pid: int) -> np.ndarray:
    """Deterministic per-id RGB in [0, 1]. Reproducible across runs."""
    rng = np.random.default_rng(int(pid))
    return rng.uniform(0.2, 1.0, size=3)


def planes_to_mesh(
    plane_params: Dict[int, np.ndarray],
    labels: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
    c2w: Optional[np.ndarray] = None,
    skip_labels=(0,),
    min_pixels_per_plane: int = 200,
    pixel_stride: int = 2,
    color_by: Union[str, np.ndarray] = "plane_id",
):
    """Build one combined TriangleMesh containing every fitted plane.

    Per plane id `pid` with parameters (a, b, c, d):
      1. Compute Z = −d / (a·xn + b·yn + c) on the mask pixels.
      2. Backproject to 3D in camera frame (or world if `c2w` is given).
      3. Triangulate via mask-grid quads at `pixel_stride` resolution.
      4. Vertex colors per plane id (deterministic palette) or sampled from
         a caller-supplied (H, W, 3) RGB array in [0, 1].

    Args:
        plane_params: {pid: (4,) array [a, b, c, d]}, ||(a,b,c)||=1.
        labels:       (H, W) int label map (typically plane_labels.h5 row).
        K:            (3, 3) camera intrinsics.
        H, W:         image dimensions (must match `labels.shape`).
        c2w:          optional (4, 4) camera-to-world; if None mesh is in
                      camera frame.
        skip_labels:  plane ids to skip (default (0,) = background).
        min_pixels_per_plane: skip planes whose mask is smaller than this.
        pixel_stride: subsample the mask grid (1 = every pixel; 2 = half-res).
        color_by:     "plane_id" (per-id palette) or (H, W, 3) RGB array
                      in [0, 1] sampled at sub-grid positions.

    Returns:
        open3d.geometry.TriangleMesh, possibly empty.

    Raises:
        RuntimeError if Open3D is not importable.
    """
    if not _O3D_AVAILABLE:
        raise RuntimeError("Open3D is required for planes_to_mesh; install via `pip install open3d`")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    skip = {int(x) for x in skip_labels}

    all_V: list = []
    all_F: list = []
    all_C: list = []
    vert_offset = 0

    use_rgb_colors = isinstance(color_by, np.ndarray)
    if use_rgb_colors:
        rgb_array = np.asarray(color_by, dtype=np.float64)
        if rgb_array.shape != (H, W, 3):
            raise ValueError(f"color_by RGB array must have shape ({H}, {W}, 3); got {rgb_array.shape}")
    elif color_by != "plane_id":
        raise ValueError(f"color_by must be 'plane_id' or an (H,W,3) array; got {color_by!r}")

    # Sub-grid coordinates (shared across planes for this frame)
    ys = np.arange(0, H, pixel_stride)
    xs = np.arange(0, W, pixel_stride)
    u_grid, v_grid = np.meshgrid(xs, ys)
    Hs, Ws = u_grid.shape
    xn_sub = (u_grid - cx) / fx
    yn_sub = (v_grid - cy) / fy

    if c2w is not None:
        c2w = np.asarray(c2w, dtype=np.float64)

    for pid, p in plane_params.items():
        if int(pid) in skip:
            continue
        mask = (labels == pid)
        if int(mask.sum()) < min_pixels_per_plane:
            continue

        a, b, c, d = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        sub_mask = mask[v_grid, u_grid]  # (Hs, Ws)
        if not sub_mask.any():
            continue

        denom = a * xn_sub + b * yn_sub + c
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = -d / denom
        bad = (np.abs(denom) < 1e-9) | (Z <= 0) | ~np.isfinite(Z)

        X = xn_sub * Z
        Y = yn_sub * Z

        if c2w is not None:
            pts_cam = np.stack([X, Y, Z], axis=-1)
            pts_h = np.concatenate([pts_cam, np.ones((Hs, Ws, 1))], axis=-1)
            pts_world = pts_h @ c2w.T
            X = pts_world[..., 0]
            Y = pts_world[..., 1]
            Z = pts_world[..., 2]

        valid = sub_mask & ~bad
        # Quad at (i, j) needs (i,j), (i,j+1), (i+1,j), (i+1,j+1) all valid.
        q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
        ii, jj = np.nonzero(q)
        if ii.size == 0:
            continue

        # Emit all sub-grid vertices; isolated ones cleaned at the end.
        V = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        n_verts = V.shape[0]

        v00 = ii * Ws + jj
        v01 = ii * Ws + (jj + 1)
        v10 = (ii + 1) * Ws + jj
        v11 = (ii + 1) * Ws + (jj + 1)
        T1 = np.stack([v00, v10, v01], axis=1)
        T2 = np.stack([v01, v10, v11], axis=1)
        F = np.concatenate([T1, T2], axis=0)

        if use_rgb_colors:
            C = rgb_array[v_grid, u_grid].reshape(-1, 3).astype(np.float64)
        else:
            color = _color_for_plane_id(int(pid))
            C = np.broadcast_to(color, (n_verts, 3)).copy()

        all_V.append(V)
        all_F.append(F + vert_offset)
        all_C.append(C)
        vert_offset += n_verts

    mesh = o3d.geometry.TriangleMesh()
    if all_V:
        # Float64 cast guards against the silent Open3D segfault documented in
        # docs/open3d_float32_segfault.md.
        V = np.concatenate(all_V, axis=0).astype(np.float64)
        F = np.concatenate(all_F, axis=0).astype(np.int32)
        C = np.concatenate(all_C, axis=0).astype(np.float64)
        mesh.vertices = o3d.utility.Vector3dVector(V)
        mesh.triangles = o3d.utility.Vector3iVector(F)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0.0, 1.0))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
    return mesh
