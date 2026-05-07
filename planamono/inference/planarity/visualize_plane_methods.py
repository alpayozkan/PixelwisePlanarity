"""
Visualize fitted planes from `compare_plane_param_methods.py` outputs.

Reads:
  <output_dir>/plane_labels.h5   ← stacked uint16 labels (per frame)
  <output_dir>/plane_params.h5   ← per (method, frame_id): plane_ids + (a,b,c,d)

Produces (per --frame_id, optionally for all frames in the stack):
  png/<frame_id>.png    2D panels: rendered_depth + |rendered − pred| residual
                        for each method, alongside the existing pred / gt panels
  meshes/<method>/<frame_id>.ply
                        3D triangle mesh built via the Render-B recipe
                        (planar depth from (a,b,c,d), backproject, triangulate)
                        — only when --save_meshes is set

Frame data needed to build the panels (depth_pred, normal_pred, planarity, K)
is read from the source moge_signals.h5; pass --signals_h5 or let the script
locate it via --signals_root + --scene_id.

Example
-------
python visualize_plane_methods.py \\
    --scene_id c50d2d1d42 \\
    --frame_id frame_000000 \\
    --save_meshes
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.paths import scannetpp_path, scannetpp_rend_plane_path
from planamono.shared.plane_fitting.visualize_planes import (
    render_planes_to_depth,
    planes_to_mesh,
    planes_to_point_cloud,
)

# Defaults match compare_plane_param_methods.py
DEFAULT_SIGNALS_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1"
DEFAULT_PARAMS_ROOT  = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1_methods"
DEFAULT_OUTPUT_ROOT  = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1_methods_vis"
DEFAULT_GT_ROOT      = scannetpp_rend_plane_path
DEFAULT_RGB_ROOT     = os.path.join(scannetpp_path, "data")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _load_signals_frame(signals_h5: str, frame_id: str):
    """Look up `frame_id` in moge_signals.h5 and return depth, normal, planarity, K."""
    with h5py.File(signals_h5, "r") as f:
        ids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
               for x in f["frame_ids"][:]]
        if frame_id not in ids:
            raise KeyError(f"{frame_id!r} not in {signals_h5}")
        i = ids.index(frame_id)
        depth = f["depth_metric"][i].astype(np.float32)
        normal = f["normal"][i].astype(np.float32)
        planarity = f["planarity"][i].astype(np.float32)
        K = f["intrinsics"][i].astype(np.float64)
    nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, np.clip(nrm, 1e-6, None), where=nrm > 1e-6)
    return depth, normal, planarity, K


def _load_labels(labels_h5: str, frame_id: str) -> np.ndarray:
    with h5py.File(labels_h5, "r") as f:
        ids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
               for x in f["frame_ids"][:]]
        if frame_id not in ids:
            raise KeyError(f"{frame_id!r} not in {labels_h5}")
        i = ids.index(frame_id)
        return f["plane_labels"][i].astype(np.int32)


def _load_plane_params(params_h5: str, method: str, frame_id: str
                       ) -> Dict[int, np.ndarray]:
    with h5py.File(params_h5, "r") as f:
        if method not in f:
            raise KeyError(f"method {method!r} not in {params_h5}; available: {list(f.keys())}")
        if frame_id not in f[method]:
            raise KeyError(f"{frame_id!r} not in {params_h5}/{method}")
        grp = f[method][frame_id]
        pids = grp["plane_ids"][:]
        params = grp["params"][:]
    return {int(pid): np.asarray(p, dtype=np.float64)
            for pid, p in zip(pids, params)}


def _load_gt_frame(rendered_h5: str, rendered_depth_h5: str, frame_id: str
                   ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Optional GT loader; returns (depth_gt, labels_gt) or (None, None)."""
    if not (os.path.exists(rendered_h5) and os.path.exists(rendered_depth_h5)):
        return None, None
    with h5py.File(rendered_h5, "r") as f:
        ids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
               for x in f["frame_ids"][:]]
        if frame_id not in ids:
            return None, None
        i = ids.index(frame_id)
        labels_gt = f["planes"][i].astype(np.int32)
    with h5py.File(rendered_depth_h5, "r") as f:
        depth_gt = f["depth"][i].astype(np.float32) / 1000.0
    return depth_gt, labels_gt


# --------------------------------------------------------------------------- #
# Headless 3D render (matplotlib Poly3DCollection / scatter — Open3D
# OffscreenRenderer requires EGL which is unavailable on the compute nodes)
# --------------------------------------------------------------------------- #

def _shared_bounds(geoms):
    pts_list = []
    for g in geoms:
        try:
            import open3d as o3d
        except Exception:
            o3d = None
        if o3d is not None and isinstance(g, o3d.geometry.TriangleMesh):
            pts_list.append(np.asarray(g.vertices))
        elif o3d is not None and isinstance(g, o3d.geometry.PointCloud):
            pts_list.append(np.asarray(g.points))
    if not pts_list:
        return None
    P = np.concatenate(pts_list, axis=0)
    if P.size == 0:
        return None
    center = (P.min(0) + P.max(0)) / 2
    half = float((P.max(0) - P.min(0)).max() / 2) * 1.05
    return center, half


def _render_3d_grid(
    geoms_by_method,
    methods,
    out_path,
    elev=-25.0,
    azim=-75.0,
    max_tris=30000,
    max_pts=30000,
    title=None,
    bg_color=(0.08, 0.08, 0.08),
):
    """Render an N-method grid of meshes-or-pcds via matplotlib 3D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import open3d as o3d

    n = len(methods)
    n_cols = 3 if n >= 3 else n
    n_rows = (n + n_cols - 1) // n_cols

    bounds = _shared_bounds([geoms_by_method[m] for m in methods if geoms_by_method.get(m) is not None])
    fig = plt.figure(figsize=(5.0 * n_cols, 5.0 * n_rows), facecolor=bg_color)

    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d", facecolor=bg_color)
        ax.set_title(m, color="white", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)

        geom = geoms_by_method.get(m)
        if geom is None:
            ax.text2D(0.5, 0.5, "(no geometry)", ha="center", va="center",
                      transform=ax.transAxes, color="lightgray", fontsize=9)
            ax.set_axis_off()
            continue

        if isinstance(geom, o3d.geometry.TriangleMesh):
            V = np.asarray(geom.vertices)
            F = np.asarray(geom.triangles)
            if len(F) == 0:
                ax.text2D(0.5, 0.5, "(empty mesh)", ha="center", va="center",
                          transform=ax.transAxes, color="lightgray", fontsize=9)
                ax.set_axis_off()
                continue
            C = np.asarray(geom.vertex_colors) if geom.has_vertex_colors() else None
            if len(F) > max_tris:
                idx = rng.choice(len(F), max_tris, replace=False)
                F = F[idx]
            polys = V[F]
            face_colors = (C[F].mean(axis=1) if C is not None
                           else np.full((len(F), 3), 0.6))
            ax.add_collection3d(Poly3DCollection(
                polys, facecolors=face_colors, edgecolors="none", linewidth=0))
        elif isinstance(geom, o3d.geometry.PointCloud):
            V = np.asarray(geom.points)
            C = np.asarray(geom.colors) if geom.has_colors() else None
            if len(V) == 0:
                ax.text2D(0.5, 0.5, "(empty pcd)", ha="center", va="center",
                          transform=ax.transAxes, color="lightgray", fontsize=9)
                ax.set_axis_off()
                continue
            if len(V) > max_pts:
                idx = rng.choice(len(V), max_pts, replace=False)
                V = V[idx]
                if C is not None:
                    C = C[idx]
            ax.scatter(V[:, 0], V[:, 1], V[:, 2],
                       c=C if C is not None else "lightblue",
                       s=0.6, depthshade=False, edgecolors="none")
        else:
            ax.text2D(0.5, 0.5, f"(unsupported: {type(geom).__name__})",
                      ha="center", va="center", transform=ax.transAxes,
                      color="lightgray", fontsize=9)
            ax.set_axis_off()
            continue

        if bounds is not None:
            center, half = bounds
            ax.set_xlim(center[0] - half, center[0] + half)
            ax.set_ylim(center[1] - half, center[1] + half)
            ax.set_zlim(center[2] - half, center[2] + half)

        ax.view_init(elev=elev, azim=azim)
        ax.invert_yaxis()  # camera image-Y is down
        ax.set_xlabel("X", color="lightgray", fontsize=8)
        ax.set_ylabel("Y", color="lightgray", fontsize=8)
        ax.set_zlabel("Z (depth)", color="lightgray", fontsize=8)
        ax.tick_params(colors="lightgray", labelsize=7)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("none")
        ax.grid(False)

    if title:
        fig.suptitle(title, color="white", fontsize=12)
    fig.savefig(out_path, dpi=120, facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2D panel
# --------------------------------------------------------------------------- #

def _render_method_grid(
    out_path: str,
    fid: str,
    rgb: np.ndarray,
    depth_pred: np.ndarray,
    normal_pred: np.ndarray,
    planarity: np.ndarray,
    labels_pred: np.ndarray,
    depth_gt: Optional[np.ndarray],
    labels_gt: Optional[np.ndarray],
    method_renders: Dict[str, Tuple[np.ndarray, np.ndarray]],
    methods: List[str],
    residual_max_m: float = 0.2,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(0)
    palette = rng.uniform(0.2, 1.0, size=(2048, 3))
    palette[0] = [0.05, 0.05, 0.05]
    cmap_lbl = ListedColormap(palette)

    H, W = depth_pred.shape

    n_method_rows = (len(methods) + 1) // 2  # 2 methods per row
    total_rows = 2 + n_method_rows
    fig_h = 4.5 * total_rows
    fig, axes = plt.subplots(total_rows, 4, figsize=(20, fig_h),
                             constrained_layout=True)

    def _show(ax, img, title, **kw):
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Row 0: rgb | depth_pred | normal_pred | planarity
    _show(axes[0, 0], rgb, "RGB")
    vmax_d = float(np.nanpercentile(depth_pred, 99))
    _show(axes[0, 1], depth_pred, "depth_pred (m)", cmap="viridis",
          vmin=0, vmax=vmax_d)
    n_vis = (np.clip(normal_pred, -1, 1) + 1) / 2
    _show(axes[0, 2], n_vis, "normal_pred")
    _show(axes[0, 3], planarity, "planarity", cmap="magma", vmin=0, vmax=1)

    # Row 1: labels_pred | depth_gt | labels_gt | metadata
    n_pred_ids = int(labels_pred.max())
    _show(axes[1, 0], labels_pred % len(palette),
          f"labels_pred ({n_pred_ids} ids)",
          cmap=cmap_lbl, vmin=0, vmax=len(palette) - 1)

    if depth_gt is not None:
        valid_gt = depth_gt > 0
        gt_vmax = float(np.nanpercentile(depth_gt[valid_gt], 99)) if valid_gt.any() else vmax_d
        _show(axes[1, 1], depth_gt, "depth_gt (m)", cmap="viridis", vmin=0, vmax=gt_vmax)
    else:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.5, 0.5, "depth_gt\n(unavailable)", ha="center", va="center",
                        transform=axes[1, 1].transAxes, fontsize=10)

    if labels_gt is not None:
        n_gt_ids = int(labels_gt.max())
        _show(axes[1, 2], labels_gt % len(palette),
              f"labels_gt ({n_gt_ids} ids)",
              cmap=cmap_lbl, vmin=0, vmax=len(palette) - 1)
    else:
        axes[1, 2].axis("off")
        axes[1, 2].text(0.5, 0.5, "labels_gt\n(unavailable)", ha="center", va="center",
                        transform=axes[1, 2].transAxes, fontsize=10)

    axes[1, 3].axis("off")
    axes[1, 3].text(0.02, 0.95, f"frame_id: {fid}",
                    transform=axes[1, 3].transAxes, fontsize=10, va="top")

    # Rows 2..N: per-method (rendered_depth | residual) pairs, 2 methods per row
    for i, m in enumerate(methods):
        r = 2 + i // 2
        c0 = (i % 2) * 2
        depth_m, _ = method_renders[m]
        plane_mask = depth_m > 0
        _show(axes[r, c0], depth_m,
              f"{m}\nrendered_depth (m)", cmap="viridis", vmin=0, vmax=vmax_d)
        residual = np.zeros_like(depth_m)
        residual[plane_mask] = np.abs(depth_m[plane_mask] - depth_pred[plane_mask])
        residual_vis = np.where(plane_mask, residual, np.nan)
        _show(axes[r, c0 + 1], residual_vis,
              f"{m}\n|rendered − pred| (clip {residual_max_m}m)",
              cmap="magma", vmin=0, vmax=residual_max_m)

    # Hide any unused method slots in the last row
    used = 2 + n_method_rows * 2
    total = 2 * 4 + n_method_rows * 4
    if used * 1 + (total_rows - 2) * 4 > 0:
        # Last row may have empty slot if odd #methods
        if len(methods) % 2 == 1:
            r = 2 + (len(methods) - 1) // 2
            axes[r, 2].axis("off")
            axes[r, 3].axis("off")

    fig.suptitle(f"visualize_plane_methods — {fid}", fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_id", required=True)
    p.add_argument("--frame_id", default=None,
                   help="Single frame to visualize. Default: every frame in plane_labels.h5.")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods to visualize; default = all groups in plane_params.h5.")
    p.add_argument("--save_meshes", action="store_true",
                   help="Also write meshes/<method>/<frame_id>.ply via planes_to_mesh.")
    p.add_argument("--mesh_pixel_stride", type=int, default=2,
                   help="Sub-grid stride for mesh triangulation (1=full res, 2=half).")
    p.add_argument("--mesh_min_pixels", type=int, default=200,
                   help="Skip planes smaller than this for mesh building.")
    p.add_argument("--save_pointclouds", action="store_true",
                   help="Also write pointclouds/<method>/<frame_id>.ply via planes_to_point_cloud.")
    p.add_argument("--pointcloud_pixel_stride", type=int, default=1,
                   help="Sub-grid stride for point cloud sampling (1=full res).")
    p.add_argument("--pointcloud_min_pixels", type=int, default=50,
                   help="Skip planes smaller than this for point-cloud building.")
    p.add_argument("--render_3d_png", action="store_true",
                   help="Render the 3D meshes / pointclouds to a PNG grid via matplotlib "
                        "(headless; no GL needed). Output: renders/{meshes,pointclouds}/<frame_id>.png")
    p.add_argument("--render_elev", type=float, default=-25.0,
                   help="Matplotlib elev angle for the 3D render (default -25).")
    p.add_argument("--render_azim", type=float, default=-75.0,
                   help="Matplotlib azim angle for the 3D render (default -75).")
    p.add_argument("--render_max_tris", type=int, default=30000,
                   help="Subsample triangles per panel before rendering.")
    p.add_argument("--render_max_pts", type=int, default=30000,
                   help="Subsample points per panel before rendering.")
    p.add_argument("--no_panels", action="store_true",
                   help="Skip the per-frame PNG (e.g. when you only want meshes).")
    p.add_argument("--residual_max_m", type=float, default=0.2,
                   help="Color-clip max for the |rendered − pred| residual panels.")
    p.add_argument("--signals_root", default=DEFAULT_SIGNALS_ROOT,
                   help="Where moge_signals.h5 lives (one dir per scene).")
    p.add_argument("--params_root", default=DEFAULT_PARAMS_ROOT,
                   help="Where plane_labels.h5 / plane_params.h5 live (the "
                        "compare_plane_param_methods.py output dir, one per scene).")
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT,
                   help="Where this script writes its outputs (png/, meshes/, "
                        "pointclouds/, renders/).")
    p.add_argument("--gt_root", default=DEFAULT_GT_ROOT)
    p.add_argument("--rgb_root", default=DEFAULT_RGB_ROOT)
    args = p.parse_args()

    params_dir = os.path.join(args.params_root, args.scene_id)
    out_dir    = os.path.join(args.output_root, args.scene_id)
    labels_h5  = os.path.join(params_dir, "plane_labels.h5")
    params_h5  = os.path.join(params_dir, "plane_params.h5")
    signals_h5 = os.path.join(args.signals_root, args.scene_id, "moge_signals.h5")

    for path in (labels_h5, params_h5, signals_h5):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    os.makedirs(out_dir, exist_ok=True)

    rendered_h5 = os.path.join(args.gt_root, args.scene_id, "rendered.h5")
    rendered_depth_h5 = os.path.join(args.gt_root, args.scene_id, "rendered_depth.h5")
    rgb_dir = os.path.join(args.rgb_root, args.scene_id, "iphone", "rgb")

    with h5py.File(labels_h5, "r") as f:
        all_fids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                    for x in f["frame_ids"][:]]

    fids_to_do: List[str] = ([args.frame_id] if args.frame_id is not None
                             else list(all_fids))
    for fid in fids_to_do:
        if fid not in all_fids:
            raise ValueError(f"{fid!r} not present in {labels_h5}")

    with h5py.File(params_h5, "r") as f:
        available_methods = list(f.keys())
    methods = args.methods if args.methods else available_methods
    for m in methods:
        if m not in available_methods:
            raise ValueError(f"method {m!r} not in {params_h5}; available: {available_methods}")

    if not args.no_panels:
        os.makedirs(os.path.join(out_dir, "png"), exist_ok=True)
    if args.save_meshes:
        for m in methods:
            os.makedirs(os.path.join(out_dir, "meshes", m), exist_ok=True)
    if args.save_pointclouds:
        for m in methods:
            os.makedirs(os.path.join(out_dir, "pointclouds", m), exist_ok=True)
    if args.render_3d_png:
        # If neither --save_meshes nor --save_pointclouds is set, default to
        # rendering both types in-memory (without writing PLYs).
        render_meshes_too = args.save_meshes or not args.save_pointclouds
        render_pcds_too = args.save_pointclouds or not args.save_meshes
        if render_meshes_too:
            os.makedirs(os.path.join(out_dir, "renders", "meshes"), exist_ok=True)
        if render_pcds_too:
            os.makedirs(os.path.join(out_dir, "renders", "pointclouds"), exist_ok=True)
    else:
        render_meshes_too = render_pcds_too = False

    print(f"[INFO] scene={args.scene_id}  frames={len(fids_to_do)}  methods={methods}")
    if args.save_meshes:
        print(f"[INFO] mesh stride={args.mesh_pixel_stride}, min_pixels={args.mesh_min_pixels}")
    if args.save_pointclouds:
        print(f"[INFO] pointcloud stride={args.pointcloud_pixel_stride}, "
              f"min_pixels={args.pointcloud_min_pixels}")
    if args.render_3d_png:
        print(f"[INFO] 3D render via matplotlib, view (elev={args.render_elev}, "
              f"azim={args.render_azim}), max_tris={args.render_max_tris}, "
              f"max_pts={args.render_max_pts}")

    n_panels = 0
    n_meshes = 0
    n_pointclouds = 0
    n_renders = 0
    for fid in fids_to_do:
        depth_pred, normal_pred, planarity, K = _load_signals_frame(signals_h5, fid)
        labels_pred = _load_labels(labels_h5, fid)
        H, W = depth_pred.shape

        method_renders: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        meshes_by_method: Dict[str, "o3d.geometry.TriangleMesh"] = {}
        pcds_by_method: Dict[str, "o3d.geometry.PointCloud"] = {}
        for m in methods:
            params = _load_plane_params(params_h5, m, fid)
            depth_m, normal_m = render_planes_to_depth(
                params, labels_pred, K, H, W)
            method_renders[m] = (depth_m, normal_m)

            need_mesh = args.save_meshes or render_meshes_too
            if need_mesh:
                mesh = planes_to_mesh(
                    params, labels_pred, K, H, W,
                    pixel_stride=args.mesh_pixel_stride,
                    min_pixels_per_plane=args.mesh_min_pixels,
                    skip_labels=(0,),
                )
                if args.save_meshes and len(mesh.vertices) > 0:
                    import open3d as o3d
                    ply_path = os.path.join(out_dir, "meshes", m, f"{fid}.ply")
                    o3d.io.write_triangle_mesh(ply_path, mesh, write_ascii=False)
                    n_meshes += 1
                meshes_by_method[m] = mesh

            need_pcd = args.save_pointclouds or render_pcds_too
            if need_pcd:
                pcd = planes_to_point_cloud(
                    params, labels_pred, K, H, W,
                    pixel_stride=args.pointcloud_pixel_stride,
                    min_pixels_per_plane=args.pointcloud_min_pixels,
                    skip_labels=(0,),
                )
                if args.save_pointclouds and len(pcd.points) > 0:
                    import open3d as o3d
                    ply_path = os.path.join(out_dir, "pointclouds", m, f"{fid}.ply")
                    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
                    n_pointclouds += 1
                pcds_by_method[m] = pcd

        if args.render_3d_png:
            if render_meshes_too and meshes_by_method:
                _render_3d_grid(
                    meshes_by_method, methods,
                    os.path.join(out_dir, "renders", "meshes", f"{fid}.png"),
                    elev=args.render_elev, azim=args.render_azim,
                    max_tris=args.render_max_tris, max_pts=args.render_max_pts,
                    title=f"meshes — {fid}",
                )
                n_renders += 1
            if render_pcds_too and pcds_by_method:
                _render_3d_grid(
                    pcds_by_method, methods,
                    os.path.join(out_dir, "renders", "pointclouds", f"{fid}.png"),
                    elev=args.render_elev, azim=args.render_azim,
                    max_tris=args.render_max_tris, max_pts=args.render_max_pts,
                    title=f"pointclouds — {fid}",
                )
                n_renders += 1

        if not args.no_panels:
            depth_gt, labels_gt = _load_gt_frame(rendered_h5, rendered_depth_h5, fid)
            rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (H, W):
                    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

            png_path = os.path.join(out_dir, "png", f"{fid}.png")
            _render_method_grid(
                png_path, fid, rgb, depth_pred, normal_pred, planarity,
                labels_pred, depth_gt, labels_gt,
                method_renders, methods,
                residual_max_m=args.residual_max_m,
            )
            n_panels += 1
            print(f"[{fid}] panel saved → {png_path}")

    print(f"\n[DONE]")
    if not args.no_panels:
        print(f"  png/*.png             {n_panels} files")
    if args.save_meshes:
        print(f"  meshes/<m>/*.ply      {n_meshes} files (across {len(methods)} methods)")
    if args.save_pointclouds:
        print(f"  pointclouds/<m>/*.ply {n_pointclouds} files (across {len(methods)} methods)")
    if args.render_3d_png:
        print(f"  renders/*.png         {n_renders} files (3D grid renders)")


if __name__ == "__main__":
    main()
