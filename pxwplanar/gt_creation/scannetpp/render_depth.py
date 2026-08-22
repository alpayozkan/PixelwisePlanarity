"""
Render ScanNet++ GT depth maps for evaluation.

For one scene, loads the full reconstructed mesh (mesh_aligned_0.05.ply) and
renders Z-depth into every frame_skip-th iPhone frame using the aligned poses,
writing one HDF5 per scene next to the plane labels from render_scene.py:

    <output_root>/<scene_id>/rendered_depth.h5
        depth      (N, H, W) uint16   Z-depth in millimeters, 0 = no surface
        frame_ids  (N,)      S<bytes> e.g. b'frame_000000'

This is the GT depth consumed by ScanNetPPPlaneDataset / the 3D metrics in
evaluate_all_baselines.py (which read depth[idx] / 1000.0). Frame selection
matches render_scene.py (same pose file iteration and frame_skip), and the
depth is raycast with the same camera model as the plane labels
(raycast_semantic_face_labels), so the two H5s stay pixel- and index-aligned.
CPU-only — no GL context required.

Usage:
    python render_depth.py <scene_id> [--input_root ...] [--output_root ...]
                           [--frame_skip 25]
"""
import os
import sys
import json
import argparse

import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pxwplanar.shared.rendering.render import raycast_depth
from pxwplanar.paths import scannetppv2_path, scannetpp_rend_plane_path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_id", type=str)
    ap.add_argument("--input_root", type=str, default=os.path.join(scannetppv2_path, "data"),
                    help="ScanNet++ data root (per-scene scans/mesh_aligned_0.05.ply "
                         "and iphone/pose_intrinsic_imu.json).")
    ap.add_argument("--output_root", type=str, default=scannetpp_rend_plane_path,
                    help="Output root for <scene>/rendered_depth.h5.")
    ap.add_argument("--frame_skip", type=int, default=25,
                    help="Render every Nth frame (default 25, matching render_scene.py).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    return ap.parse_args()


def main():
    args = parse_args()
    scene_id = args.scene_id

    mesh_path = os.path.join(args.input_root, scene_id, "scans", "mesh_aligned_0.05.ply")
    print(f"[INFO] Rendering depth for scene: {scene_id}")
    print(f"[INFO] Reading {mesh_path} ...")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    print(f"[INFO] Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    pose_file = os.path.join(args.input_root, scene_id, "iphone", "pose_intrinsic_imu.json")
    with open(pose_file, "r") as f:
        data = json.load(f)

    # Intrinsics scaled from the iPhone native resolution to the render size.
    first_key = next(iter(data))
    K = np.array(data[first_key]["intrinsic"])
    W_orig, H_orig = 1920, 1440
    W, H = args.width, args.height
    K_scaled = K.copy()
    K_scaled[0, 0] *= W / W_orig   # fx
    K_scaled[0, 2] *= W / W_orig   # cx
    K_scaled[1, 1] *= H / H_orig   # fy
    K_scaled[1, 2] *= H / H_orig   # cy

    out_dir = os.path.join(args.output_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "rendered_depth.h5")

    print("[INFO] Rendering depth ...")
    frame_ids = []
    depth_list = []
    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        if i % max(1, args.frame_skip) != 0:
            continue

        c2w = np.array(frame_data["aligned_pose"])
        depth = raycast_depth(mesh, K_scaled, (W, H), c2w)

        # uint16 millimeters (the loaders read depth / 1000.0)
        depth_mm = (depth * 1000.0).astype(np.uint16)

        frame_ids.append(frame_id)
        depth_list.append(depth_mm)

    if not depth_list:
        print("[WARN] No frames rendered (check frame_skip / pose file).")
        return

    print(f"[SAVE] Writing {len(depth_list)} frames to {h5_path}")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("depth", data=np.stack(depth_list), compression="gzip")
        f.create_dataset("frame_ids", data=np.array(frame_ids, dtype="S"))
    print(f"[DONE] Saved depth H5 for scene: {scene_id}")


if __name__ == "__main__":
    main()
