"""
Render (raycast) ScanNet++ plane-GT meshes to per-frame 2D label maps.

For one scene, loads the extracted plane mesh (planes.ply with per-face plane
ids from plane_extraction.py / scene_runner.py), raycasts it into every
frame_skip-th iPhone frame using the aligned poses, and writes one HDF5 per
scene in the format consumed by ScanNetPPPlaneDataset and the evaluation:

    <output_root>/<scene_id>/rendered.h5
        planes     (N, H, W) uint16   plane labels, 0 = non-planar
        frame_ids  (N,)      S<bytes> e.g. b'frame_000000'

Label convention: raycast miss / non-planar (-1) is shifted globally by +1 so
that 0 = non-planar and plane ids stay consistent across frames.

Usage:
    python render_scene.py <scene_id> [--input_root ...] [--plane_root ...]
                           [--output_root ...] [--frame_skip 25]
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

from pxwplanar.shared.rendering.mesh_io import read_ply_faces_with_plane_ids
from pxwplanar.shared.rendering.render import raycast_semantic_face_labels
from pxwplanar.paths import scannetppv2_path, scannetpp_plane_path, scannetpp_rend_plane_path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_id", type=str)
    ap.add_argument("--input_root", type=str, default=os.path.join(scannetppv2_path, "data"),
                    help="ScanNet++ data root (per-scene iphone/pose_intrinsic_imu.json).")
    ap.add_argument("--plane_root", type=str, default=scannetpp_plane_path,
                    help="Root with extracted plane meshes (<scene>/planes.ply).")
    ap.add_argument("--output_root", type=str, default=scannetpp_rend_plane_path,
                    help="Output root for <scene>/rendered.h5.")
    ap.add_argument("--frame_skip", type=int, default=25,
                    help="Render every Nth frame (default 25).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    return ap.parse_args()


def main():
    args = parse_args()
    scene_id = args.scene_id

    mesh_path = os.path.join(args.plane_root, scene_id, "planes.ply")
    print(f"[INFO] Rendering scene: {scene_id}")
    print(f"[INFO] Reading {mesh_path} ...")
    V, F, plane_id_face, _ = read_ply_faces_with_plane_ids(mesh_path)

    sem_mesh = o3d.geometry.TriangleMesh()
    sem_mesh.vertices = o3d.utility.Vector3dVector(V)
    sem_mesh.triangles = o3d.utility.Vector3iVector(F)
    sem_mesh.compute_vertex_normals()

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
    h5_path = os.path.join(out_dir, "rendered.h5")

    print("[INFO] Raycasting plane labels ...")
    frame_ids = []
    planes_list = []
    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        if i % max(1, args.frame_skip) != 0:
            continue

        c2w = np.array(frame_data["aligned_pose"])
        semantic_img = raycast_semantic_face_labels(sem_mesh, plane_id_face, K_scaled, (W, H), c2w)

        # Global +1 shift: raycast miss / non-planar (-1) -> 0, plane k -> k+1.
        # (A per-frame compaction or clamping -1 to 0 would either make plane ids
        # inconsistent across frames or collide plane 0 with non-planar.)
        semantic_img = np.where(semantic_img < 0, 0, semantic_img + 1)
        semantic_img = np.clip(semantic_img, 0, 65535).astype(np.uint16)

        frame_ids.append(frame_id)
        planes_list.append(semantic_img)

    if not planes_list:
        print("[WARN] No frames rendered (check frame_skip / pose file).")
        return

    print(f"[SAVE] Writing {len(planes_list)} frames to {h5_path}")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("planes", data=np.stack(planes_list), compression="gzip")
        f.create_dataset("frame_ids", data=np.array(frame_ids, dtype="S"))
    print(f"[DONE] Saved H5 for scene: {scene_id}")


if __name__ == "__main__":
    main()
