"""
Stage 2: Raycast PlaneRCNN planes.ply → rendered.h5

Reads planes.ply (binary PLY with per-face plane_id), raycasts to each
camera frame, and writes rendered.h5 with the standard label convention:
  0 = non-planar, 1+ = plane IDs (shifted by +1 from PLY's -1/0+ scheme).

Adapted from render_scene.py.
Supports YAML configuration via --config flag (see planercnn_default.yml).
"""

import argparse
import os
import json
import struct
import numpy as np
import open3d as o3d
import h5py
import yaml
from tqdm import tqdm

from planamono.shared.rendering.render import raycast_semantic_face_labels
from planamono.paths import scannetppv2_path


# Default paths (used when no config is provided)
DEFAULT_MESH_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset_mesh/scannetpp_planercnn"
DEFAULT_OUTPUT_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp_planercnn"


def read_ply_faces_fast(ply_path):
    """
    Fast binary PLY reader for our planes.ply format.
    Reads all face data in a single block instead of per-face loops.

    Returns:
        V: (N, 3) float32 vertices
        F: (M, 3) int32 face indices
        plane_id_face: (M,) int32 per-face plane IDs
        label_int_face: (M,) int32 per-face semantic labels
    """
    with open(ply_path, "rb") as f:
        # Parse header
        num_verts = None
        num_faces = None
        while True:
            line = f.readline().decode("ascii", errors="strict").rstrip("\r\n")
            if line.startswith("element vertex"):
                num_verts = int(line.split()[-1])
            elif line.startswith("element face"):
                num_faces = int(line.split()[-1])
            elif line.startswith("end_header"):
                break

        # Read vertices: N * 3 float32
        V = np.frombuffer(f.read(num_verts * 12), dtype="<f4").reshape(num_verts, 3)

        # Read faces: each face = 1B (count) + 3*4B (indices) + 4B (plane_id) + 4B (label_int) = 21B
        face_bytes = f.read(num_faces * 21)

    # Parse face data from raw bytes
    raw = np.frombuffer(face_bytes, dtype=np.uint8).reshape(num_faces, 21)

    # Indices at bytes 1-12 (3 int32)
    F = np.frombuffer(raw[:, 1:13].tobytes(), dtype="<i4").reshape(num_faces, 3)
    # plane_id at bytes 13-16
    plane_id_face = np.frombuffer(raw[:, 13:17].tobytes(), dtype="<i4").reshape(num_faces)
    # label_int at bytes 17-20
    label_int_face = np.frombuffer(raw[:, 17:21].tobytes(), dtype="<i4").reshape(num_faces)

    return V.copy(), F.copy(), plane_id_face.copy(), label_int_face.copy()


def main():
    parser = argparse.ArgumentParser(description="Raycast planes.ply → rendered.h5")
    parser.add_argument("scene_id", type=str, help="ScanNet++ scene ID")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: use hardcoded parameters)")
    parser.add_argument("--mesh_root", type=str, default=None,
                        help="Root for planes.ply files (overrides config)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Output root for rendered.h5 files (overrides config)")
    parser.add_argument("--frame_skip", type=int, default=None,
                        help="Render every N-th frame (overrides config)")
    args = parser.parse_args()

    # Resolve parameters: CLI > config > defaults
    if args.config is not None:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"[INFO] Using config: {args.config}")
        mesh_root = args.mesh_root or cfg.get("mesh_root", DEFAULT_MESH_ROOT)
        output_root = args.output_root or cfg.get("output_root", DEFAULT_OUTPUT_ROOT)
        frame_skip = args.frame_skip if args.frame_skip is not None else cfg.get("frame_skip", 25)
        W = cfg.get("target_width", 640)
        H = cfg.get("target_height", 480)
    else:
        mesh_root = args.mesh_root or DEFAULT_MESH_ROOT
        output_root = args.output_root or DEFAULT_OUTPUT_ROOT
        frame_skip = args.frame_skip if args.frame_skip is not None else 25
        W, H = 640, 480

    scene_id = args.scene_id

    # --- Load planes.ply ---
    mesh_path = os.path.join(mesh_root, scene_id, "planes.ply")
    if not os.path.exists(mesh_path):
        print(f"[ERR] planes.ply not found: {mesh_path}")
        print("      Run fit_planes.py first.")
        return

    print(f"[INFO] Loading {mesh_path}")
    V, F, plane_id_face, _ = read_ply_faces_fast(mesh_path)
    print(f"[INFO] {V.shape[0]} vertices, {F.shape[0]} faces, "
          f"{len(np.unique(plane_id_face[plane_id_face >= 0]))} planes")

    # Build Open3D mesh
    sem_mesh = o3d.geometry.TriangleMesh()
    sem_mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
    sem_mesh.triangles = o3d.utility.Vector3iVector(F)
    sem_mesh.compute_vertex_normals()

    # --- Load camera poses ---
    root_dir = os.path.join(scannetppv2_path, "data")
    iphone_dir = os.path.join(root_dir, scene_id, "iphone")
    pose_file = os.path.join(iphone_dir, "pose_intrinsic_imu.json")

    if not os.path.exists(pose_file):
        print(f"[ERR] Pose file not found: {pose_file}")
        return

    with open(pose_file, "r") as f:
        data = json.load(f)

    # --- Intrinsics (scaled from 1920x1440 → target resolution) ---
    first_key = next(iter(data))
    K = np.array(data[first_key]["intrinsic"])
    W_orig, H_orig = 1920, 1440
    scale_x = W / W_orig
    scale_y = H / H_orig
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[1, 2] *= scale_y

    # --- Raycast and collect ---
    print(f"[INFO] Raycasting {len(data)} frames (skip={frame_skip}) …")
    frame_ids = []
    planes_list = []

    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        if i % frame_skip != 0:
            continue

        c2w = np.array(frame_data["aligned_pose"])

        semantic_img = raycast_semantic_face_labels(
            sem_mesh, plane_id_face, K_scaled, (W, H), c2w
        )

        # Shift labels: -1 → 0 (non-planar), 0 → 1, 1 → 2, …
        semantic_img = np.where(semantic_img < 0, 0, semantic_img + 1)
        semantic_img = np.clip(semantic_img, 0, 65535).astype(np.uint16)

        frame_ids.append(frame_id)
        planes_list.append(semantic_img)

    # --- Save H5 ---
    if len(planes_list) == 0:
        print("[WARN] No frames rendered. Check frame_skip or pose file.")
        return

    out_dir = os.path.join(output_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "rendered.h5")

    print(f"[SAVE] Writing {len(planes_list)} frames to {h5_path}")
    planes_stack = np.stack(planes_list)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("planes", data=planes_stack, compression="gzip")
        f.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))

    print(f"[DONE] rendered.h5: shape={planes_stack.shape}, dtype={planes_stack.dtype}")
    print(f"  Unique labels: {np.unique(planes_stack)[:20]}{'…' if len(np.unique(planes_stack)) > 20 else ''}")
    print(f"  Saved to: {h5_path}")


if __name__ == "__main__":
    main()
