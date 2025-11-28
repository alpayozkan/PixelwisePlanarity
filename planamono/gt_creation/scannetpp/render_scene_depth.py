import os
import json
import argparse
import numpy as np
import open3d as o3d
import h5py
from tqdm import tqdm
import imageio

from plyfile import PlyData

from planamono.shared.utils.utils import *
from planamono.shared.parsers.parse_scannetpp import *
from planamono.shared.rendering.mesh_utils import *
from planamono.shared.rendering.render import *
from planamono.paths import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id", type=str)
    args_cli = parser.parse_args()
    scene_id = args_cli.scene_id

    # --- Path to the ScanNet++ dataset ---
    # main_dir = '/cluster/project/cvg/Shared_datasets/scannetpp_v2'
    main_dir = scannetppv2_path
    root_dir = f"{main_dir}/data"
    iphone_dir = os.path.join(root_dir, scene_id, "iphone")

    # --- Pose and intrinsics ---
    pose_file = os.path.join(iphone_dir, "pose_intrinsic_imu.json")
    with open(pose_file, "r") as f:
        data = json.load(f)

    # --- Load mesh ---
    print("[DEBUG] reading mesh...")
    mesh_path = os.path.join(root_dir, scene_id, "scans", "mesh_aligned_0.05.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    print(f"[INFO] Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    print("[DEBUG] mesh read success...")

    # --- Optional: semantic mesh ---
    print("[DEBUG] reading semantic mesh...")
    sem_path = os.path.join(root_dir, scene_id, "scans", "mesh_aligned_0.05_semantic.ply")
    sem_mesh = o3d.io.read_triangle_mesh(sem_path)
    sem_mesh.compute_vertex_normals()
    sem_labels = np.asarray(sem_mesh.vertex_colors)
    print("[DEBUG] semantic mesh read success...")

    # --- Optional: segments ---
    seg_path = os.path.join(root_dir, scene_id, "scans", "segments.json")
    with open(seg_path) as f:
        seg_json = json.load(f)
        segmentation = seg_json["segIndices"]

    # --- Metadata for semantic classes ---
    semantic_classes_path = os.path.join(main_dir, 'metadata', 'semantic_classes.txt')
    id_to_name = load_semantic_id_to_name_list(semantic_classes_path)
    print("[DEBUG] semantic id to name read success...")

    # --- Intrinsics ---
    first_key = next(iter(data))
    K = np.array(data[first_key]["intrinsic"])
    W_orig, H_orig = 1920, 1440
    W, H = 640, 480  # target resolution

    scale_x = W / W_orig
    scale_y = H / H_orig
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy

    # --- Output directory (scene-specific) ---
    # rgb_h5_dir = f'/cluster/scratch/aoezkan/dataset/scannetpp/rgb_gt_rendered/{scene_id}'
    # depth_h5_dir = f'/cluster/scratch/aoezkan/dataset/scannetpp/depth_gt_rendered/{scene_id}'
    depth_h5_dir = f'{scannetpp_rend_plane_path}/{scene_id}'
    # os.makedirs(rgb_h5_dir, exist_ok=True)
    os.makedirs(depth_h5_dir, exist_ok=True)

    # rgb_h5_path = os.path.join(rgb_h5_dir, "rendered_rgb.h5")
    depth_h5_path = os.path.join(depth_h5_dir, "rendered_depth.h5")

    # --- Collect rendered frames ---
    frame_ids = []
    # rgb_list = []
    depth_list = []

    print("[DEBUG] starting rendering...")
    frame_skip = 25
    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        if i % frame_skip != 0:
            continue

        c2w = np.array(frame_data["aligned_pose"])
        rgb, depth = render_rgb_depth(mesh, K_scaled, (W, H), c2w)

        # Convert depth to uint16 (millimeters)
        depth_mm = (depth * 1000.0).astype(np.uint16)

        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        frame_ids.append(frame_id)
        # rgb_list.append(rgb)
        depth_list.append(depth_mm)

    # --- Save all frames to .h5 ---
    # print(f"[SAVE] RGB → {rgb_h5_path}")
    print(f"[SAVE] DEPTH → {depth_h5_path}")

    with h5py.File(depth_h5_path, "w") as f_depth:
    # with h5py.File(rgb_h5_path, "w") as f_rgb, h5py.File(depth_h5_path, "w") as f_depth:
        # f_rgb.create_dataset("rgb", data=np.stack(rgb_list), compression="gzip")
        # f_rgb.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))

        f_depth.create_dataset("depth", data=np.stack(depth_list), compression="gzip")
        f_depth.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))
