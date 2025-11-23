#!/usr/bin/env python3
import sys
sys.path.append('/cluster/home/aoezkan/planeseg/3d_vision/gt_gen')

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from visualize_planes_v1 import *
from utils import *
from mesh_utils import *
from render import *


def compute_intrinsics_from_proj(M_proj, width, height):
    """Convert Hypersim projection matrix to Open3D intrinsics."""
    fx = M_proj[0, 0] * 0.5 * width
    fy = -M_proj[1, 1] * 0.5 * height  # Note: Y-axis flipped
    cx = M_proj[0, 2] * 0.5 * width + 0.5 * width
    cy = -M_proj[1, 2] * 0.5 * height + 0.5 * height
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,   0,  1]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id", type=str)
    args_cli = parser.parse_args()
    scene_id = args_cli.scene_id

    print(f"[INFO] Rendering Hypersim scene: {scene_id}")

    # === Paths ===
    root_dir = "/cluster/scratch/ayavuz/dataset/Hypersim"
    root_save_dir = "/cluster/scratch/aoezkan/dataset/Hypersim"
    mesh_path = f"{root_save_dir}/plane_ours_gt/{scene_id}/planes.ply"

    if not os.path.exists(mesh_path):
        mesh_path = f"{root_dir}/plane_ours_gt/{scene_id}/planes.ply"

    print(f"[INFO] Reading mesh: {mesh_path}")
    V, F, plane_id_face, _ = read_ply_faces_with_plane_ids(mesh_path)
    print(f"[INFO] V={V.shape[0]}  F={F.shape[0]}")

    # === Create mesh ===
    sem_mesh = o3d.geometry.TriangleMesh()
    sem_mesh.vertices = o3d.utility.Vector3dVector(V)
    sem_mesh.triangles = o3d.utility.Vector3iVector(F)
    sem_mesh.compute_vertex_normals()

    scene_dir = os.path.join(root_dir, scene_id)
    detail_dir = os.path.join(scene_dir, "_detail")

    # === Intrinsics ===
    meta_cam_file = os.path.join(root_dir, "metadata_camera_parameters.csv")
    df_meta = pd.read_csv(meta_cam_file, index_col="scene_name")
    df_scene = df_meta.loc[scene_id]
    width = int(df_scene["settings_output_img_width"])
    height = int(df_scene["settings_output_img_height"])
    M_proj = np.array([[df_scene[f"M_proj_{i}{j}"] for j in range(4)] for i in range(4)])
    K = compute_intrinsics_from_proj(M_proj, width, height)

    # === Find available cameras ===
    cam_names = sorted([d for d in os.listdir(detail_dir)
                        if d.startswith("cam_") and os.path.isdir(os.path.join(detail_dir, d))])
    print(f"[INFO] Found {len(cam_names)} cameras: {cam_names}")

    frame_skip = 1  # e.g., use 2 or 4 to subsample frames

    for cam_name in cam_names:
        print(f"\n[INFO] Processing {cam_name}...")
        camera_dir = os.path.join(detail_dir, cam_name)
        cam_pos_path = os.path.join(camera_dir, "camera_keyframe_positions.hdf5")
        cam_rot_path = os.path.join(camera_dir, "camera_keyframe_orientations.hdf5")

        if not os.path.exists(cam_pos_path) or not os.path.exists(cam_rot_path):
            print(f"[WARN] Missing pose files for {cam_name}, skipping.")
            continue

        # --- Load poses ---
        with h5py.File(cam_pos_path, "r") as f:
            cam_positions = f["dataset"][:]
        with h5py.File(cam_rot_path, "r") as f:
            cam_orientations = f["dataset"][:]

        total_frames = len(cam_positions)
        frame_ids, planes_list = [], []

        print(f"[INFO] Raycasting {total_frames} frames for {cam_name}...")
        for frame_id in tqdm(range(total_frames), desc=f"{cam_name}"):
            if frame_id % frame_skip != 0:
                continue

            R = cam_orientations[frame_id]
            T = cam_positions[frame_id]

            # --- Build camera-to-world ---
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T

            # --- Flip for Open3D convention (Y,Z axes) ---
            c2w = c2w @ np.diag([1, -1, -1, 1])

            # --- Raycast plane IDs per face ---
            semantic_img_face = raycast_semantic_face_labels(
                sem_mesh, plane_id_face, K, (width, height), c2w
            )

            # --- Map per-face to per-pixel IDs ---
            semantic_img = remap_semantic(semantic_img_face)
            semantic_img = np.flipud(semantic_img)  # OpenGL → image coordinates
            semantic_img = np.where(semantic_img < 0, 0, semantic_img)
            semantic_img = np.clip(semantic_img, 0, 65535).astype(np.uint16)

            frame_ids.append(f"{frame_id:04d}")
            planes_list.append(semantic_img)

        # === Save HDF5 for this camera ===
        h5_save_path = f"{root_save_dir}/plane_ours_gt/{scene_id}/rendered_planes_{cam_name}.h5"
        os.makedirs(os.path.dirname(h5_save_path), exist_ok=True)

        print(f"[SAVE] → {h5_save_path}")
        with h5py.File(h5_save_path, "w") as f:
            f.create_dataset("planes", data=np.stack(planes_list), compression="gzip")
            f.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))

        print(f"[DONE] Rendered {len(planes_list)} frames for {scene_id}/{cam_name}")

    print(f"\n[FINISHED] All cameras processed for scene {scene_id}")
