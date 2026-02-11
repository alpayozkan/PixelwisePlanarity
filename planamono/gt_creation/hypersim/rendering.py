#!/usr/bin/env python3
"""
Hypersim Plane Rendering Script

Renders extracted planes to HDF5 files for each camera in a Hypersim scene.
Uses raycasting to project 3D plane meshes to 2D images.

Usage:
    python rendering.py ai_001_001 --input_root /data/hypersim --plane_root /data/planes --output_root /data/output
"""
import sys
from pathlib import Path

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from planamono.shared.rendering.mesh_io import read_ply_faces_with_plane_ids


def raycast_with_cam_from_uv(sem_mesh, face_labels, M_cam_from_uv, img_res, c2w):
    """
    Raycast per-face labels using M_cam_from_uv for ray generation.
    Handles oblique/sheared projections that a simple pinhole K cannot represent.

    M_cam_from_uv maps NDC coordinates [-1,1] to camera-space ray directions.
    """
    W, H = img_res
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sem_mesh))

    # Pixel grid → NDC coordinates
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    ndc_x = 2.0 * (px + 0.5) / W - 1.0
    ndc_y = 1.0 - 2.0 * (py + 0.5) / H  # OpenGL: Y up
    ndc = np.stack([ndc_x, ndc_y, np.ones_like(ndc_x)], axis=-1)  # (H, W, 3)

    # Ray directions in camera space
    dirs_cam = ndc @ M_cam_from_uv.T  # (H, W, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # Camera-to-world (raw Hypersim c2w, OpenGL convention)
    R = c2w[:3, :3]
    cam_origin = c2w[:3, 3]

    dirs_world = dirs_cam @ R.T
    rays_o = np.tile(cam_origin, (H, W, 1))

    # Raycast
    rays = np.concatenate([rays_o, dirs_world], axis=-1).astype(np.float32)
    rays_o3d = o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays_o3d)

    triangle_ids = ans['primitive_ids'].numpy().reshape(H, W)
    hit_mask = triangle_ids != o3d.t.geometry.RaycastingScene.INVALID_ID

    semantic_img = np.full((H, W), fill_value=-1, dtype=np.int32)
    semantic_img[hit_mask] = face_labels[triangle_ids[hit_mask]]

    return semantic_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render Hypersim planes to HDF5 for each camera"
    )
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., ai_001_001)")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory of Hypersim dataset")
    parser.add_argument("--plane_root", type=str, required=True,
                        help="Root directory containing extracted planes (planes.ply)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory to save rendered plane HDF5 files")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Frame skip interval (default: 1 = all frames)")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Path to metadata_camera_parameters.csv (default: input_root/metadata_camera_parameters.csv)")
    args = parser.parse_args()

    scene_id = args.scene_id
    input_root = args.input_root
    plane_root = args.plane_root
    output_root = args.output_root
    frame_skip = args.frame_skip

    print(f"[INFO] Rendering Hypersim scene: {scene_id}")
    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Plane root: {plane_root}")
    print(f"[INFO] Output root: {output_root}")

    # === Paths ===
    mesh_path = os.path.join(plane_root, scene_id, "planes.ply")

    if not os.path.exists(mesh_path):
        # Try alternative path structure
        mesh_path = os.path.join(plane_root, "plane_ours_gt", scene_id, "planes.ply")

    if not os.path.exists(mesh_path):
        print(f"[ERROR] Mesh file not found: {mesh_path}")
        sys.exit(1)

    print(f"[INFO] Reading mesh: {mesh_path}")
    V, F, plane_id_face, _ = read_ply_faces_with_plane_ids(mesh_path)
    print(f"[INFO] V={V.shape[0]}  F={F.shape[0]}")

    # === Create mesh ===
    sem_mesh = o3d.geometry.TriangleMesh()
    sem_mesh.vertices = o3d.utility.Vector3dVector(V)
    sem_mesh.triangles = o3d.utility.Vector3iVector(F)
    sem_mesh.compute_vertex_normals()

    scene_dir = os.path.join(input_root, scene_id)
    detail_dir = os.path.join(scene_dir, "_detail")

    # === Camera parameters ===
    if args.metadata_csv:
        meta_cam_file = args.metadata_csv
    else:
        meta_cam_file = os.path.join(input_root, "metadata_camera_parameters.csv")

    if not os.path.exists(meta_cam_file):
        print(f"[ERROR] Metadata file not found: {meta_cam_file}")
        sys.exit(1)

    df_meta = pd.read_csv(meta_cam_file, index_col="scene_name")
    df_scene = df_meta.loc[scene_id]
    width = int(df_scene["settings_output_img_width"])
    height = int(df_scene["settings_output_img_height"])

    # M_cam_from_uv: maps NDC [-1,1] to camera-space ray directions
    # Correctly handles oblique/sheared projections (268/482 Hypersim scenes)
    M_cam_from_uv = np.array([
        [df_scene[f"M_cam_from_uv_{i}{j}"] for j in range(3)]
        for i in range(3)
    ], dtype=np.float64)

    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] M_cam_from_uv:\n{M_cam_from_uv}")

    # === Find available cameras ===
    if not os.path.exists(detail_dir):
        print(f"[ERROR] Detail directory not found: {detail_dir}")
        sys.exit(1)

    cam_names = sorted([d for d in os.listdir(detail_dir)
                        if d.startswith("cam_") and os.path.isdir(os.path.join(detail_dir, d))])
    print(f"[INFO] Found {len(cam_names)} cameras: {cam_names}")

    if len(cam_names) == 0:
        print(f"[ERROR] No cameras found in {detail_dir}")
        sys.exit(1)

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

        print(f"[INFO] Raycasting {total_frames} frames for {cam_name} (skip={frame_skip})...")
        for frame_id in tqdm(range(total_frames), desc=f"{cam_name}"):
            if frame_id % frame_skip != 0:
                continue

            R = cam_orientations[frame_id]
            T = cam_positions[frame_id]

            # --- Build camera-to-world (raw Hypersim, OpenGL convention) ---
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T

            # --- Raycast plane IDs per face ---
            semantic_img_face = raycast_with_cam_from_uv(
                sem_mesh, plane_id_face, M_cam_from_uv, (width, height), c2w
            )

            # --- Shift plane IDs by +1 so 0 = no plane ---
            # raycaster returns -1 for misses, 0+ for valid planes
            # +1 makes: -1 → 0 (no plane), 0 → 1, 1 → 2, etc.
            semantic_img = semantic_img_face + 1
            semantic_img = np.where(semantic_img < 0, 0, semantic_img)
            semantic_img = np.clip(semantic_img, 0, 65535).astype(np.uint16)

            frame_ids.append(f"{frame_id:04d}")
            planes_list.append(semantic_img)

        # === Save HDF5 for this camera ===
        h5_save_dir = os.path.join(output_root, scene_id)
        os.makedirs(h5_save_dir, exist_ok=True)
        h5_save_path = os.path.join(h5_save_dir, f"rendered_planes_{cam_name}.h5")

        print(f"[SAVE] -> {h5_save_path}")
        with h5py.File(h5_save_path, "w") as f:
            f.create_dataset("planes", data=np.stack(planes_list), compression="gzip")
            f.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))

        print(f"[DONE] Rendered {len(planes_list)} frames for {scene_id}/{cam_name}")

    print(f"\n[FINISHED] All cameras processed for scene {scene_id}")
