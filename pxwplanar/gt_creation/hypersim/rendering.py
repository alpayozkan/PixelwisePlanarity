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
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from pxwplanar.shared.rendering import read_ply_faces_with_plane_ids, raycast_semantic_face_labels_mcam


def remap_semantic_old(semantic_img):
    """Remap semantic labels, replacing -1 with 0.

    BUGGY: plane_id=0 (the largest plane) collides with non-planar after
    this remap — both end up as 0.  Use remap_plane_ids() instead.
    """
    return np.where(semantic_img < 0, 0, semantic_img)


def remap_plane_ids(semantic_img):
    """Shift plane IDs by +1 so that 0 is reserved exclusively for non-planar.

    Input convention  (from plane_extraction + raycast):
        -1 = non-planar mesh face OR raycast miss
         0, 1, 2, … = valid plane IDs (0 = largest plane)

    Output convention (stored in HDF5, consumed by dataset / eval):
         0 = non-planar
         1, 2, 3, … = valid plane IDs
    """
    return np.where(semantic_img < 0, 0, semantic_img + 1)


def load_M_cam_from_uv(df_scene):
    """Load the 3x3 M_cam_from_uv matrix from a metadata CSV row."""
    return np.array([[df_scene[f"M_cam_from_uv_{i}{j}"] for j in range(3)]
                     for i in range(3)])


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
    sem_mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
    sem_mesh.triangles = o3d.utility.Vector3iVector(F)
    sem_mesh.compute_vertex_normals()

    scene_dir = os.path.join(input_root, scene_id)
    detail_dir = os.path.join(scene_dir, "_detail")

    # === Intrinsics ===
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
    M_cam_from_uv = load_M_cam_from_uv(df_scene)

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

            R = cam_orientations[frame_id]  # (3,3) R_world_from_cam
            T = cam_positions[frame_id]    # (3,)  camera position in world

            # --- Raycast plane IDs using M_cam_from_uv (no flip needed) ---
            semantic_img_face = raycast_semantic_face_labels_mcam(
                sem_mesh, plane_id_face, M_cam_from_uv, R, T, width, height
            )

            # --- Shift plane IDs: -1→0 (non-planar), 0→1, 1→2, … ---
            # semantic_img = remap_semantic_old(semantic_img_face)  # BUGGY: plane_id=0 collides with non-planar
            semantic_img = remap_plane_ids(semantic_img_face)
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
