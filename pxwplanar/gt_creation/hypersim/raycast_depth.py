#!/usr/bin/env python3
"""
Hypersim Raycasted Depth Script

Raycasts the extracted plane mesh to produce per-frame depth (meters)
that is geometrically consistent with the plane labels.

This avoids two problems with the original V-Ray depth_meters.hdf5:
  1. Geometry mismatch: V-Ray depth comes from the full scene mesh (all objects),
     while plane labels come from our plane mesh (planes.ply).
  2. Camera model artifacts: pinhole backprojection introduces radial errors
     compared to the true V-Ray ray directions.

Supports two depth types (--depth_type):
  - zdepth (default): perpendicular distance to the image plane.
      Saved to scene_{cam}_geometry_hdf5_raycast/
      Use with backproject (pinhole K). No further conversion needed.
  - euclidean: raw Euclidean ray distance in meters (t_hit * MPAU).
      Saved to scene_{cam}_geometry_hdf5_raycast_euc/
      Use with backproject_mcam (M_cam_from_uv). No further conversion needed.

Usage:
    # z-depth (default, for pinhole K backprojection)
    python raycast_depth.py ai_001_001 \\
        --params_root /data/Hypersim_params \\
        --plane_root /data/hypersim_mesh_ours \\
        --output_root /data/hypersim \\
        --metadata_csv metadata_camera_parameters.csv

    # Euclidean depth (for M_cam_from_uv backprojection)
    python raycast_depth.py ai_001_001 \\
        --depth_type euclidean \\
        --params_root /data/Hypersim_params \\
        --plane_root /data/hypersim_mesh_ours \\
        --output_root /data/hypersim \\
        --metadata_csv metadata_camera_parameters.csv
"""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import os
import h5py
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from pxwplanar.shared.rendering import read_ply_faces_with_plane_ids


def get_mpau(df_scene, params_root, scene_id):
    """Get meters_per_asset_unit from metadata CSV, with per-scene fallback.

    Args:
        df_scene: Row from metadata_camera_parameters.csv for this scene.
        params_root: Root directory containing per-scene _detail folders.
        scene_id: Scene identifier.

    Returns:
        float: meters_per_asset_unit scale factor.
    """
    # Primary: metadata_camera_parameters.csv
    if "settings_units_info_meters_scale" in df_scene.index:
        mpau = float(df_scene["settings_units_info_meters_scale"])
        if mpau > 0:
            return mpau

    # Fallback: per-scene metadata_scene.csv
    fallback_csv = os.path.join(params_root, scene_id, "_detail", "metadata_scene.csv")
    if os.path.exists(fallback_csv):
        df_meta_scene = pd.read_csv(fallback_csv)
        for _, row in df_meta_scene.iterrows():
            key = str(row.iloc[0]).strip() if len(row) > 0 else ""
            if key == "meters_per_asset_unit":
                return float(row.iloc[1])

    raise ValueError(f"Cannot determine MPAU for scene {scene_id}")


def build_raycasting_scene(mesh_path):
    """Load planes.ply and build an Open3D RaycastingScene.

    Args:
        mesh_path: Path to planes.ply file.

    Returns:
        rc_scene: Open3D RaycastingScene ready for cast_rays().
        V: (N, 3) float32 vertex positions.
        F: (M, 3) int32 face indices.
    """
    V, F, _, _ = read_ply_faces_with_plane_ids(mesh_path)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(F)

    rc_scene = o3d.t.geometry.RaycastingScene()
    rc_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    return rc_scene


def load_M_cam_from_uv(df_scene):
    """Load the 3x3 M_cam_from_uv matrix from a metadata CSV row."""
    return np.array([[df_scene[f"M_cam_from_uv_{i}{j}"] for j in range(3)]
                     for i in range(3)])


def raycast_depth(rc_scene, M_cam_from_uv, R_world_from_cam,
                  cam_position, W, H, mpau, depth_type="zdepth"):
    """Raycast the plane mesh and return depth in meters.

    Uses the exact V-Ray NDC grid and M_cam_from_uv for ray generation,
    matching the rendering pipeline in render.py.

    Args:
        rc_scene: Open3D RaycastingScene with plane mesh.
        M_cam_from_uv: (3, 3) camera matrix from metadata CSV.
        R_world_from_cam: (3, 3) rotation (camera-to-world).
        cam_position: (3,) camera centre in world space.
        W, H: Image dimensions in pixels.
        mpau: meters_per_asset_unit scale factor.
        depth_type: "zdepth" for perpendicular distance to image plane
                    (t_hit * mpau * cos_theta), or "euclidean" for raw
                    Euclidean ray distance (t_hit * mpau).

    Returns:
        (H, W) float32 depth in meters (0.0 where no hit).
    """
    # Official Hypersim UV grid: half-pixel offsets within [-1, 1]
    half_du = 1.0 / W
    half_dv = 1.0 / H
    u = np.linspace(-1 + half_du, 1 - half_du, W)
    v = np.linspace(-1 + half_dv, 1 - half_dv, H)[::-1]  # top row = largest v
    uu, vv = np.meshgrid(u, v)

    # Camera-space ray directions via M_cam_from_uv
    uvs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)
    dirs_cam = uvs @ M_cam_from_uv.T                      # (H, W, 3)

    # World-space ray directions (normalized for Open3D raycasting)
    dirs_world = dirs_cam @ R_world_from_cam.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True)

    # Raycast
    origins = np.broadcast_to(cam_position, dirs_world.shape).copy()
    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    ans = rc_scene.cast_rays(
        o3d.core.Tensor(rays.reshape(-1, 6), dtype=o3d.core.Dtype.Float32)
    )

    t_hit = ans['t_hit'].numpy().reshape(H, W)
    hit_mask = np.isfinite(t_hit)

    depth = np.zeros((H, W), dtype=np.float32)
    if depth_type == "euclidean":
        # t_hit (asset units) -> Euclidean meters
        depth[hit_mask] = (t_hit[hit_mask] * mpau).astype(np.float32)
    else:
        # t_hit (asset units) -> Euclidean meters -> z-depth meters
        # cos(theta) = |d_cam.z| / |d_cam|
        # V-Ray camera looks along -Z so d_cam.z is typically negative.
        ray_lengths = np.linalg.norm(dirs_cam, axis=-1)        # (H, W)
        z_abs = np.abs(dirs_cam[:, :, 2])                      # (H, W)
        cos_theta = np.where(ray_lengths > 0, z_abs / ray_lengths, 1.0)
        depth[hit_mask] = (t_hit[hit_mask] * mpau * cos_theta[hit_mask]).astype(np.float32)

    return depth


def save_depth_hdf5(depth, path):
    """Save depth array as HDF5 (matching Hypersim depth_meters.hdf5 format).

    Args:
        depth: (H, W) float32 z-depth in meters.
        path: Output HDF5 file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("dataset", data=depth, dtype=np.float32,
                         compression="gzip", chunks=(96, 128))


def process_scene(scene_id, params_root, plane_root, output_root,
                  metadata_csv, frame_skip=1, depth_type="zdepth"):
    """Raycast plane mesh to produce per-frame depth for all cameras in a scene.

    Args:
        scene_id: Scene identifier (e.g. 'ai_001_001').
        params_root: Root directory of Hypersim parameters (_detail, poses).
        plane_root: Root directory containing extracted plane meshes.
        output_root: Root directory to save raycasted depth HDF5 files.
        metadata_csv: Path to metadata_camera_parameters.csv.
        frame_skip: Frame skip interval (1 = all frames).
        depth_type: "zdepth" or "euclidean".
    """
    print(f"[INFO] Raycasting depth for scene: {scene_id} (depth_type={depth_type})")

    # === Locate mesh ===
    mesh_path = os.path.join(plane_root, scene_id, "planes.ply")
    if not os.path.exists(mesh_path):
        mesh_path = os.path.join(plane_root, "plane_ours_gt", scene_id, "planes.ply")
    if not os.path.exists(mesh_path):
        print(f"[ERROR] Mesh not found: {mesh_path}")
        return False

    print(f"[INFO] Loading mesh: {mesh_path}")
    rc_scene = build_raycasting_scene(mesh_path)

    # === Metadata ===
    if not os.path.exists(metadata_csv):
        print(f"[ERROR] Metadata CSV not found: {metadata_csv}")
        return False

    df_meta = pd.read_csv(metadata_csv, index_col="scene_name")
    if scene_id not in df_meta.index:
        print(f"[ERROR] Scene {scene_id} not in metadata CSV")
        return False

    df_scene = df_meta.loc[scene_id]
    W = int(df_scene["settings_output_img_width"])
    H = int(df_scene["settings_output_img_height"])
    M_cam_from_uv = load_M_cam_from_uv(df_scene)
    mpau = get_mpau(df_scene, params_root, scene_id)

    print(f"[INFO] Resolution: {W}x{H}, MPAU: {mpau:.6f}")

    # === Find cameras ===
    detail_dir = os.path.join(params_root, scene_id, "_detail")
    if not os.path.exists(detail_dir):
        print(f"[ERROR] Detail directory not found: {detail_dir}")
        return False

    cam_names = sorted([d for d in os.listdir(detail_dir)
                        if d.startswith("cam_") and os.path.isdir(os.path.join(detail_dir, d))])
    print(f"[INFO] Found {len(cam_names)} cameras: {cam_names}")

    if len(cam_names) == 0:
        print(f"[ERROR] No cameras found in {detail_dir}")
        return False

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
        dir_suffix = "raycast_euc" if depth_type == "euclidean" else "raycast"
        out_dir = os.path.join(output_root, scene_id, "images",
                               f"scene_{cam_name}_geometry_hdf5_{dir_suffix}")

        processed = 0
        print(f"[INFO] Raycasting {total_frames} frames for {cam_name} (skip={frame_skip})...")

        for frame_id in tqdm(range(total_frames), desc=f"{cam_name}"):
            if frame_id % frame_skip != 0:
                continue

            fid = f"{frame_id:04d}"
            out_path = os.path.join(out_dir, f"frame.{fid}.depth_meters.hdf5")

            R = cam_orientations[frame_id]  # (3,3) R_world_from_cam
            T = cam_positions[frame_id]     # (3,)  camera position in world

            depth = raycast_depth(
                rc_scene, M_cam_from_uv, R, T, W, H, mpau,
                depth_type=depth_type,
            )
            save_depth_hdf5(depth, out_path)
            processed += 1

        print(f"[DONE] {cam_name}: {processed} frames written")

    print(f"\n[FINISHED] All cameras processed for scene {scene_id}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raycast Hypersim plane mesh to produce per-frame depth"
    )
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., ai_001_001)")
    parser.add_argument("--params_root", type=str, required=True,
                        help="Root directory of Hypersim parameters")
    parser.add_argument("--plane_root", type=str, required=True,
                        help="Root directory containing extracted plane meshes")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory to save raycasted depth HDF5 files")
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="Path to metadata_camera_parameters.csv")
    parser.add_argument("--frame_skip", type=int, default=1,
                        help="Frame skip interval (default: 1 = all frames)")
    parser.add_argument("--depth_type", type=str, default="zdepth",
                        choices=["zdepth", "euclidean"],
                        help="Depth type: 'zdepth' (default) saves z-depth "
                             "(use with pinhole K), 'euclidean' saves raw "
                             "Euclidean ray distance (use with M_cam_from_uv)")
    args = parser.parse_args()

    success = process_scene(
        scene_id=args.scene_id,
        params_root=args.params_root,
        plane_root=args.plane_root,
        output_root=args.output_root,
        metadata_csv=args.metadata_csv,
        frame_skip=args.frame_skip,
        depth_type=args.depth_type,
    )

    sys.exit(0 if success else 1)
