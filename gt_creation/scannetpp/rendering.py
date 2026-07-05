#!/usr/bin/env python3
"""
ScanNet++ Plane Rendering Script

Renders extracted planes to PNG images for each frame in a ScanNet++ scene.
Uses raycasting to project 3D plane meshes to 2D images.

Usage:
    python rendering.py scene_id --input_root /path/to/scannetpp --plane_root /path/to/planes --output_root /path/to/output
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.rendering import load_mesh_with_vertex_labels, raycast_semantic
from shared.utils import save_label_image
import open3d as o3d
import numpy as np
import json
import os
import argparse
from tqdm import tqdm


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render ScanNet++ planes to PNG images"
    )
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., 0a5c013435)")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory of ScanNet++ dataset")
    parser.add_argument("--plane_root", type=str, required=True,
                        help="Root directory containing extracted planes (planes_v2.ply)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory to save rendered plane images")
    parser.add_argument("--frame_skip", type=int, default=25,
                        help="Frame skip interval (default: 25)")
    parser.add_argument("--width", type=int, default=640,
                        help="Output image width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Output image height (default: 480)")
    args = parser.parse_args()

    scene_id = args.scene_id
    input_root = args.input_root
    plane_root = args.plane_root
    output_root = args.output_root
    frame_skip = args.frame_skip

    print(f"[INFO] Rendering ScanNet++ scene: {scene_id}")
    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Plane root: {plane_root}")
    print(f"[INFO] Output root: {output_root}")

    # === Paths ===
    mesh_path = os.path.join(plane_root, scene_id, "planes_v2.ply")
    if not os.path.exists(mesh_path):
        # Try alternative naming
        mesh_path = os.path.join(plane_root, scene_id, "planes.ply")

    if not os.path.exists(mesh_path):
        print(f"[ERROR] Mesh file not found: {mesh_path}")
        sys.exit(1)

    render_save_path = os.path.join(output_root, scene_id, "rendered_v2")
    os.makedirs(render_save_path, exist_ok=True)

    print(f"[INFO] Reading mesh: {mesh_path}")
    sem_mesh, vertex_labels = load_mesh_with_vertex_labels(mesh_path)

    # === Load pose data ===
    iphone_dir = os.path.join(input_root, scene_id, "iphone")
    pose_file = os.path.join(iphone_dir, "pose_intrinsic_imu.json")

    if not os.path.exists(pose_file):
        print(f"[ERROR] Pose file not found: {pose_file}")
        sys.exit(1)

    with open(pose_file, "r") as f:
        data = json.load(f)

    # === Intrinsics ===
    first_key = next(iter(data))
    K = np.array(data[first_key]["intrinsic"])

    # Native iPhone resolution for ScanNet++ (1920×1440)
    W_orig, H_orig = 1920, 1440
    W, H = args.width, args.height

    # Scale intrinsics to match target resolution
    scale_x = W / W_orig
    scale_y = H / H_orig
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[1, 2] *= scale_y  # cy

    print(f"[INFO] Output resolution: {W}x{H}")
    print(f"[INFO] Frame skip: {frame_skip}")
    print(f"[INFO] Total frames: {len(data)}")

    # === Raycast ===
    print("[INFO] Raycasting...")
    for i, (frame_id, frame_data) in enumerate(tqdm(data.items(), total=len(data))):
        if i % frame_skip != 0:
            continue

        c2w = np.array(frame_data["aligned_pose"])
        semantic_img = raycast_semantic(sem_mesh, vertex_labels, K_scaled, (W, H), c2w)
        # semantic_img = remap_semantic_old(semantic_img)  # BUGGY: plane_id=0 collides with non-planar
        semantic_img = remap_plane_ids(semantic_img)

        seg_path = os.path.join(render_save_path, f"{frame_id}.png")
        save_label_image(seg_path, semantic_img)

    print(f"[DONE] Finished scene: {scene_id}")
    print(f"[INFO] Saved to: {render_save_path}")
