#!/usr/bin/env python3
"""
VKITTI2 Plane Extraction Scene Runner.

Processes all frames in a VKITTI2 scene+variant, extracts planes via
LO-RANSAC per semantic label, and saves everything (RGB, depth, semantics,
planes, calibration) to a single HDF5 file per scene+variant.

Usage:
    python -m planamono.gt_creation.vkitti2.scene_runner \
        --scene Scene01 --variant clone \
        --config planamono/gt_creation/configs/vkitti2_default.yml

    python -m planamono.gt_creation.vkitti2.scene_runner \
        --scene Scene01 --variant clone \
        --rgb_root /path/to/vkitti_2.0.3_rgb \
        --depth_root /path/to/vkitti_2.0.3_depth \
        --semseg_root /path/to/vkitti_2.0.3_classSegmentation \
        --output_root /path/to/output
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

from planamono.shared.plane_fitting.ransac_image import (
    depth_to_xyz, compute_normals, extract_planes_from_frame,
)

# ── VKITTI2 constants ────────────────────────────────────────────────

# Camera intrinsics
FX = FY = 725.0087
CX, CY = 620.5, 187.0

# Semantic classes: RGB color -> (name, class_id)
CLASS_MAP = {
    (0, 0, 0):       ('undefined', 0),
    (210, 0, 200):   ('terrain', 1),
    (90, 200, 255):  ('sky', 2),
    (0, 199, 0):     ('tree', 3),
    (90, 240, 0):    ('vegetation', 4),
    (140, 140, 140): ('building', 5),
    (100, 60, 100):  ('road', 6),
    (250, 100, 255): ('guard_rail', 7),
    (255, 255, 0):   ('traffic_sign', 8),
    (200, 200, 0):   ('traffic_light', 9),
    (255, 130, 0):   ('pole', 10),
    (80, 80, 80):    ('misc', 11),
    (160, 60, 60):   ('truck', 12),
    (255, 127, 80):  ('car', 13),
    (0, 139, 139):   ('van', 14),
}

CLASS_NAMES = {v[1]: v[0] for v in CLASS_MAP.values()}

# Planar semantic classes
PLANAR = {1, 5, 6, 8, 11}  # terrain, building, road, traffic_sign, misc (walls/sidewalks)

# No cross-class merging for VKITTI2
MERGE_COMPATIBLE = []

SCENES = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
VARIANTS = [
    'clone', 'fog', 'morning', 'overcast', 'rain', 'sunset',
    '15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right',
]


def load_extrinsics(data_root, scene, variant, camera=0):
    """
    Load per-frame camera-to-world (c2w) matrices from extrinsic.txt.

    The file lives at data_root/scene/variant/extrinsic.txt.
    Each line: frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1

    Returns:
        dict mapping frame_id (str, zero-padded) -> (4, 4) numpy array
    """
    ext_path = os.path.join(data_root, scene, variant, "extrinsic.txt")
    if not os.path.exists(ext_path):
        return None

    c2w_dict = {}
    with open(ext_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 18:
                continue
            frame_idx = int(parts[0])
            cam_id = int(parts[1])
            if cam_id != camera:
                continue
            vals = [float(x) for x in parts[2:]]
            mat = np.array(vals).reshape(4, 4)
            frame_id = f"{frame_idx:05d}"
            c2w_dict[frame_id] = mat

    return c2w_dict


def rgb_to_class_ids(seg_rgb):
    """Convert RGB segmentation image to integer class IDs."""
    class_ids = np.full(seg_rgb.shape[:2], -1, dtype=np.int32)
    for color, (name, cid) in CLASS_MAP.items():
        mask = np.all(seg_rgb == color, axis=-1)
        class_ids[mask] = cid
    return class_ids


def discover_frames(rgb_root, depth_root, semseg_root, scene, variant, camera=0):
    """
    Discover available frames for a scene+variant.

    Returns:
        list of (frame_id, rgb_path, depth_path, semseg_path) tuples
    """
    cam_str = f"Camera_{camera}"
    rgb_dir = os.path.join(rgb_root, scene, variant, "frames", "rgb", cam_str)
    depth_dir = os.path.join(depth_root, scene, variant, "frames", "depth", cam_str)
    seg_dir = os.path.join(semseg_root, scene, variant, "frames",
                           "classSegmentation", cam_str)

    if not os.path.isdir(rgb_dir):
        print(f"[ERROR] RGB directory not found: {rgb_dir}")
        return []

    frames = []
    for fname in sorted(os.listdir(rgb_dir)):
        if not fname.startswith("rgb_") or not fname.endswith(".jpg"):
            continue
        frame_id = fname.replace("rgb_", "").replace(".jpg", "")

        rgb_path = os.path.join(rgb_dir, fname)
        depth_path = os.path.join(depth_dir, f"depth_{frame_id}.png")
        seg_path = os.path.join(seg_dir, f"classgt_{frame_id}.png")

        if os.path.exists(depth_path) and os.path.exists(seg_path):
            frames.append((frame_id, rgb_path, depth_path, seg_path))

    return frames


def process_scene(args):
    """Process all frames in a VKITTI2 scene+variant."""
    scene = args.scene
    variant = args.variant
    camera = getattr(args, 'camera', 0)

    print(f"[INFO] Processing VKITTI2: {scene}/{variant}/Camera_{camera}")

    # Discover frames
    frames = discover_frames(
        args.rgb_root, args.depth_root, args.semseg_root,
        scene, variant, camera
    )
    if not frames:
        print("[ERROR] No frames found. Check paths.")
        return

    max_frames = getattr(args, 'max_frames', None)
    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]
        print(f"[INFO] Limited to {max_frames} frames (--max_frames)")

    print(f"[INFO] Processing {len(frames)} frames")

    # Load extrinsics (c2w) — try data_root first, then rgb_root
    ext_root = getattr(args, 'data_root', None) or args.rgb_root
    c2w_dict = load_extrinsics(ext_root, scene, variant, camera)
    if c2w_dict is not None:
        print(f"[INFO] Loaded {len(c2w_dict)} extrinsic matrices (c2w)")
    else:
        print("[WARN] No extrinsic.txt found, skipping c2w")

    # Output directory
    out_dir = os.path.join(args.output_root, scene, variant)
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "scene_data.h5")

    # RANSAC parameters
    ransac_iters = getattr(args, 'ransac_iters', 200)
    dist_thresh = getattr(args, 'dist_thresh', 0.05)
    normal_cos = getattr(args, 'normal_cos', 0.94)
    min_inliers = getattr(args, 'min_inliers', 500)
    max_planes = getattr(args, 'max_planes', 50)
    merge_normal = getattr(args, 'merge_normal', 0.95)
    merge_rel_dist = getattr(args, 'merge_rel_dist', 0.02)

    frame_ids = []
    rgb_list = []
    depth_list = []
    semseg_list = []
    planes_list = []
    c2w_list = []
    all_planes_json = []

    for frame_id, rgb_path, depth_path, seg_path in tqdm(frames, desc=f"{scene}/{variant}"):
        # Load RGB
        rgb = np.array(Image.open(rgb_path))[:, :, :3]

        # Load depth: 16-bit PNG in centimeters -> meters
        depth_png = np.array(Image.open(depth_path))
        depth = depth_png.astype(np.float32) / 100.0
        valid = depth > 0

        # Load semantic segmentation (RGB-encoded)
        seg_rgb = np.array(Image.open(seg_path))[:, :, :3]
        class_ids = rgb_to_class_ids(seg_rgb)

        # Compute 3D point cloud and normals
        xyz = depth_to_xyz(depth, FX, FY, CX, CY)
        normals = compute_normals(xyz, valid)

        # Extract planes
        plane_map, planes_info = extract_planes_from_frame(
            depth, class_ids, valid, xyz, normals,
            planar_classes=PLANAR,
            class_names=CLASS_NAMES,
            merge_compatible=MERGE_COMPATIBLE,
            ransac_iters=ransac_iters,
            dist_thresh=dist_thresh,
            normal_cos=normal_cos,
            min_inliers=min_inliers,
            max_planes=max_planes,
            merge_normal=merge_normal,
            merge_rel_dist=merge_rel_dist,
        )

        # Store everything
        frame_ids.append(frame_id)
        rgb_list.append(rgb.astype(np.uint8))
        depth_list.append(depth)
        semseg_list.append(class_ids.astype(np.int8))
        planes_list.append(plane_map.astype(np.uint16))
        if c2w_dict is not None and frame_id in c2w_dict:
            c2w_list.append(c2w_dict[frame_id].astype(np.float32))
        elif c2w_dict is not None:
            c2w_list.append(np.eye(4, dtype=np.float32))

        # JSON metadata per frame
        for pid, p in enumerate(planes_info):
            all_planes_json.append({
                'frame_id': frame_id,
                'plane_id': pid + 1,
                'n': p['n'].tolist(),
                'd': float(p['d']),
                'num_pixels': p['num_pixels'],
                'p95': p['p95'],
                'class_id': p['class_id'],
                'class_name': p['class_name'],
            })

    if not planes_list:
        print("[WARN] No frames processed.")
        return

    # Save H5 with everything
    print(f"[SAVE] Writing {len(planes_list)} frames to {h5_path}")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("rgb", data=np.stack(rgb_list), compression="gzip", compression_opts=4)
        f.create_dataset("depth", data=np.stack(depth_list), compression="gzip", compression_opts=4)
        f.create_dataset("semantic", data=np.stack(semseg_list), compression="gzip", compression_opts=4)
        f.create_dataset("planes", data=np.stack(planes_list), compression="gzip", compression_opts=4)
        f.create_dataset("frame_ids", data=np.array(frame_ids, dtype='S'))
        if c2w_list:
            f.create_dataset("c2w", data=np.stack(c2w_list))
        f.attrs['dataset'] = 'vkitti2'
        f.attrs['scene'] = scene
        f.attrs['variant'] = variant
        f.attrs['fx'] = FX
        f.attrs['fy'] = FY
        f.attrs['cx'] = CX
        f.attrs['cy'] = CY
        f.attrs['num_frames'] = len(frame_ids)
        f.attrs['image_height'] = rgb_list[0].shape[0]
        f.attrs['image_width'] = rgb_list[0].shape[1]

    # Save JSON
    json_path = os.path.join(out_dir, "planes.json")
    with open(json_path, 'w') as f:
        json.dump(all_planes_json, f, indent=2)

    # Print summary
    total_planes = sum(pm.max() for pm in planes_list)
    print(f"[DONE] {scene}/{variant}: {len(planes_list)} frames, "
          f"avg {total_planes / len(planes_list):.1f} planes/frame")


def main():
    parser = argparse.ArgumentParser(description='VKITTI2 Plane Extraction')
    parser.add_argument('--scene', type=str, required=True,
                        help='Scene name (e.g., Scene01)')
    parser.add_argument('--variant', type=str, default='clone',
                        help='Variant (e.g., clone, fog, morning)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (0=left, 1=right)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')

    # Data paths
    parser.add_argument('--data_root', type=str, default=None,
                        help='Single root for rgb/depth/semseg (if all under one dir)')
    parser.add_argument('--rgb_root', type=str, default=None)
    parser.add_argument('--depth_root', type=str, default=None)
    parser.add_argument('--semseg_root', type=str, default=None)
    parser.add_argument('--output_root', type=str, default=None)

    # RANSAC parameters
    parser.add_argument('--ransac_iters', type=int, default=200)
    parser.add_argument('--dist_thresh', type=float, default=0.05)
    parser.add_argument('--normal_cos', type=float, default=0.94)
    parser.add_argument('--min_inliers', type=int, default=500)
    parser.add_argument('--max_planes', type=int, default=50)
    parser.add_argument('--merge_normal', type=float, default=0.95)
    parser.add_argument('--merge_rel_dist', type=float, default=0.02)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Process only first N frames per scene (for testing)')

    args = parser.parse_args()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            if getattr(args, key, None) is None:
                setattr(args, key, val)

    # If data_root is set, use it for any unset roots
    if args.data_root is not None:
        if args.rgb_root is None:
            args.rgb_root = args.data_root
        if args.depth_root is None:
            args.depth_root = args.data_root
        if args.semseg_root is None:
            args.semseg_root = args.data_root

    # Validate required paths
    for attr in ['rgb_root', 'depth_root', 'semseg_root', 'output_root']:
        if getattr(args, attr, None) is None:
            print(f"[ERROR] Missing required argument: --{attr} (or --data_root)")
            sys.exit(1)

    process_scene(args)


if __name__ == '__main__':
    main()
