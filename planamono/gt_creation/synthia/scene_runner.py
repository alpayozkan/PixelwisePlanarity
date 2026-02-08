#!/usr/bin/env python3
"""
SYNTHIA-AL Plane Extraction Scene Runner.

Processes all frames in a SYNTHIA scene directory, extracts planes via
LO-RANSAC per semantic label, and saves everything (RGB, depth, semantics,
planes, calibration) to a single HDF5 file per scene.

Usage:
    python -m planamono.gt_creation.synthia.scene_runner \
        --scene_dir /path/to/test/test5_10segs_weather_0_.../DD-MM-YYYY_HH-MM-SS \
        --output_root /path/to/output

    python -m planamono.gt_creation.synthia.scene_runner \
        --scene_dir /path/to/scene_dir \
        --config planamono/gt_creation/configs/synthia_default.yml
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

# ── SYNTHIA constants ────────────────────────────────────────────────

# Camera intrinsics
FX = FY = 895.692
CX, CY = 320.0, 240.0

# Planar semantic classes
PLANAR = {1, 2, 3, 20}  # Ground, Sidewalk, Building, Lanemarking

CLASS_NAMES = {
    0: 'Void', 1: 'Ground', 2: 'Sidewalk', 3: 'Building',
    4: 'Sidewalk2', 5: 'Fence', 6: 'Pole', 7: 'TrafficLight',
    8: 'Car', 9: 'Vegetation', 10: 'Unknown', 11: 'Sky',
    12: 'Human', 14: 'Unknown2', 20: 'Lanemarking',
    21: 'Human', 25: 'Cyclist',
}

# Ground + Lanemarking can merge (same road surface)
MERGE_COMPATIBLE = [
    {1, 20},
]


def decode_depth(depth_png):
    """Decode SYNTHIA RGB-encoded depth to meters."""
    R = depth_png[:, :, 0].astype(np.float64)
    G = depth_png[:, :, 1].astype(np.float64)
    B = depth_png[:, :, 2].astype(np.float64)
    return (5000.0 * (R + G * 256 + B * 256 * 256) / (256**3 - 1)).astype(np.float32)


def discover_frames(scene_dir):
    """
    Discover available frames in a SYNTHIA scene directory.

    Expects structure:
        scene_dir/
            RGB/000101.png, 000102.png, ...
            Depth/000101.png, ...
            SemSeg/000101.png, ...

    Returns:
        list of (frame_id, rgb_path, depth_path, semseg_path) tuples
    """
    rgb_dir = os.path.join(scene_dir, "RGB")
    depth_dir = os.path.join(scene_dir, "Depth")
    semseg_dir = os.path.join(scene_dir, "SemSeg")

    if not os.path.isdir(rgb_dir):
        print(f"[ERROR] RGB directory not found: {rgb_dir}")
        return []

    frames = []
    for fname in sorted(os.listdir(rgb_dir)):
        if not fname.endswith(".png"):
            continue
        frame_id = os.path.splitext(fname)[0]

        rgb_path = os.path.join(rgb_dir, fname)
        depth_path = os.path.join(depth_dir, fname)
        seg_path = os.path.join(semseg_dir, fname)

        if os.path.exists(depth_path) and os.path.exists(seg_path):
            frames.append((frame_id, rgb_path, depth_path, seg_path))

    return frames


def get_scene_name(scene_dir):
    """Extract a compact scene name from SYNTHIA directory path."""
    parts = os.path.normpath(scene_dir).split(os.sep)
    for part in reversed(parts):
        if part.startswith("test5_"):
            return part
    return "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def process_scene(args):
    """Process all frames in a SYNTHIA scene directory."""
    scene_dir = args.scene_dir

    # Handle scenes with timestamp subdirectories
    if not os.path.isdir(os.path.join(scene_dir, "RGB")):
        subdirs = [d for d in os.listdir(scene_dir)
                   if os.path.isdir(os.path.join(scene_dir, d))
                   and os.path.isdir(os.path.join(scene_dir, d, "RGB"))]
        if len(subdirs) == 1:
            scene_dir = os.path.join(scene_dir, subdirs[0])
        elif len(subdirs) > 1:
            print(f"[INFO] Found {len(subdirs)} timestamp subdirs, processing all")
            for subdir in sorted(subdirs):
                sub_args = argparse.Namespace(**vars(args))
                sub_args.scene_dir = os.path.join(args.scene_dir, subdir)
                process_scene(sub_args)
            return
        else:
            print(f"[ERROR] No RGB directory found in {scene_dir}")
            return

    scene_name = get_scene_name(scene_dir)
    print(f"[INFO] Processing SYNTHIA scene: {scene_name}")

    # Discover frames
    frames = discover_frames(scene_dir)
    if not frames:
        print("[ERROR] No frames found. Check directory structure.")
        return

    max_frames = getattr(args, 'max_frames', None)
    if max_frames is not None and len(frames) > max_frames:
        frames = frames[:max_frames]
        print(f"[INFO] Limited to {max_frames} frames (--max_frames)")

    print(f"[INFO] Processing {len(frames)} frames")

    # Output directory
    out_dir = os.path.join(args.output_root, scene_name)
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
    all_planes_json = []

    for frame_id, rgb_path, depth_path, seg_path in tqdm(frames, desc=scene_name):
        # Load RGB
        rgb = np.array(Image.open(rgb_path))
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        # Load depth: RGB-encoded depth
        depth_png = np.array(Image.open(depth_path))
        depth = decode_depth(depth_png)
        valid = (depth > 0.1) & (depth < 4999.0)  # exclude sky/invalid

        # Load semantic segmentation (channel 0 = class ID)
        semseg = np.array(Image.open(seg_path))
        class_ids = semseg[:, :, 0].astype(np.int32)

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
        semseg_list.append(class_ids.astype(np.uint8))
        planes_list.append(plane_map.astype(np.uint16))

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
        f.attrs['dataset'] = 'synthia'
        f.attrs['scene_name'] = scene_name
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
    print(f"[DONE] {scene_name}: {len(planes_list)} frames, "
          f"avg {total_planes / len(planes_list):.1f} planes/frame")


def main():
    parser = argparse.ArgumentParser(description='SYNTHIA Plane Extraction')
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='Path to SYNTHIA scene directory (containing RGB/Depth/SemSeg)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Output root directory')

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
    parser.add_argument('--output_root_override', type=str, default=None,
                        help='Override output_root (takes priority over config)')

    args = parser.parse_args()

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for key, val in cfg.items():
            if getattr(args, key, None) is None:
                setattr(args, key, val)

    if args.output_root_override is not None:
        args.output_root = args.output_root_override

    if args.output_root is None:
        print("[ERROR] Missing --output_root")
        sys.exit(1)

    process_scene(args)


if __name__ == '__main__':
    main()
