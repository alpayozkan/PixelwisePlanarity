#!/usr/bin/env python3
"""
ScanNet++ Video Generation Script

Generates visualization videos from rendered plane HDF5 files.

Usage:
    python video_gen.py scene_id --h5_root /path/to/h5 --rgb_root /path/to/scannetpp --output_root /path/to/output
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pxwplanar.shared.utils import visualize_top_components

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import cv2
from tqdm import tqdm
import argparse


def generate_video_from_plane_h5(scene_id, h5_path, rgb_root, save_video_path, fps=10, top_n=20):
    """
    Generate visualization video from rendered plane HDF5 file.

    Args:
        scene_id: Scene ID
        h5_path: Path to HDF5 file with rendered planes
        rgb_root: Root directory of ScanNet++ RGB images
        save_video_path: Output video path
        fps: Frames per second
        top_n: Number of top components to visualize
    """
    # === Load HDF5 ===
    with h5py.File(h5_path, "r") as f:
        planes = f["rendered_planes"][:]  # (N, H, W), dtype=uint16
        frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]  # (N,)

    print(f"[INFO] Loaded {len(planes)} frames from {scene_id}")

    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
    writer = imageio.get_writer(save_video_path, fps=fps)

    rgb_base_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")

    N = len(planes)
    for i in tqdm(range(N), desc=f"[{scene_id}] Writing video"):
        plane_img = planes[i]
        frame_id = frame_ids[i]

        # === Load and process RGB ===
        rgb_path = os.path.join(rgb_base_dir, f"{frame_id}.jpg")
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (plane_img.shape[1], plane_img.shape[0]))
        else:
            print(f"[WARN] Missing RGB: {rgb_path}")
            rgb_img = np.zeros((plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8)

        # === Visualize plane ===
        plane_vis = visualize_top_components(plane_img, k=top_n, ignore_label=0,
                                             return_colors=True)

        # === Plot RGB + Plane side-by-side ===
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(rgb_img)
        axs[0].set_title("RGB")
        axs[0].axis("off")

        axs[1].imshow(plane_vis, cmap="tab20", interpolation="nearest")
        axs[1].set_title("Plane Segmentation")
        axs[1].axis("off")

        fig.suptitle(f"Scene: {scene_id} | Frame: {frame_id}", fontsize=12)
        fig.tight_layout()

        # Render to image
        fig.canvas.draw()
        img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(img_np)

        plt.close(fig)

    writer.close()
    print(f"[DONE] Saved video: {save_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visualization video from rendered plane HDF5"
    )
    parser.add_argument("scene_id", type=str, help="Scene ID (e.g., 0a5c013435)")
    parser.add_argument("--h5_root", type=str, required=True,
                        help="Root directory containing rendered HDF5 files")
    parser.add_argument("--rgb_root", type=str, required=True,
                        help="Root directory of ScanNet++ dataset (contains scene_id/iphone/rgb)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for output videos")
    parser.add_argument("--fps", type=int, default=5,
                        help="Video frames per second (default: 5)")
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top components to visualize (default: 20)")
    args = parser.parse_args()

    scene_id = args.scene_id

    # Build paths
    h5_path = os.path.join(args.h5_root, scene_id, "rendered.h5")
    video_dir = os.path.join(args.output_root, "videos")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{scene_id}.mp4")

    print(f"[INFO] Scene: {scene_id}")
    print(f"[INFO] H5 path: {h5_path}")
    print(f"[INFO] RGB root: {args.rgb_root}")
    print(f"[INFO] Output: {video_path}")

    if not os.path.exists(h5_path):
        print(f"[ERROR] HDF5 not found: {h5_path}")
        sys.exit(1)

    generate_video_from_plane_h5(
        scene_id, h5_path, args.rgb_root, video_path,
        fps=args.fps, top_n=args.top_n
    )
