#!/usr/bin/env python3
"""
Qualitative Comparison Visualization Script

Generates side-by-side comparison videos of different plane segmentation methods.

Usage:
    python visualize_comparison.py --rgb_root /path/to/scannet --results_root /path/to/results --gt_root /path/to/gt
"""
import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import cv2
import time
from tqdm import tqdm
import glob
import argparse

from PIL import Image
from natsort import natsorted
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from planamono.shared.utils import visualize_top_components_v1


def merge_plane_masks(seg_pred):
    """
    Convert multi-channel binary plane masks into a single-channel instance mask.

    Args:
        seg_pred (np.ndarray): shape (C, H, W), each channel is a binary mask for a plane.

    Returns:
        np.ndarray: shape (H, W), each pixel has the plane instance ID (0 = background).
    """
    C, H, W = seg_pred.shape
    instance_mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(C):
        mask = seg_pred[i] > 0
        instance_mask[mask] = i + 1

    return instance_mask


def generate_comparison_video(
    scene_id,
    rgb_root,
    results_root,
    gt_root,
    output_dir,
    frame_skip=50,
    fps=1,
    top_k=10
):
    """Generate comparison video for a single scene."""
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{scene_id}_baseline.mp4")

    frame_size = (1600, 400)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Find images
    color_dir = os.path.join(rgb_root, scene_id, 'color')
    image_list = natsorted(glob.glob(os.path.join(color_dir, '*.jpg')))

    if len(image_list) == 0:
        print(f"[WARN] No images found for {scene_id}")
        return None

    # Find GT planes
    plane_gt_dir = os.path.join(gt_root, scene_id)
    plane_gt_list = natsorted(glob.glob(os.path.join(plane_gt_dir, '*.png')))

    # Find predictions
    plane_ours_dir = os.path.join(results_root, 'moge', 'seg_pred', scene_id)
    plane_ours_list = natsorted(glob.glob(os.path.join(plane_ours_dir, '*.npy')))

    plane_rcnn_dir = os.path.join(rgb_root, scene_id, 'seg_pred', 'planercnn')
    plane_rcnn_list = natsorted(glob.glob(os.path.join(plane_rcnn_dir, '*.npy')))

    plane_zero_dir = os.path.join(rgb_root, scene_id, 'seg_pred', 'zeroplane')
    plane_zero_list = natsorted(glob.glob(os.path.join(plane_zero_dir, '*.npy')))

    for subindx, idx in enumerate(range(0, len(image_list), frame_skip)):
        image_path = image_list[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Load image
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)

        # Load GT
        plane_gt_arr = None
        if subindx < len(plane_gt_list):
            plane_gt = Image.open(plane_gt_list[subindx]).convert('L')
            plane_gt_arr = np.array(plane_gt).astype(np.uint8)

        # Load predictions
        plane_ours = None
        if subindx < len(plane_ours_list):
            plane_ours = np.load(plane_ours_list[subindx])

        plane_rcnn = None
        if subindx < len(plane_rcnn_list):
            plane_rcnn = merge_plane_masks(np.load(plane_rcnn_list[subindx]))

        plane_zero = None
        if subindx < len(plane_zero_list):
            plane_zero = np.load(plane_zero_list[subindx])

        # Create figure
        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        fig.suptitle(f"ScanNet: {scene_id} | {base_name}", fontsize=12)

        # Image
        axs[0].imshow(img_np)
        axs[0].set_title("Image")
        axs[0].axis('off')

        # GT
        if plane_gt_arr is not None:
            n = min(top_k, len(np.unique(plane_gt_arr)))
            seg_vis = visualize_top_components_v1(plane_gt_arr, k=n, return_colors=True)
            axs[1].imshow(seg_vis)
            axs[1].set_title(f"GT: Top-{n}")
        else:
            axs[1].imshow(np.zeros_like(img_np))
            axs[1].set_title("GT: N/A")
        axs[1].axis('off')

        # Ours (MoGe)
        if plane_ours is not None:
            n = min(top_k, len(np.unique(plane_ours)))
            seg_vis = visualize_top_components_v1(plane_ours, k=n, return_colors=True)
            axs[2].imshow(seg_vis)
            axs[2].set_title(f"Ours: Top-{n}")
        else:
            axs[2].imshow(np.zeros_like(img_np))
            axs[2].set_title("Ours: N/A")
        axs[2].axis('off')

        # PlaneRCNN
        if plane_rcnn is not None:
            n = min(top_k, len(np.unique(plane_rcnn)))
            seg_vis = visualize_top_components_v1(plane_rcnn, k=n, return_colors=True)
            axs[3].imshow(seg_vis)
            axs[3].set_title(f"PlaneRCNN: Top-{n}")
        else:
            axs[3].imshow(np.zeros_like(img_np))
            axs[3].set_title("PlaneRCNN: N/A")
        axs[3].axis('off')

        # ZeroPlane
        if plane_zero is not None:
            n = min(top_k, len(np.unique(plane_zero)))
            seg_vis = visualize_top_components_v1(plane_zero, k=n, return_colors=True)
            axs[4].imshow(seg_vis)
            axs[4].set_title(f"ZeroPlane: Top-{n}")
        else:
            axs[4].imshow(np.zeros_like(img_np))
            axs[4].set_title("ZeroPlane: N/A")
        axs[4].axis('off')

        plt.tight_layout()

        # Render to video frame
        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        fig_img = cv2.resize(fig_img, frame_size)
        video_writer.write(cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"[DONE] Video saved: {output_video_path}")
    return output_video_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate qualitative comparison videos"
    )
    parser.add_argument("--rgb_root", type=str, required=True,
                        help="Root directory of ScanNet RGB images (contains scene_id/color/)")
    parser.add_argument("--results_root", type=str, required=True,
                        help="Root directory of prediction results")
    parser.add_argument("--gt_root", type=str, required=True,
                        help="Root directory of ground truth planes")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for output videos")
    parser.add_argument("--frame_skip", type=int, default=50,
                        help="Process every Nth frame (default: 50)")
    parser.add_argument("--fps", type=int, default=1,
                        help="Video FPS (default: 1)")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Maximum number of scenes to process")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top components to visualize (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("Qualitative Comparison Video Generator")
    print("=" * 60)
    print(f"RGB root: {args.rgb_root}")
    print(f"Results root: {args.results_root}")
    print(f"GT root: {args.gt_root}")
    print(f"Output root: {args.output_root}")
    print("-" * 60)

    # Find scenes
    scene_id_list = [name for name in os.listdir(args.rgb_root)
                     if os.path.isdir(os.path.join(args.rgb_root, name))]
    scene_id_list = natsorted(scene_id_list)

    if len(scene_id_list) == 0:
        print(f"[ERROR] No scenes found in {args.rgb_root}")
        sys.exit(1)

    if args.max_scenes:
        scene_id_list = scene_id_list[:args.max_scenes]

    print(f"[INFO] Found {len(scene_id_list)} scenes")

    output_dir = os.path.join(args.output_root, 'comparison_videos')
    os.makedirs(output_dir, exist_ok=True)

    for scene_id in tqdm(scene_id_list, desc="Generating videos"):
        generate_comparison_video(
            scene_id,
            args.rgb_root,
            args.results_root,
            args.gt_root,
            output_dir,
            frame_skip=args.frame_skip,
            fps=args.fps,
            top_k=args.top_k
        )

    print("=" * 60)
    print(f"[SUCCESS] Generated {len(scene_id_list)} videos")
    print(f"[INFO] Videos saved to: {output_dir}")


if __name__ == "__main__":
    main()
