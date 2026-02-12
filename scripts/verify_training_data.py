#!/usr/bin/env python3
"""
Verify the PixelwisePlanarity training data pipeline.

Loads samples through the actual dataset classes used during training
and saves visualizations: RGB, plane mask, plane overlay on RGB,
depth, and semantic labels — at the training resolution.

Usage:
    python scripts/verify_training_data.py --dataset scannetpp \
        --num_samples 5 --output_dir output/training_data_verify/

    python scripts/verify_training_data.py --dataset hypersim \
        --num_samples 5 --output_dir output/training_data_verify/

    python scripts/verify_training_data.py --dataset scannetpp \
        --resolution 576 644 --num_samples 3
"""
import argparse
import os
import sys
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from planamono.shared.datasets.scannetpp import ScanNetPPPlanarityDataset
from planamono.shared.datasets.hypersim import HypersimPlanarityDataset

# Default training resolution
DEFAULT_RESOLUTION = (476, 644)

# Colors for plane overlay
PLANE_COLOR = np.array([0, 255, 128], dtype=np.float32)


def plane_overlay(rgb_uint8, plane_mask, alpha=0.45):
    """
    Overlay plane mask on RGB. Planar regions get green tint, non-planar dimmed.

    Args:
        rgb_uint8: (H, W, 3) uint8
        plane_mask: (H, W) binary float/bool
    """
    overlay = rgb_uint8.copy().astype(np.float32)
    planar = plane_mask > 0.5

    overlay[planar] = overlay[planar] * (1 - alpha) + PLANE_COLOR * alpha
    overlay[~planar] = overlay[~planar] * 0.4

    return np.clip(overlay, 0, 255).astype(np.uint8)


def depth_to_colormap(depth, max_depth=None):
    """JET colormap for depth."""
    valid = depth > 0
    if max_depth is None:
        max_depth = np.percentile(depth[valid], 95) if valid.any() else 10.0
    max_depth = max(max_depth, 0.1)

    norm = np.clip(depth / max_depth, 0, 1)
    colored = cv2.applyColorMap((255 - (norm * 255).astype(np.uint8)), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def semantic_to_colormap(sem):
    """Tab20 colormap for semantic labels."""
    cmap = plt.cm.get_cmap('tab20', max(sem.max() + 1, 20))
    colored = (cmap(sem % 20)[:, :, :3] * 255).astype(np.uint8)
    colored[sem == 0] = 0  # background black
    return colored


def process_sample(sample, output_dir, idx):
    """Save visualizations for one training sample."""
    image = sample["image"].numpy()        # (3, H, W) float [0,1]
    depth = sample["depth"].numpy()        # (1, H, W) meters
    plane = sample["plane"].numpy()        # (1, H, W) binary
    semantic = sample["semantic"].numpy()  # (1, H, W) int

    scene_id = sample["scene_id"]
    frame_idx = sample["frame_idx"]

    H, W = image.shape[1], image.shape[2]

    rgb_uint8 = (image.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    depth_2d = depth[0]
    plane_2d = plane[0]
    sem_2d = semantic[0]

    out_dir = os.path.join(output_dir, f"{idx:04d}_{scene_id}_f{frame_idx}")
    os.makedirs(out_dir, exist_ok=True)

    n_planar = int((plane_2d > 0.5).sum())
    n_total = H * W
    pct = 100.0 * n_planar / n_total
    depth_valid = depth_2d[depth_2d > 0]

    print(f"  Resolution: {H}x{W}, planar: {n_planar}/{n_total} ({pct:.1f}%)", end="")
    if len(depth_valid) > 0:
        print(f", depth: [{depth_valid.min():.2f}, {depth_valid.max():.2f}] m")
    else:
        print()

    # 01: RGB
    cv2.imwrite(os.path.join(out_dir, "01_rgb.png"),
                cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))

    # 02: Plane mask
    plane_vis = (plane_2d * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "02_plane_mask.png"), plane_vis)

    # 03: Plane overlay on RGB
    overlay = plane_overlay(rgb_uint8, plane_2d)
    cv2.imwrite(os.path.join(out_dir, "03_plane_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # 04: Depth
    depth_vis = depth_to_colormap(depth_2d)
    cv2.imwrite(os.path.join(out_dir, "04_depth.png"),
                cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

    # 05: Semantic
    sem_vis = semantic_to_colormap(sem_2d)
    cv2.imwrite(os.path.join(out_dir, "05_semantic.png"),
                cv2.cvtColor(sem_vis, cv2.COLOR_RGB2BGR))

    # 06: Combined figure
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(rgb_uint8)
    axes[0].set_title(f"RGB ({H}x{W})")
    axes[1].imshow(plane_2d, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Plane mask ({pct:.1f}%)")
    axes[2].imshow(overlay)
    axes[2].set_title("Plane overlay")
    axes[3].imshow(depth_vis)
    axes[3].set_title("Depth")
    axes[4].imshow(sem_vis)
    axes[4].set_title("Semantic")
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f"{scene_id} / frame {frame_idx}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_combined.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Verify PixelwisePlanarity training data pipeline")
    parser.add_argument("--dataset", required=True, choices=["scannetpp", "hypersim"],
                        help="Dataset to verify")
    parser.add_argument("--resolution", nargs=2, type=int, default=None, metavar=("H", "W"),
                        help="Target resolution H W (default: 476 644)")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Filter to specific scene IDs (e.g. 0a5c013435 ai_001_001)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to visualize per scene (default: 10)")
    parser.add_argument("--output_dir", default="output/training_data_verify/",
                        help="Output directory")
    parser.add_argument("--split", default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--sample_indices", nargs="+", type=int, default=None,
                        help="Specific dataset indices to visualize (overrides --num_samples)")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sample every N-th item (for large datasets)")

    # ScanNet++ paths
    parser.add_argument("--rgb_root", default=None)
    parser.add_argument("--plane_label_root", default=None)
    parser.add_argument("--sem_label_root", default=None)
    parser.add_argument("--depth_label_root", default=None)
    parser.add_argument("--split_txt_dir", default=None)

    # Hypersim paths
    parser.add_argument("--hypersim_root", default=None)
    parser.add_argument("--hypersim_plane_root", default=None)
    parser.add_argument("--hypersim_csv", default=None)

    args = parser.parse_args()

    # Resolve resolution
    if args.resolution is not None:
        target_h, target_w = args.resolution
    else:
        target_h, target_w = DEFAULT_RESOLUTION

    # Build dataset
    if args.dataset == "scannetpp":
        rgb_root = args.rgb_root or "/cluster/project/cvg/Shared_datasets/scannet++/data"
        plane_label_root = args.plane_label_root or "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
        sem_label_root = args.sem_label_root or "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
        depth_label_root = args.depth_label_root or "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
        split_txt_dir = args.split_txt_dir or "/cluster/home/ayavuz/PixelwisePlanarity/planamono/splits/scannetpp"

        dataset = ScanNetPPPlanarityDataset(
            rgb_root=rgb_root,
            plane_label_root=plane_label_root,
            sem_label_root=sem_label_root,
            depth_label_root=depth_label_root,
            split_txt_dir=split_txt_dir,
            split=args.split,
            image_height=target_h,
            image_width=target_w,
        )

    elif args.dataset == "hypersim":
        root_dir = args.hypersim_root or "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
        plane_root = args.hypersim_plane_root or "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
        csv_path = args.hypersim_csv or "/cluster/home/ayavuz/PixelwisePlanarity/planamono/splits/hypersim/metadata_images_split_with_planes_filtered.csv"

        dataset = HypersimPlanarityDataset(
            root_dir=root_dir,
            plane_label_root=plane_root,
            filtered_csv_path=csv_path,
            split=args.split,
            image_height=target_h,
            image_width=target_w,
        )

    print(f"\nDataset: {args.dataset}, split: {args.split}, "
          f"resolution: {target_h}x{target_w}, size: {len(dataset)}")

    # Filter to specific scenes if requested
    if args.scenes is not None:
        scene_set = set(args.scenes)
        if args.dataset == "scannetpp":
            # valid_pairs: (rgb_path, plane_h5, sem_h5, depth_h5, frame_idx)
            # scene_id is 4 dirs up from rgb_path
            scene_indices = [
                i for i, (rgb_path, *_) in enumerate(dataset.valid_pairs)
                if rgb_path.split("/")[-4] in scene_set
            ]
        elif args.dataset == "hypersim":
            # valid_pairs: (scene_id, frame_id, cam_name, ...)
            scene_indices = [
                i for i, (scene_id, *_) in enumerate(dataset.valid_pairs)
                if scene_id in scene_set
            ]
        print(f"Filtered to scenes {args.scenes}: {len(scene_indices)} samples")
    else:
        scene_indices = None

    # Determine which indices to visualize
    if args.sample_indices is not None:
        indices = args.sample_indices
    elif scene_indices is not None:
        # Use scene-filtered indices, take up to num_samples evenly spaced
        if len(scene_indices) <= args.num_samples:
            indices = scene_indices
        else:
            step = max(1, len(scene_indices) // args.num_samples)
            indices = scene_indices[::step][:args.num_samples]
    elif args.stride is not None:
        indices = list(range(0, len(dataset), args.stride))[:args.num_samples]
    else:
        step = max(1, len(dataset) // args.num_samples)
        indices = list(range(0, len(dataset), step))[:args.num_samples]

    print(f"Visualizing {len(indices)} samples: {indices}\n")

    for i, idx in enumerate(indices):
        if idx >= len(dataset):
            print(f"[SKIP] Index {idx} out of range (dataset size: {len(dataset)})")
            continue

        sample = dataset[idx]
        scene_id = sample["scene_id"]
        frame_idx = sample["frame_idx"]
        print(f"[{i+1}/{len(indices)}] idx={idx}, {scene_id}/frame_{frame_idx}")

        try:
            process_sample(sample, args.output_dir, idx)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
