#!/usr/bin/env python3
"""
Segmentation Prediction Script

Runs full pipeline: RGB → MoGe (planarity + depth + normals) → plane segmentation.

Usage:
    python predict.py --model_path /path/to/model.pt --input_root /path/to/scannet --output_root /path/to/output
"""
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import argparse
import numpy as np
import glob
import cv2
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pxwplanar.shared.segmentation import compute_vectorized_planar_segments
from pxwplanar.shared.utils.label_utils import remap_labels
from pxwplanar.shared.utils import visualize_top_components
from pxwplanar.inference.planarity.moge_inference import MoGePlanarityInference


def process_scene(scene_id, image_list, inference_model, output_dir, args):
    """Process a single scene."""
    seg_dir = os.path.join(output_dir, 'seg_pred', scene_id)
    os.makedirs(seg_dir, exist_ok=True)

    segvis_dir = os.path.join(output_dir, 'seg_vis', scene_id)
    os.makedirs(segvis_dir, exist_ok=True)

    for idx in range(0, len(image_list), args.frame_skip):
        image_path = image_list[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        # Run MoGe inference
        res = inference_model.predict(image_path, num_tokens=args.num_tokens, return_all_heads=True)

        depth = res['points'][:, :, 2]
        normal = res['normal']  # (H, W, 3)
        planarity = res['planarity_probability']

        # Resize to original resolution
        depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        normal = cv2.resize(normal.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        planarity = cv2.resize(planarity.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

        # Create planarity mask
        planarity_mask = (planarity > args.threshold_planarity).astype(np.int16)

        # Compute segmentation
        normal_threshold_rad = np.deg2rad(args.normal_threshold_deg)
        labels, n_components = compute_vectorized_planar_segments(
            planarity_mask, normal, depth,
            normal_threshold_rad, args.depth_threshold,
            neighbor_match_count_thresh=args.neighbor_match_count_thresh
        )
        filtered_segmentation = labels.copy()
        filtered_segmentation, _ = remap_labels(filtered_segmentation)

        # Save segmentation
        seg_save_path = os.path.join(seg_dir, f"{base_name}_seg_pred.npy")
        np.save(seg_save_path, filtered_segmentation)

        # Save visualization
        if args.save_visualization:
            segvis_save_path = os.path.join(segvis_dir, f"{base_name}_seg_vis.png")

            fig, axs = plt.subplots(1, 7, figsize=(16, 4))
            fig.suptitle(f"Scene: {scene_id} | {base_name}", fontsize=12)

            axs[0].imshow(img_np)
            axs[0].set_title("Image")
            axs[0].axis('off')

            axs[1].imshow(depth, cmap='viridis')
            axs[1].set_title("Depth")
            axs[1].axis('off')

            axs[2].imshow(np.transpose(normal, (1, 2, 0)) * 0.5 + 0.5)
            axs[2].set_title("Normal")
            axs[2].axis('off')

            axs[3].imshow(planarity, cmap='gray', vmin=0, vmax=1)
            axs[3].set_title("Planarity")
            axs[3].axis('off')

            axs[4].imshow(planarity_mask, cmap='gray')
            axs[4].set_title("Planarity Mask")
            axs[4].axis('off')

            n = len(np.unique(filtered_segmentation))
            n1 = min(10, n)
            seg_vis = visualize_top_components(filtered_segmentation, k=n1, return_colors=True)
            axs[5].imshow(seg_vis)
            axs[5].set_title(f"Seg: Top-{n1}")
            axs[5].axis('off')

            n2 = max(n // 2, 1)
            seg_vis2 = visualize_top_components(filtered_segmentation, k=n2, return_colors=True)
            axs[6].imshow(seg_vis2)
            axs[6].set_title(f"Seg: Top-{n2}")
            axs[6].axis('off')

            plt.tight_layout()
            fig.savefig(segvis_save_path, dpi=100)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run segmentation prediction on ScanNet scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required paths
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained MoGe checkpoint (.pt file)")
    parser.add_argument("--input_root", type=str, required=True,
                        help="Root directory of dataset (contains scene folders)")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for output")

    # Model configuration
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache dir for MoGe base weights (sets HF_HOME)")

    # Processing parameters
    parser.add_argument("--frame_skip", type=int, default=50,
                        help="Process every Nth frame (default: 50)")
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Limit number of scenes to process")

    # Segmentation parameters
    parser.add_argument("--threshold_planarity", type=float, default=0.3)
    parser.add_argument("--normal_threshold_deg", type=float, default=5.0)
    parser.add_argument("--depth_threshold", type=float, default=0.025,
                        help="Relative depth threshold (fraction of center depth)")
    parser.add_argument("--neighbor_match_count_thresh", type=int, default=8)

    # Output options
    parser.add_argument("--save_visualization", action="store_true",
                        help="Save visualization images")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    if not os.path.isdir(args.input_root):
        print(f"[ERROR] Input directory not found: {args.input_root}")
        sys.exit(1)

    os.makedirs(args.output_root, exist_ok=True)

    print("=" * 60)
    print("Segmentation Prediction")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_root}")
    print(f"Output: {args.output_root}")
    print(f"Frame skip: {args.frame_skip}")
    print("-" * 60)

    # Load model (base MoGe weights come from HuggingFace; cache via HF_HOME)
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
    print("[INFO] Loading model...")
    inference_model = MoGePlanarityInference(
        model_path=args.model_path,
        device=args.device
    )

    print("[INFO] Model loaded")
    print("-" * 60)

    # Find scenes
    scene_id_list = [name for name in os.listdir(args.input_root)
                     if os.path.isdir(os.path.join(args.input_root, name))]
    scene_id_list = natsorted(scene_id_list)

    if len(scene_id_list) == 0:
        print(f"[ERROR] No scene folders found in {args.input_root}")
        sys.exit(1)

    if args.max_scenes:
        scene_id_list = scene_id_list[:args.max_scenes]

    print(f"[INFO] Found {len(scene_id_list)} scenes")

    # Process scenes
    for scene_id in tqdm(scene_id_list, desc="Processing scenes"):
        color_dir = os.path.join(args.input_root, scene_id, 'color')
        if not os.path.isdir(color_dir):
            print(f"[WARN] No color folder for {scene_id}, skipping")
            continue

        image_list = natsorted(glob.glob(os.path.join(color_dir, '*.jpg')))
        if len(image_list) == 0:
            print(f"[WARN] No images found for {scene_id}, skipping")
            continue

        process_scene(scene_id, image_list, inference_model, args.output_root, args)

    print("=" * 60)
    print(f"[SUCCESS] Processed {len(scene_id_list)} scenes")
    print(f"[INFO] Results saved to: {args.output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
