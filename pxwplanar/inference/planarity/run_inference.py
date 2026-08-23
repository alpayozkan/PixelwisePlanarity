#!/usr/bin/env python3
"""
MoGe Planarity Inference Runner

Runs planarity prediction on a directory of images using the MoGe 4-head model.

Usage:
    python run_inference.py --model_path /path/to/model.pt --input_dir /path/to/images --output_dir /path/to/output

MoGe base weights are pulled from HuggingFace into the standard cache
(~/.cache/huggingface; override via HF_HOME).
"""
import sys
from pathlib import Path

# Add project root to path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
from natsort import natsorted

from pxwplanar.inference.planarity.moge_inference import MoGePlanarityInference


def main():
    parser = argparse.ArgumentParser(
        description="Run MoGe planarity inference on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --model_path model.pt --input_dir ./images --output_dir ./results
        """
    )

    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save results")

    # Model configuration
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")

    # Output options
    parser.add_argument("--save_raw", action="store_true",
                        help="Save raw probability maps as .npy files")
    parser.add_argument("--save_binary", action="store_true",
                        help="Save binary masks as .png files")
    parser.add_argument("--save_visualization", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary mask (default: 0.5)")

    # Processing options
    parser.add_argument("--num_tokens", type=int, default=1600,
                        help="Number of tokens for the model (default 1600, matching the benchmark)")
    parser.add_argument("--extensions", type=str, default="jpg,jpeg,png",
                        help="Comma-separated image extensions to process (default: jpg,jpeg,png)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model file not found: {args.model_path}")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MoGe Planarity Inference")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("-" * 60)

    # Initialize model (base MoGe weights come from the standard HuggingFace cache)
    print("[INFO] Loading model...")
    model = MoGePlanarityInference(
        model_path=args.model_path,
        device=args.device
    )

    print("[INFO] Model loaded successfully")
    print("-" * 60)

    # Find all images
    extensions = args.extensions.split(",")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext}")))
        image_paths.extend(glob.glob(os.path.join(args.input_dir, f"*.{ext.upper()}")))

    image_paths = natsorted(list(set(image_paths)))

    if len(image_paths) == 0:
        print(f"[ERROR] No images found in {args.input_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} images")

    # Create subdirectories
    if args.save_raw:
        os.makedirs(os.path.join(args.output_dir, "raw"), exist_ok=True)
    if args.save_binary:
        os.makedirs(os.path.join(args.output_dir, "binary"), exist_ok=True)
    if args.save_visualization:
        os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)

    # Process images
    print("[INFO] Running inference...")
    for image_path in tqdm(image_paths, desc="Processing"):
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        try:
            results = model.predict(image_path, num_tokens=args.num_tokens)

            # Get full resolution results if available
            if 'planarity_probability_full' in results:
                prob = results['planarity_probability_full']
                binary = results['planarity_binary_full']
            else:
                prob = results['planarity_probability']
                binary = results['planarity_binary']

            # Save raw probability
            if args.save_raw:
                np.save(os.path.join(args.output_dir, "raw", f"{base_name}_planarity.npy"), prob)

            # Save binary mask
            if args.save_binary:
                import cv2
                binary_mask = (prob > args.threshold).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(args.output_dir, "binary", f"{base_name}_binary.png"), binary_mask)

            # Save visualization
            if args.save_visualization:
                import matplotlib
                matplotlib.use('Agg')
                fig = model.visualize_prediction(
                    image_path,
                    save_path=os.path.join(args.output_dir, "vis", f"{base_name}_vis.png"),
                    show_overlay=True
                )
                import matplotlib.pyplot as plt
                plt.close(fig)

        except Exception as e:
            print(f"[WARN] Failed to process {image_path}: {e}")
            continue

    print("=" * 60)
    print(f"[SUCCESS] Processed {len(image_paths)} images")
    print(f"[INFO] Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
