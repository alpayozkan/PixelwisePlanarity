#!/usr/bin/env python3
"""
Generate PDF visualization of predictions with RGB, Depth, GT Planes, and Predicted Planes.

Each page shows one frame with 4 panels:
- RGB image
- Depth map
- Ground truth plane segmentation
- Predicted plane segmentation

Usage:
    # Visualize 10 random samples from all scenes
    python visualize_predictions_pdf.py --n-samples 10

    # Visualize 5 samples per scene from 3 scenes
    python visualize_predictions_pdf.py --n-samples 5 --n-scenes 3

    # Visualize specific scene
    python visualize_predictions_pdf.py --scene 0d2ee665be --n-samples 10

    # Visualize all frames from a specific scene
    python visualize_predictions_pdf.py --scene 0d2ee665be --all-frames

    # Custom output path
    python visualize_predictions_pdf.py --n-samples 20 --output my_vis.pdf
"""

import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from PIL import Image

from planamono.paths import scannetpp_path, scannetpp_rend_plane_path

# Adjust these paths based on your setup
PRED_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_mixed_bce_h5")
GT_ROOT = Path(scannetpp_rend_plane_path)
RGB_ROOT = Path(os.path.join(scannetpp_path, "data"))
OUTPUT_DIR = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/visualizations")


def load_h5_data(h5_path: Path, frame_idx: str) -> Optional[np.ndarray]:
    """Load data from H5 file for a specific frame."""
    if not h5_path.exists():
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            frame_ids = [fid.decode() if isinstance(fid, bytes) else fid
                        for fid in f["frame_ids"][:]]

            if frame_idx not in frame_ids:
                return None

            idx = frame_ids.index(frame_idx)

            # Load appropriate dataset
            if "planes" in f:
                data = f["planes"][idx]
            elif "depth" in f:
                data = f["depth"][idx]
            else:
                return None

            return data
    except Exception as e:
        print(f"Error loading {h5_path}: {e}")
        return None


def load_rgb(scene_id: str, frame_idx: str, rgb_root: Path = RGB_ROOT) -> Optional[np.ndarray]:
    """Load RGB image."""
    # Try both .jpg and .JPG extensions
    for ext in ['.jpg', '.JPG']:
        rgb_path = rgb_root / scene_id / "iphone" / "rgb" / f"{frame_idx}{ext}"
        if rgb_path.exists():
            try:
                img = cv2.imread(str(rgb_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
            except Exception as e:
                print(f"Error loading RGB {rgb_path}: {e}")
                continue

    return None


def colorize_segmentation(seg: np.ndarray, max_colors: int = 256) -> np.ndarray:
    """Colorize segmentation map with distinct colors."""
    # Create a colormap
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, size=(max_colors, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    # Map labels to colors
    h, w = seg.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(seg)
    for label in unique_labels:
        if label < max_colors:
            colored[seg == label] = colors[label]

    return colored


def colorize_depth(depth: np.ndarray, vmin: Optional[float] = None,
                   vmax: Optional[float] = None) -> np.ndarray:
    """Colorize depth map using viridis colormap."""
    # Handle invalid depth values
    valid_mask = (depth > 0) & np.isfinite(depth)

    if not valid_mask.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    # Compute percentiles for robust visualization
    if vmin is None:
        vmin = np.percentile(depth[valid_mask], 1)
    if vmax is None:
        vmax = np.percentile(depth[valid_mask], 99)

    # Normalize
    depth_norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    depth_norm[~valid_mask] = 0

    # Apply colormap
    cmap = plt.get_cmap('viridis')
    colored = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    return colored


def visualize_frame(rgb: np.ndarray, depth: np.ndarray,
                   gt_planes: np.ndarray, pred_planes: np.ndarray,
                   scene_id: str, frame_idx: str,
                   fig: Optional[plt.Figure] = None) -> plt.Figure:
    """Create a 1x4 visualization of one frame."""
    if fig is None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        axes = fig.subplots(1, 4)

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title(f'RGB\n{scene_id} - {frame_idx}', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Depth
    depth_colored = colorize_depth(depth)
    axes[1].imshow(depth_colored)
    axes[1].set_title('Depth (GT)', fontsize=10, fontweight='bold')
    axes[1].axis('off')

    # GT Planes
    # Resize GT to match RGB resolution for fair comparison
    h, w = rgb.shape[:2]
    gt_planes_resized = cv2.resize(gt_planes.astype(np.uint16), (w, h),
                                   interpolation=cv2.INTER_NEAREST)
    gt_colored = colorize_segmentation(gt_planes_resized)
    axes[2].imshow(gt_colored)
    n_gt_planes = len(np.unique(gt_planes_resized)) - 1  # Exclude background
    axes[2].set_title(f'GT Planes (n={n_gt_planes})', fontsize=10, fontweight='bold')
    axes[2].axis('off')

    # Predicted Planes
    pred_planes_resized = cv2.resize(pred_planes.astype(np.uint16), (w, h),
                                     interpolation=cv2.INTER_NEAREST)
    pred_colored = colorize_segmentation(pred_planes_resized)
    axes[3].imshow(pred_colored)
    n_pred_planes = len(np.unique(pred_planes_resized)) - 1  # Exclude background
    axes[3].set_title(f'Predicted Planes (n={n_pred_planes})', fontsize=10, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    return fig


def get_available_scenes(pred_root: Path) -> List[str]:
    """Get list of scenes with prediction results."""
    if not pred_root.exists():
        return []

    scenes = [d.name for d in pred_root.iterdir() if d.is_dir()]
    scenes.sort()
    return scenes


def get_scene_frames(scene_id: str, pred_root: Path) -> List[str]:
    """Get list of frames available for a scene."""
    pred_h5 = pred_root / scene_id / "planes.h5"

    if not pred_h5.exists():
        return []

    try:
        with h5py.File(pred_h5, "r") as f:
            frame_ids = [fid.decode() if isinstance(fid, bytes) else fid
                        for fid in f["frame_ids"][:]]
        return frame_ids
    except Exception as e:
        print(f"Error reading frames from {pred_h5}: {e}")
        return []


def sample_frames(scenes: List[str], n_samples: int, n_scenes: Optional[int] = None,
                 pred_root: Path = PRED_ROOT, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Sample frames from scenes.

    Args:
        scenes: List of scene IDs
        n_samples: Total number of samples (or samples per scene if n_scenes is specified)
        n_scenes: Number of scenes to sample from (None = all scenes)
        pred_root: Root directory of predictions
        seed: Random seed

    Returns:
        List of (scene_id, frame_idx) tuples
    """
    random.seed(seed)

    # Select scenes
    if n_scenes is not None and n_scenes < len(scenes):
        scenes = random.sample(scenes, n_scenes)

    samples = []

    if n_scenes is not None:
        # Sample n_samples per scene
        for scene_id in scenes:
            frames = get_scene_frames(scene_id, pred_root)
            if not frames:
                continue

            n_to_sample = min(n_samples, len(frames))
            sampled_frames = random.sample(frames, n_to_sample)
            samples.extend([(scene_id, frame) for frame in sampled_frames])
    else:
        # Sample n_samples total across all scenes
        # Build pool of all (scene, frame) pairs
        pool = []
        for scene_id in scenes:
            frames = get_scene_frames(scene_id, pred_root)
            pool.extend([(scene_id, frame) for frame in frames])

        if not pool:
            return []

        n_to_sample = min(n_samples, len(pool))
        samples = random.sample(pool, n_to_sample)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF visualization of predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--pred-root", type=Path, default=PRED_ROOT,
                       help="Root directory of predictions")
    parser.add_argument("--gt-root", type=Path, default=GT_ROOT,
                       help="Root directory of ground truth")
    parser.add_argument("--rgb-root", type=Path, default=RGB_ROOT,
                       help="Root directory of RGB images")

    parser.add_argument("--n-samples", type=int, default=10,
                       help="Number of samples (total or per scene if --n-scenes is used)")
    parser.add_argument("--n-scenes", type=int, default=None,
                       help="Number of scenes to sample from (None = all scenes)")
    parser.add_argument("--scene", type=str, default=None,
                       help="Specific scene ID to visualize")
    parser.add_argument("--all-frames", action="store_true",
                       help="Visualize all frames from the scene (use with --scene)")

    parser.add_argument("--output", type=Path, default=None,
                       help="Output PDF path (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")

    args = parser.parse_args()

    # Use paths from args
    pred_root = args.pred_root
    gt_root = args.gt_root
    rgb_root = args.rgb_root

    # Get scenes to process
    if args.scene:
        scenes = [args.scene]
        if args.all_frames:
            frames = get_scene_frames(args.scene, pred_root)
            samples = [(args.scene, frame) for frame in frames]
        else:
            samples = sample_frames(scenes, args.n_samples, n_scenes=None,
                                   pred_root=pred_root, seed=args.seed)
    else:
        available_scenes = get_available_scenes(pred_root)
        if not available_scenes:
            print(f"No scenes found in {pred_root}")
            return

        print(f"Found {len(available_scenes)} scenes with predictions")
        samples = sample_frames(available_scenes, args.n_samples,
                               n_scenes=args.n_scenes, pred_root=pred_root,
                               seed=args.seed)

    if not samples:
        print("No samples to visualize!")
        return

    print(f"Generating visualizations for {len(samples)} frames...")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.scene:
            suffix = f"{args.scene}_{len(samples)}frames"
        elif args.n_scenes:
            suffix = f"{args.n_scenes}scenes_{args.n_samples}samples_per_scene"
        else:
            suffix = f"{len(samples)}samples"
        output_path = OUTPUT_DIR / f"predictions_{suffix}.pdf"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate PDF
    skipped = 0
    with PdfPages(output_path) as pdf:
        for scene_id, frame_idx in tqdm(samples, desc="Generating PDF"):
            # Load data
            rgb = load_rgb(scene_id, frame_idx, rgb_root)
            if rgb is None:
                print(f"Skipping {scene_id}/{frame_idx}: RGB not found")
                skipped += 1
                continue

            # Load depth
            depth_h5 = gt_root / scene_id / "rendered_depth.h5"
            depth = load_h5_data(depth_h5, frame_idx)
            if depth is None:
                print(f"Skipping {scene_id}/{frame_idx}: Depth not found")
                skipped += 1
                continue

            # Load GT planes
            gt_h5 = gt_root / scene_id / "rendered.h5"
            gt_planes = load_h5_data(gt_h5, frame_idx)
            if gt_planes is None:
                print(f"Skipping {scene_id}/{frame_idx}: GT planes not found")
                skipped += 1
                continue

            # Load predicted planes
            pred_h5 = pred_root / scene_id / "planes.h5"
            pred_planes = load_h5_data(pred_h5, frame_idx)
            if pred_planes is None:
                print(f"Skipping {scene_id}/{frame_idx}: Predicted planes not found")
                skipped += 1
                continue

            # Create visualization
            fig = plt.figure(figsize=(20, 5))
            visualize_frame(rgb, depth, gt_planes, pred_planes,
                          scene_id, frame_idx, fig=fig)

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"\n{'='*60}")
    print(f"PDF saved to: {output_path}")
    print(f"Total pages: {len(samples) - skipped}")
    if skipped > 0:
        print(f"Skipped: {skipped} frames (missing data)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
