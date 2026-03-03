#!/usr/bin/env python3
"""
Generate a PDF with 20 sample slides from Hypersim dataset.
Each slide shows: RGB, Depth, Plane Segmentation, Semantics.

Usage:
    python generate_hypersim_samples_pdf.py
    python generate_hypersim_samples_pdf.py --n-samples 50 --split train
    python generate_hypersim_samples_pdf.py --output my_samples.pdf
"""

import sys
import os
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset


def create_sample_slide(sample, slide_idx):
    """
    Create a single slide with 4 subplots: RGB, Depth, Plane, Semantics.

    Args:
        sample: Dataset sample dictionary
        slide_idx: Slide number for title

    Returns:
        matplotlib figure
    """
    # Extract data
    rgb = sample['image'].permute(1, 2, 0).numpy()  # (H, W, 3)
    depth = sample['depth'][0].numpy()  # (H, W)
    plane = sample['plane'][0].numpy()  # (H, W)
    sem = sample['sem'][0].numpy()  # (H, W)

    scene_id = sample['scene_id']
    frame_idx = sample['frame_idx']

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Sample {slide_idx + 1}: {scene_id} - Frame {frame_idx}',
                 fontsize=16, fontweight='bold')

    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Add RGB stats as text
    rgb_text = f'Range: [{rgb.min():.3f}, {rgb.max():.3f}]\nMean: {rgb.mean():.3f}'
    axes[0, 0].text(0.02, 0.98, rgb_text, transform=axes[0, 0].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Depth
    depth_vis = axes[0, 1].imshow(depth, cmap='plasma')
    axes[0, 1].set_title('Depth Map', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(depth_vis, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Add depth stats
    depth_text = f'Range: [{depth.min():.2f}, {depth.max():.2f}] m\nMean: {depth.mean():.2f} m'
    axes[0, 1].text(0.02, 0.98, depth_text, transform=axes[0, 1].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plane Segmentation
    n_planes = len(np.unique(plane)) - 1  # Exclude background (0)
    plane_vis = axes[1, 0].imshow(plane, cmap='tab20', vmin=0, vmax=20)
    axes[1, 0].set_title(f'Plane Segmentation ({n_planes} planes)',
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(plane_vis, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Add plane stats
    plane_text = f'Unique planes: {n_planes}\nMin ID: {plane.min()}\nMax ID: {plane.max()}'
    axes[1, 0].text(0.02, 0.98, plane_text, transform=axes[1, 0].transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Semantic Segmentation
    n_semantic = len(np.unique(sem))
    sem_vis = axes[1, 1].imshow(sem, cmap='tab20')
    axes[1, 1].set_title(f'Semantic Segmentation ({n_semantic} classes)',
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(sem_vis, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Add semantic note
    if n_semantic <= 1:
        sem_text = 'Not available for this dataset'
        axes[1, 1].text(0.02, 0.98, sem_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    return fig


def generate_pdf(output_path, n_samples=20, split='val', max_scenes=None):
    """
    Generate a multi-page PDF with sample slides.

    Args:
        output_path: Path to save PDF
        n_samples: Number of samples to include
        split: Dataset split ('train', 'val', 'test')
        max_scenes: Maximum number of scenes to load (None = all)
    """
    print("="*80)
    print(f"Generating Hypersim Sample PDF")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Samples: {n_samples}")
    print(f"Split: {split}")
    print()

    # Dataset configuration
    hypersim_root = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
    plane_label_root = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
    params_root = "/cluster/scratch/ayavuz/dataset/Hypersim_params"
    split_txt_dir = str(project_root / "planamono" / "splits" / "hypersim")

    # Load dataset
    print(f"Loading dataset...")
    dataset = HypersimPlaneDataset(
        hypersim_root=hypersim_root,
        plane_label_root=plane_label_root,
        params_root=params_root,
        split_txt_dir=split_txt_dir,
        split=split,
        max_scenes=max_scenes
    )

    print(f"Dataset loaded: {len(dataset)} samples from {len(dataset.scene_ids)} scenes")
    print()

    # Limit to requested number of samples
    n_samples = min(n_samples, len(dataset))

    # Sample indices (evenly spaced to get diverse samples)
    if len(dataset) > n_samples:
        sample_indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)
    else:
        sample_indices = list(range(len(dataset)))

    # Generate PDF
    print(f"Generating {n_samples} slides...")
    with PdfPages(output_path) as pdf:
        for i, idx in enumerate(sample_indices):
            print(f"  Processing slide {i+1}/{n_samples} (dataset index {idx})...", end='')

            try:
                # Load sample
                sample = dataset[idx]

                # Create slide
                fig = create_sample_slide(sample, i)

                # Save to PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                print(" ✓")

            except Exception as e:
                print(f" ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Add metadata
        d = pdf.infodict()
        d['Title'] = f'Hypersim Dataset Samples ({split} split)'
        d['Author'] = 'Hypersim Plane Dataset'
        d['Subject'] = f'{n_samples} sample frames with RGB, Depth, Plane Segmentation, Semantics'
        d['Keywords'] = 'Hypersim, Plane Segmentation, Dataset Visualization'

    print()
    print("="*80)
    print(f"✓ PDF generated successfully: {output_path}")
    print(f"  Total slides: {n_samples}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate PDF with Hypersim dataset sample slides'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='hypersim_samples.pdf',
        help='Output PDF path (default: hypersim_samples.pdf)'
    )
    parser.add_argument(
        '--n-samples', '-n',
        type=int,
        default=20,
        help='Number of samples to include (default: 20)'
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        choices=['train', 'val', 'test'],
        default='val',
        help='Dataset split (default: val)'
    )
    parser.add_argument(
        '--max-scenes',
        type=int,
        default=None,
        help='Maximum number of scenes to load (default: None = all)'
    )

    args = parser.parse_args()

    # Generate PDF
    generate_pdf(
        output_path=args.output,
        n_samples=args.n_samples,
        split=args.split,
        max_scenes=args.max_scenes
    )


if __name__ == "__main__":
    main()
