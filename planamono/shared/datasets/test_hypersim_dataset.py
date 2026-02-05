#!/usr/bin/env python3
"""
Test script for HypersimPlaneDataset.

Usage:
    python test_hypersim_dataset.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from torch.utils.data import DataLoader


def test_dataset():
    """Test HypersimPlaneDataset initialization and data loading."""

    # Configuration - update these paths according to user's data
    hypersim_root = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
    plane_label_root = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
    params_root = "/cluster/scratch/ayavuz/dataset/Hypersim_params"
    split_txt_dir = str(project_root / "splits" / "hypersim")

    print("=" * 80)
    print("Testing HypersimPlaneDataset")
    print("=" * 80)

    print(f"\nPaths:")
    print(f"  Hypersim root: {hypersim_root}")
    print(f"  Plane label root: {plane_label_root}")
    print(f"  Params root: {params_root}")
    print(f"  Split txt dir: {split_txt_dir}")

    # Test with validation split and limited scenes
    print("\n" + "=" * 80)
    print("Initializing dataset (val split, max 2 scenes)...")
    print("=" * 80)

    try:
        dataset = HypersimPlaneDataset(
            hypersim_root=hypersim_root,
            plane_label_root=plane_label_root,
            params_root=params_root,
            split_txt_dir=split_txt_dir,
            split='val',
            image_height=768,
            image_width=1024,
            max_scenes=2
        )

        print(f"\n✓ Dataset initialized successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Scenes: {len(dataset.scene_ids)}")
        print(f"  Scene IDs: {dataset.scene_ids}")

    except Exception as e:
        print(f"\n✗ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading a few samples
    print("\n" + "=" * 80)
    print("Testing sample loading...")
    print("=" * 80)

    num_test_samples = min(3, len(dataset))
    for i in range(num_test_samples):
        print(f"\n--- Sample {i} ---")
        try:
            sample = dataset[i]

            print(f"  Scene ID: {sample['scene_id']}")
            print(f"  Frame ID: {sample['frame_idx']}")
            print(f"  RGB path: {sample['rgb_path']}")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Depth shape: {sample['depth'].shape}")
            print(f"  Plane shape: {sample['plane'].shape}")
            print(f"  Semantic shape: {sample['sem'].shape}")
            print(f"  K shape: {sample['K'].shape}")
            print(f"  c2w shape: {sample['c2w'].shape}")

            # Check data types
            print(f"  Image dtype: {sample['image'].dtype}, range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            print(f"  Depth dtype: {sample['depth'].dtype}, range: [{sample['depth'].min():.3f}, {sample['depth'].max():.3f}]")
            print(f"  Plane dtype: {sample['plane'].dtype}, unique values: {sample['plane'].unique().shape[0]}")

            print(f"  ✓ Sample {i} loaded successfully!")

        except Exception as e:
            print(f"  ✗ Failed to load sample {i}: {e}")
            import traceback
            traceback.print_exc()

    # Test DataLoader
    print("\n" + "=" * 80)
    print("Testing DataLoader...")
    print("=" * 80)

    try:
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )

        batch = next(iter(loader))
        print(f"\n  Batch keys: {list(batch.keys())}")
        print(f"  Batch size: {batch['image'].shape[0]}")
        print(f"  Image batch shape: {batch['image'].shape}")
        print(f"  Depth batch shape: {batch['depth'].shape}")
        print(f"  Plane batch shape: {batch['plane'].shape}")
        print(f"  ✓ DataLoader works!")

    except Exception as e:
        print(f"\n  ✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset()
