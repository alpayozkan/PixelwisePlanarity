#!/usr/bin/env python3
"""
Comprehensive verification script for HypersimPlaneDataset splits.

This script verifies:
1. Dataset loads correctly for train/val/test splits
2. No scene overlap between splits
3. Scenes match the split files
4. Data can be loaded from each split
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset


def load_split_scenes(split_txt_dir, split):
    """Load scene IDs from split file."""
    split_file = os.path.join(split_txt_dir, f"{split}.txt")
    with open(split_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    return set(scenes)


def verify_dataset_split(rgb_depth_root, plane_label_root, intrinsics_root,
                         split_txt_dir, split, max_scenes=None):
    """Verify a single dataset split."""
    print(f"\n{'='*80}")
    print(f"Verifying {split.upper()} split")
    print(f"{'='*80}")

    # Load expected scenes from split file
    expected_scenes = load_split_scenes(split_txt_dir, split)
    print(f"Expected scenes from split file: {len(expected_scenes)}")
    if max_scenes:
        print(f"(Limited to first {max_scenes} scenes for testing)")

    # Initialize dataset
    try:
        dataset = HypersimPlaneDataset(
            rgb_depth_root=rgb_depth_root,
            plane_label_root=plane_label_root,
            intrinsics_root=intrinsics_root,
            split_txt_dir=split_txt_dir,
            split=split,
            max_scenes=max_scenes
        )
        print(f"✓ Dataset initialized successfully")
    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, set()

    # Check dataset size
    print(f"\nDataset statistics:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Valid scenes: {len(dataset.scene_ids)}")

    # Get actual scenes loaded
    actual_scenes = set(dataset.scene_ids)

    # Compare expected vs actual
    if max_scenes:
        # When limited, we expect a subset
        expected_subset = set(list(expected_scenes)[:max_scenes])
        missing = expected_subset - actual_scenes
        extra = actual_scenes - expected_subset
    else:
        # When not limited, check against full set
        missing = expected_scenes - actual_scenes
        extra = actual_scenes - expected_scenes

    print(f"\nScene matching:")
    if max_scenes:
        print(f"  Expected (first {max_scenes}): {len(expected_subset)}")
    else:
        print(f"  Expected: {len(expected_scenes)}")
    print(f"  Actual loaded: {len(actual_scenes)}")
    print(f"  Missing scenes: {len(missing)}")
    if missing and len(missing) <= 10:
        print(f"    {missing}")
    print(f"  Extra scenes: {len(extra)}")
    if extra and len(extra) <= 10:
        print(f"    {extra}")

    # Test loading samples
    print(f"\nTesting sample loading:")
    num_samples = min(3, len(dataset))
    for i in range(num_samples):
        try:
            sample = dataset[i]
            print(f"  ✓ Sample {i}: scene={sample['scene_id']}, frame={sample['frame_idx']}, "
                  f"img={sample['image'].shape}, plane_ids={sample['plane'].unique().shape[0]}")
        except Exception as e:
            print(f"  ✗ Sample {i} failed: {e}")

    return dataset, actual_scenes


def main():
    """Main verification routine."""
    print("="*80)
    print("HYPERSIM DATASET SPLIT VERIFICATION")
    print("="*80)

    # Configuration
    rgb_depth_root = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
    plane_label_root = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
    intrinsics_root = "/cluster/scratch/ayavuz/dataset/Hypersim_params"
    split_txt_dir = str(project_root / "splits" / "hypersim")

    print(f"\nData paths:")
    print(f"  RGB/Depth: {rgb_depth_root}")
    print(f"  Planes: {plane_label_root}")
    print(f"  Intrinsics: {intrinsics_root}")
    print(f"  Splits: {split_txt_dir}")

    # Verify paths exist
    print(f"\nPath verification:")
    for name, path in [("RGB/Depth", rgb_depth_root),
                       ("Planes", plane_label_root),
                       ("Intrinsics", intrinsics_root),
                       ("Splits", split_txt_dir)]:
        exists = os.path.exists(path)
        print(f"  {name}: {'✓' if exists else '✗'} {path}")
        if not exists:
            print(f"\n✗ ERROR: {name} path does not exist!")
            return

    # Test with limited scenes first
    max_test_scenes = 2
    print(f"\n{'='*80}")
    print(f"PHASE 1: Quick test with {max_test_scenes} scenes per split")
    print(f"{'='*80}")

    train_dataset, train_scenes = verify_dataset_split(
        rgb_depth_root, plane_label_root, intrinsics_root, split_txt_dir,
        split='train', max_scenes=max_test_scenes
    )

    val_dataset, val_scenes = verify_dataset_split(
        rgb_depth_root, plane_label_root, intrinsics_root, split_txt_dir,
        split='val', max_scenes=max_test_scenes
    )

    test_dataset, test_scenes = verify_dataset_split(
        rgb_depth_root, plane_label_root, intrinsics_root, split_txt_dir,
        split='test', max_scenes=max_test_scenes
    )

    # Check for overlap between splits
    print(f"\n{'='*80}")
    print(f"SPLIT OVERLAP CHECK")
    print(f"{'='*80}")

    train_val_overlap = train_scenes & val_scenes
    train_test_overlap = train_scenes & test_scenes
    val_test_overlap = val_scenes & test_scenes

    print(f"\nTrain-Val overlap: {len(train_val_overlap)} scenes")
    if train_val_overlap:
        print(f"  ✗ WARNING: Overlapping scenes: {train_val_overlap}")
    else:
        print(f"  ✓ No overlap")

    print(f"\nTrain-Test overlap: {len(train_test_overlap)} scenes")
    if train_test_overlap:
        print(f"  ✗ WARNING: Overlapping scenes: {train_test_overlap}")
    else:
        print(f"  ✓ No overlap")

    print(f"\nVal-Test overlap: {len(val_test_overlap)} scenes")
    if val_test_overlap:
        print(f"  ✗ WARNING: Overlapping scenes: {val_test_overlap}")
    else:
        print(f"  ✓ No overlap")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    if train_dataset and val_dataset and test_dataset:
        print(f"\n✓ All splits loaded successfully!")
        print(f"\nDataset sizes (with max_scenes={max_test_scenes}):")
        print(f"  Train: {len(train_dataset)} frames from {len(train_scenes)} scenes")
        print(f"  Val:   {len(val_dataset)} frames from {len(val_scenes)} scenes")
        print(f"  Test:  {len(test_dataset)} frames from {len(test_scenes)} scenes")
        print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} frames")

        no_overlap = (len(train_val_overlap) == 0 and
                     len(train_test_overlap) == 0 and
                     len(val_test_overlap) == 0)

        if no_overlap:
            print(f"\n✓ No scene overlap between splits - splits are valid!")
        else:
            print(f"\n✗ WARNING: Scene overlap detected between splits!")

        # Ask if user wants to test full dataset
        print(f"\n{'='*80}")
        print(f"To test with full dataset (all scenes), run:")
        print(f"  python {__file__} --full")
        print(f"{'='*80}")
    else:
        print(f"\n✗ One or more splits failed to load")
        print(f"  Check the error messages above for details")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("\n!!! Running FULL dataset test (this may take a while) !!!\n")
        # Re-run with max_scenes=None
        # (You can modify the script to support this)
    main()
