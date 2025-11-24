"""
Create train/validation/test splits for ScanNet++ and Hypersim datasets.

This script reads the full scene lists and creates standard train/val/test splits
with configurable ratios.
"""

import os
import random
from pathlib import Path


def create_splits(
    input_file,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Create train/val/test splits from a scene list.

    Args:
        input_file: Path to full scene list (one scene ID per line)
        output_dir: Directory to write split files
        train_ratio: Fraction of scenes for training
        val_ratio: Fraction of scenes for validation
        test_ratio: Fraction of scenes for testing
        seed: Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Read scene list
    with open(input_file, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(scenes)

    # Split
    n_total = len(scenes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_scenes = scenes[:n_train]
    val_scenes = scenes[n_train:n_train + n_val]
    test_scenes = scenes[n_train + n_val:]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Write splits
    splits = {
        'train': train_scenes,
        'val': val_scenes,
        'test': test_scenes
    }

    for split_name, split_scenes in splits.items():
        output_file = os.path.join(output_dir, f'{split_name}.txt')
        with open(output_file, 'w') as f:
            f.write('\n'.join(split_scenes) + '\n')
        print(f"[{split_name:5s}] {len(split_scenes):4d} scenes -> {output_file}")

    print(f"\nTotal: {n_total} scenes")
    print(f"  Train: {len(train_scenes)} ({len(train_scenes)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_scenes)} ({len(val_scenes)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_scenes)} ({len(test_scenes)/n_total*100:.1f}%)")


def main():
    """Create splits for both datasets."""
    base_dir = Path(__file__).parent

    print("="*60)
    print("Creating Train/Val/Test Splits")
    print("="*60)

    # ScanNet++ splits
    print("\nScanNet++ Dataset:")
    print("-" * 60)
    create_splits(
        input_file=base_dir / 'scannetpp' / 'all_scenes.txt',
        output_dir=base_dir / 'scannetpp',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    # Hypersim splits
    print("\n\nHypersim Dataset:")
    print("-" * 60)
    create_splits(
        input_file=base_dir / 'hypersim' / 'all_scenes.txt',
        output_dir=base_dir / 'hypersim',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )

    print("\n" + "="*60)
    print("Done! Split files created successfully.")
    print("="*60)


if __name__ == '__main__':
    main()
