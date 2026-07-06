#!/usr/bin/env python3
"""
Generate train/val/test split files for SYNTHIA-AL.

Train scenes come from synthia_planes/train, split 95/5 into train/val.
Test scenes come from synthia_planes/test (kept as-is).
"""

import os
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pxwplanar.paths import synthia_path
import random


def main():
    parser = argparse.ArgumentParser(description="Generate SYNTHIA train/val/test splits")
    parser.add_argument("--train_root", type=str,
                        default=os.path.join(synthia_path, "train"),
                        help="Path to synthia_planes/train with scene_data.h5 per scene")
    parser.add_argument("--test_root", type=str,
                        default=os.path.join(synthia_path, "test"),
                        help="Path to synthia_planes/test with scene_data.h5 per scene")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "synthia"),
                        help="Output directory for split .txt files")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover train scenes (directories with test5_ prefix)
    train_scenes = sorted([
        d for d in os.listdir(args.train_root)
        if os.path.isdir(os.path.join(args.train_root, d)) and d.startswith('test5_')
    ])
    print(f"Found {len(train_scenes)} train scenes in {args.train_root}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(train_scenes)
    n_val = max(1, int(len(train_scenes) * args.val_fraction))
    val_scenes = sorted(train_scenes[:n_val])
    train_scenes_final = sorted(train_scenes[n_val:])

    print(f"  Train: {len(train_scenes_final)} scenes")
    print(f"  Val:   {len(val_scenes)} scenes")

    # Discover test scenes
    test_scenes = sorted([
        d for d in os.listdir(args.test_root)
        if os.path.isdir(os.path.join(args.test_root, d)) and d.startswith('test5_')
    ])
    print(f"  Test:  {len(test_scenes)} scenes from {args.test_root}")

    # Write split files
    for split_name, scenes in [("train", train_scenes_final), ("val", val_scenes), ("test", test_scenes)]:
        path = os.path.join(args.output_dir, f"{split_name}.txt")
        with open(path, 'w') as f:
            for s in scenes:
                f.write(s + '\n')
        print(f"Wrote {path} ({len(scenes)} scenes)")


if __name__ == "__main__":
    main()
