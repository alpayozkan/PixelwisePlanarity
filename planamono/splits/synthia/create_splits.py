#!/usr/bin/env python3
"""
Create train/val/test splits for SYNTHIA-AL dataset.

Discovers all scene directories and splits them 70/15/15.
Writes train.txt, val.txt, test.txt to the output directory.

Usage:
    python create_splits.py --data_root /path/to/synthia/test --out_dir .
"""

import os
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root of SYNTHIA test/ directory')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Output directory for split files')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Discover scene directories
    scenes = []
    for name in sorted(os.listdir(args.data_root)):
        scene_path = os.path.join(args.data_root, name)
        if not os.path.isdir(scene_path):
            continue
        if not name.startswith('test5_'):
            continue
        scenes.append(name)

    print(f"Found {len(scenes)} SYNTHIA scenes")

    if len(scenes) == 0:
        print("No scenes found. Check --data_root path.")
        return

    # Shuffle and split 70/15/15
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(scenes))

    n_train = int(len(scenes) * 0.70)
    n_val = int(len(scenes) * 0.15)

    train_scenes = [scenes[i] for i in indices[:n_train]]
    val_scenes = [scenes[i] for i in indices[n_train:n_train + n_val]]
    test_scenes = [scenes[i] for i in indices[n_train + n_val:]]

    os.makedirs(args.out_dir, exist_ok=True)

    for split_name, split_scenes in [('train', train_scenes),
                                     ('val', val_scenes),
                                     ('test', test_scenes)]:
        path = os.path.join(args.out_dir, f'{split_name}.txt')
        with open(path, 'w') as f:
            for s in sorted(split_scenes):
                f.write(s + '\n')
        print(f"  {split_name}: {len(split_scenes)} scenes -> {path}")

    # Also write full scene list
    path = os.path.join(args.out_dir, 'scene_list.txt')
    with open(path, 'w') as f:
        for s in sorted(scenes):
            f.write(s + '\n')
    print(f"  all: {len(scenes)} scenes -> {path}")


if __name__ == '__main__':
    main()
