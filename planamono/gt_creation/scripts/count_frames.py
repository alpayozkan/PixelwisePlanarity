#!/usr/bin/env python3
"""Count total frames in SYNTHIA and VKITTI2 source datasets."""
import os
import sys
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "..", "configs")


def count_synthia(train_root, test_root):
    total = 0
    for root, split in [(train_root, "train"), (test_root, "test")]:
        split_count = 0
        if not os.path.isdir(root):
            print(f"  [SKIP] {root} not found")
            continue
        for scene in sorted(os.listdir(root)):
            scene_dir = os.path.join(root, scene)
            if not os.path.isdir(scene_dir):
                continue
            for ts in os.listdir(scene_dir):
                rgb_dir = os.path.join(scene_dir, ts, "RGB")
                if not os.path.isdir(rgb_dir):
                    continue
                n = len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
                split_count += n
        print(f"  {split}: {split_count} frames")
        total += split_count
    return total


def count_vkitti2(data_root, scenes=None):
    total = 0
    if not os.path.isdir(data_root):
        print(f"  [SKIP] {data_root} not found")
        return 0
    for scene in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        if scenes and scene not in scenes:
            continue
        for variant in sorted(os.listdir(scene_dir)):
            rgb_dir = os.path.join(scene_dir, variant, "frames", "rgb", "Camera_0")
            if not os.path.isdir(rgb_dir):
                continue
            n = len([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
            print(f"  {scene}/{variant}: {n} frames")
            total += n
    return total


if __name__ == "__main__":
    # SYNTHIA
    synthia_cfg = os.path.join(CONFIG_DIR, "synthia_default.yml")
    if os.path.exists(synthia_cfg):
        cfg = yaml.safe_load(open(synthia_cfg))
        train_root = cfg.get("synthia_train", "/cluster/scratch/ayavuz/dataset/synthia/train")
        test_root = cfg.get("synthia_test", "/cluster/scratch/ayavuz/dataset/synthia/test")
    else:
        train_root = "/cluster/scratch/ayavuz/dataset/synthia/train"
        test_root = "/cluster/scratch/ayavuz/dataset/synthia/test"

    print("SYNTHIA")
    print("=" * 40)
    synthia_total = count_synthia(train_root, test_root)
    print(f"  TOTAL: {synthia_total} frames")

    # VKITTI2
    vkitti_cfg = os.path.join(CONFIG_DIR, "vkitti2_default.yml")
    if os.path.exists(vkitti_cfg):
        cfg = yaml.safe_load(open(vkitti_cfg))
        data_root = cfg.get("data_root", "/cluster/scratch/ayavuz/dataset/virtual_kitti")
    else:
        data_root = "/cluster/scratch/ayavuz/dataset/virtual_kitti"

    print()
    print("VKITTI2")
    print("=" * 40)
    vkitti_total = count_vkitti2(data_root)
    print(f"  TOTAL: {vkitti_total} frames")

    print()
    print(f"GRAND TOTAL: {synthia_total + vkitti_total} frames")
