#!/usr/bin/env python3
"""Count scenes and frames per split for all 4 datasets.

Replicates exactly how training code discovers and counts data.

Usage:
    python planamono/splits/count_splits.py
"""

import os
import sys
import argparse
import h5py
import pandas as pd
from pathlib import Path

SPLITS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPLITS_ROOT.parents[1]))
from planamono.paths import (
    scannetpp_rend_plane_path,
    vkitti2_plane_path,
    synthia_plane_train_path,
    synthia_plane_test_path,
)


def count_scannetpp(scannet_plane_root, scannet_rgb_root):
    """ScanNet++: split txt -> scenes -> {plane_root}/{scene}/rendered.h5 -> frame_ids"""
    print("\n=== ScanNet++ ===")
    total = 0
    for split_name, fname in [
        ("train", "nvs_sem_train_with_planes_fixed.txt"),
        ("val", "nvs_sem_val_with_planes_fixed.txt"),
        ("test", "nvs_sem_test_with_planes.txt"),
    ]:
        path = SPLITS_ROOT / "scannetpp" / fname
        if not path.exists():
            print(f"  {split_name}: split file not found")
            continue
        with open(path) as f:
            scenes = [l.strip() for l in f if l.strip()]
        n_frames = 0
        n_valid = 0
        for scene_id in scenes:
            h5 = os.path.join(scannet_plane_root, scene_id, "rendered.h5")
            if not os.path.exists(h5):
                continue
            with h5py.File(h5, "r") as hf:
                n_frames += hf["frame_ids"].shape[0]
            n_valid += 1
        print(f"  {split_name}: {n_valid}/{len(scenes)} scenes, {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def count_hypersim():
    """Hypersim: CSV split -> per-frame rows (scene_name, camera_name, frame_id, split)"""
    print("\n=== Hypersim ===")
    csv_path = SPLITS_ROOT / "hypersim" / "metadata_images_split_with_planes_filtered.csv"
    if not csv_path.exists():
        print("  CSV not found")
        return
    df = pd.read_csv(csv_path)
    total = 0
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split_partition_name"] == split_name]
        n_frames = len(split_df)
        n_groups = split_df.groupby(["scene_name", "camera_name"]).ngroups
        print(f"  {split_name}: {n_groups} scene/cam groups, {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def count_vkitti2(vkitti2_plane_root):
    """VKITTI2: split txt -> scene list -> {root}/{scene}/{variant}/scene_data.h5 (all variants)"""
    print("\n=== VKITTI2 ===")
    total = 0
    for split_name in ["train", "val", "test"]:
        path = SPLITS_ROOT / "vkitti2" / f"{split_name}.txt"
        if not path.exists():
            print(f"  {split_name}: split file not found")
            continue
        with open(path) as f:
            scenes = [l.strip() for l in f if l.strip()]
        n_frames = 0
        n_scene_variants = 0
        for scene in scenes:
            scene_dir = os.path.join(vkitti2_plane_root, scene)
            if not os.path.isdir(scene_dir):
                continue
            for variant in sorted(os.listdir(scene_dir)):
                h5_path = os.path.join(scene_dir, variant, "scene_data.h5")
                if not os.path.exists(h5_path):
                    continue
                with h5py.File(h5_path, "r") as hf:
                    n_frames += int(hf.attrs["num_frames"])
                n_scene_variants += 1
        print(f"  {split_name}: {len(scenes)} scenes ({n_scene_variants} scene/variants), {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def count_synthia(synthia_plane_train_root, synthia_plane_test_root):
    """SYNTHIA: split txt -> scene list -> {split_root}/{scene}/scene_data.h5"""
    print("\n=== SYNTHIA ===")
    split_roots = {
        "train": synthia_plane_train_root,
        "val": synthia_plane_train_root,  # val scenes also live under train root
        "test": synthia_plane_test_root,
    }
    total = 0
    for split_name in ["train", "val", "test"]:
        path = SPLITS_ROOT / "synthia" / f"{split_name}.txt"
        if not path.exists():
            print(f"  {split_name}: split file not found")
            continue
        with open(path) as f:
            scenes = [l.strip() for l in f if l.strip()]
        root = split_roots[split_name]
        n_frames = 0
        n_valid = 0
        for scene in scenes:
            h5_path = os.path.join(root, scene, "scene_data.h5")
            if not os.path.exists(h5_path):
                continue
            with h5py.File(h5_path, "r") as hf:
                n_frames += int(hf.attrs["num_frames"])
            n_valid += 1
        print(f"  {split_name}: {n_valid}/{len(scenes)} scenes, {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def main():
    p = argparse.ArgumentParser(description="Count scenes and frames per split (matches training code)")
    p.add_argument("--scannet_plane_root", type=str, default=scannetpp_rend_plane_path)
    p.add_argument("--scannet_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--vkitti2_plane_root", type=str, default=vkitti2_plane_path)
    p.add_argument("--synthia_plane_train_root", type=str, default=synthia_plane_train_path)
    p.add_argument("--synthia_plane_test_root", type=str, default=synthia_plane_test_path)
    args = p.parse_args()

    count_scannetpp(args.scannet_plane_root, args.scannet_rgb_root)
    count_hypersim()
    count_vkitti2(args.vkitti2_plane_root)
    count_synthia(args.synthia_plane_train_root, args.synthia_plane_test_root)


if __name__ == "__main__":
    main()
