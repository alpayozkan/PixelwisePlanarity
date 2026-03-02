#!/usr/bin/env python3
"""Count scenes and frames per split for all 4 datasets.

Reads actual H5 files on cluster to get frame counts.

Usage:
    python planamono/splits/count_splits.py
"""

import os
import argparse
import h5py
import pandas as pd
from pathlib import Path

SPLITS_ROOT = Path(__file__).resolve().parent


def count_scannetpp(args):
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
        for scene_id in scenes:
            gt_h5 = os.path.join(args.scannetpp_gt_root, scene_id, "rendered.h5")
            if os.path.exists(gt_h5):
                with h5py.File(gt_h5, "r") as hf:
                    n_frames += hf["frame_ids"].shape[0]
        print(f"  {split_name}: {len(scenes)} scenes, {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def count_hypersim(args):
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
        n_scenes = split_df.groupby(["scene_name", "camera_name"]).ngroups
        print(f"  {split_name}: {n_scenes} scene/cam groups, {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def count_vkitti2(args):
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
        for scene in scenes:
            h5_path = os.path.join(args.vkitti2_plane_root, scene, "clone", "scene_data.h5")
            if os.path.exists(h5_path):
                with h5py.File(h5_path, "r") as hf:
                    n_frames += hf["rgb"].shape[0]
        print(f"  {split_name}: {len(scenes)} scenes, {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def count_synthia(args):
    print("\n=== SYNTHIA ===")
    total = 0
    for split_name in ["train", "val", "test"]:
        path = SPLITS_ROOT / "synthia" / f"{split_name}.txt"
        if not path.exists():
            print(f"  {split_name}: split file not found")
            continue
        with open(path) as f:
            scenes = [l.strip() for l in f if l.strip()]
        n_frames = 0
        for scene in scenes:
            h5_path = os.path.join(args.synthia_plane_root, split_name, scene, "scene_data.h5")
            if os.path.exists(h5_path):
                with h5py.File(h5_path, "r") as hf:
                    n_frames += hf["rgb"].shape[0]
        print(f"  {split_name}: {len(scenes)} scenes, {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def main():
    p = argparse.ArgumentParser(description="Count scenes and frames per split")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--hypersim_data_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/hypersim")
    p.add_argument("--vkitti2_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/vkitti2_planes")
    p.add_argument("--synthia_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/synthia_planes")
    args = p.parse_args()

    count_scannetpp(args)
    count_hypersim(args)
    count_vkitti2(args)
    count_synthia(args)


if __name__ == "__main__":
    main()
