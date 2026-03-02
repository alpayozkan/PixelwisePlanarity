#!/usr/bin/env python3
"""Count scenes and frames per split for all 4 datasets.

Replicates the exact same logic as the training dataset classes:
- ScanNetPPPlanarityDataset: checks rgb_dir, plane_h5, sem_h5, depth_h5 exist per scene,
  then checks per-frame RGB jpg exists
- HypersimPlanarityDataset: counts rows from filtered CSV per split
- VKITTI2PlanarityDataset: iterates scene/variant dirs, reads num_frames from H5 attrs
- SYNTHIAPlanarityDataset: reads scene list from split txt, reads num_frames from H5 attrs

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


def count_scannetpp(plane_root, rgb_root):
    """Exact same logic as ScanNetPPPlanarityDataset.__init__"""
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
        n_valid_scenes = 0
        n_skipped_scenes = 0
        for scene_id in scenes:
            rgb_dir = os.path.join(rgb_root, scene_id, "iphone", "rgb")
            plane_h5 = os.path.join(plane_root, scene_id, "rendered.h5")
            sem_h5 = os.path.join(plane_root, scene_id, "rendered_sem.h5")
            depth_h5 = os.path.join(plane_root, scene_id, "rendered_depth.h5")

            if not (os.path.isdir(rgb_dir) and os.path.exists(plane_h5)
                    and os.path.exists(sem_h5) and os.path.exists(depth_h5)):
                n_skipped_scenes += 1
                continue

            with h5py.File(plane_h5, "r") as hf:
                frame_ids = [fid.decode("utf-8") if isinstance(fid, bytes) else str(fid)
                             for fid in hf["frame_ids"][:]]

            scene_frames = 0
            for fid in frame_ids:
                rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")
                if os.path.exists(rgb_path):
                    scene_frames += 1

            n_frames += scene_frames
            n_valid_scenes += 1

        print(f"  {split_name}: {n_valid_scenes} scenes ({n_skipped_scenes} skipped), {n_frames} frames")
        total += n_frames
    print(f"  TOTAL: {total} frames")


def count_hypersim(csv_path):
    """Exact same logic as HypersimPlanarityDataset.__init__ — just counts CSV rows per split"""
    print("\n=== Hypersim ===")
    if not os.path.exists(csv_path):
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


def count_vkitti2(data_root):
    """Exact same logic as VKITTI2PlanarityDataset.__init__ — iterates all scene/variant combos"""
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
        for scene in sorted(scenes):
            scene_dir = os.path.join(data_root, scene)
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


def count_synthia(plane_train_root, plane_test_root):
    """Exact same logic as SYNTHIAPlanarityDataset.__init__ — scene list + H5 num_frames"""
    print("\n=== SYNTHIA ===")
    split_roots = {
        "train": plane_train_root,
        "val": plane_train_root,
        "test": plane_test_root,
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
        for scene in sorted(scenes):
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
    p = argparse.ArgumentParser(description="Count scenes/frames per split (same logic as training code)")
    p.add_argument("--scannet_plane_root", type=str, default=scannetpp_rend_plane_path)
    p.add_argument("--scannet_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--hypersim_csv", type=str,
                   default=str(SPLITS_ROOT / "hypersim" / "metadata_images_split_with_planes_filtered.csv"))
    p.add_argument("--vkitti2_plane_root", type=str, default=vkitti2_plane_path)
    p.add_argument("--synthia_plane_train_root", type=str, default=synthia_plane_train_path)
    p.add_argument("--synthia_plane_test_root", type=str, default=synthia_plane_test_path)
    args = p.parse_args()

    count_scannetpp(args.scannet_plane_root, args.scannet_rgb_root)
    count_hypersim(args.hypersim_csv)
    count_vkitti2(args.vkitti2_plane_root)
    count_synthia(args.synthia_plane_train_root, args.synthia_plane_test_root)


if __name__ == "__main__":
    main()
