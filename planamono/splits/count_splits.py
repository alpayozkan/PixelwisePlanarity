#!/usr/bin/env python3
"""Count scenes and frames per split by scanning GT H5 files on disk.

Usage:
    python planamono/splits/count_splits.py
"""

import os
import argparse
import glob
import h5py


def count_h5_frames(h5_path, key="frame_ids"):
    """Read frame count from an H5 file."""
    with h5py.File(h5_path, "r") as hf:
        if key in hf:
            return hf[key].shape[0]
        if "num_frames" in hf.attrs:
            return int(hf.attrs["num_frames"])
        if "rgb" in hf:
            return hf["rgb"].shape[0]
    return 0


def count_scannetpp(gt_root):
    print("\n=== ScanNet++ ===")
    # {gt_root}/{scene}/rendered.h5
    h5_files = sorted(glob.glob(os.path.join(gt_root, "*/rendered.h5")))
    total_frames = 0
    for h5 in h5_files:
        total_frames += count_h5_frames(h5)
    print(f"  {len(h5_files)} scenes, {total_frames} frames")


def count_hypersim(data_root):
    print("\n=== Hypersim ===")
    # {data_root}/{scene}/rendered_planes_{cam}.h5
    h5_files = sorted(glob.glob(os.path.join(data_root, "*/rendered_planes_*.h5")))
    total_frames = 0
    for h5 in h5_files:
        total_frames += count_h5_frames(h5)
    print(f"  {len(h5_files)} scene/cam groups, {total_frames} frames")


def count_vkitti2(plane_root):
    print("\n=== VKITTI2 ===")
    # {plane_root}/{scene}/clone/scene_data.h5
    h5_files = sorted(glob.glob(os.path.join(plane_root, "*/clone/scene_data.h5")))
    total_frames = 0
    for h5 in h5_files:
        total_frames += count_h5_frames(h5)
    print(f"  {len(h5_files)} scenes, {total_frames} frames")


def count_synthia(plane_root):
    print("\n=== SYNTHIA ===")
    # {plane_root}/{split}/{scene}/scene_data.h5
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(plane_root, split)
        if not os.path.isdir(split_dir):
            print(f"  {split}: directory not found")
            continue
        h5_files = sorted(glob.glob(os.path.join(split_dir, "*/scene_data.h5")))
        total_frames = 0
        for h5 in h5_files:
            total_frames += count_h5_frames(h5)
        print(f"  {split}: {len(h5_files)} scenes, {total_frames} frames")


def main():
    p = argparse.ArgumentParser(description="Count scenes and frames from GT H5 files")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--hypersim_data_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/hypersim")
    p.add_argument("--vkitti2_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/vkitti2_planes")
    p.add_argument("--synthia_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/synthia_planes")
    args = p.parse_args()

    count_scannetpp(args.scannetpp_gt_root)
    count_hypersim(args.hypersim_data_root)
    count_vkitti2(args.vkitti2_plane_root)
    count_synthia(args.synthia_plane_root)


if __name__ == "__main__":
    main()
