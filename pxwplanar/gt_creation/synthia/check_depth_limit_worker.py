#!/usr/bin/env python3
"""Check depth limit for a list of scenes. Called by parallel job submission."""

import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

PLANAR = {1, 2, 3, 20}


def decode_depth(depth_png):
    R = depth_png[:, :, 0].astype(np.float64)
    G = depth_png[:, :, 1].astype(np.float64)
    B = depth_png[:, :, 2].astype(np.float64)
    return (5000.0 * (R + G * 256 + B * 256 * 256) / (256**3 - 1)).astype(
        np.float32
    )


scene_list = sys.argv[1]
out_file = sys.argv[2]

affected = []

with open(scene_list) as f:
    lines = [ln.strip() for ln in f if ln.strip()]

for line in tqdm(lines, desc="checking"):
    split, scene_dir = line.split("\t")

    # Find timestamp subdir
    for ts in sorted(os.listdir(scene_dir)):
        ts_dir = os.path.join(scene_dir, ts)
        if not os.path.isdir(os.path.join(ts_dir, "RGB")):
            continue

        total_lost = 0
        total_planar = 0
        n_frames = 0

        for fname in sorted(os.listdir(os.path.join(ts_dir, "Depth"))):
            if not fname.endswith(".png"):
                continue
            depth_png = np.array(
                Image.open(os.path.join(ts_dir, "Depth", fname))
            )
            depth = decode_depth(depth_png)
            seg = np.array(Image.open(os.path.join(ts_dir, "SemSeg", fname)))
            class_ids = seg[:, :, 0].astype(np.int32)

            planar_mask = np.isin(class_ids, list(PLANAR))
            lost = (
                planar_mask & (depth > 0) & ((depth <= 0.1) | (depth >= 300.0))
            )
            total_lost += lost.sum()
            total_planar += (planar_mask & (depth > 0)).sum()
            n_frames += 1

        if total_lost > 0:
            pct = 100 * total_lost / max(total_planar, 1)
            print(
                f"[{split}/{os.path.basename(scene_dir)}] {n_frames} frames: "
                f"{total_lost} px lost ({pct:.2f}%)"
            )
            affected.append(f"{split}\t{scene_dir}")

with open(out_file, "w") as f:
    for line in affected:
        f.write(line + "\n")

print(f"{len(affected)} affected scenes written to {out_file}")
