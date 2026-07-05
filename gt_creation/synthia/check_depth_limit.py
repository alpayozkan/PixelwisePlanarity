#!/usr/bin/env python3
"""Check how many SYNTHIA planar pixels were filtered by the old 300m depth cap."""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

SYNTHIA_TRAIN = '/cluster/scratch/aoezkan/planeseg/synthia/raw/train'
SYNTHIA_TEST = '/cluster/scratch/aoezkan/planeseg/synthia/raw/test'
PLANAR = {1, 2, 3, 20}  # Ground, Sidewalk, Building, Lanemarking


def decode_depth(depth_png):
    R = depth_png[:, :, 0].astype(np.float64)
    G = depth_png[:, :, 1].astype(np.float64)
    B = depth_png[:, :, 2].astype(np.float64)
    return (5000.0 * (R + G * 256 + B * 256 * 256) / (256**3 - 1)).astype(np.float32)


OUT_FILE = '/cluster/scratch/aoezkan/planeseg/synthia/synthia_planes/scenes_to_rerun.txt'
affected = []

for data_root, split in [(SYNTHIA_TRAIN, 'train'), (SYNTHIA_TEST, 'test')]:
    if not os.path.isdir(data_root):
        continue
    scenes = sorted([s for s in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, s))])
    for scene_name in tqdm(scenes, desc=split):
        scene_dir = os.path.join(data_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        # Find timestamp subdir
        for ts in sorted(os.listdir(scene_dir)):
            ts_dir = os.path.join(scene_dir, ts)
            if not os.path.isdir(os.path.join(ts_dir, 'RGB')):
                continue

            total_lost = 0
            total_planar = 0
            max_lost_depth = 0
            n_frames = 0

            for fname in sorted(os.listdir(os.path.join(ts_dir, 'Depth'))):
                if not fname.endswith('.png'):
                    continue
                depth_png = np.array(Image.open(os.path.join(ts_dir, 'Depth', fname)))
                depth = decode_depth(depth_png)

                seg = np.array(Image.open(os.path.join(ts_dir, 'SemSeg', fname)))
                class_ids = seg[:, :, 0].astype(np.int32)

                planar_mask = np.isin(class_ids, list(PLANAR))
                valid_new = depth > 0
                valid_old = (depth > 0.1) & (depth < 300.0)

                lost = planar_mask & valid_new & ~valid_old
                total_lost += lost.sum()
                total_planar += (planar_mask & valid_new).sum()
                if lost.sum() > 0:
                    max_lost_depth = max(max_lost_depth, depth[lost].max())
                n_frames += 1

            if total_lost > 0:
                pct = 100 * total_lost / max(total_planar, 1)
                print(f'[{split}/{scene_name}] {n_frames} frames: {total_lost} planar px lost ({pct:.2f}%), max depth: {max_lost_depth:.1f}m')
                affected.append((split, os.path.join(data_root, scene_name)))
            else:
                print(f'[{split}/{scene_name}] {n_frames} frames: no pixels lost')

with open(OUT_FILE, 'w') as f:
    for split, scene_dir in affected:
        f.write(f'{split}\t{scene_dir}\n')
print(f'\n{len(affected)} affected scenes saved to {OUT_FILE}')
