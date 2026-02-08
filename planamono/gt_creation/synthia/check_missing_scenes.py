#!/usr/bin/env python3
"""Find SYNTHIA scenes that are missing from the generated planes output."""
import os

SYNTHIA_TRAIN = '/cluster/scratch/ayavuz/dataset/synthia/train'
SYNTHIA_TEST = '/cluster/scratch/ayavuz/dataset/synthia/test'
PLANES_TRAIN = '/cluster/scratch/ayavuz/dataset/synthia_planes/train'
PLANES_TEST = '/cluster/scratch/ayavuz/dataset/synthia_planes/test'
OUT_FILE = '/cluster/scratch/ayavuz/dataset/synthia_planes/scenes_to_rerun.txt'

missing = []

for src_root, planes_root, split in [
    (SYNTHIA_TRAIN, PLANES_TRAIN, 'train'),
    (SYNTHIA_TEST, PLANES_TEST, 'test'),
]:
    if not os.path.isdir(src_root):
        print(f'[SKIP] {src_root} not found')
        continue

    src_scenes = sorted([s for s in os.listdir(src_root)
                         if s.startswith('test5_') and os.path.isdir(os.path.join(src_root, s))])

    for scene in src_scenes:
        h5_path = os.path.join(planes_root, scene, 'scene_data.h5')
        if not os.path.exists(h5_path):
            scene_dir = os.path.join(src_root, scene)
            missing.append((split, scene_dir))
            print(f'[MISSING] {split}/{scene}')

    done = 0
    if os.path.isdir(planes_root):
        done = len([s for s in os.listdir(planes_root)
                    if os.path.exists(os.path.join(planes_root, s, 'scene_data.h5'))])
    print(f'{split}: {len(src_scenes)} total, {done} done, {len([m for m in missing if m[0] == split])} missing')

with open(OUT_FILE, 'w') as f:
    for split, scene_dir in missing:
        f.write(f'{split}\t{scene_dir}\n')

print(f'\n{len(missing)} missing scenes saved to {OUT_FILE}')
