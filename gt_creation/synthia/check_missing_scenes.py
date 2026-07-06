#!/usr/bin/env python3
"""Find SYNTHIA scenes that are missing or incomplete from the generated planes output.

Checks for planes.json (last file written per scene) as the completion marker.
Scenes with scene_data.h5 but no planes.json are flagged as incomplete.
"""
import os

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import synthia_raw_path, synthia_path

SYNTHIA_TRAIN = os.path.join(synthia_raw_path, 'train')
SYNTHIA_TEST = os.path.join(synthia_raw_path, 'test')
PLANES_TRAIN = os.path.join(synthia_path, 'train')
PLANES_TEST = os.path.join(synthia_path, 'test')
OUT_FILE = os.path.join(synthia_path, 'scenes_to_rerun.txt')

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
        json_path = os.path.join(planes_root, scene, 'planes.json')
        h5_path = os.path.join(planes_root, scene, 'scene_data.h5')

        if not os.path.exists(json_path):
            scene_dir = os.path.join(src_root, scene)
            missing.append((split, scene_dir))
            if os.path.exists(h5_path):
                print(f'[INCOMPLETE] {split}/{scene} (h5 exists but no planes.json)')
            else:
                print(f'[MISSING] {split}/{scene}')

    done = 0
    if os.path.isdir(planes_root):
        done = len([s for s in os.listdir(planes_root)
                    if os.path.exists(os.path.join(planes_root, s, 'planes.json'))])
    print(f'{split}: {len(src_scenes)} total, {done} done, {len([m for m in missing if m[0] == split])} missing')

with open(OUT_FILE, 'w') as f:
    for split, scene_dir in missing:
        f.write(f'{split}\t{scene_dir}\n')

print(f'\n{len(missing)} scenes to rerun saved to {OUT_FILE}')
