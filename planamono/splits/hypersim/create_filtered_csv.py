"""
Generate filtered CSV for HypersimPlanarityDataset from scene splits and rendered data.

This script:
1. Reads train/val/test scene splits
2. Scans rendered plane H5 files to find valid (scene, camera, frame) tuples
3. Outputs a CSV with columns: scene_name, camera_name, frame_id, split_partition_name
"""

import os
import argparse
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_split(split_file):
    """Load scene IDs from a split file."""
    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def scan_rendered_planes(plane_label_root, scene_id):
    """
    Scan rendered plane H5 files for a scene to get (camera, frame_id) pairs.

    Expected structure:
        {plane_label_root}/{scene_id}/rendered_planes_cam_XX.h5

    Each H5 file contains:
        - frame_ids: list of frame IDs (e.g., ['0000', '0001', ...])
        - planes: array of plane masks
    """
    scene_dir = os.path.join(plane_label_root, scene_id)
    if not os.path.isdir(scene_dir):
        return []

    results = []
    for fname in os.listdir(scene_dir):
        if fname.startswith('rendered_planes_') and fname.endswith('.h5'):
            # Extract camera name: rendered_planes_cam_00.h5 -> cam_00
            cam_name = fname.replace('rendered_planes_', '').replace('.h5', '')
            h5_path = os.path.join(scene_dir, fname)

            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'frame_ids' in f:
                        frame_ids = [fid.decode('utf-8') if isinstance(fid, bytes) else str(fid)
                                     for fid in f['frame_ids'][:]]
                        for fid in frame_ids:
                            results.append((cam_name, fid))
            except Exception as e:
                print(f"[WARN] Error reading {h5_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate filtered CSV for Hypersim training")
    parser.add_argument("--splits_dir", type=str,
                        default="/Users/ahmetcanyavuz/Developer/3D_vision/PixelwisePlanarity/planamono/splits/hypersim",
                        help="Directory containing train.txt, val.txt, test.txt")
    parser.add_argument("--plane_label_root", type=str,
                        default="/cluster/scratch/ayavuz/dataset/Hypersim_rendered",
                        help="Root directory with rendered plane H5 files")
    parser.add_argument("--output_csv", type=str,
                        default="metadata_images_split_with_planes.csv",
                        help="Output CSV filename (saved in splits_dir)")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)

    # Load splits
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = splits_dir / f'{split_name}.txt'
        if split_file.exists():
            splits[split_name] = load_split(split_file)
            print(f"[{split_name:5s}] Loaded {len(splits[split_name])} scenes")
        else:
            print(f"[WARN] Missing split file: {split_file}")
            splits[split_name] = []

    # Build CSV rows
    rows = []
    all_scenes = []
    for split_name, scene_ids in splits.items():
        for scene_id in scene_ids:
            all_scenes.append((scene_id, split_name))

    print(f"\nScanning {len(all_scenes)} scenes for rendered planes...")

    for scene_id, split_name in tqdm(all_scenes):
        cam_frame_pairs = scan_rendered_planes(args.plane_label_root, scene_id)

        if not cam_frame_pairs:
            print(f"[SKIP] No rendered planes found for {scene_id}")
            continue

        for cam_name, frame_id in cam_frame_pairs:
            rows.append({
                'scene_name': scene_id,
                'camera_name': cam_name,
                'frame_id': frame_id,
                'split_partition_name': split_name
            })

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    output_path = splits_dir / args.output_csv
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"CSV saved to: {output_path}")
    print(f"Total entries: {len(df)}")
    print(f"\nPer-split breakdown:")
    print(df['split_partition_name'].value_counts())
    print(f"\nSample rows:")
    print(df.head(10))


if __name__ == "__main__":
    main()
