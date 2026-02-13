#!/usr/bin/env python3
"""
Download Hypersim data required by HypersimPlaneDataset.

Only downloads the files actually used by the dataset class:
  - RGB:   scene_cam_XX_final_hdf5/frame.XXXX.color.hdf5
  - Depth: scene_cam_XX_geometry_hdf5/frame.XXXX.depth_meters.hdf5

Only downloads scenes present in our split files (train/val/test).

Based on the official Hypersim downloader by Thomas Germer (MIT License).
Uses HTTP range requests to stream directly from Apple's servers without
downloading full zip files.

Usage:
    # Download RGB + depth for all splits
    python download_hypersim.py -d /cluster/scratch/aoezkan/planeseg/dataset/hypersim

    # Download only val split
    python download_hypersim.py -d /cluster/scratch/aoezkan/planeseg/dataset/hypersim --splits val

    # Download only RGB (no depth)
    python download_hypersim.py -d /cluster/scratch/aoezkan/planeseg/dataset/hypersim --rgb-only

    # Download only depth (no RGB)
    python download_hypersim.py -d /cluster/scratch/aoezkan/planeseg/dataset/hypersim --depth-only

    # Dry run: list files without downloading
    python download_hypersim.py -d /cluster/scratch/aoezkan/planeseg/dataset/hypersim --list

    # Download specific scenes
    python download_hypersim.py -d /path/to/output --scenes ai_001_001 ai_001_002

    # Parallel download with N workers
    python download_hypersim.py -d /path/to/output --workers 4
"""

import os
import sys
import argparse
import requests
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Increase download speed
zipfile.ZipExtFile.MIN_READ_SIZE = 2 ** 20

# Base URL pattern for Hypersim scenes
BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"

# Split file directory (relative to this script)
SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../splits/hypersim")

# File patterns that HypersimPlaneDataset actually loads
RGB_PATTERN = "color.hdf5"
DEPTH_PATTERN = "depth_meters.hdf5"


class WebFile:
    """Read a remote file via HTTP range requests (seekable file-like object)."""

    def __init__(self, url, session):
        with session.head(url) as response:
            response.raise_for_status()
            size = int(response.headers["content-length"])
        self.url = url
        self.session = session
        self.offset = 0
        self.size = size

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def available(self):
        return self.size - self.offset

    def seek(self, offset, whence=0):
        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset = min(self.offset + offset, self.size)
        elif whence == 2:
            self.offset = max(0, self.size + offset)

    def read(self, n=None):
        if n is None:
            n = self.available()
        else:
            n = min(n, self.available())
        end_inclusive = self.offset + n - 1
        headers = {"Range": f"bytes={self.offset}-{end_inclusive}"}
        with self.session.get(self.url, headers=headers) as response:
            data = response.content
        self.offset += len(data)
        return data


def load_scene_ids(splits):
    """Load scene IDs from split files."""
    all_scenes = set()
    for split in splits:
        split_file = os.path.join(SPLIT_DIR, f"{split}.txt")
        if not os.path.exists(split_file):
            print(f"[WARN] Split file not found: {split_file}")
            continue
        with open(split_file) as f:
            scenes = [line.strip() for line in f if line.strip()]
        all_scenes.update(scenes)
        print(f"[INFO] {split} split: {len(scenes)} scenes")
    return sorted(all_scenes)


def should_extract(filename, rgb, depth):
    """Check if a file matches the patterns we need."""
    if rgb and filename.endswith(RGB_PATTERN):
        return True
    if depth and filename.endswith(DEPTH_PATTERN):
        return True
    return False


def process_scene(scene_id, output_dir, session, download_rgb, download_depth,
                  list_only, overwrite, silent):
    """Download files for a single scene. Returns (scene_id, n_downloaded, n_skipped, n_listed, error)."""
    url = f"{BASE_URL}/{scene_id}.zip"
    n_downloaded = 0
    n_skipped = 0
    n_listed = 0

    try:
        f = WebFile(url, session)
        z = zipfile.ZipFile(f)

        for entry in z.infolist():
            if entry.is_dir():
                continue

            if not should_extract(entry.filename, download_rgb, download_depth):
                continue

            if list_only:
                print(entry.filename)
                n_listed += 1
                continue

            path = os.path.join(output_dir, entry.filename)
            if os.path.isfile(path) and not overwrite:
                if not silent:
                    n_skipped += 1
                continue

            z.extract(entry.filename, output_dir)
            n_downloaded += 1

        return (scene_id, n_downloaded, n_skipped, n_listed, None)

    except Exception as e:
        return (scene_id, n_downloaded, n_skipped, n_listed, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Download Hypersim RGB + depth for scenes in our splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all splits to output directory
  python download_hypersim.py -d /cluster/scratch/ayavuz/dataset/Hypersim_merged

  # Download only val+test splits
  python download_hypersim.py -d /path/to/output --splits val test

  # Dry run
  python download_hypersim.py -d /path/to/output --list

  # Only download depth (RGB already exists)
  python download_hypersim.py -d /path/to/output --depth-only
""",
    )
    parser.add_argument("-d", "--directory", type=str, required=True,
                        help="Output directory (e.g., /cluster/scratch/.../Hypersim_merged)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"],
                        help="Which splits to download (default: all)")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Download specific scene IDs instead of using splits")
    parser.add_argument("--rgb-only", action="store_true",
                        help="Only download RGB (color.hdf5)")
    parser.add_argument("--depth-only", action="store_true",
                        help="Only download depth (depth_meters.hdf5)")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("-s", "--silent", action="store_true",
                        help="Suppress skip messages")
    parser.add_argument("-l", "--list", action="store_true",
                        help="Only list files, do not download")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel download workers (default: 1)")
    args = parser.parse_args()

    # Determine what to download
    download_rgb = True
    download_depth = True
    if args.rgb_only:
        download_depth = False
    if args.depth_only:
        download_rgb = False
    if args.rgb_only and args.depth_only:
        download_rgb = True
        download_depth = True

    dl_types = []
    if download_rgb:
        dl_types.append("RGB")
    if download_depth:
        dl_types.append("depth")
    print(f"[INFO] Downloading: {' + '.join(dl_types)}")

    # Get scene list
    if args.scenes:
        scene_ids = sorted(args.scenes)
        print(f"[INFO] {len(scene_ids)} scenes specified via --scenes")
    else:
        scene_ids = load_scene_ids(args.splits)

    if not scene_ids:
        print("[ERROR] No scenes to download")
        sys.exit(1)

    print(f"[INFO] Total scenes to process: {len(scene_ids)}")
    print(f"[INFO] Output directory: {args.directory}")
    if args.list:
        print("[INFO] DRY RUN — listing files only")
    print()

    os.makedirs(args.directory, exist_ok=True)

    total_downloaded = 0
    total_skipped = 0
    total_listed = 0
    total_errors = 0

    if args.workers > 1 and not args.list:
        # Parallel download
        print(f"[INFO] Using {args.workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for scene_id in scene_ids:
                session = requests.Session()
                future = executor.submit(
                    process_scene, scene_id, args.directory, session,
                    download_rgb, download_depth, args.list, args.overwrite, args.silent,
                )
                futures[future] = scene_id

            for i, future in enumerate(as_completed(futures)):
                scene_id, n_dl, n_skip, n_list, error = future.result()
                total_downloaded += n_dl
                total_skipped += n_skip
                total_listed += n_list
                if error:
                    total_errors += 1
                    print(f"  [{i+1}/{len(scene_ids)}] {scene_id}: ERROR — {error}")
                else:
                    print(f"  [{i+1}/{len(scene_ids)}] {scene_id}: {n_dl} downloaded, {n_skip} skipped")
    else:
        # Sequential download
        session = requests.Session()
        for i, scene_id in enumerate(scene_ids):
            print(f"[{i+1}/{len(scene_ids)}] Processing {scene_id}...")
            scene_id, n_dl, n_skip, n_list, error = process_scene(
                scene_id, args.directory, session,
                download_rgb, download_depth, args.list, args.overwrite, args.silent,
            )
            total_downloaded += n_dl
            total_skipped += n_skip
            total_listed += n_list
            if error:
                total_errors += 1
                print(f"  ERROR: {error}")
            elif not args.list:
                print(f"  {n_dl} downloaded, {n_skip} skipped")

    # Summary
    print()
    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  Scenes processed: {len(scene_ids)}")
    if args.list:
        print(f"  Files listed:     {total_listed}")
    else:
        print(f"  Files downloaded: {total_downloaded}")
        print(f"  Files skipped:    {total_skipped}")
    if total_errors:
        print(f"  Scenes with errors: {total_errors}")
    print(f"  Output directory: {args.directory}")


if __name__ == "__main__":
    main()
