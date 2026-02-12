#!/usr/bin/env python3
"""
Verify that camera intrinsics are constant across all frames/scenes
for SYNTHIA-AL and VKITTI2 datasets.

Usage:
    python scripts/check_intrinsics_constant.py \
        --synthia /path/to/synthia \
        --vkitti2 /path/to/virtual_kitti

Either or both flags can be provided. Omit a flag to skip that dataset.
"""
import argparse
import hashlib
import os
import sys
import glob
from tqdm import tqdm


def check_synthia(synthia_root):
    """
    Check every calib_kitti/*.txt file via MD5 hash.
    All files should be byte-identical.
    """
    print("=" * 60)
    print("SYNTHIA-AL Intrinsics Check")
    print("=" * 60)

    print("  Scanning for calib_kitti/*.txt files...")
    calib_files = glob.glob(
        os.path.join(synthia_root, "**", "calib_kitti", "*.txt"),
        recursive=True
    )

    if not calib_files:
        print(f"  No calib_kitti/*.txt files found under {synthia_root}")
        return False

    total = len(calib_files)
    print(f"  Found {total} calibration files")

    hashes = {}
    for fpath in tqdm(calib_files, desc="  Hashing", unit="file"):
        h = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
        if h not in hashes:
            hashes[h] = {'count': 0, 'first_file': fpath}
        hashes[h]['count'] += 1

    print(f"\n  Unique file contents: {len(hashes)}")
    for i, (h, info) in enumerate(hashes.items()):
        rel = os.path.relpath(info['first_file'], synthia_root)
        content = open(info['first_file']).read().strip()
        print(f"\n  #{i+1} ({info['count']} files, e.g. {rel}):")
        for line in content.splitlines():
            print(f"    {line}")

    if len(hashes) == 1:
        print(f"\n  PASS: All {total} files are byte-identical")
    else:
        print(f"\n  FAIL: Found {len(hashes)} distinct file contents")

    return len(hashes) == 1


def check_vkitti2(vkitti2_root):
    """
    Check every intrinsic.txt row across all scenes/variants.
    Format: 'frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]'
    All rows (excluding frame/cameraID) should have identical K values.
    """
    print("=" * 60)
    print("VKITTI2 Intrinsics Check")
    print("=" * 60)

    intrinsic_files = sorted(glob.glob(
        os.path.join(vkitti2_root, "**", "intrinsic.txt"),
        recursive=True
    ))

    if not intrinsic_files:
        print(f"  No intrinsic.txt files found under {vkitti2_root}")
        return False

    print(f"  Found {len(intrinsic_files)} intrinsic.txt files")

    unique_params = {}
    total_rows = 0

    for fpath in tqdm(intrinsic_files, desc="  Parsing", unit="file"):
        rel = os.path.relpath(fpath, vkitti2_root)
        with open(fpath, 'r') as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_rows += 1
                parts = line.split()
                # parts[0]=frame, parts[1]=cameraID, rest=K values
                k_vals = " ".join(parts[2:])
                if k_vals not in unique_params:
                    unique_params[k_vals] = {
                        'count': 0,
                        'first_file': rel,
                        'first_line': line,
                    }
                unique_params[k_vals]['count'] += 1

    print(f"\n  Total rows checked: {total_rows}")
    print(f"  Unique K parameter sets: {len(unique_params)}")
    for i, (k_vals, info) in enumerate(unique_params.items()):
        print(f"\n  #{i+1} ({info['count']} rows, e.g. {info['first_file']}):")
        print(f"    K values: {k_vals}")

    if len(unique_params) == 1:
        print(f"\n  PASS: All {total_rows} rows have identical K values")
    else:
        print(f"\n  FAIL: Found {len(unique_params)} distinct K parameter sets")

    return len(unique_params) == 1


def main():
    parser = argparse.ArgumentParser(
        description="Verify camera intrinsics are constant across SYNTHIA and VKITTI2 datasets"
    )
    parser.add_argument('--synthia', type=str, default=None,
                        help='Path to SYNTHIA root (containing train/ and test/)')
    parser.add_argument('--vkitti2', type=str, default=None,
                        help='Path to VKITTI2 root (containing Scene01/, Scene02/, ...)')

    args = parser.parse_args()

    if args.synthia is None and args.vkitti2 is None:
        parser.print_help()
        print("\nProvide at least one of --synthia or --vkitti2")
        sys.exit(1)

    results = {}

    if args.synthia:
        results['synthia'] = check_synthia(args.synthia)
        print()

    if args.vkitti2:
        results['vkitti2'] = check_vkitti2(args.vkitti2)
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
