#!/usr/bin/env python3
"""
Verify that camera intrinsics are constant across all frames/scenes
for SYNTHIA-AL and VKITTI2 datasets.

Usage:
    python scripts/check_intrinsics_constant.py \
        --synthia /path/to/synthia/test \
        --vkitti2-textgt /path/to/vkitti_2.0.3_textgt

Either or both flags can be provided. Omit a flag to skip that dataset.
"""
import argparse
import os
import sys
import glob
import numpy as np


def parse_kitti_calib(filepath):
    """Parse KITTI-format calibration file, return K matrices as dict."""
    matrices = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            nums = list(map(float, vals.strip().split()))
            if len(nums) == 12:
                # 3x4 projection matrix -> extract 3x3 intrinsic
                P = np.array(nums).reshape(3, 4)
                matrices[key.strip()] = P[:, :3]
            elif len(nums) == 9:
                matrices[key.strip()] = np.array(nums).reshape(3, 3)
    return matrices


def check_synthia(synthia_root):
    """
    Check all calib_kitti/*.txt files across all SYNTHIA scenes.
    Expected: fx=fy=895.692, cx=320.0, cy=240.0 everywhere.
    """
    print("=" * 60)
    print("SYNTHIA-AL Intrinsics Check")
    print("=" * 60)

    calib_files = sorted(glob.glob(
        os.path.join(synthia_root, "**", "calib_kitti", "*.txt"),
        recursive=True
    ))

    if not calib_files:
        print(f"  No calib_kitti/*.txt files found under {synthia_root}")
        return False

    print(f"  Found {len(calib_files)} calibration files")

    reference_K = None
    reference_file = None
    all_same = True
    unique_Ks = {}

    for fpath in calib_files:
        try:
            matrices = parse_kitti_calib(fpath)
        except Exception as e:
            print(f"  ERROR parsing {fpath}: {e}")
            all_same = False
            continue

        # Use P0 or P2 (left camera) — try common keys
        K = None
        for key in ['P0', 'P2', 'P_rect_00']:
            if key in matrices:
                K = matrices[key]
                break

        if K is None:
            # Fall back to first matrix found
            if matrices:
                first_key = list(matrices.keys())[0]
                K = matrices[first_key]
            else:
                print(f"  WARNING: No matrices in {fpath}")
                continue

        # Track unique K values
        K_tuple = tuple(K.flatten().round(6))
        if K_tuple not in unique_Ks:
            unique_Ks[K_tuple] = {
                'K': K,
                'count': 0,
                'first_file': fpath,
            }
        unique_Ks[K_tuple]['count'] += 1

        if reference_K is None:
            reference_K = K
            reference_file = fpath
        elif not np.allclose(K, reference_K, atol=1e-4):
            all_same = False

    print(f"\n  Unique K matrices found: {len(unique_Ks)}")
    for i, (k_tuple, info) in enumerate(unique_Ks.items()):
        K = info['K']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        skew = K[0, 1]
        print(f"\n  K#{i+1} ({info['count']} files, e.g. {os.path.relpath(info['first_file'], synthia_root)}):")
        print(f"    fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}  skew={skew:.6f}")

    if all_same and len(unique_Ks) == 1:
        print(f"\n  PASS: All {len(calib_files)} files have identical intrinsics")
    else:
        print(f"\n  FAIL: Found {len(unique_Ks)} distinct intrinsic matrices")

    return all_same and len(unique_Ks) == 1


def parse_vkitti2_intrinsic(filepath):
    """
    Parse VKITTI2 intrinsic.txt file.
    Format: one header line, then per-frame rows with K values.
    Returns list of (frame_id, K_3x3) tuples.
    """
    results = []
    with open(filepath, 'r') as f:
        header = f.readline().strip()
        cols = header.split()

        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split()
            row = dict(zip(cols, vals))

            frame_id = int(row.get('frame', row.get('Frame', -1)))
            cam_id = int(row.get('cameraID', row.get('cam', 0)))

            # Build K from columns
            K = np.eye(3)
            for key in row:
                kl = key.lower()
                try:
                    v = float(row[key])
                except ValueError:
                    continue

                if kl in ('k00', 'k[0,0]', 'k_00'):
                    K[0, 0] = v
                elif kl in ('k01', 'k[0,1]', 'k_01'):
                    K[0, 1] = v
                elif kl in ('k02', 'k[0,2]', 'k_02'):
                    K[0, 2] = v
                elif kl in ('k10', 'k[1,0]', 'k_10'):
                    K[1, 0] = v
                elif kl in ('k11', 'k[1,1]', 'k_11'):
                    K[1, 1] = v
                elif kl in ('k12', 'k[1,2]', 'k_12'):
                    K[1, 2] = v
                elif kl in ('k20', 'k[2,0]', 'k_20'):
                    K[2, 0] = v
                elif kl in ('k21', 'k[2,1]', 'k_21'):
                    K[2, 1] = v
                elif kl in ('k22', 'k[2,2]', 'k_22'):
                    K[2, 2] = v

            results.append((frame_id, cam_id, K))

    return results


def check_vkitti2(textgt_root):
    """
    Check all intrinsic.txt files across all VKITTI2 scenes/variants.
    Expected: fx=fy=725.0087, cx=620.5, cy=187.0 everywhere.
    """
    print("=" * 60)
    print("VKITTI2 Intrinsics Check")
    print("=" * 60)

    intrinsic_files = sorted(glob.glob(
        os.path.join(textgt_root, "**", "intrinsic.txt"),
        recursive=True
    ))

    if not intrinsic_files:
        print(f"  No intrinsic.txt files found under {textgt_root}")
        return False

    print(f"  Found {len(intrinsic_files)} intrinsic.txt files")

    all_same = True
    unique_Ks = {}
    total_frames = 0

    for fpath in intrinsic_files:
        rel = os.path.relpath(fpath, textgt_root)
        try:
            entries = parse_vkitti2_intrinsic(fpath)
        except Exception as e:
            print(f"  ERROR parsing {fpath}: {e}")
            # Print first few lines to help debug format
            try:
                with open(fpath, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 3:
                            print(f"    line {i}: {line.rstrip()}")
            except:
                pass
            all_same = False
            continue

        if not entries:
            print(f"  WARNING: No entries parsed from {rel}")
            # Print first few lines
            try:
                with open(fpath, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 3:
                            print(f"    line {i}: {line.rstrip()}")
            except:
                pass
            continue

        total_frames += len(entries)

        for frame_id, cam_id, K in entries:
            K_tuple = tuple(K.flatten().round(6))
            if K_tuple not in unique_Ks:
                unique_Ks[K_tuple] = {
                    'K': K,
                    'count': 0,
                    'first_file': rel,
                    'first_frame': frame_id,
                    'cam_id': cam_id,
                }
            unique_Ks[K_tuple]['count'] += 1

    # Check per camera (left/right may differ)
    print(f"\n  Total frame entries: {total_frames}")
    print(f"  Unique K matrices found: {len(unique_Ks)}")

    for i, (k_tuple, info) in enumerate(unique_Ks.items()):
        K = info['K']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        skew = K[0, 1]
        print(f"\n  K#{i+1} ({info['count']} entries, cam={info['cam_id']}, "
              f"e.g. {info['first_file']} frame {info['first_frame']}):")
        print(f"    fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}  skew={skew:.6f}")
        print(f"    Full K:")
        for row in range(3):
            print(f"      [{K[row, 0]:10.4f} {K[row, 1]:10.4f} {K[row, 2]:10.4f}]")

    if all_same and len(unique_Ks) <= 2:
        # Allow 2 unique Ks: one per camera (left/right)
        print(f"\n  PASS: {'1 unique K (mono)' if len(unique_Ks) == 1 else '2 unique Ks (left/right camera)'}")
    else:
        print(f"\n  FAIL: Found {len(unique_Ks)} distinct intrinsic matrices (expected 1 or 2)")
        all_same = False

    return all_same


def main():
    parser = argparse.ArgumentParser(
        description="Verify camera intrinsics are constant across SYNTHIA and VKITTI2 datasets"
    )
    parser.add_argument('--synthia', type=str, default=None,
                        help='Path to SYNTHIA test/ directory containing scene folders')
    parser.add_argument('--vkitti2-textgt', type=str, default=None,
                        help='Path to vkitti_2.0.3_textgt/ directory')

    args = parser.parse_args()

    if args.synthia is None and args.vkitti2_textgt is None:
        parser.print_help()
        print("\nProvide at least one of --synthia or --vkitti2-textgt")
        sys.exit(1)

    results = {}

    if args.synthia:
        results['synthia'] = check_synthia(args.synthia)
        print()

    if args.vkitti2_textgt:
        results['vkitti2'] = check_vkitti2(args.vkitti2_textgt)
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
