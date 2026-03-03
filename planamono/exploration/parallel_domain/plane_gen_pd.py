#!/usr/bin/env python3
"""
Parallel Domain Plane Extraction.

Processes PD NPZ samples, extracts planes via LO-RANSAC from depth maps
(no semantic labels — all valid depth pixels are candidates), and saves
per-sample results for comparison with PD's provided plane ground truth.

PD dataset has no semantic segmentation, so we use purely geometric
plane extraction on all valid depth pixels.

Usage:
    python plane_gen_pd.py \
        --data_root /cluster/scratch/ayavuz/dataset/pd_zero/parallel_domain_plane \
        --output_root /cluster/scratch/ayavuz/dataset/pd_zero/pd_ours_planes \
        --split val

    # Quick test (5 samples):
    python plane_gen_pd.py \
        --data_root /cluster/scratch/ayavuz/dataset/pd_zero/parallel_domain_plane \
        --output_root /tmp/pd_planes_test \
        --split val --max_samples 5

Output per sample (NPZ):
    plane_map:    (H, W) int32   (0 = non-planar, 1..N = plane instances)
    num_planes:   scalar int
    planes_n:     (N, 3) float64 (unit normals)
    planes_d:     (N,)   float64 (distances in meters, n.x = d)
    planes_p95:   (N,)   float64 (95th percentile residual)
    planes_npix:  (N,)   int64   (pixel count per plane)
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Local import — ransac_outdoor.py lives next to this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ransac_outdoor import depth_to_xyz, compute_normals, extract_planes_from_frame

# ── PD camera intrinsics at native 192x256 ──────────────────────────
PD_FX = 166.53
PD_FY = 166.36
PD_CX = 128.0
PD_CY = 96.0
PD_H, PD_W = 192, 256
HIRES_H, HIRES_W = 480, 640


def process_pd_dataset(args):
    """Process all samples in a PD split."""
    data_root = args.data_root
    split = args.split

    # Discover NPZ files
    npz_files = sorted(
        [f for f in os.listdir(data_root)
         if f.startswith(f"{split}_") and f.endswith("_d2.npz")],
        key=lambda f: int(f.split("_")[1]),
    )
    if not npz_files:
        print(f"[ERROR] No {split}_*_d2.npz files in {data_root}")
        sys.exit(1)

    if args.max_samples is not None:
        npz_files = npz_files[:args.max_samples]

    print(f"[INFO] Processing {len(npz_files)} PD {split} samples")

    # Intrinsics at working resolution
    if args.use_hires:
        fx = PD_FX * HIRES_W / PD_W
        fy = PD_FY * HIRES_H / PD_H
        cx = PD_CX * HIRES_W / PD_W
        cy = PD_CY * HIRES_H / PD_H
        print(f"[INFO] Using high-res ({HIRES_H}x{HIRES_W}), "
              f"K: fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")
    else:
        fx, fy, cx, cy = PD_FX, PD_FY, PD_CX, PD_CY
        print(f"[INFO] Using low-res ({PD_H}x{PD_W}), "
              f"K: fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")

    # Output directory
    out_dir = os.path.join(args.output_root, split)
    os.makedirs(out_dir, exist_ok=True)

    stats = []

    for npz_file in tqdm(npz_files, desc=f"PD {split}"):
        idx = int(npz_file.split("_")[1])
        d = np.load(os.path.join(data_root, npz_file), allow_pickle=True)

        # Load depth at chosen resolution
        if args.use_hires:
            depth = d['high_res_raw_depth'].astype(np.float32)
        else:
            depth = d['raw_depth'].astype(np.float32)

        valid = depth > 0
        H, W = depth.shape

        # Backproject to 3D and compute normals
        xyz = depth_to_xyz(depth, fx, fy, cx, cy)
        normals = compute_normals(xyz, valid)

        # No semantic labels → all valid pixels are "class 1"
        class_ids = np.ones((H, W), dtype=np.int32)
        class_ids[~valid] = 0

        # Extract planes via RANSAC
        plane_map, planes_info = extract_planes_from_frame(
            depth, class_ids, valid, xyz, normals,
            planar_classes={1},
            class_names={0: 'invalid', 1: 'all'},
            merge_compatible=[],
            ransac_iters=args.ransac_iters,
            dist_thresh=args.dist_thresh,
            normal_cos=args.normal_cos,
            min_inliers=args.min_inliers,
            max_planes=args.max_planes,
            merge_normal=args.merge_normal,
            merge_rel_dist=args.merge_rel_dist,
        )

        num_planes = len(planes_info)

        # Extract plane parameters
        if num_planes > 0:
            planes_n = np.array([p['n'] for p in planes_info])
            planes_d = np.array([p['d'] for p in planes_info])
            planes_p95 = np.array([p['p95'] for p in planes_info])
            planes_npix = np.array([p['num_pixels'] for p in planes_info])
        else:
            planes_n = np.zeros((0, 3), dtype=np.float64)
            planes_d = np.zeros((0,), dtype=np.float64)
            planes_p95 = np.zeros((0,), dtype=np.float64)
            planes_npix = np.zeros((0,), dtype=np.int64)

        # Save per-sample NPZ
        out_path = os.path.join(out_dir, f"{split}_{idx}_planes.npz")
        np.savez_compressed(
            out_path,
            plane_map=plane_map.astype(np.int32),
            num_planes=np.int64(num_planes),
            planes_n=planes_n,
            planes_d=planes_d,
            planes_p95=planes_p95,
            planes_npix=planes_npix,
        )

        stats.append({
            'idx': idx,
            'num_planes': num_planes,
            'planar_frac': float((plane_map > 0).sum()) / (H * W),
        })

    # Print summary
    if stats:
        n_planes = [s['num_planes'] for s in stats]
        p_fracs = [s['planar_frac'] for s in stats]
        print(f"\n[DONE] {len(stats)} samples processed")
        print(f"  Planes per frame: mean={np.mean(n_planes):.1f}, "
              f"median={np.median(n_planes):.0f}, "
              f"range=[{min(n_planes)}, {max(n_planes)}]")
        print(f"  Planar fraction: mean={np.mean(p_fracs):.3f}, "
              f"range=[{min(p_fracs):.3f}, {max(p_fracs):.3f}]")
        print(f"  Output: {out_dir}")
    else:
        print("[WARN] No samples processed.")


def main():
    parser = argparse.ArgumentParser(
        description='Parallel Domain Plane Extraction (RANSAC on depth)')

    parser.add_argument('--data_root', type=str, required=True,
                        help='PD data directory containing NPZ files')
    parser.add_argument('--output_root', type=str, required=True,
                        help='Output directory for extracted planes')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split to process')
    parser.add_argument('--use_hires', action='store_true',
                        help='Use 480x640 high-res depth (default: 192x256)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples (for testing)')

    # RANSAC parameters (outdoor defaults)
    parser.add_argument('--ransac_iters', type=int, default=500)
    parser.add_argument('--dist_thresh', type=float, default=0.01,
                        help='RANSAC inlier distance threshold (meters)')
    parser.add_argument('--normal_cos', type=float, default=0.985,
                        help='Minimum cosine similarity for normal consistency')
    parser.add_argument('--min_inliers', type=int, default=50,
                        help='Minimum inliers to accept a plane')
    parser.add_argument('--max_planes', type=int, default=50,
                        help='Maximum planes to extract per frame')
    parser.add_argument('--merge_normal', type=float, default=0.985,
                        help='Normal cosine threshold for merging')
    parser.add_argument('--merge_rel_dist', type=float, default=0.02,
                        help='Relative distance threshold for merging')

    args = parser.parse_args()
    process_pd_dataset(args)


if __name__ == '__main__':
    main()
