#!/usr/bin/env python3
"""
Segment moge_signals.h5 → planes.h5 (for evaluate_all_baselines.py).

Reads per-scene moge_signals.h5 (keys: planarity, normal, depth_metric) produced by
save_moge_signals_planarity.py and writes per-scene planes.h5 with the schema
evaluate_all_baselines.py consumes (key "planes" uint16, 0 = non-planar; "frame_ids").

Segmentation = OUR method: compute_planar_segments with
    threshold_planarity        0.3   (planarity mask cutoff)
    normal_threshold_rad       deg2rad(5.0)
    depth_threshold            0.025 (relative, 2.5% of center depth)
    neighbor_match_count_thresh 8
These are the canonical parameters used across the benchmark (see evaluate_all_baselines.py).

Scene sharding via --part_id / --num_parts (contiguous slices of the sorted scene list),
so this can be fanned out across SLURM jobs. All parts write to the same --output_root
(one subdir per scene → no collisions).

Usage:
    python segment_signals_to_planes.py \
        --input_root  /.../moge_signals_4ds_ep1_640x480/scannetpp \
        --output_root /.../moge_planes_4ds_ep1_640x480 \
        --part_id 0 --num_parts 15
"""
import argparse
import os
import sys
import time

import numpy as np
import h5py
from tqdm import tqdm

from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pxwplanar.shared.segmentation import compute_planar_segments
from pxwplanar.shared.utils.label_utils import remap_labels

# ── OUR-method segmentation params (match the benchmark's internal segmentation) ──
THRESHOLD_PLANARITY = 0.3
NORMAL_THRESHOLD_DEG = 5.0
DEPTH_THRESHOLD_REL = 0.025
NEIGHBOR_MATCH_COUNT = 8


def split_scenes(scenes, part_id, num_parts):
    """Contiguous slice of a sorted scene list; remainder → first parts."""
    n = len(scenes)
    base, rem = divmod(n, num_parts)
    start = part_id * base + min(part_id, rem)
    end = start + base + (1 if part_id < rem else 0)
    return scenes[start:end]


def segment_scene(sig_h5_path, out_h5_path, device):
    """Read one scene's moge_signals.h5, segment every frame, write planes.h5."""
    with h5py.File(sig_h5_path, "r") as f:
        frame_ids = [x.decode() if isinstance(x, bytes) else x for x in f["frame_ids"][:]]
        N = len(frame_ids)
        H, W = f["planarity"].shape[1:3]
        planes = np.zeros((N, H, W), dtype=np.uint16)
        for i in range(N):
            planarity = f["planarity"][i].astype(np.float32)
            normal = f["normal"][i].astype(np.float32)
            depth = f["depth_metric"][i].astype(np.float32)
            mask = (planarity > THRESHOLD_PLANARITY).astype(np.int32)
            labels, _ = compute_planar_segments(
                mask, normal, depth,
                np.deg2rad(NORMAL_THRESHOLD_DEG),
                DEPTH_THRESHOLD_REL,
                neighbor_match_count_thresh=NEIGHBOR_MATCH_COUNT,
                device=device,
            )
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            labels, _ = remap_labels(labels)
            if labels.max() >= 65536:
                raise ValueError(f"{sig_h5_path} frame {i}: {labels.max()} labels "
                                 "exceed the uint16 planes schema")
            planes[i] = labels.astype(np.uint16)

    # Write to a temp name and rename on success so a killed run cannot leave a
    # truncated planes.h5 that a rerun skips as complete (same pattern as the
    # signals exporter).
    os.makedirs(os.path.dirname(out_h5_path), exist_ok=True)
    tmp_h5 = out_h5_path + ".tmp"
    with h5py.File(tmp_h5, "w") as f:
        f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
        f.create_dataset("frame_ids", data=np.array(frame_ids, dtype="S"))
    os.replace(tmp_h5, out_h5_path)
    return N


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", required=True,
                    help="Dir containing <scene>/moge_signals.h5")
    ap.add_argument("--output_root", required=True,
                    help="Dir to write <scene>/planes.h5")
    ap.add_argument("--part_id", type=int, default=0)
    ap.add_argument("--num_parts", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.num_parts < 1 or not (0 <= args.part_id < args.num_parts):
        ap.error(f"--part_id {args.part_id} / --num_parts {args.num_parts}: "
                 "need num_parts >= 1 and 0 <= part_id < num_parts")
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available — falling back to --device cpu")
            args.device = "cpu"

    scenes = sorted([
        d for d in os.listdir(args.input_root)
        if os.path.isfile(os.path.join(args.input_root, d, "moge_signals.h5"))
    ])
    if not scenes:
        sys.exit(f"[ERROR] no <scene>/moge_signals.h5 under {args.input_root}")

    part = split_scenes(scenes, args.part_id, args.num_parts)
    print(f"[INFO] part {args.part_id}/{args.num_parts}: {len(part)}/{len(scenes)} scenes "
          f"(seg=compute_planar_segments plan>{THRESHOLD_PLANARITY} n<{NORMAL_THRESHOLD_DEG}° "
          f"d_rel<{DEPTH_THRESHOLD_REL} match≥{NEIGHBOR_MATCH_COUNT}, device={args.device})")

    t0 = time.perf_counter()
    total = 0
    for sid in tqdm(part, desc=f"part{args.part_id}", unit="scene"):
        out_h5 = os.path.join(args.output_root, sid, "planes.h5")
        if os.path.isfile(out_h5) and not args.overwrite:
            print(f"  [skip] {sid}: planes.h5 exists")
            continue
        sig_h5 = os.path.join(args.input_root, sid, "moge_signals.h5")
        n = segment_scene(sig_h5, out_h5, args.device)
        total += n
    dt = time.perf_counter() - t0
    print(f"[DONE] part {args.part_id}: {total} frames, {len(part)} scenes in {dt:.1f}s "
          f"({total/dt:.1f} fps)" if dt > 0 else f"[DONE] part {args.part_id}")


if __name__ == "__main__":
    main()
