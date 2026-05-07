#!/usr/bin/env python3
"""
Sensitivity analysis for v5_relative segmentation parameters.

Sweeps one parameter at a time while holding others at their baseline values.
For each config: re-segments from raw MoGe H5 → evaluates full metrics
(SC, RI, VOI, Precision/Recall @ 1/5/10mm, binary planarity F1).

Usage:
    # Sweep a single parameter (submit as separate SLURM job per param)
    python sensitivity_analysis_v5rel.py \
        --param threshold_planarity \
        --raw_root /path/to/raw_h5 \
        --output_dir /path/to/sensitivity_results

    # Sweep all parameters sequentially
    python sensitivity_analysis_v5rel.py \
        --param all \
        --raw_root /path/to/raw_h5 \
        --output_dir /path/to/sensitivity_results
"""

import os
import sys
import argparse
import time
import json
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import h5py
import cv2
import pandas as pd
from tqdm import tqdm

# Ensure planamono is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_relative
from planamono.shared.utils.label_utils import remap_labels
from planamono.evaluation.quantitative.eval_utils import (
    evaluate_single_frame,
    save_results_csv,
)
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path


# ============================================================
# CONSTANTS
# ============================================================

THRESHOLDS = (0.001, 0.005, 0.01)  # 1mm, 5mm, 10mm
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9

DATASET_DIR = scannetpp_rend_plane_path
RGB_ROOT = os.path.join(scannetpp_path, "data")
SPLIT_DIR = os.path.join(repo_path, "splits", "scannetpp")

# Baseline: MoGe 4ds ep2 v5_rel
BASELINE = {
    "threshold_planarity": 0.3,
    "normal_threshold_deg": 5.0,
    "depth_threshold": 0.025,
    "neighbor_match_count_thresh": 8,
}

# One-at-a-time sweeps
SWEEPS = {
    "threshold_planarity":          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "normal_threshold_deg":         [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
    "depth_threshold":              [0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
    "neighbor_match_count_thresh":  [2, 4, 6, 8, 12, 16, 20, 24],
}


# ============================================================
# RAW H5 LOADER
# ============================================================

class RawH5SceneLoader:
    """Lazy loader for raw MoGe H5 files — one frame at a time."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.frame_ids = [
                fid.decode() if isinstance(fid, bytes) else fid
                for fid in f["frame_ids"][:]
            ]
            self.num_frames = len(self.frame_ids)

    def __len__(self):
        return self.num_frames

    def load_frame(self, idx: int):
        with h5py.File(self.h5_path, "r") as f:
            planarity = f["planarity"][idx]   # (H, W)
            depth = f["depth"][idx]           # (H, W)
            normal = f["normal"][idx]         # (H, W, 3)
        return self.frame_ids[idx], planarity, depth, normal


# ============================================================
# SEGMENTATION
# ============================================================

def segment_frame(planarity, normal, depth, config):
    """Segment a single frame with v5_relative."""
    planarity_mask = (planarity > config["threshold_planarity"]).astype(np.int32)
    labels, _ = compute_vectorized_planar_segments_v5_relative(
        planarity_mask,
        normal,
        depth,
        np.deg2rad(config["normal_threshold_deg"]),
        config["depth_threshold"],
        neighbor_match_count_thresh=config["neighbor_match_count_thresh"],
        device="cuda",
    )
    labels, _ = remap_labels(labels)
    return labels.astype(np.int32)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset_frames(raw_root, max_scenes, split="test", max_frames_per_scene=None):
    """
    Load dataset GT and match with raw H5 files.

    Returns list of dicts with GT + raw H5 info per frame.
    """
    dataset = ScanNetPPPlaneDataset(
        rgb_root=RGB_ROOT,
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=DATASET_DIR,
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=SPLIT_DIR,
        split=split,
        max_scenes=max_scenes,
    )

    # Discover available raw scenes
    raw_scenes = set()
    if os.path.isdir(raw_root):
        raw_scenes = {
            d for d in os.listdir(raw_root)
            if os.path.isdir(os.path.join(raw_root, d))
            and os.path.isfile(os.path.join(raw_root, d, "moge_raw.h5"))
        }

    # Group dataset frames by scene
    scene_frames = defaultdict(list)
    for idx in range(len(dataset)):
        pair = dataset.valid_pairs[idx]
        rgb_path = pair[0]
        scene_id = rgb_path.split("/")[-4]
        if scene_id in raw_scenes:
            scene_frames[scene_id].append(idx)

    # Build frame list, optionally subsampling
    frames = []
    for scene_id in sorted(scene_frames.keys()):
        idxs = scene_frames[scene_id]
        if max_frames_per_scene and len(idxs) > max_frames_per_scene:
            step = max(1, len(idxs) // max_frames_per_scene)
            idxs = idxs[::step][:max_frames_per_scene]

        # Load raw H5 loader for this scene
        raw_h5 = os.path.join(raw_root, scene_id, "moge_raw.h5")
        loader = RawH5SceneLoader(raw_h5)
        frame_id_to_raw_idx = {fid: i for i, fid in enumerate(loader.frame_ids)}

        for ds_idx in idxs:
            sample = dataset[ds_idx]
            frame_idx = sample["frame_idx"]

            if frame_idx not in frame_id_to_raw_idx:
                continue

            raw_idx = frame_id_to_raw_idx[frame_idx]
            _, planarity, depth_moge, normal = loader.load_frame(raw_idx)

            gt_seg = sample["plane"].squeeze(0).numpy().astype(np.int32)
            depth_gt = sample["depth"].squeeze(0).numpy()
            K = sample["K"].numpy()
            c2w = sample["c2w"].numpy()

            # Resize raw outputs to GT resolution if needed
            H_gt, W_gt = gt_seg.shape
            H_raw, W_raw = planarity.shape
            if (H_raw, W_raw) != (H_gt, W_gt):
                planarity = cv2.resize(planarity, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
                depth_moge = cv2.resize(depth_moge, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
                normal = cv2.resize(normal, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)

            frames.append({
                "scene_id": scene_id,
                "frame_idx": frame_idx,
                "planarity": planarity.astype(np.float32),
                "depth_moge": depth_moge.astype(np.float32),
                "normal": normal.astype(np.float32),
                "gt_seg": gt_seg,
                "depth_gt": depth_gt.astype(np.float32),
                "K": K,
                "c2w": c2w,
            })

    return frames


# ============================================================
# SWEEP
# ============================================================

def evaluate_config(frames, config):
    """Segment + evaluate all frames with one config. Returns per-frame metrics."""
    results = {}

    for frame in frames:
        labels = segment_frame(
            frame["planarity"], frame["normal"], frame["depth_moge"], config
        )

        metrics, _ = evaluate_single_frame(
            scene_id=frame["scene_id"],
            frame_idx=frame["frame_idx"],
            depth_np=frame["depth_gt"],
            gt_seg_np=frame["gt_seg"],
            K_np=frame["K"],
            c2w_np=frame["c2w"],
            labels=labels,
            thresholds=THRESHOLDS,
            compute_plane_metrics_flag=True,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE,
        )
        results[(frame["scene_id"], frame["frame_idx"])] = metrics

    return results


def run_sweep(param_name, frames, output_dir):
    """Sweep one parameter over its range, save per-value results."""
    values = SWEEPS[param_name]
    summary_rows = []

    param_dir = os.path.join(output_dir, f"sweep_{param_name}")
    os.makedirs(param_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Sweeping: {param_name} ({len(values)} values)")
    print(f"Baseline: {BASELINE}")
    print(f"Output:   {param_dir}")
    print(f"{'='*60}")

    for val in values:
        config = BASELINE.copy()
        config[param_name] = val
        is_baseline = (val == BASELINE[param_name])
        tag = " <-- BASELINE" if is_baseline else ""

        t0 = time.time()
        results = evaluate_config(frames, config)
        elapsed = time.time() - t0

        # Save per-frame results
        val_dir = os.path.join(param_dir, f"{param_name}={val}")
        df_frames, df_scenes, df_dataset = save_results_csv(results, val_dir)

        # Aggregate for summary
        numeric_cols = df_frames.select_dtypes(include="number").columns
        means = df_frames[numeric_cols].mean().to_dict()
        means["param_value"] = val
        means["n_frames"] = len(df_frames)
        means["time_s"] = elapsed
        summary_rows.append(means)

        sc = means.get("sc", 0)
        ri = means.get("rand_index", 0)
        p5 = means.get("prec@0.5cm", 0)
        r5 = means.get("rec@0.5cm", 0)
        f1 = means.get("bp_f1", 0)
        print(f"  {param_name}={val:>8}: SC={sc:.4f} RI={ri:.4f} "
              f"P@5mm={p5:.4f} R@5mm={r5:.4f} F1={f1:.4f} "
              f"({elapsed:.1f}s){tag}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(param_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] {summary_path}")

    return summary_df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: v5_relative segmentation parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--param", type=str, required=True,
                        choices=list(SWEEPS.keys()) + ["all"],
                        help="Which parameter to sweep (or 'all')")
    parser.add_argument("--raw_root", type=str, required=True,
                        help="Root of raw MoGe H5 files (from save_moge_raw.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max_scenes", type=int, default=10,
                        help="Number of scenes to evaluate (default: 10)")
    parser.add_argument("--max_frames_per_scene", type=int, default=None,
                        help="Max frames per scene (None = all frames)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])

    args = parser.parse_args()

    print("=" * 60)
    print("Sensitivity Analysis: v5_relative Segmentation")
    print("=" * 60)
    print(f"Parameter:   {args.param}")
    print(f"Raw root:    {args.raw_root}")
    print(f"Output:      {args.output_dir}")
    print(f"Max scenes:  {args.max_scenes}")
    print(f"Max frames:  {args.max_frames_per_scene or 'all'}")
    print(f"Split:       {args.split}")
    print(f"Baseline:    {BASELINE}")
    print("=" * 60)

    # Load data
    print("\n[INFO] Loading frames...")
    frames = load_dataset_frames(
        args.raw_root,
        max_scenes=args.max_scenes,
        split=args.split,
        max_frames_per_scene=args.max_frames_per_scene,
    )
    print(f"[INFO] Loaded {len(frames)} frames")

    if len(frames) == 0:
        print("[ERROR] No frames found. Check raw_root and dataset paths.")
        sys.exit(1)

    # Save config
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "baseline": BASELINE,
            "sweeps": SWEEPS if args.param == "all" else {args.param: SWEEPS[args.param]},
            "n_scenes": args.max_scenes,
            "n_frames": len(frames),
            "split": args.split,
            "thresholds": list(THRESHOLDS),
            "ransac_iterations": RANSAC_ITERATIONS,
            "inlier_ratio_gate": INLIER_RATIO_GATE,
        }, f, indent=2)

    # Run sweeps
    params_to_sweep = list(SWEEPS.keys()) if args.param == "all" else [args.param]

    for param_name in params_to_sweep:
        run_sweep(param_name, frames, args.output_dir)

    print("\n[DONE] Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
