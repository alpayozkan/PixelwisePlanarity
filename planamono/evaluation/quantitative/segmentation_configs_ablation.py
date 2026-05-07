#!/usr/bin/env python3
"""
Segmentation parameter configs ablation on ScanNet++.

Evaluates 5 predefined configs (loose → strict) using raw MoGe H5 outputs.
Each config varies planarity threshold, normal threshold, depth threshold,
and neighbor match count. Config 3 is the default.

Outputs per-config results to RESULTS_DIR/<config_name>/ and a summary CSV.

Usage:
    # All configs
    python segmentation_configs_ablation.py \
        --raw_root /path/to/raw_h5 \
        --output_dir /path/to/results

    # Single config
    python segmentation_configs_ablation.py \
        --raw_root /path/to/raw_h5 \
        --output_dir /path/to/results \
        --configs config3_default

    # With SLURM parallelism (one config per job)
    python segmentation_configs_ablation.py \
        --raw_root /path/to/raw_h5 \
        --output_dir /path/to/results \
        --configs config1_loose
"""

import os
import sys
import argparse
import time
import json
from collections import defaultdict, OrderedDict

import numpy as np
import h5py
import cv2
import pandas as pd
from tqdm import tqdm

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


# ============================================================
# 5 CONFIGS: loose (config1) → strict (config5)
# config3 = default parameters
# ============================================================

CONFIGS = OrderedDict([
    ("config1_loose", {
        "threshold_planarity": 0.1,
        "normal_threshold_deg": 10.0,
        "depth_threshold": 0.05,
        "neighbor_match_count_thresh": 4,
        "description": "Loose: low planarity threshold, permissive merging",
    }),
    ("config2_relaxed", {
        "threshold_planarity": 0.2,
        "normal_threshold_deg": 7.0,
        "depth_threshold": 0.035,
        "neighbor_match_count_thresh": 6,
        "description": "Relaxed: moderate-low thresholds",
    }),
    ("config3_default", {
        "threshold_planarity": 0.3,
        "normal_threshold_deg": 5.0,
        "depth_threshold": 0.025,
        "neighbor_match_count_thresh": 8,
        "description": "Default: standard parameters",
    }),
    ("config4_moderate", {
        "threshold_planarity": 0.5,
        "normal_threshold_deg": 3.0,
        "depth_threshold": 0.015,
        "neighbor_match_count_thresh": 12,
        "description": "Moderate: tighter thresholds",
    }),
    ("config5_strict", {
        "threshold_planarity": 0.7,
        "normal_threshold_deg": 2.0,
        "depth_threshold": 0.01,
        "neighbor_match_count_thresh": 16,
        "description": "Strict: high planarity threshold, tight geometric checks",
    }),
])


# ============================================================
# RAW H5 LOADER
# ============================================================

class RawH5SceneLoader:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.frame_ids = [
                fid.decode() if isinstance(fid, bytes) else fid
                for fid in f["frame_ids"][:]
            ]
            self.num_frames = len(self.frame_ids)

    def __len__(self):
        return self.num_frames

    def load_frame(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            planarity = f["planarity"][idx]
            depth = f["depth"][idx]
            normal = f["normal"][idx]
        return self.frame_ids[idx], planarity, depth, normal


# ============================================================
# SEGMENTATION
# ============================================================

def segment_frame(planarity, normal, depth, config):
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

def load_dataset_frames(raw_root, max_scenes, split="test"):
    dataset = ScanNetPPPlaneDataset(
        rgb_root=RGB_ROOT,
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=DATASET_DIR,
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=SPLIT_DIR,
        split=split,
        max_scenes=max_scenes,
    )

    raw_scenes = set()
    if os.path.isdir(raw_root):
        raw_scenes = {
            d for d in os.listdir(raw_root)
            if os.path.isdir(os.path.join(raw_root, d))
            and os.path.isfile(os.path.join(raw_root, d, "moge_raw.h5"))
        }

    scene_frames = defaultdict(list)
    for idx in range(len(dataset)):
        pair = dataset.valid_pairs[idx]
        rgb_path = pair[0]
        scene_id = rgb_path.split("/")[-4]
        if scene_id in raw_scenes:
            scene_frames[scene_id].append(idx)

    frames = []
    for scene_id in sorted(scene_frames.keys()):
        idxs = scene_frames[scene_id]
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
# EVALUATE ONE CONFIG
# ============================================================

def evaluate_config(frames, config, config_name, output_dir):
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    results = {}
    t0 = time.time()

    for frame in tqdm(frames, desc=config_name):
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

    elapsed = time.time() - t0

    # Save per-frame, per-scene, dataset-level CSVs
    df_frames, df_scenes, df_dataset = save_results_csv(results, config_dir)

    # Save config
    with open(os.path.join(config_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Summary line
    numeric_cols = df_frames.select_dtypes(include="number").columns
    means = df_frames[numeric_cols].mean().to_dict()
    means["config"] = config_name
    means["n_frames"] = len(df_frames)
    means["time_s"] = elapsed

    sc = means.get("sc", 0)
    ri = means.get("rand_index", 0)
    p5 = means.get("prec@0.5cm", 0)
    r5 = means.get("rec@0.5cm", 0)
    bp_f1 = means.get("bp_f1", 0)
    print(f"  {config_name}: SC={sc:.4f} RI={ri:.4f} "
          f"P@5mm={p5:.4f} R@5mm={r5:.4f} bp_F1={bp_f1:.4f} ({elapsed:.1f}s)")

    return means


# ============================================================
# AGGREGATE
# ============================================================

def aggregate_results(output_dir, config_names):
    """Read results_dataset.csv from each config dir and build a summary table."""
    from planamono.evaluation.quantitative.create_unified_tables import (
        read_results, fmt, compute_f1,
    )

    thresholds = ["0.1", "0.5", "1.0"]
    header = ["Config", "Description", "plan_thresh", "normal_deg", "depth_thresh", "match_count",
              "num_frames"]
    header.extend(["RI", "VOI", "SC"])
    for t in thresholds:
        header.extend([f"P@{t}cm", f"R@{t}cm", f"F1@{t}cm"])
    header.extend(["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"])

    rows = []
    for config_name in config_names:
        cfg = CONFIGS[config_name]
        data = read_results(output_dir, config_name)
        if data is None:
            print(f"  Skipping {config_name} (no results_dataset.csv)")
            continue

        row = [config_name, cfg.get("description", "")]
        row.append(cfg["threshold_planarity"])
        row.append(cfg["normal_threshold_deg"])
        row.append(cfg["depth_threshold"])
        row.append(cfg["neighbor_match_count_thresh"])
        row.append(data.get("num_frames_total", ""))
        row.append(fmt(data.get("rand_index_mean", "")))
        row.append(fmt(data.get("voi_mean", "")))
        row.append(fmt(data.get("sc_mean", "")))

        for t in thresholds:
            p = data.get(f"prec@{t}cm_mean", "")
            r = data.get(f"rec@{t}cm_mean", "")
            f1 = compute_f1(p, r)
            row.extend([fmt(p), fmt(r), fmt(f1)])

        row.append(fmt(data.get("bp_accuracy_mean", "")))
        row.append(fmt(data.get("bp_precision_mean", "")))
        row.append(fmt(data.get("bp_recall_mean", "")))
        row.append(fmt(data.get("bp_f1_mean", "")))
        row.append(fmt(data.get("bp_iou_mean", "")))

        rows.append(row)
        print(f"  OK: {config_name}")

    import csv
    out_path = os.path.join(output_dir, "table_configs_ablation.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"\n  Written: {out_path} ({len(rows)} configs)")
    return out_path


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Segmentation configs ablation (5 configs, loose → strict)",
    )
    parser.add_argument("--raw_root", type=str, required=True,
                        help="Root of raw MoGe H5 files (from save_moge_raw.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--configs", nargs="+",
                        default=list(CONFIGS.keys()),
                        choices=list(CONFIGS.keys()),
                        help="Which configs to evaluate (default: all)")
    parser.add_argument("--max_scenes", type=int, default=10)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results (skip evaluation)")

    args = parser.parse_args()

    if args.aggregate_only:
        print("=" * 60)
        print("Aggregating existing results")
        print("=" * 60)
        aggregate_results(args.output_dir, args.configs)
        return

    print("=" * 60)
    print("Segmentation Configs Ablation")
    print("=" * 60)
    print(f"Raw root:    {args.raw_root}")
    print(f"Output:      {args.output_dir}")
    print(f"Configs:     {args.configs}")
    print(f"Max scenes:  {args.max_scenes}")
    print(f"Split:       {args.split}")
    print("=" * 60)

    # Print config summary
    for name in args.configs:
        cfg = CONFIGS[name]
        print(f"  {name}: plan={cfg['threshold_planarity']} "
              f"norm={cfg['normal_threshold_deg']}° "
              f"depth={cfg['depth_threshold']} "
              f"match={cfg['neighbor_match_count_thresh']} "
              f"({cfg['description']})")
    print()

    # Load data once
    print("[INFO] Loading frames...")
    frames = load_dataset_frames(
        args.raw_root,
        max_scenes=args.max_scenes,
        split=args.split,
    )
    print(f"[INFO] Loaded {len(frames)} frames from {len(set(f['scene_id'] for f in frames))} scenes\n")

    if len(frames) == 0:
        print("[ERROR] No frames found. Check raw_root and dataset paths.")
        sys.exit(1)

    # Save run config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump({
            "configs": {k: v for k, v in CONFIGS.items() if k in args.configs},
            "n_scenes": args.max_scenes,
            "n_frames": len(frames),
            "split": args.split,
            "thresholds": list(THRESHOLDS),
        }, f, indent=2)

    # Evaluate each config
    summary_rows = []
    for config_name in args.configs:
        cfg = CONFIGS[config_name]
        seg_params = {k: v for k, v in cfg.items() if k != "description"}
        means = evaluate_config(frames, seg_params, config_name, args.output_dir)
        summary_rows.append(means)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] {summary_path}")

    # Aggregate into table
    print("\n[INFO] Aggregating results...")
    aggregate_results(args.output_dir, args.configs)

    print("\n[DONE] Segmentation configs ablation complete.")


if __name__ == "__main__":
    main()
