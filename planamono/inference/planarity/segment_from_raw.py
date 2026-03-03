#!/usr/bin/env python3
"""
Stage 2: Raw H5 → Segmented Labels H5

Reads raw MoGe outputs (planarity, depth, normals) from Stage 1 H5 files,
applies planarity threshold + segmentation, and writes evaluation-compatible
planes.h5 files.

Supports single-config and grid-search modes for fast parameter tuning
without re-running GPU inference.

Usage (single config):
    python segment_from_raw.py \
        --raw_root /path/to/moge_raw \
        --output_root /path/to/labels_h5 \
        --dataset scannetpp \
        --seg_version v10 \
        --threshold_planarity 0.3

Usage (grid search):
    python segment_from_raw.py \
        --raw_root /path/to/moge_raw \
        --output_root /path/to/labels_h5 \
        --dataset scannetpp \
        --grid_config grid_config.yaml

Grid config YAML format:
    base:
        seg_version: v10
        normal_threshold_deg: 5.0
        depth_threshold: 0.025
        adaptive_frac: 0.75
        min_valid_neighbors: 3
        min_segment_pixels: 50
    sweep:
        threshold_planarity: [0.2, 0.3, 0.4, 0.5]
"""

import os
import sys
import argparse
import itertools
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import h5py
import yaml
from tqdm import tqdm

from planamono.shared.segmentation import compute_vectorized_planar_segments_v5
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_relative
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_no_sobel
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_dotprod_relative
from planamono.shared.segmentation import compute_vectorized_planar_segments_v6
from planamono.shared.segmentation import compute_vectorized_planar_segments_v10
from planamono.shared.segmentation import compute_vectorized_planar_segments_v11
from planamono.shared.utils.label_utils import remap_labels

SEG_FN_MAP = {
    "v5": compute_vectorized_planar_segments_v5,
    "v5_relative": compute_vectorized_planar_segments_v5_relative,
    "v5_no_sobel": compute_vectorized_planar_segments_v5_no_sobel,
    "v5_dotprod_relative": compute_vectorized_planar_segments_v5_dotprod_relative,
    "v6": compute_vectorized_planar_segments_v6,
    "v10": compute_vectorized_planar_segments_v10,
    "v11": compute_vectorized_planar_segments_v11,
}


# ============================================================
# TIMING
# ============================================================

class Timer:
    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.start_time = time.perf_counter()

    @contextmanager
    def __call__(self, name: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.timings[name] += elapsed
        self.counts[name] += 1

    def total_elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def print_summary(self, num_frames: int = 0):
        total = self.total_elapsed()
        print("\n" + "=" * 60)
        print("RUNTIME BREAKDOWN")
        print("=" * 60)
        for k, v in sorted(self.timings.items(), key=lambda x: -x[1]):
            count = self.counts[k]
            avg_ms = (v / count * 1000) if count > 0 else 0
            pct = (v / total * 100) if total > 0 else 0
            print(f"{k:25s} {v:>8.2f}s ({count:>6d} calls, {avg_ms:>8.2f}ms avg, {pct:>5.1f}%)")
        print("-" * 60)
        print(f"{'TOTAL':25s} {total:>8.2f}s")
        if num_frames > 0:
            print(f"{'Throughput':25s} {num_frames / total:>8.2f} fps")
        print("=" * 60)


# ============================================================
# RAW H5 LOADER
# ============================================================

class RawH5SceneLoader:
    """Lazy loader for raw MoGe H5 files with per-frame chunked reads."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        # Read frame_ids once to know the mapping
        with h5py.File(h5_path, "r") as f:
            self.frame_ids = [fid.decode() if isinstance(fid, bytes) else fid
                              for fid in f["frame_ids"][:]]
            self.num_frames = len(self.frame_ids)

    def __len__(self):
        return self.num_frames

    def load_frame(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        """Load a single frame by index. Returns (frame_id, planarity, depth, normal)."""
        with h5py.File(self.h5_path, "r") as f:
            planarity = f["planarity"][idx]   # (H, W)
            depth = f["depth"][idx]           # (H, W)
            normal = f["normal"][idx]         # (H, W, 3)
        return self.frame_ids[idx], planarity, depth, normal


# ============================================================
# SEGMENTATION
# ============================================================

def segment_single_frame(
    planarity: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Apply planarity threshold + segmentation to a single frame.

    Args:
        planarity: (H, W) float32, [0, 1] probability
        depth: (H, W) float32
        normal: (H, W, 3) float32, unit normals
        config: Segmentation parameters

    Returns:
        labels: (H, W) uint16, 0 = non-planar, 1..N = plane instances
    """
    seg_version = config["seg_version"]
    seg_fn = SEG_FN_MAP[seg_version]

    # Apply planarity threshold
    planarity_mask = (planarity > config["threshold_planarity"]).astype(np.int32)

    # Build segmentation kwargs
    seg_kwargs = dict(
        neighbor_match_count_thresh=config.get("neighbor_match_count_thresh", 18),
    )
    if "device" in config:
        seg_kwargs["device"] = config["device"]

    if seg_version in ("v10", "v11"):
        seg_kwargs.update(
            adaptive_frac=config.get("adaptive_frac", 0.75),
            min_valid_neighbors=config.get("min_valid_neighbors", 3),
            min_segment_pixels=config.get("min_segment_pixels", 50),
        )
    if seg_version == "v11":
        seg_kwargs.update(
            normal_adaptive_frac=config.get("normal_adaptive_frac", None),
            depth_adaptive_frac=config.get("depth_adaptive_frac", None),
            merge_enabled=config.get("merge_enabled", False),
            merge_normal_deg=config.get("merge_normal_deg", 10.0),
            merge_offset_m=config.get("merge_offset_m", 0.02),
            merge_min_pixels=config.get("merge_min_pixels", 50),
            merge_gap_px=config.get("merge_gap_px", 5),
        )

    labels, _ = seg_fn(
        planarity_mask,
        normal,          # (H, W, 3) — already in correct format from raw H5
        depth,
        np.deg2rad(config["normal_threshold_deg"]),
        config["depth_threshold"],
        **seg_kwargs,
    )

    labels, _ = remap_labels(labels)
    return labels.astype(np.uint16)


# ============================================================
# SCENE PROCESSING
# ============================================================

def process_scene_scannetpp(
    raw_h5_path: str,
    output_h5_path: str,
    config: dict,
    timer: Timer,
) -> int:
    """Process a single ScanNet++ scene: read raw → segment → write planes.h5."""
    loader = RawH5SceneLoader(raw_h5_path)
    all_labels = []
    frame_ids_list = []

    for idx in range(len(loader)):
        with timer("load_frame"):
            frame_id, planarity, depth, normal = loader.load_frame(idx)

        with timer("segment"):
            labels = segment_single_frame(planarity, depth, normal, config)

        all_labels.append(labels)
        frame_ids_list.append(frame_id)

    # Write output
    with timer("save_h5"):
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        planes = np.stack(all_labels, axis=0)  # (N, H, W)
        with h5py.File(output_h5_path, "w") as f:
            f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
            f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))

    return len(loader)


def process_scene_hypersim(
    scene_dir: str,
    output_scene_dir: str,
    config: dict,
    timer: Timer,
) -> int:
    """Process a single Hypersim scene: iterate per-camera raw H5 files."""
    total_frames = 0

    # Find all moge_raw_cam_XX.h5 files
    raw_files = sorted([
        f for f in os.listdir(scene_dir)
        if f.startswith("moge_raw_") and f.endswith(".h5")
    ])

    for raw_filename in raw_files:
        # Extract cam_name: moge_raw_cam_XX.h5 → cam_XX
        cam_name = raw_filename.replace("moge_raw_", "").replace(".h5", "")

        raw_h5_path = os.path.join(scene_dir, raw_filename)
        output_h5_path = os.path.join(output_scene_dir, f"planes_{cam_name}.h5")

        loader = RawH5SceneLoader(raw_h5_path)
        all_labels = []
        frame_ids_list = []

        for idx in range(len(loader)):
            with timer("load_frame"):
                frame_id, planarity, depth, normal = loader.load_frame(idx)

            with timer("segment"):
                labels = segment_single_frame(planarity, depth, normal, config)

            all_labels.append(labels)
            frame_ids_list.append(frame_id)

        with timer("save_h5"):
            os.makedirs(output_scene_dir, exist_ok=True)
            planes = np.stack(all_labels, axis=0)
            with h5py.File(output_h5_path, "w") as f:
                f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
                f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))

        total_frames += len(loader)

    return total_frames


# ============================================================
# GRID SEARCH
# ============================================================

def load_grid_configs(grid_config_path: str) -> List[dict]:
    """Load grid search config and generate cartesian product of sweep parameters."""
    with open(grid_config_path, "r") as f:
        grid = yaml.safe_load(f)

    base = grid.get("base", {})
    sweep = grid.get("sweep", {})

    if not sweep:
        return [base]

    # Generate cartesian product
    sweep_keys = list(sweep.keys())
    sweep_values = [sweep[k] if isinstance(sweep[k], list) else [sweep[k]] for k in sweep_keys]

    configs = []
    for combo in itertools.product(*sweep_values):
        config = dict(base)
        for key, val in zip(sweep_keys, combo):
            config[key] = val
        configs.append(config)

    return configs


def config_to_dirname(config: dict) -> str:
    """Generate a directory name from config parameters for grid search output."""
    parts = []
    parts.append(f"plan{config['threshold_planarity']}")
    parts.append(config["seg_version"])

    # Include non-default params
    if config.get("normal_threshold_deg", 5.0) != 5.0:
        parts.append(f"ndeg{config['normal_threshold_deg']}")
    if config.get("depth_threshold", 0.025) != 0.025:
        parts.append(f"dt{config['depth_threshold']}")
    if config.get("adaptive_frac", 0.75) != 0.75:
        parts.append(f"af{config['adaptive_frac']}")
    if config.get("min_segment_pixels", 50) != 50:
        parts.append(f"msp{config['min_segment_pixels']}")
    if config.get("neighbor_match_count_thresh", 18) != 18:
        parts.append(f"nmc{config['neighbor_match_count_thresh']}")

    return "_".join(parts)


# ============================================================
# MAIN
# ============================================================

def run_segmentation(raw_root: str, output_root: str, dataset: str, config: dict, max_scenes: int = None):
    """Run segmentation on all scenes with a single config."""
    timer = Timer()
    total_frames = 0

    # Discover scenes
    scene_dirs = sorted([
        d for d in os.listdir(raw_root)
        if os.path.isdir(os.path.join(raw_root, d))
    ])
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    print(f"[INFO] Processing {len(scene_dirs)} scenes...")

    for scene_id in tqdm(scene_dirs, desc="Scenes"):
        scene_raw_dir = os.path.join(raw_root, scene_id)

        if dataset == "scannetpp":
            raw_h5_path = os.path.join(scene_raw_dir, "moge_raw.h5")
            if not os.path.isfile(raw_h5_path):
                print(f"[WARN] Missing {raw_h5_path}, skipping")
                continue
            output_h5_path = os.path.join(output_root, scene_id, "planes.h5")
            nf = process_scene_scannetpp(raw_h5_path, output_h5_path, config, timer)
        elif dataset == "hypersim":
            output_scene_dir = os.path.join(output_root, scene_id)
            nf = process_scene_hypersim(scene_raw_dir, output_scene_dir, config, timer)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        total_frames += nf

    timer.print_summary(num_frames=total_frames)

    # Save config
    config_path = os.path.join(output_root, "config.yml")
    os.makedirs(output_root, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[CONFIG] Saved parameters to {config_path}")

    return total_frames


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Raw H5 → Segmented Labels H5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--raw_root", type=str, required=True,
                        help="Root directory of Stage 1 raw H5 files")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory for segmented label H5 output")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["scannetpp", "hypersim"],
                        help="Dataset type (determines H5 naming convention)")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Limit number of scenes (for testing)")

    # Grid search mode
    parser.add_argument("--grid_config", type=str, default=None,
                        help="Path to grid search YAML config (overrides individual params)")

    # Single-config segmentation parameters
    parser.add_argument("--seg_version", type=str, default="v10",
                        choices=list(SEG_FN_MAP.keys()))
    parser.add_argument("--threshold_planarity", type=float, default=0.3)
    parser.add_argument("--normal_threshold_deg", type=float, default=5.0)
    parser.add_argument("--depth_threshold", type=float, default=0.025)
    parser.add_argument("--neighbor_match_count_thresh", type=int, default=18)
    parser.add_argument("--adaptive_frac", type=float, default=0.75)
    parser.add_argument("--min_valid_neighbors", type=int, default=3)
    parser.add_argument("--min_segment_pixels", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")

    # v11-specific
    parser.add_argument("--normal_adaptive_frac", type=float, default=None,
                        help="[v11] Separate fraction for normal voting (None = use adaptive_frac)")
    parser.add_argument("--depth_adaptive_frac", type=float, default=None,
                        help="[v11] Separate fraction for depth voting (None = use adaptive_frac)")
    parser.add_argument("--merge_enabled", action="store_true",
                        help="[v11] Enable post-merge of adjacent similar segments")
    parser.add_argument("--merge_normal_deg", type=float, default=15.0)
    parser.add_argument("--merge_offset_m", type=float, default=0.05)
    parser.add_argument("--merge_min_pixels", type=int, default=50)
    parser.add_argument("--merge_gap_px", type=int, default=5)

    args = parser.parse_args()

    if args.grid_config:
        # Grid search mode
        configs = load_grid_configs(args.grid_config)
        print(f"[GRID] Loaded {len(configs)} configurations from {args.grid_config}")

        for i, config in enumerate(configs):
            # Ensure required keys exist with defaults
            config.setdefault("device", args.device)
            dirname = config_to_dirname(config)
            output_root = os.path.join(args.output_root, dirname)

            print(f"\n{'=' * 60}")
            print(f"[GRID] Config {i+1}/{len(configs)}: {dirname}")
            print(f"{'=' * 60}")
            for k, v in sorted(config.items()):
                print(f"  {k}: {v}")

            run_segmentation(args.raw_root, output_root, args.dataset, config, args.max_scenes)
    else:
        # Single config mode
        config = {
            "seg_version": args.seg_version,
            "threshold_planarity": args.threshold_planarity,
            "normal_threshold_deg": args.normal_threshold_deg,
            "depth_threshold": args.depth_threshold,
            "neighbor_match_count_thresh": args.neighbor_match_count_thresh,
            "adaptive_frac": args.adaptive_frac,
            "min_valid_neighbors": args.min_valid_neighbors,
            "min_segment_pixels": args.min_segment_pixels,
            "device": args.device,
            "normal_adaptive_frac": args.normal_adaptive_frac,
            "depth_adaptive_frac": args.depth_adaptive_frac,
            "merge_enabled": args.merge_enabled,
            "merge_normal_deg": args.merge_normal_deg,
            "merge_offset_m": args.merge_offset_m,
            "merge_min_pixels": args.merge_min_pixels,
            "merge_gap_px": args.merge_gap_px,
        }

        print("=" * 60)
        print("Stage 2: Raw H5 → Segmented Labels H5")
        print("=" * 60)
        print(f"Raw root:     {args.raw_root}")
        print(f"Output:       {args.output_root}")
        print(f"Dataset:      {args.dataset}")
        print(f"Seg version:  {args.seg_version}")
        print(f"Planarity θ:  {args.threshold_planarity}")
        print(f"Normal θ:     {args.normal_threshold_deg}°")
        print(f"Depth θ:      {args.depth_threshold}")
        print("=" * 60)

        run_segmentation(args.raw_root, args.output_root, args.dataset, config, args.max_scenes)

    print("\n[DONE] Segmentation complete.")


if __name__ == "__main__":
    main()
