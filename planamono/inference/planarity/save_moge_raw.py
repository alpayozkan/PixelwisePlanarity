#!/usr/bin/env python3
"""
Stage 1: MoGe Inference → Raw H5 (ScanNet++)

Runs MoGe GPU inference only (no segmentation) and saves raw outputs
(planarity, depth, normals) to HDF5 files. This enables fast iteration
on segmentation parameters without re-running expensive GPU inference.

Writes frames incrementally to H5 (one frame at a time) to avoid OOM
on scenes with 300-450+ frames.

Usage:
    python save_moge_raw.py --model_path /path/to/model.pt --output_root /path/to/output

    # Quick test on 2 scenes
    python save_moge_raw.py --model_path /path/to/model.pt --output_root /path/to/output --max_scenes 2

Stage 2 (segmentation): Use segment_from_raw.py to convert raw H5 to eval-compatible planes.h5.
"""

import os
import sys
import argparse
import time
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from contextlib import contextmanager

import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import pandas as pd

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.inference.planarity.moge_inference_v1 import MoGePlanarityInference
from planamono.inference.planarity.moge_inference_neck_head import (
    MoGePlanarityNeckHeadInference,
    MoGePlanarityProjNeckHeadInference,
)
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path

ARCH_CLASSES = {
    "4head": MoGePlanarityInference,
    "neck_head": MoGePlanarityNeckHeadInference,
    "proj_neck_head": MoGePlanarityProjNeckHeadInference,
}


# ============================================================
# TIMING INFRASTRUCTURE
# ============================================================

class Timer:
    """Timing infrastructure for profiling."""

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

    def save_csv(self, output_path: str):
        rows = []
        total = self.total_elapsed()
        for name, seconds in self.timings.items():
            rows.append({
                "stage": name,
                "time_seconds": seconds,
                "calls": self.counts[name],
                "avg_ms": (seconds / self.counts[name] * 1000) if self.counts[name] > 0 else 0,
                "percent": (seconds / total * 100) if total > 0 else 0
            })
        df = pd.DataFrame(rows).sort_values(by="time_seconds", ascending=False)
        df.to_csv(output_path, index=False)
        print(f"[TIMING] Saved runtime breakdown to {output_path}")


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    "model_path": "/cluster/scratch/ayavuz/moge_mixed_output_476644_fixed_cosLR_singlePhase_mixed_HiRes/model_epoch3.pt",
    "output_root": "/cluster/scratch/aoezkan/planeseg/scannetpp/inference_raw/moge_hires_ep3_raw",
    "dataset_dir": scannetpp_rend_plane_path,
    "rgb_root": os.path.join(scannetpp_path, "data"),
    "num_tokens": 1600,
    "batch_size": 16,
    "split": "test",
    "max_scenes": None,
    "num_workers": 4,
}


# ============================================================
# INCREMENTAL H5 WRITER
# ============================================================

class IncrementalH5Writer:
    """
    Manages per-scene H5 files with incremental frame writes.

    Pre-creates resizable datasets so frames can be written one at a time
    without accumulating the entire scene in RAM. Peak memory usage is
    O(batch_size) instead of O(scene_frames).
    """

    def __init__(self, h5_root: str, scene_frame_counts: Dict[str, int], attrs: dict):
        self.h5_root = h5_root
        self.scene_frame_counts = scene_frame_counts
        self.attrs = attrs
        # Track per-scene: next write index and frame_ids collected so far
        self.scene_write_idx: Dict[str, int] = defaultdict(int)
        self.scene_frame_ids: Dict[str, List[str]] = defaultdict(list)
        self.scene_h5_created: set = set()
        os.makedirs(h5_root, exist_ok=True)

    def _h5_path(self, scene_id: str) -> str:
        scene_dir = os.path.join(self.h5_root, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        return os.path.join(scene_dir, "moge_raw.h5")

    def _ensure_h5(self, scene_id: str, H: int, W: int):
        """Create H5 file with pre-allocated datasets on first write."""
        if scene_id in self.scene_h5_created:
            return
        N = self.scene_frame_counts[scene_id]
        h5_path = self._h5_path(scene_id)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "planarity", shape=(N, H, W), dtype=np.float32,
                chunks=(1, H, W), compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "depth", shape=(N, H, W), dtype=np.float32,
                chunks=(1, H, W), compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "normal", shape=(N, H, W, 3), dtype=np.float32,
                chunks=(1, H, W, 3), compression="gzip", compression_opts=4,
            )
            # Placeholder for frame_ids — will be overwritten at finalization
            f.create_dataset(
                "frame_ids", shape=(N,), dtype=h5py.string_dtype(),
            )
            for k, v in self.attrs.items():
                f.attrs[k] = v
        self.scene_h5_created.add(scene_id)

    def write_frame(self, scene_id: str, frame_id: str,
                    planarity: np.ndarray, depth: np.ndarray, normal: np.ndarray):
        """Write a single frame to the scene's H5 file."""
        H, W = planarity.shape
        self._ensure_h5(scene_id, H, W)

        idx = self.scene_write_idx[scene_id]
        h5_path = self._h5_path(scene_id)
        with h5py.File(h5_path, "a") as f:
            f["planarity"][idx] = planarity.astype(np.float32)
            f["depth"][idx] = depth.astype(np.float32)
            f["normal"][idx] = normal.astype(np.float32)
            f["frame_ids"][idx] = frame_id

        self.scene_write_idx[scene_id] = idx + 1
        self.scene_frame_ids[scene_id].append(frame_id)

    def get_completed_scenes(self) -> List[str]:
        """Return list of scenes where all frames have been written."""
        completed = []
        for scene_id, count in self.scene_frame_counts.items():
            if self.scene_write_idx.get(scene_id, 0) >= count:
                completed.append(scene_id)
        return completed

    def summary(self) -> str:
        written = sum(self.scene_write_idx.values())
        total = sum(self.scene_frame_counts.values())
        n_scenes = len(self.scene_h5_created)
        return f"{written}/{total} frames across {n_scenes} scenes"


# ============================================================
# BATCH INFERENCE (no segmentation)
# ============================================================

def process_batch(
    batch,
    inference_model,
    args,
    timer,
) -> List[Dict]:
    """
    Run batch MoGe inference. Returns raw planarity/depth/normal arrays.
    """
    rgb_paths = batch["rgb_path"]
    scene_ids = batch["scene_id"]
    frame_ids = batch["frame_idx"]

    # Batch GPU inference
    with timer("gpu_inference"):
        if args.metric_depth:
            results = inference_model.predict_batch_fast_metric(
                rgb_paths,
                num_tokens=args.num_tokens,
                return_all_heads=True,
            )
        else:
            results = inference_model.predict_batch_fast(
                rgb_paths,
                num_tokens=args.num_tokens,
                return_all_heads=True,
            )

    batch_outputs = []

    for res, rgb_path, scene_id, frame_id in zip(results, rgb_paths, scene_ids, frame_ids):
        with timer("postprocess_io"):
            img = Image.open(rgb_path).convert("RGB")
            img_np = np.array(img)
            H_rgb, W_rgb = img_np.shape[:2]

        with timer("postprocess_extract"):
            planarity = res["planarity_probability"]
            depth_moge = res["depth"] if args.metric_depth else res["points"][:, :, 2]
            # MoGe v2 outputs (H, W, 3) — store directly, no transpose needed
            normal = res["normal"]  # (H, W, 3)

        with timer("postprocess_resize"):
            planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            depth_moge_rgb = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            normal_rgb = cv2.resize(normal, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)

        batch_outputs.append({
            "scene_id": scene_id,
            "frame_id": frame_id,
            "planarity": planarity_rgb.astype(np.float32),
            "depth": depth_moge_rgb.astype(np.float32),
            "normal": normal_rgb.astype(np.float32),  # (H, W, 3)
        })

    return batch_outputs


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: MoGe Inference → Raw H5 (ScanNet++)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--output_root", type=str, default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_CONFIG["dataset_dir"])
    parser.add_argument("--rgb_root", type=str, default=DEFAULT_CONFIG["rgb_root"])
    parser.add_argument("--num_tokens", type=int, default=DEFAULT_CONFIG["num_tokens"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--split", type=str, default=DEFAULT_CONFIG["split"],
                        choices=["train", "val", "test"])
    parser.add_argument("--max_scenes", type=int, default=DEFAULT_CONFIG["max_scenes"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--architecture", type=str, default="4head",
                        choices=["4head", "neck_head", "proj_neck_head"])
    parser.add_argument("--metric_depth", action="store_true",
                        help="Use MoGe metric depth (model.infer) instead of affine points[:,:,2]")

    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    print("=" * 60)
    print("Stage 1: MoGe Inference → Raw H5 (ScanNet++)")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Output:       {args.output_root}")
    print(f"Split:        {args.split}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Num tokens:   {args.num_tokens}")
    print(f"Metric depth: {args.metric_depth}")
    print(f"Architecture: {args.architecture}")
    print("=" * 60)

    # Load dataset
    print("[INFO] Loading dataset...")
    dataset = ScanNetPPPlaneDataset(
        rgb_root=args.rgb_root,
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=args.dataset_dir,
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split=args.split,
        max_scenes=args.max_scenes,
    )
    print(f"[INFO] Dataset size: {len(dataset)} frames")

    def collate_fn(batch):
        return {
            "rgb_path": [b["rgb_path"] for b in batch],
            "scene_id": [b["scene_id"] for b in batch],
            "frame_idx": [b["frame_idx"] for b in batch],
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load model
    print(f"[INFO] Loading MoGe model (architecture={args.architecture})...")
    model_cls = ARCH_CLASSES[args.architecture]
    inference_model = model_cls(args.model_path, device=args.device)
    inference_model.model.encoder.use_memory_efficient_attention = False
    torch.set_grad_enabled(False)
    inference_model.model.eval()
    print("[INFO] Model loaded")

    timer = Timer()

    # Count frames per scene for pre-allocation
    scene_frame_counts = Counter()
    for pair in dataset.valid_pairs:
        rgb_path = pair[0]
        scene_id = rgb_path.split("/")[-4]
        scene_frame_counts[scene_id] += 1

    print(f"[INFO] Scenes: {len(scene_frame_counts)}, "
          f"frames: {sum(scene_frame_counts.values())}, "
          f"max frames/scene: {max(scene_frame_counts.values())}")

    h5_attrs = {
        "model_path": args.model_path,
        "num_tokens": args.num_tokens,
        "metric_depth": args.metric_depth,
        "split": args.split,
        "architecture": args.architecture,
    }

    # Incremental writer — writes each frame to H5 immediately
    writer = IncrementalH5Writer(args.output_root, dict(scene_frame_counts), h5_attrs)

    print("[INFO] Running inference...")
    for batch in tqdm(dataloader, desc="Processing"):
        outputs = process_batch(batch, inference_model, args, timer)

        with timer("write_h5"):
            for out in outputs:
                writer.write_frame(
                    out["scene_id"],
                    out["frame_id"],
                    out["planarity"],
                    out["depth"],
                    out["normal"],
                )

    print(f"\n[INFO] {writer.summary()}")

    timer.print_summary(num_frames=len(dataset))
    timing_csv = os.path.join(args.output_root, "runtime_breakdown.csv")
    timer.save_csv(timing_csv)

    # Save config
    import yaml
    config = {
        "model_path": args.model_path,
        "num_tokens": args.num_tokens,
        "metric_depth": args.metric_depth,
        "split": args.split,
        "batch_size": args.batch_size,
        "architecture": args.architecture,
    }
    config_path = os.path.join(args.output_root, "config.yml")
    os.makedirs(args.output_root, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[CONFIG] Saved parameters to {config_path}")

    completed = writer.get_completed_scenes()
    print("=" * 60)
    print(f"[DONE] Saved {len(completed)} scenes to {args.output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
