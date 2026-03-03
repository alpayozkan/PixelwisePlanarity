#!/usr/bin/env python3
"""
Stage 1: MoGe Inference → Raw H5 (Hypersim)

Runs MoGe GPU inference only (no segmentation) and saves raw outputs
(planarity, depth, normals) to per-camera HDF5 files. This enables fast
iteration on segmentation parameters without re-running expensive GPU inference.

Usage:
    python save_moge_raw_hypersim.py --model_path /path/to/model.pt --output_root /path/to/output

    # Quick test on 2 scenes
    python save_moge_raw_hypersim.py --model_path /path/to/model.pt --output_root /path/to/output --max_scenes 2

Stage 2 (segmentation): Use segment_from_raw.py --dataset hypersim to convert raw H5 to planes_cam_XX.h5.
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
import pandas as pd

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.inference.planarity.moge_inference_v1 import MoGePlanarityInference
from planamono.inference.planarity.moge_inference_neck_head import (
    MoGePlanarityNeckHeadInference,
    MoGePlanarityProjNeckHeadInference,
)
from planamono.paths import repo_path

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
    "output_root": "/cluster/scratch/aoezkan/planeseg/hypersim/inference_raw/moge_hires_ep3_raw",
    "hypersim_root": "/cluster/scratch/aoezkan/planeseg/dataset/hypersim",
    "plane_label_root": "/cluster/scratch/aoezkan/planeseg/dataset/hypersim",
    "params_root": "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params",
    "num_tokens": 1600,
    "batch_size": 8,
    "split": "test",
    "max_scenes": None,
    "num_workers": 4,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def tonemap_hdr(hdr, gamma=2.2):
    """Apply robust tone mapping for Hypersim HDR images."""
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
    hdr = np.clip(hdr, 0, None)
    max_val = np.percentile(hdr, 99)
    if max_val > 0:
        hdr = hdr / max_val
    hdr = np.power(hdr, 1.0 / gamma)
    return np.clip(hdr, 0, 1)


# ============================================================
# H5 SAVING
# ============================================================

def save_raw_h5(
    scene_data: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]],
    h5_root: str,
    attrs: dict,
):
    """
    Save raw MoGe outputs to per-camera H5 files (Hypersim convention).

    Args:
        scene_data: {scene_id: [(cam_name, frame_id, planarity, depth, normal), ...]}
        h5_root: Root directory for H5 files
        attrs: Metadata attributes to store in the H5 file
    """
    os.makedirs(h5_root, exist_ok=True)

    # Group by scene and camera
    scene_cam_data = defaultdict(lambda: defaultdict(list))
    for scene_id, frame_list in scene_data.items():
        for cam_name, frame_id, planarity, depth, normal in frame_list:
            scene_cam_data[scene_id][cam_name].append((frame_id, planarity, depth, normal))

    for scene_id, cam_dict in tqdm(scene_cam_data.items(), desc="Writing H5"):
        scene_h5_dir = os.path.join(h5_root, scene_id)
        os.makedirs(scene_h5_dir, exist_ok=True)

        for cam_name, frame_data in cam_dict.items():
            frame_data.sort(key=lambda x: x[0])
            frame_ids_list = [fd[0] for fd in frame_data]
            planarity = np.stack([fd[1] for fd in frame_data], axis=0)  # (N, H, W)
            depth = np.stack([fd[2] for fd in frame_data], axis=0)      # (N, H, W)
            normal = np.stack([fd[3] for fd in frame_data], axis=0)     # (N, H, W, 3)

            h5_path = os.path.join(scene_h5_dir, f"moge_raw_{cam_name}.h5")
            N, H, W = planarity.shape
            with h5py.File(h5_path, "w") as f:
                f.create_dataset(
                    "planarity", data=planarity.astype(np.float32),
                    chunks=(1, H, W), compression="gzip", compression_opts=4,
                )
                f.create_dataset(
                    "depth", data=depth.astype(np.float32),
                    chunks=(1, H, W), compression="gzip", compression_opts=4,
                )
                f.create_dataset(
                    "normal", data=normal.astype(np.float32),
                    chunks=(1, H, W, 3), compression="gzip", compression_opts=4,
                )
                f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))
                for k, v in attrs.items():
                    f.attrs[k] = v

    print(f"[H5] Written {len(scene_data)} scene files to {h5_root}")


# ============================================================
# BATCH INFERENCE (no segmentation)
# ============================================================

def process_batch(
    batch,
    inference_model,
    args,
    timer,
    hypersim_root,
) -> List[Dict]:
    """
    Run batch MoGe inference for Hypersim. Returns raw planarity/depth/normal arrays.
    """
    formatted_paths = batch["rgb_path"]
    scene_ids = batch["scene_id"]
    frame_ids = batch["frame_id"]

    # Reconstruct actual RGB HDF5 file paths
    rgb_paths = []
    cam_names = []
    for formatted_path in formatted_paths:
        parts = formatted_path.split("/")
        scene_id, cam_name, fid = parts[0], parts[1], parts[2]
        cam_names.append(cam_name)
        rgb_path = os.path.join(
            hypersim_root, scene_id, "images",
            f"scene_{cam_name}_final_hdf5",
            f"frame.{fid}.color.hdf5",
        )
        rgb_paths.append(rgb_path)

    # Load RGB images from HDF5
    with timer("load_hdf5"):
        rgb_images = []
        for rgb_path in rgb_paths:
            try:
                with h5py.File(rgb_path, "r") as f:
                    key = list(f.keys())[0]
                    rgb = f[key][:]
                if rgb.dtype == np.uint8:
                    rgb = rgb.astype(np.float32) / 255.0
                elif rgb.dtype == np.uint16:
                    rgb = rgb.astype(np.float32) / 65535.0
                elif rgb.dtype in [np.float16, np.float32, np.float64]:
                    rgb = tonemap_hdr(rgb)
                else:
                    rgb = tonemap_hdr(rgb.astype(np.float32))
                rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                rgb_images.append(rgb_uint8)
            except Exception as e:
                print(f"[ERROR] Failed to load {rgb_path}: {e}")
                rgb_images.append(np.zeros((768, 1024, 3), dtype=np.uint8))

    # Batch GPU inference
    with timer("gpu_inference"):
        images_tensor, original_sizes = inference_model.preprocess_images(rgb_images)
        with torch.no_grad():
            if hasattr(inference_model, '_forward'):
                outputs = inference_model._forward(images_tensor, num_tokens=args.num_tokens)
            else:
                outputs = inference_model.model(images_tensor, num_tokens=args.num_tokens)

        results = []
        for i in range(len(rgb_images)):
            res = {}
            planarity = outputs["planarity"][i].squeeze().cpu().numpy()
            h0, w0 = original_sizes[i]
            res["planarity_probability"] = planarity
            if "normal" in outputs:
                res["normal"] = outputs["normal"][i].cpu().numpy()
            if "points" in outputs:
                res["points"] = outputs["points"][i].cpu().numpy()
            res["original_size"] = (h0, w0)
            results.append(res)

    batch_outputs = []

    for res, rgb_path, scene_id, cam_name, frame_id in zip(results, rgb_paths, scene_ids, cam_names, frame_ids):
        with timer("postprocess_io"):
            H_rgb, W_rgb = res["original_size"]

        with timer("postprocess_extract"):
            planarity = res["planarity_probability"]
            depth_moge = res["points"][:, :, 2]  # Hypersim always uses affine depth
            normal = res["normal"]  # (H, W, 3)

        with timer("postprocess_resize"):
            planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            depth_moge_rgb = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            normal_rgb = cv2.resize(normal, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)

        batch_outputs.append({
            "scene_id": scene_id,
            "cam_name": cam_name,
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
        description="Stage 1: MoGe Inference → Raw H5 (Hypersim)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--output_root", type=str, default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--hypersim_root", type=str, default=DEFAULT_CONFIG["hypersim_root"])
    parser.add_argument("--plane_label_root", type=str, default=DEFAULT_CONFIG["plane_label_root"])
    parser.add_argument("--params_root", type=str, default=DEFAULT_CONFIG["params_root"])
    parser.add_argument("--num_tokens", type=int, default=DEFAULT_CONFIG["num_tokens"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--split", type=str, default=DEFAULT_CONFIG["split"],
                        choices=["train", "val", "test"])
    parser.add_argument("--max_scenes", type=int, default=DEFAULT_CONFIG["max_scenes"])
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--architecture", type=str, default="4head",
                        choices=["4head", "neck_head", "proj_neck_head"])

    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    print("=" * 60)
    print("Stage 1: MoGe Inference → Raw H5 (Hypersim)")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Output:       {args.output_root}")
    print(f"Split:        {args.split}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Num tokens:   {args.num_tokens}")
    print(f"Architecture: {args.architecture}")
    print("=" * 60)

    # Load dataset
    print("[INFO] Loading dataset...")
    dataset = HypersimPlaneDataset(
        hypersim_root=args.hypersim_root,
        plane_label_root=args.plane_label_root,
        params_root=args.params_root,
        split_txt_dir=os.path.join(repo_path, "splits", "hypersim"),
        split=args.split,
        max_scenes=args.max_scenes,
    )
    print(f"[INFO] Dataset size: {len(dataset)} frames")

    def collate_fn(batch):
        return {
            "rgb_path": [b["rgb_path"] for b in batch],
            "scene_id": [b["scene_id"] for b in batch],
            "frame_id": [b["frame_idx"] for b in batch],
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

    scene_data: Dict[str, List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    total_frames = 0

    h5_attrs = {
        "model_path": args.model_path,
        "num_tokens": args.num_tokens,
        "metric_depth": False,
        "split": args.split,
        "architecture": args.architecture,
    }

    print("[INFO] Running inference...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        batch_outputs = process_batch(batch, inference_model, args, timer, args.hypersim_root)

        for output in batch_outputs:
            scene_data[output["scene_id"]].append((
                output["cam_name"],
                output["frame_id"],
                output["planarity"],
                output["depth"],
                output["normal"],
            ))
            total_frames += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"[PROGRESS] Processed {total_frames} frames from {len(scene_data)} scenes")

    print(f"\n[DONE] Processed {total_frames} frames from {len(scene_data)} scenes")

    # Save to H5
    print("[INFO] Saving raw predictions to H5...")
    with timer("h5_save"):
        save_raw_h5(dict(scene_data), args.output_root, h5_attrs)

    timer.print_summary(num_frames=total_frames)
    timing_csv = os.path.join(args.output_root, "runtime_breakdown.csv")
    timer.save_csv(timing_csv)

    # Save config
    import yaml
    config = {
        "model_path": args.model_path,
        "num_tokens": args.num_tokens,
        "metric_depth": False,
        "split": args.split,
        "batch_size": args.batch_size,
        "architecture": args.architecture,
    }
    config_path = os.path.join(args.output_root, "config.yml")
    os.makedirs(args.output_root, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[CONFIG] Saved parameters to {config_path}")

    print("=" * 60)
    print(f"[DONE] Saved {len(scene_data)} scenes to {args.output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
