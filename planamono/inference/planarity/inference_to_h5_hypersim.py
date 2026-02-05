#!/usr/bin/env python3
"""
MoGe Planarity Inference + Segmentation → H5 (Hypersim Dataset)

Runs full pipeline for Hypersim dataset:
1. MoGe inference (planarity, depth, normals)
2. Vectorized segmentation
3. Saves planes.h5 to output directory

Usage:
    python inference_to_h5_hypersim.py --model_path /path/to/model.pt --output_root /path/to/output

    # With custom parameters
    python inference_to_h5_hypersim.py --model_path /path/to/model.pt --output_root /path/to/output \
        --threshold_planarity 0.5 --normal_threshold_deg 10.0 --depth_threshold 0.05
"""

import os
import sys
import argparse
import pickle
import time
from pathlib import Path
from collections import defaultdict
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

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.paths import repo_path


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


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    # Paths
    "model_path": "/cluster/scratch/aoezkan/moge_runs/hypersim/moge_hypersim_4heads_v1/final_planarity_4heads_model.pt",
    "output_root": "/cluster/scratch/aoezkan/planeseg/hypersim/inference/moge_ours_h5",
    "hypersim_root": "/cluster/scratch/ayavuz/dataset/Hypersim_merged",
    "plane_label_root": "/cluster/scratch/ayavuz/dataset/Hypersim_rendered",
    "params_root": "/cluster/scratch/ayavuz/dataset/Hypersim_params",

    # Inference parameters
    "num_tokens": 1024,
    "batch_size": 8,

    # Segmentation parameters
    "threshold_planarity": 0.6,
    "normal_threshold_deg": 10.0,
    "depth_threshold": 0.05,
    "neighbor_match_count_thresh": 24,

    # Processing
    "split": "val",
    "max_scenes": None,
    "num_workers": 4,
}


# ============================================================
# H5 SAVING
# ============================================================

def save_predictions_h5(
    scene_predictions: Dict[str, List[Tuple[str, str, np.ndarray]]],
    h5_root: str
):
    """
    Save predictions to H5 files (one per scene-camera).

    Args:
        scene_predictions: {scene_id: [(cam_name, frame_id, labels), ...]}
        h5_root: Root directory for H5 files
    """
    os.makedirs(h5_root, exist_ok=True)

    # Group by scene and camera
    scene_cam_data = defaultdict(lambda: defaultdict(list))
    for scene_id, frame_list in scene_predictions.items():
        for cam_name, frame_id, labels in frame_list:
            scene_cam_data[scene_id][cam_name].append((frame_id, labels))

    for scene_id, cam_dict in tqdm(scene_cam_data.items(), desc="Writing H5"):
        scene_h5_dir = os.path.join(h5_root, scene_id)
        os.makedirs(scene_h5_dir, exist_ok=True)

        for cam_name, frame_data in cam_dict.items():
            # Sort by frame ID
            frame_data.sort(key=lambda x: x[0])
            frame_ids_list = [fd[0] for fd in frame_data]
            planes = np.stack([fd[1] for fd in frame_data], axis=0).astype(np.uint16)

            h5_path = os.path.join(scene_h5_dir, f"planes_{cam_name}.h5")
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
                f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))

    print(f"[H5] Written {len(scene_predictions)} scene files to {h5_root}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def tonemap_hdr(hdr, gamma=2.2):
    """Apply robust tone mapping for Hypersim HDR images."""
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0)
    hdr = np.clip(hdr, 0, None)

    # Simple normalization
    max_val = np.percentile(hdr, 99)
    if max_val > 0:
        hdr = hdr / max_val

    # Gamma correction
    hdr = np.power(hdr, 1.0 / gamma)
    return np.clip(hdr, 0, 1)


# ============================================================
# BATCH INFERENCE + SEGMENTATION
# ============================================================

def process_batch(
    batch,
    inference_model,
    args,
    timer,
    hypersim_root,
) -> List[Dict]:
    """
    Run batch inference and segmentation for Hypersim.

    Returns list of dicts with scene_id, cam_name, frame_id, labels.
    """
    # Parse formatted rgb_path strings (format: "scene_id/cam_name/frame_id")
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

        # Hypersim structure: <hypersim_root>/<scene_id>/images/scene_<cam_name>_final_hdf5/frame.<fid>.color.hdf5
        rgb_path = os.path.join(
            hypersim_root,
            scene_id,
            "images",
            f"scene_{cam_name}_final_hdf5",  # Use "scene_" prefix, not scene_id
            f"frame.{fid}.color.hdf5"
        )
        rgb_paths.append(rgb_path)

    # Load RGB images from HDF5 (Hypersim format)
    with timer("load_hdf5"):
        rgb_images = []
        for rgb_path in rgb_paths:
            try:
                with h5py.File(rgb_path, "r") as f:
                    key = list(f.keys())[0]
                    rgb = f[key][:]  # (H, W, 3)

                # Handle different dtypes (same as dataset)
                if rgb.dtype == np.uint8:
                    rgb = rgb.astype(np.float32) / 255.0
                elif rgb.dtype == np.uint16:
                    rgb = rgb.astype(np.float32) / 65535.0
                elif rgb.dtype in [np.float16, np.float32, np.float64]:
                    # Hypersim HDR - apply tone mapping
                    rgb = tonemap_hdr(rgb)
                else:
                    rgb = tonemap_hdr(rgb.astype(np.float32))

                # Convert to uint8 for preprocess_images
                rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                rgb_images.append(rgb_uint8)
            except Exception as e:
                print(f"[ERROR] Failed to load {rgb_path}: {e}")
                # Create dummy black image
                rgb_images.append(np.zeros((768, 1024, 3), dtype=np.uint8))

    # Batch GPU inference (preprocess_images can handle numpy arrays)
    with timer("gpu_inference"):
        # preprocess_images accepts numpy arrays (not just paths)
        images_tensor, original_sizes = inference_model.preprocess_images(rgb_images)

        with torch.no_grad():
            outputs = inference_model.model(images_tensor, num_tokens=args.num_tokens)

        results = []

        for i in range(len(rgb_images)):
            res = {}
            planarity = outputs["planarity"][i].squeeze().cpu().numpy()
            planarity_bin = (planarity > 0.5).astype(np.uint8)

            h0, w0 = original_sizes[i]
            planarity_full = cv2.resize(planarity, (w0, h0))
            planarity_bin_full = cv2.resize(planarity_bin, (w0, h0), interpolation=cv2.INTER_NEAREST)

            res["planarity_probability"] = planarity
            res["planarity_probability_full"] = planarity_full
            res["planarity_binary"] = planarity_bin
            res["planarity_binary_full"] = planarity_bin_full

            if "normal" in outputs:
                normal = outputs["normal"][i].cpu().numpy()
                res["normal"] = normal
            if "points" in outputs:
                points = outputs["points"][i].cpu().numpy()
                res["points"] = points

            results.append(res)

    batch_outputs = []

    for res, rgb_path, scene_id, cam_name, frame_id in zip(results, rgb_paths, scene_ids, cam_names, frame_ids):
        # Get image dimensions from RGB HDF5
        with timer("postprocess_io"):
            with h5py.File(rgb_path, "r") as f:
                key = list(f.keys())[0]
                rgb_data = f[key][:]
            H_rgb, W_rgb = rgb_data.shape[:2]

        with timer("postprocess_extract"):
            # Get MoGe outputs
            planarity = res["planarity_probability"]
            depth_moge = res["points"][:, :, 2]
            normal = res["normal"].transpose(2, 0, 1)  # (3, H, W)

        # Resize to RGB resolution
        with timer("postprocess_resize"):
            planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            depth_moge_rgb = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            normal_rgb = cv2.resize(
                normal.transpose(1, 2, 0), (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)

        with timer("postprocess_threshold"):
            # Apply planarity threshold
            planarity_mask = (planarity_rgb > args.threshold_planarity).astype(np.int32)

        # Run vectorized segmentation (v5 is faster)
        with timer("segmentation_compute"):
            labels, _ = compute_vectorized_planar_segments_v5(
                planarity_mask,
                normal_rgb.transpose(1, 2, 0),  # (H, W, 3)
                depth_moge_rgb,
                np.deg2rad(args.normal_threshold_deg),
                args.depth_threshold,
                args.neighbor_match_count_thresh,
            )

        with timer("segmentation_remap"):
            labels, _ = remap_labels(labels)

        batch_outputs.append({
            "scene_id": scene_id,
            "cam_name": cam_name,
            "frame_id": frame_id,
            "labels": labels.astype(np.uint16),
        })

    return batch_outputs


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MoGe Planarity Inference + Segmentation → H5 (Hypersim)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"],
                        help="Path to trained MoGe checkpoint (.pt file)")
    parser.add_argument("--output_root", type=str, default=DEFAULT_CONFIG["output_root"],
                        help="Root directory for H5 output")
    parser.add_argument("--hypersim_root", type=str, default=DEFAULT_CONFIG["hypersim_root"],
                        help="Hypersim dataset root (merged)")
    parser.add_argument("--plane_label_root", type=str, default=DEFAULT_CONFIG["plane_label_root"],
                        help="Rendered plane labels root")
    parser.add_argument("--params_root", type=str, default=DEFAULT_CONFIG["params_root"],
                        help="Hypersim params root")

    # Inference parameters
    parser.add_argument("--num_tokens", type=int, default=DEFAULT_CONFIG["num_tokens"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])

    # Segmentation parameters
    parser.add_argument("--threshold_planarity", type=float,
                        default=DEFAULT_CONFIG["threshold_planarity"],
                        help="Planarity threshold for binary mask")
    parser.add_argument("--normal_threshold_deg", type=float,
                        default=DEFAULT_CONFIG["normal_threshold_deg"],
                        help="Normal angle threshold in degrees")
    parser.add_argument("--depth_threshold", type=float,
                        default=DEFAULT_CONFIG["depth_threshold"],
                        help="Depth difference threshold in meters")
    parser.add_argument("--neighbor_match_count_thresh", type=int,
                        default=DEFAULT_CONFIG["neighbor_match_count_thresh"],
                        help="Minimum matching neighbors for connectivity")

    # Processing
    parser.add_argument("--split", type=str, default=DEFAULT_CONFIG["split"],
                        choices=["train", "val", "test"])
    parser.add_argument("--max_scenes", type=int, default=DEFAULT_CONFIG["max_scenes"],
                        help="Limit number of scenes (for testing)")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Validate
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MoGe Planarity Inference + Segmentation → H5 (Hypersim)")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Output:       {args.output_root}")
    print(f"Split:        {args.split}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Planarity θ:  {args.threshold_planarity}")
    print(f"Normal θ:     {args.normal_threshold_deg}°")
    print(f"Depth θ:      {args.depth_threshold}m")
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

    # Custom collate to handle variable-sized data
    def collate_fn(batch):
        return {
            "rgb_path": [b["rgb_path"] for b in batch],  # Formatted paths: "scene_id/cam_name/fid"
            "scene_id": [b["scene_id"] for b in batch],
            "frame_id": [b["frame_idx"] for b in batch],  # Note: frame_idx from dataset
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
    print("[INFO] Loading MoGe model...")
    inference_model = MoGePlanarityInference(args.model_path, device=args.device)

    # Optimizations
    inference_model.model.encoder.use_memory_efficient_attention = False
    torch.set_grad_enabled(False)
    inference_model.model.eval()

    # Optional: Use half precision for faster inference (comment out if issues)
    # inference_model.model = inference_model.model.half()

    print("[INFO] Model loaded")

    # Initialize timer
    timer = Timer()

    # Process all batches
    print("[INFO] Running inference + segmentation...")
    scene_predictions = defaultdict(list)
    total_frames = 0

    with timer("total_pipeline"):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            batch_outputs = process_batch(batch, inference_model, args, timer, args.hypersim_root)

            # Accumulate results by scene
            for output in batch_outputs:
                scene_predictions[output["scene_id"]].append((
                    output["cam_name"],
                    output["frame_id"],
                    output["labels"]
                ))
                total_frames += 1

            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"[PROGRESS] Processed {total_frames} frames from {len(scene_predictions)} scenes")

    print(f"\n[DONE] Processed {total_frames} frames from {len(scene_predictions)} scenes")

    # Save to H5
    print("[INFO] Saving predictions to H5...")
    with timer("h5_save"):
        save_predictions_h5(scene_predictions, args.output_root)

    # Print timing summary
    timer.print_summary(num_frames=total_frames)

    print(f"\n[SUCCESS] Results saved to {args.output_root}")


if __name__ == "__main__":
    main()
