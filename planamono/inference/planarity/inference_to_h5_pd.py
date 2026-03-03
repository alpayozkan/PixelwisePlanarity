#!/usr/bin/env python3
"""
MoGe Planarity Inference + Segmentation -> H5 (Parallel Domain Dataset)

Runs full pipeline for Parallel Domain dataset:
1. MoGe inference (planarity, depth, normals)
2. Vectorized segmentation
3. Saves planes.h5 to output directory (one per scene_id)

PD data is stored as flat NPZ files (not H5 per scene).
RGB is loaded from NPZ at 480x640 (high-res) resolution.
scene_id is extracted from origin_img_path in each NPZ.

Usage:
    python inference_to_h5_pd.py --model_path /path/to/model.pt --output_root /path/to/output

    # With custom parameters
    python inference_to_h5_pd.py --model_path /path/to/model.pt --output_root /path/to/output \
        --threshold_planarity 0.5 --normal_threshold_deg 10.0 --depth_threshold 0.05
"""

import os
import sys
import argparse
import time
from collections import defaultdict
from typing import Dict, List, Tuple
from contextlib import contextmanager

import numpy as np
import cv2
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from planamono.shared.datasets.pd_plane_dataset import PDPlaneDataset
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_relative
from planamono.shared.segmentation import compute_vectorized_planar_segments_v6
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference import MoGePlanarityInference
from planamono.inference.planarity.moge_inference_neck_head import (
    MoGePlanarityNeckHeadInference,
    MoGePlanarityProjNeckHeadInference,
)

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


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_CONFIG = {
    # Paths
    "model_path": "/cluster/scratch/aoezkan/moge_runs/mixed/moge_mixed_4heads_v1/final_planarity_4heads_model.pt",
    "output_root": "/cluster/scratch/aoezkan/planeseg/pd/inference/moge_ours_h5",
    "data_root": "/cluster/scratch/ayavuz/dataset/pd_zero/parallel_domain_plane",

    # Inference parameters
    "num_tokens": 1024,
    "batch_size": 16,  # PD images are 480x640

    # Segmentation parameters
    "threshold_planarity": 0.6,
    "normal_threshold_deg": 10.0,
    "depth_threshold": 0.05,
    "neighbor_match_count_thresh": 24,

    # Processing
    "split": "val",
    "max_samples": None,
    "num_workers": 4,
}


# ============================================================
# H5 SAVING (one planes.h5 per scene_id)
# ============================================================

def save_predictions_h5(
    scene_predictions: Dict[str, List[Tuple[str, np.ndarray]]],
    h5_root: str
):
    """
    Save predictions to H5 files (one per scene_id).

    Args:
        scene_predictions: {scene_id: [(frame_id, labels), ...]}
        h5_root: Root directory for H5 files
    """
    os.makedirs(h5_root, exist_ok=True)

    for scene_id, frame_data in tqdm(scene_predictions.items(), desc="Writing H5"):
        frame_data.sort(key=lambda x: x[0])
        frame_ids_list = [fd[0] for fd in frame_data]
        planes = np.stack([fd[1] for fd in frame_data], axis=0).astype(np.uint16)

        scene_h5_dir = os.path.join(h5_root, scene_id)
        os.makedirs(scene_h5_dir, exist_ok=True)

        h5_path = os.path.join(scene_h5_dir, "planes.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
            f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))

    print(f"[H5] Written {len(scene_predictions)} scene files to {h5_root}")


# ============================================================
# BATCH INFERENCE + SEGMENTATION
# ============================================================

def process_batch(
    batch,
    inference_model,
    args,
    timer,
    data_root,
    split,
    seg_fn=compute_vectorized_planar_segments_v5,
) -> List[Dict]:
    """
    Run batch inference and segmentation for Parallel Domain.

    Returns list of dicts with scene_id, frame_id, labels.
    """
    formatted_paths = batch["rgb_path"]
    scene_ids = batch["scene_id"]
    frame_ids = batch["frame_id"]

    # Load RGB images from NPZ files
    with timer("load_npz"):
        rgb_images = []
        for formatted_path in formatted_paths:
            # rgb_path format: "pd/<split>_<idx>"
            _, split_idx = formatted_path.split("/")
            npz_file = f"{split_idx}_d2.npz"
            npz_path = os.path.join(data_root, npz_file)

            try:
                d = np.load(npz_path, allow_pickle=True)
                # Use high-res RGB (480x640), BGR -> RGB
                rgb = d["raw_image"][:, :, ::-1].copy()  # (480, 640, 3) uint8
                rgb_images.append(rgb)
            except Exception as e:
                print(f"[ERROR] Failed to load {npz_path}: {e}")
                rgb_images.append(np.zeros((480, 640, 3), dtype=np.uint8))

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

    for res, formatted_path, scene_id, frame_id in zip(results, formatted_paths, scene_ids, frame_ids):
        # PD high-res is always 480x640
        H_rgb, W_rgb = 480, 640

        with timer("postprocess_extract"):
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
            planarity_mask = (planarity_rgb > args.threshold_planarity).astype(np.int32)

        # Run vectorized segmentation
        with timer("segmentation_compute"):
            labels, _ = seg_fn(
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
            "frame_id": frame_id,
            "labels": labels.astype(np.uint16),
        })

    return batch_outputs


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MoGe Planarity Inference + Segmentation -> H5 (Parallel Domain)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"],
                        help="Path to trained MoGe checkpoint (.pt file)")
    parser.add_argument("--output_root", type=str, default=DEFAULT_CONFIG["output_root"],
                        help="Root directory for H5 output")
    parser.add_argument("--data_root", type=str, default=DEFAULT_CONFIG["data_root"],
                        help="Parallel Domain dataset root")

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
    parser.add_argument("--seg_version", type=str, default="v5",
                        choices=["v5", "v5_relative", "v6"],
                        help="Segmentation algorithm version (v6: pairwise normals, relative depth)")

    # Processing
    parser.add_argument("--split", type=str, default=DEFAULT_CONFIG["split"],
                        choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=DEFAULT_CONFIG["max_samples"],
                        help="Limit number of samples (for testing)")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--architecture", type=str, default="4head",
                        choices=["4head", "neck_head", "proj_neck_head"],
                        help="Model architecture")

    args = parser.parse_args()

    # Validate
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MoGe Planarity Inference + Segmentation -> H5 (Parallel Domain)")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Output:       {args.output_root}")
    print(f"Data root:    {args.data_root}")
    print(f"Split:        {args.split}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Planarity th: {args.threshold_planarity}")
    print(f"Normal th:    {args.normal_threshold_deg} deg")
    print(f"Depth th:     {args.depth_threshold}{'(relative)' if args.seg_version == 'v6' else 'm'}")
    print(f"Seg version:  {args.seg_version}")
    print("=" * 60)

    seg_fn_map = {
        "v5": compute_vectorized_planar_segments_v5,
        "v5_relative": compute_vectorized_planar_segments_v5_relative,
        "v6": compute_vectorized_planar_segments_v6,
    }
    seg_fn = seg_fn_map[args.seg_version]

    # Load dataset
    print("[INFO] Loading dataset...")
    dataset = PDPlaneDataset(
        data_root=args.data_root,
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"[INFO] Dataset size: {len(dataset)} frames")

    # Custom collate to handle variable-sized data
    def collate_fn(batch):
        return {
            "rgb_path": [b["rgb_path"] for b in batch],     # "pd/<split>_<idx>"
            "scene_id": [b["scene_id"] for b in batch],     # scene name
            "frame_id": [b["frame_idx"] for b in batch],    # sample index string
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

    # Optimizations
    inference_model.model.encoder.use_memory_efficient_attention = False
    torch.set_grad_enabled(False)
    inference_model.model.eval()

    print("[INFO] Model loaded")

    # Initialize timer
    timer = Timer()

    # Process all batches
    print("[INFO] Running inference + segmentation...")
    scene_predictions = defaultdict(list)
    total_frames = 0

    with timer("total_pipeline"):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
            batch_outputs = process_batch(
                batch, inference_model, args, timer,
                args.data_root, args.split, seg_fn=seg_fn,
            )

            for output in batch_outputs:
                scene_predictions[output["scene_id"]].append((
                    output["frame_id"],
                    output["labels"]
                ))
                total_frames += 1

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
