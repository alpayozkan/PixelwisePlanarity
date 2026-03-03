#!/usr/bin/env python3
"""
MoGe Planarity Inference + Segmentation → H5

Runs full pipeline without metric evaluation:
1. MoGe inference (planarity, depth, normals)
2. Vectorized segmentation
3. Saves planes.h5 to output directory

Usage:
    python inference_to_h5.py --model_path /path/to/model.pt --output_root /path/to/output

    # With custom parameters
    python inference_to_h5.py --model_path /path/to/model.pt --output_root /path/to/output \
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

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_relative
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_no_sobel
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_dotprod_relative
from planamono.shared.segmentation import compute_vectorized_planar_segments_v6
from planamono.shared.segmentation import compute_vectorized_planar_segments_v9_vote
from planamono.shared.segmentation import compute_vectorized_planar_segments_v10
from planamono.shared.segmentation import compute_vectorized_planar_segments_v11
from planamono.shared.segmentation.merge import merge_v5
from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference_v1 import MoGePlanarityInference
from planamono.inference.planarity.moge_inference_neck_head import (
    MoGePlanarityNeckHeadInference,
    MoGePlanarityProjNeckHeadInference,
)
from planamono.paths import repo_path, scannetpp_rend_plane_path

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
# CONFIGURATION (defaults, can be overridden via CLI)
# ============================================================

DEFAULT_CONFIG = {
    "model_path": "/cluster/scratch/aoezkan/moge_runs/scannetpp/moge_scannetpp_4heads_v3/final_planarity_4heads_model.pt",
    "output_root": "/cluster/scratch/aoezkan/planeseg/scannetpp/inference/moge_ours_v3_h5",
    "dataset_dir": "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp",
    "rgb_root": "/cluster/project/cvg/Shared_datasets/scannet++/data",

    # Inference parameters
    "num_tokens": 1024,
    "batch_size": 16,  # Increased from 8 (evaluation uses 32)

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
    scene_predictions: Dict[str, List[Tuple[str, np.ndarray]]],
    h5_root: str
):
    """
    Save predictions to H5 files (one per scene).

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
    seg_fn=compute_vectorized_planar_segments_v5,
) -> List[Dict]:
    """
    Run batch inference and segmentation.

    Returns list of dicts with scene_id, frame_id, labels.
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
                return_all_heads=True
            )
        else:
            results = inference_model.predict_batch_fast(
                rgb_paths,
                num_tokens=args.num_tokens,
                return_all_heads=True
            )

    batch_outputs = []

    for res, rgb_path, scene_id, frame_id in zip(results, rgb_paths, scene_ids, frame_ids):
        # Get image dimensions
        with timer("postprocess_io"):
            img = Image.open(rgb_path).convert("RGB")
            img_np = np.array(img)
            H_rgb, W_rgb = img_np.shape[:2]

        with timer("postprocess_extract"):
            # Get MoGe outputs
            planarity = res["planarity_probability"]
            depth_moge = res["depth"] if args.metric_depth else res["points"][:, :, 2]
            normal = res["normal"].transpose(2, 0, 1)  # (3, H, W)

        # Resize to RGB resolution
        with timer("postprocess_resize"):
            planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            depth_moge_rgb = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
            normal_rgb = cv2.resize(
                normal.transpose(1, 2, 0), (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)

        with timer("postprocess_threshold"):
            # Apply planarity threshold (not used for v9_vote which does its own voting)
            if args.seg_version != "v9_vote":
                planarity_mask = (planarity_rgb > args.threshold_planarity).astype(np.int32)

        # Run vectorized segmentation
        with timer("segmentation_compute"):
            seg_kwargs = dict(
                neighbor_match_count_thresh=args.neighbor_match_count_thresh
            )
            if args.seg_version in ("v10", "v11"):
                seg_kwargs.update(
                    adaptive_frac=args.adaptive_frac,
                    min_valid_neighbors=args.min_valid_neighbors,
                    min_segment_pixels=args.min_segment_pixels,
                )
            if args.seg_version == "v11":
                seg_kwargs.update(
                    merge_normal_deg=args.merge_normal_deg,
                    merge_offset_m=args.merge_offset_m,
                    merge_min_pixels=args.merge_min_pixels,
                    merge_gap_px=args.merge_gap_px,
                )
            if args.seg_version == "v9_vote":
                seg_kwargs.update(
                    planarity_threshold=args.planarity_threshold,
                    planarity_ratio=args.planarity_ratio,
                )
                # v9_vote takes raw planarity (float), not binary mask
                labels, _ = seg_fn(
                    planarity_rgb,
                    normal_rgb.transpose(1, 2, 0),  # (H, W, 3)
                    depth_moge_rgb,
                    np.deg2rad(args.normal_threshold_deg),
                    args.depth_threshold,
                    **seg_kwargs
                )
            else:
                labels, _ = seg_fn(
                    planarity_mask,
                    normal_rgb.transpose(1, 2, 0),  # (H, W, 3)
                    depth_moge_rgb,
                    np.deg2rad(args.normal_threshold_deg),
                    args.depth_threshold,
                    **seg_kwargs
                )

        with timer("segmentation_remap"):
            labels, _ = remap_labels(labels)

        # Optional post-segmentation merge
        if args.merge_version == "v5":
            with timer("merge_v5"):
                # Get K and c2w from batch
                batch_idx = list(zip(batch["scene_id"], batch["frame_idx"])).index(
                    (scene_id, frame_id)
                )
                K_mat = batch["K"][batch_idx]
                c2w_mat = batch["c2w"][batch_idx]
                # K and c2w come as tensors from dataloader, convert to numpy
                if hasattr(K_mat, 'numpy'):
                    K_mat = K_mat.numpy()
                if hasattr(c2w_mat, 'numpy'):
                    c2w_mat = c2w_mat.numpy()
                labels, _ = merge_v5(
                    labels.astype(np.int32),
                    depth_moge_rgb,
                    normal_rgb.transpose(1, 2, 0),
                    K_mat, c2w_mat,
                    merge_normal_deg=args.merge_normal_deg,
                    merge_offset_m=args.merge_offset_m,
                    merge_min_pixels=args.merge_min_pixels,
                    merge_gap_px=args.merge_gap_px,
                    nn_dist_m=args.merge_nn_dist_m,
                    topk=args.merge_topk,
                )

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
        description="MoGe Planarity Inference + Segmentation → H5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Paths
    parser.add_argument("--model_path", type=str, default=DEFAULT_CONFIG["model_path"],
                        help="Path to trained MoGe checkpoint (.pt file)")
    parser.add_argument("--output_root", type=str, default=DEFAULT_CONFIG["output_root"],
                        help="Root directory for H5 output")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_CONFIG["dataset_dir"],
                        help="Directory with rendered GT data")
    parser.add_argument("--rgb_root", type=str, default=DEFAULT_CONFIG["rgb_root"],
                        help="Root directory for RGB images")

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
                        choices=["v5", "v5_relative", "v5_no_sobel", "v5_dotprod_relative",
                                 "v6", "v9_vote", "v10", "v11"],
                        help="Segmentation algorithm version")
    parser.add_argument("--adaptive_frac", type=float, default=0.75,
                        help="[v10/v11] Fraction of valid neighbors required to match")
    parser.add_argument("--min_valid_neighbors", type=int, default=3,
                        help="[v10/v11] Minimum valid neighbor count (absolute floor)")
    parser.add_argument("--min_segment_pixels", type=int, default=50,
                        help="[v10/v11] Segments smaller than this are removed")
    parser.add_argument("--merge_normal_deg", type=float, default=10.0,
                        help="[v11] Max angle between mean normals for merge (degrees)")
    parser.add_argument("--merge_offset_m", type=float, default=0.02,
                        help="[v11] Max mean-depth difference for merge (meters)")
    parser.add_argument("--merge_min_pixels", type=int, default=50,
                        help="[v11] Ignore segments smaller than this for merge")
    parser.add_argument("--merge_gap_px", type=int, default=5,
                        help="[v11] Dilation radius to bridge non-planar gaps (pixels)")

    # v9_vote parameters
    parser.add_argument("--planarity_threshold", type=float, default=0.6,
                        help="[v9_vote] Per-pixel planarity threshold for segment voting")
    parser.add_argument("--planarity_ratio", type=float, default=0.5,
                        help="[v9_vote] Minimum fraction of planar pixels to keep a segment")

    # Post-segmentation merge (merge_v5)
    parser.add_argument("--merge_version", type=str, default="none",
                        choices=["none", "v5"],
                        help="Post-segmentation merge algorithm (none or v5)")
    parser.add_argument("--merge_nn_dist_m", type=float, default=0.2,
                        help="[merge_v5] Max nearest-neighbor distance for 3D adjacency (meters)")
    parser.add_argument("--merge_topk", type=int, default=20,
                        help="[merge_v5] Number of largest segments to consider for merging")

    # Processing
    parser.add_argument("--split", type=str, default=DEFAULT_CONFIG["split"],
                        choices=["train", "val", "test"])
    parser.add_argument("--max_scenes", type=int, default=DEFAULT_CONFIG["max_scenes"],
                        help="Limit number of scenes (for testing)")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--architecture", type=str, default="4head",
                        choices=["4head", "neck_head", "proj_neck_head"],
                        help="Model architecture (4head=original, neck_head/proj_neck_head=separate modules)")
    parser.add_argument("--metric_depth", action="store_true",
                        help="Use MoGe metric depth (model.infer) instead of affine points[:,:,2]")

    args = parser.parse_args()

    # Validate
    if not os.path.isfile(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("MoGe Planarity Inference + Segmentation → H5")
    print("=" * 60)
    print(f"Model:        {args.model_path}")
    print(f"Output:       {args.output_root}")
    print(f"Split:        {args.split}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Planarity θ:  {args.threshold_planarity}")
    print(f"Normal θ:     {args.normal_threshold_deg}°")
    print(f"Depth θ:      {args.depth_threshold}{'(relative)' if args.seg_version in ('v6', 'v10') else 'm'}")
    print(f"Seg version:  {args.seg_version}")
    print(f"Merge:        {args.merge_version}")
    print(f"Metric depth: {args.metric_depth}")
    print("=" * 60)

    seg_fn_map = {
        "v5": compute_vectorized_planar_segments_v5,
        "v5_relative": compute_vectorized_planar_segments_v5_relative,
        "v5_no_sobel": compute_vectorized_planar_segments_v5_no_sobel,
        "v5_dotprod_relative": compute_vectorized_planar_segments_v5_dotprod_relative,
        "v6": compute_vectorized_planar_segments_v6,
        "v9_vote": compute_vectorized_planar_segments_v9_vote,
        "v10": compute_vectorized_planar_segments_v10,
        "v11": compute_vectorized_planar_segments_v11,
    }
    seg_fn = seg_fn_map[args.seg_version]

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

    # Custom collate to handle variable-sized data
    def collate_fn(batch):
        result = {
            "rgb_path": [b["rgb_path"] for b in batch],
            "scene_id": [b["scene_id"] for b in batch],
            "frame_idx": [b["frame_idx"] for b in batch],
        }
        if args.merge_version != "none":
            result["K"] = [b["K"] for b in batch]
            result["c2w"] = [b["c2w"] for b in batch]
        return result

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

    # Optional: Use half precision for faster inference (comment out if issues)
    # inference_model.model = inference_model.model.half()

    print("[INFO] Model loaded")

    # Initialize timer
    timer = Timer()

    # Process all batches with incremental saving
    scene_predictions: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    scenes_completed = set()

    # Get total frames per scene for tracking (fast: use internal valid_pairs, avoid __getitem__)
    from collections import Counter
    scene_frame_counts = Counter()
    for pair in dataset.valid_pairs:
        # pair = (rgb_path, plane_h5, sem_h5, depth_h5, idx, K, c2w)
        rgb_path = pair[0]
        scene_id = rgb_path.split("/")[-4]  # extract scene_id from path
        scene_frame_counts[scene_id] += 1

    print("[INFO] Running inference...")
    for batch in tqdm(dataloader, desc="Processing"):
        outputs = process_batch(batch, inference_model, args, timer, seg_fn=seg_fn)

        with timer("accumulate"):
            for out in outputs:
                scene_id = out["scene_id"]
                scene_predictions[scene_id].append(
                    (out["frame_id"], out["labels"])
                )

        # Save scene when complete to free memory
        with timer("save_h5"):
            for scene_id in list(scene_predictions.keys()):
                if scene_id not in scenes_completed and len(scene_predictions[scene_id]) >= scene_frame_counts[scene_id]:
                    print(f"\n[H5] Saving {scene_id} ({len(scene_predictions[scene_id])} frames)...")
                    save_predictions_h5({scene_id: scene_predictions[scene_id]}, args.output_root)
                    scenes_completed.add(scene_id)
                    del scene_predictions[scene_id]  # Free memory
                    torch.cuda.empty_cache()  # Clear GPU cache too

    # Save any remaining scenes
    if scene_predictions:
        with timer("save_h5"):
            print(f"\n[H5] Saving {len(scene_predictions)} remaining scenes...")
            save_predictions_h5(dict(scene_predictions), args.output_root)

    # Print timing breakdown
    timer.print_summary(num_frames=len(dataset))

    # Save timing to CSV
    timing_csv = os.path.join(args.output_root, "runtime_breakdown.csv")
    timer.save_csv(timing_csv)

    # Save all parameters to YAML for reproducibility
    import yaml
    config = {
        "model_path": args.model_path,
        "seg_version": args.seg_version,
        "metric_depth": args.metric_depth,
        "threshold_planarity": args.threshold_planarity,
        "normal_threshold_deg": args.normal_threshold_deg,
        "depth_threshold": args.depth_threshold,
        "neighbor_match_count_thresh": args.neighbor_match_count_thresh,
        "adaptive_frac": args.adaptive_frac,
        "min_valid_neighbors": args.min_valid_neighbors,
        "min_segment_pixels": args.min_segment_pixels,
        "split": args.split,
        "batch_size": args.batch_size,
        "num_tokens": args.num_tokens,
        "architecture": args.architecture,
        "merge_version": args.merge_version,
        "merge_nn_dist_m": args.merge_nn_dist_m,
        "merge_topk": args.merge_topk,
        "planarity_threshold": args.planarity_threshold,
        "planarity_ratio": args.planarity_ratio,
    }
    config_path = os.path.join(args.output_root, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[CONFIG] Saved parameters to {config_path}")

    print("=" * 60)
    print(f"[DONE] Saved {len(scenes_completed) + len(scene_predictions)} scenes to {args.output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
