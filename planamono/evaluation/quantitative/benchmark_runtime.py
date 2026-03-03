#!/usr/bin/env python3
"""
Runtime benchmark for MoGe planarity + plane segmentation pipeline.

Measures per-frame timing for every pipeline component:
  - MoGe inference (GPU)
  - Postprocessing (resize, threshold)
  - Segmentation (with internal breakdown)
  - Label remapping

Saves per-frame CSV and dataset-aggregated CSV. No predictions are kept.

Usage:
  # ScanNet++
  python benchmark_runtime.py --method v5 --dataset scannetpp
  python benchmark_runtime.py --method v6 --dataset scannetpp

  # Hypersim
  python benchmark_runtime.py --method v5 --dataset hypersim
  python benchmark_runtime.py --method v6 --dataset hypersim

  # Quick test
  python benchmark_runtime.py --method v5 --dataset scannetpp --max-scenes 2
"""

import os
import sys
import time
import argparse
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Tuple

import numpy as np
import cv2
import h5py
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from planamono.shared.utils.label_utils import remap_labels
from planamono.inference.planarity.moge_inference_v1 import MoGePlanarityInference
from planamono.paths import repo_path, scannetpp_rend_plane_path

import cc3d


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/cluster/scratch/ayavuz/moge_mixed_output_bce_476644_fixed/model_epoch6.pt"
NUM_TOKENS = 1024

# ScanNet++ paths
SCANNETPP_DATASET_DIR = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"
SCANNETPP_RGB_ROOT = "/cluster/project/cvg/Shared_datasets/scannet++/data"

# Hypersim paths (unified dataset from paths.py)
HYPERSIM_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
HYPERSIM_PLANE_ROOT = "/cluster/scratch/aoezkan/planeseg/dataset/hypersim"
HYPERSIM_PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/HP_all/Hypersim_params"

SPLIT = "test"

METHODS = {
    "v5": {
        "seg_version": "v5",
        "threshold_planarity": 0.6,
        "normal_threshold_deg": 10.0,
        "depth_threshold": 0.05,
        "neighbor_match_count_thresh": 24,
        "output_name": "moge_mixed_bce_476644_ep6_v6",
    },
    "v6": {
        "seg_version": "v6",
        "threshold_planarity": 0.6,
        "normal_threshold_deg": 10.0,
        "depth_threshold": 0.02,
        "neighbor_match_count_thresh": 18,
        "output_name": "moge_mixed_bce_476644_ep6_v6seg_v6",
    },
}


# ============================================================
# HYPERSIM HDR TONEMAP
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


def load_hypersim_rgb(rgb_hdf5_path):
    """Load Hypersim RGB from HDF5, tonemap, return uint8 (H, W, 3)."""
    with h5py.File(rgb_hdf5_path, "r") as f:
        key = list(f.keys())[0]
        rgb = f[key][:]
    if rgb.dtype == np.uint8:
        return rgb
    elif rgb.dtype == np.uint16:
        return (rgb.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
    else:
        rgb = tonemap_hdr(rgb.astype(np.float32))
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ============================================================
# TIMED SEGMENTATION (v5 with internal breakdown)
# ============================================================

def segmentation_v5_timed(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int,
    device: str = "cuda",
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """v5 segmentation with per-stage timing. Returns (labels, n_components, timings_dict)."""
    timings = {}
    H, W = planarity_mask.shape

    # Stage 1: CPU→GPU transfer
    t0 = time.perf_counter()
    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )
    if device == "cuda":
        torch.cuda.synchronize()
    timings["seg_cpu_to_gpu"] = time.perf_counter() - t0

    # Stage 2: Sobel normal gradient
    t0 = time.perf_counter()
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32)
    sobel_x_batch = sobel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    sobel_y_batch = sobel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    normal_batch = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_dx = F.conv2d(normal_batch, sobel_x_batch, padding=1, groups=3)
    normal_dy = F.conv2d(normal_batch, sobel_y_batch, padding=1, groups=3)
    normal_grad_mag = torch.sqrt((normal_dx ** 2 + normal_dy ** 2).sum(dim=1)).squeeze(0)
    grad_mag_threshold = torch.sqrt(
        torch.tensor(2.0, device=device) -
        2 * torch.cos(torch.tensor(normal_threshold_rad, device=device))
    )
    normal_similar = (normal_grad_mag <= grad_mag_threshold)
    if device == "cuda":
        torch.cuda.synchronize()
    timings["seg_sobel_normal"] = time.perf_counter() - t0

    # Stage 3: Unfold + neighbor matching
    t0 = time.perf_counter()
    kernel_size = 5
    pad = kernel_size // 2

    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode='constant', value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size).view(25, H, W)

    mask_padded = F.pad(planarity_t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size).view(25, H, W).bool()

    normal_sim_padded = F.pad(normal_similar[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    normal_sim_patches = F.unfold(normal_sim_padded, kernel_size=kernel_size).view(25, H, W).bool()

    center_idx = 12
    neighbor_indices = [i for i in range(25) if i != center_idx]
    neighbor_depths = depth_patches[neighbor_indices]
    neighbor_masks = mask_patches[neighbor_indices]
    neighbor_normals = normal_sim_patches[neighbor_indices]

    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks
    depth_diff = torch.abs(depth_t.unsqueeze(0) - neighbor_depths)
    depth_close = (depth_diff < depth_threshold)
    matches = valid_pair & neighbor_normals & depth_close
    neighbor_match_count = matches.sum(dim=0)
    connected = (neighbor_match_count >= neighbor_match_count_thresh)
    if device == "cuda":
        torch.cuda.synchronize()
    timings["seg_unfold_match"] = time.perf_counter() - t0

    # Stage 4: GPU→CPU + connected components
    t0 = time.perf_counter()
    connected_np = connected.cpu().numpy()
    timings["seg_gpu_to_cpu"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = cc3d.connected_components(connected_np)
    num_components = int(labels.max())
    timings["seg_connected_components"] = time.perf_counter() - t0

    return labels, num_components, timings


def segmentation_v6_timed(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int,
    device: str = "cuda",
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """v6 segmentation with per-stage timing. Returns (labels, n_components, timings_dict)."""
    timings = {}
    H, W = planarity_mask.shape

    # Stage 1: CPU→GPU transfer
    t0 = time.perf_counter()
    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )
    if device == "cuda":
        torch.cuda.synchronize()
    timings["seg_cpu_to_gpu"] = time.perf_counter() - t0

    # Stage 2: Unfold (depth, mask, normals)
    t0 = time.perf_counter()
    kernel_size = 5
    pad = kernel_size // 2

    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode='constant', value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size).view(25, H, W)

    mask_padded = F.pad(planarity_t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size).view(25, H, W).bool()

    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode='constant', value=0)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size).view(3, 25, H, W)
    if device == "cuda":
        torch.cuda.synchronize()
    timings["seg_unfold"] = time.perf_counter() - t0

    # Stage 3: Pairwise normal similarity + relative depth matching
    t0 = time.perf_counter()
    center_idx = 12
    neighbor_indices = [i for i in range(25) if i != center_idx]

    neighbor_depths = depth_patches[neighbor_indices]
    neighbor_masks = mask_patches[neighbor_indices]
    neighbor_normals = normal_patches[:, neighbor_indices]

    center_normal = normal_t.permute(2, 0, 1)
    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0)
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = torch.acos(dot)
    normal_similar = angle < normal_threshold_rad

    center_depth = depth_t.unsqueeze(0)
    depth_diff = torch.abs(center_depth - neighbor_depths)
    depth_close = depth_diff < (depth_threshold * center_depth)

    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks
    matches = valid_pair & normal_similar & depth_close
    neighbor_match_count = matches.sum(dim=0)
    connected = (neighbor_match_count >= neighbor_match_count_thresh)
    if device == "cuda":
        torch.cuda.synchronize()
    timings["seg_pairwise_match"] = time.perf_counter() - t0

    # Stage 4: GPU→CPU + connected components
    t0 = time.perf_counter()
    connected_np = connected.cpu().numpy()
    timings["seg_gpu_to_cpu"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = cc3d.connected_components(connected_np)
    num_components = int(labels.max())
    timings["seg_connected_components"] = time.perf_counter() - t0

    return labels, num_components, timings


# ============================================================
# DATASET LOADERS
# ============================================================

def load_scannetpp_dataset(max_scenes):
    from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
    dataset = ScanNetPPPlaneDataset(
        rgb_root=SCANNETPP_RGB_ROOT,
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=SCANNETPP_DATASET_DIR,
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split=SPLIT,
        max_scenes=max_scenes,
    )
    return dataset


def load_hypersim_dataset(max_scenes):
    from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
    dataset = HypersimPlaneDataset(
        hypersim_root=HYPERSIM_ROOT,
        plane_label_root=HYPERSIM_PLANE_ROOT,
        params_root=HYPERSIM_PARAMS_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "hypersim"),
        split=SPLIT,
        max_scenes=max_scenes,
    )
    return dataset


# ============================================================
# PER-FRAME BENCHMARK FUNCTIONS
# ============================================================

def benchmark_scannetpp_frame(
    sample, inference_model, seg_fn_timed,
    threshold_planarity, normal_threshold_rad, depth_threshold,
    neighbor_match_count_thresh, device,
):
    """Benchmark one ScanNet++ frame. Returns timing dict."""
    scene_id = sample["scene_id"]
    frame_idx = sample["frame_idx"]
    rgb_path = sample["rgb_path"]

    row = {"scene_id": scene_id, "frame_idx": frame_idx}

    # --- MoGe inference (includes image load + preprocess inside predict_batch_fast) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = inference_model.predict_batch_fast(
        [rgb_path], num_tokens=NUM_TOKENS, return_all_heads=True
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    row["moge_inference"] = time.perf_counter() - t0

    res = results[0]

    # --- Load image for resolution ---
    t0 = time.perf_counter()
    img = Image.open(rgb_path).convert("RGB")
    H_rgb, W_rgb = np.array(img).shape[:2]
    row["load_image"] = time.perf_counter() - t0

    # --- Resize MoGe outputs to RGB resolution ---
    t0 = time.perf_counter()
    planarity = res["planarity_probability"]
    depth_moge = res["points"][:, :, 2]
    normal = res["normal"]  # (h, w, 3) — MoGe v2 format

    planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
    depth_moge_rgb = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
    normal_rgb = cv2.resize(normal, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
    row["resize_outputs"] = time.perf_counter() - t0

    # --- Threshold ---
    t0 = time.perf_counter()
    planarity_mask = (planarity_rgb > threshold_planarity).astype(np.int32)
    row["threshold"] = time.perf_counter() - t0

    # --- Segmentation ---
    t0 = time.perf_counter()
    labels, n_comp, seg_timings = seg_fn_timed(
        planarity_mask, normal_rgb, depth_moge_rgb,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh, device=str(device),
    )
    row["seg_total"] = time.perf_counter() - t0
    row.update(seg_timings)

    # --- Label remap ---
    t0 = time.perf_counter()
    labels, _ = remap_labels(labels)
    row["label_remap"] = time.perf_counter() - t0

    # --- Total ---
    row["total"] = (
        row["moge_inference"] + row["load_image"] + row["resize_outputs"]
        + row["threshold"] + row["seg_total"] + row["label_remap"]
    )
    row["n_segments"] = n_comp
    row["resolution"] = f"{W_rgb}x{H_rgb}"

    return row


def benchmark_hypersim_frame(
    sample, inference_model, seg_fn_timed,
    threshold_planarity, normal_threshold_rad, depth_threshold,
    neighbor_match_count_thresh, device,
):
    """Benchmark one Hypersim frame. Returns timing dict."""
    scene_id = sample["scene_id"]
    frame_idx = sample["frame_idx"]
    # rgb_path is "scene_id/cam_name/fid" for Hypersim
    formatted_path = sample["rgb_path"]
    parts = formatted_path.split("/")
    cam_name, fid = parts[1], parts[2]

    # Reconstruct actual HDF5 path
    rgb_hdf5_path = os.path.join(
        HYPERSIM_ROOT, scene_id, "images",
        f"scene_{cam_name}_final_hdf5", f"frame.{fid}.color.hdf5"
    )

    row = {"scene_id": scene_id, "frame_idx": frame_idx}

    # --- Load HDF5 RGB + tonemap ---
    t0 = time.perf_counter()
    rgb_uint8 = load_hypersim_rgb(rgb_hdf5_path)
    H_rgb, W_rgb = rgb_uint8.shape[:2]
    row["load_image"] = time.perf_counter() - t0

    # --- MoGe inference (pass numpy array) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    # preprocess_images accepts numpy arrays
    images_tensor, original_sizes = inference_model.preprocess_images([rgb_uint8])
    with torch.no_grad():
        outputs = inference_model.model(images_tensor, num_tokens=NUM_TOKENS)

    # Extract single-frame results
    planarity = outputs["planarity"][0].squeeze().cpu().numpy()
    h0, w0 = original_sizes[0]
    planarity_full = cv2.resize(planarity, (w0, h0))
    normal = outputs["normal"][0].cpu().numpy()  # (H, W, 3)
    points = outputs["points"][0].cpu().numpy()   # (H, W, 3)

    if device.type == "cuda":
        torch.cuda.synchronize()
    row["moge_inference"] = time.perf_counter() - t0

    # --- Resize MoGe outputs to native RGB resolution ---
    t0 = time.perf_counter()
    depth_moge = points[:, :, 2]
    planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
    depth_moge_rgb = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
    normal_rgb = cv2.resize(normal, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
    row["resize_outputs"] = time.perf_counter() - t0

    # --- Threshold ---
    t0 = time.perf_counter()
    planarity_mask = (planarity_rgb > threshold_planarity).astype(np.int32)
    row["threshold"] = time.perf_counter() - t0

    # --- Segmentation ---
    t0 = time.perf_counter()
    labels, n_comp, seg_timings = seg_fn_timed(
        planarity_mask, normal_rgb, depth_moge_rgb,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh, device=str(device),
    )
    row["seg_total"] = time.perf_counter() - t0
    row.update(seg_timings)

    # --- Label remap ---
    t0 = time.perf_counter()
    labels, _ = remap_labels(labels)
    row["label_remap"] = time.perf_counter() - t0

    # --- Total ---
    row["total"] = (
        row["moge_inference"] + row["load_image"] + row["resize_outputs"]
        + row["threshold"] + row["seg_total"] + row["label_remap"]
    )
    row["n_segments"] = n_comp
    row["resolution"] = f"{W_rgb}x{H_rgb}"

    return row


# ============================================================
# RESULTS SAVING + PRINTING
# ============================================================

def save_and_print_results(all_rows, output_dir, output_name):
    """Save per-frame, per-scene, and dataset CSVs. Print summary."""
    if not all_rows:
        print(f"\n[WARN] No frames benchmarked for {output_name}. Skipping CSV generation.")
        return

    df = pd.DataFrame(all_rows)
    per_frame_path = os.path.join(output_dir, "runtime_per_frame.csv")
    df.to_csv(per_frame_path, index=False)
    print(f"\n[CSV] Per-frame: {per_frame_path}")

    # Per-scene aggregation
    time_cols = [c for c in df.columns if c not in ("scene_id", "frame_idx", "n_segments", "resolution")]
    df_scene = df.groupby("scene_id")[time_cols].agg(["mean", "std", "count"])
    df_scene.columns = [f"{c[0]}_{c[1]}" for c in df_scene.columns]
    scene_path = os.path.join(output_dir, "runtime_per_scene.csv")
    df_scene.to_csv(scene_path)
    print(f"[CSV] Per-scene: {scene_path}")

    # Dataset-level summary
    summary = {"num_frames": len(df), "num_scenes": df["scene_id"].nunique()}
    for col in time_cols:
        summary[f"{col}_mean_ms"] = df[col].mean() * 1000
        summary[f"{col}_std_ms"] = df[col].std() * 1000
        summary[f"{col}_median_ms"] = df[col].median() * 1000

    summary["fps_mean"] = 1.0 / df["total"].mean() if df["total"].mean() > 0 else 0
    summary["fps_median"] = 1.0 / df["total"].median() if df["total"].median() > 0 else 0

    df_summary = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "runtime_dataset.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[CSV] Dataset:   {summary_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"RUNTIME SUMMARY — {output_name}")
    print("=" * 60)
    print(f"{'Stage':<30s} {'Mean (ms)':>10s} {'Std (ms)':>10s} {'Median (ms)':>12s}")
    print("-" * 62)
    for col in time_cols:
        mean_ms = df[col].mean() * 1000
        std_ms = df[col].std() * 1000
        med_ms = df[col].median() * 1000
        print(f"{col:<30s} {mean_ms:>10.2f} {std_ms:>10.2f} {med_ms:>12.2f}")
    print("-" * 62)
    print(f"{'FPS (mean)':<30s} {summary['fps_mean']:>10.2f}")
    print(f"{'FPS (median)':<30s} {summary['fps_median']:>10.2f}")
    print(f"{'Frames':<30s} {len(df):>10d}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Runtime benchmark for MoGe + segmentation")
    parser.add_argument("--method", type=str, required=True, choices=["v5", "v6"],
                        help="Segmentation method to benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=["scannetpp", "hypersim"],
                        help="Dataset to benchmark on")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Limit scenes for testing")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    method_cfg = METHODS[args.method]
    output_name = method_cfg["output_name"]
    output_root = f"/cluster/scratch/aoezkan/planeseg/{args.dataset}/runtime"
    output_dir = os.path.join(output_root, output_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Runtime Benchmark: {output_name}")
    print("=" * 60)
    print(f"Dataset:     {args.dataset}")
    print(f"Model:       {MODEL_PATH}")
    print(f"Split:       {SPLIT}")
    print(f"Seg version: {method_cfg['seg_version']}")
    print(f"Params:      planarity={method_cfg['threshold_planarity']}, "
          f"normal={method_cfg['normal_threshold_deg']}deg, "
          f"depth={method_cfg['depth_threshold']}, "
          f"neighbors={method_cfg['neighbor_match_count_thresh']}")
    print(f"Output:      {output_dir}")
    print("=" * 60)

    # Load dataset
    if args.dataset == "scannetpp":
        dataset = load_scannetpp_dataset(args.max_scenes)
    else:
        dataset = load_hypersim_dataset(args.max_scenes)
    print(f"[DATA] {len(dataset)} frames")

    def collate_fn(batch):
        return batch[0]

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True,
    )

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    inference_model = MoGePlanarityInference(MODEL_PATH, device=str(device))
    inference_model.model.encoder.use_memory_efficient_attention = False
    torch.set_grad_enabled(False)
    inference_model.model.eval()

    # Select segmentation function
    seg_fn_timed = segmentation_v6_timed if method_cfg["seg_version"] == "v6" else segmentation_v5_timed

    # Segmentation params
    threshold_planarity = method_cfg["threshold_planarity"]
    normal_threshold_rad = np.deg2rad(method_cfg["normal_threshold_deg"])
    depth_threshold = method_cfg["depth_threshold"]
    neighbor_match_count_thresh = method_cfg["neighbor_match_count_thresh"]

    # Pick per-frame benchmark function
    if args.dataset == "scannetpp":
        bench_frame = benchmark_scannetpp_frame
    else:
        bench_frame = benchmark_hypersim_frame

    # Warmup (2 frames)
    print("[INFO] Warming up GPU...")
    for i, sample in enumerate(dataloader):
        if i >= 2:
            break
        if args.dataset == "scannetpp":
            inference_model.predict_batch_fast(
                [sample["rgb_path"]], num_tokens=NUM_TOKENS, return_all_heads=True
            )
        else:
            # Hypersim: load RGB array, preprocess, forward
            parts = sample["rgb_path"].split("/")
            cam_name, fid = parts[1], parts[2]
            rgb_hdf5 = os.path.join(
                HYPERSIM_ROOT, sample["scene_id"], "images",
                f"scene_{cam_name}_final_hdf5", f"frame.{fid}.color.hdf5"
            )
            rgb_uint8 = load_hypersim_rgb(rgb_hdf5)
            imgs, _ = inference_model.preprocess_images([rgb_uint8])
            with torch.no_grad():
                inference_model.model(imgs, num_tokens=NUM_TOKENS)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[INFO] Warmup done")

    # Benchmark loop
    all_rows = []
    for sample in tqdm(dataloader, desc="Benchmarking"):
        row = bench_frame(
            sample, inference_model, seg_fn_timed,
            threshold_planarity, normal_threshold_rad, depth_threshold,
            neighbor_match_count_thresh, device,
        )
        all_rows.append(row)

    # Save and print
    save_and_print_results(all_rows, output_dir, f"{output_name} ({args.dataset})")


if __name__ == "__main__":
    main()
