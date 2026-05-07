#!/usr/bin/env python3
"""
Grid search over v10 segmentation parameters on ScanNet++ val split.

Supports metric depth (via model.infer()) and v10-specific parameters
(adaptive_frac, min_valid_neighbors, min_segment_pixels).

Three modes:
  --mode cache    GPU job: run MoGe inference once, cache raw outputs to H5 (gzip)
  --mode eval     CPU job: load cache, run segmentation + evaluation for config range
  --mode aggregate  Merge all config results into a summary CSV

Usage:
  python grid_search_v10.py --yaml grid_search_v10_config.yaml --mode cache
  python grid_search_v10.py --yaml grid_search_v10_config.yaml --mode eval --config-start 0 --config-end 3
  python grid_search_v10.py --yaml grid_search_v10_config.yaml --mode aggregate
"""

import os
import sys
import argparse
import random
import itertools
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import cv2
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import h5py
from scipy.ndimage import label

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.utils.label_utils import remap_labels
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path

from eval_utils import (
    Timer,
    save_results_csv,
    evaluate_single_frame,
)


# ============================================================
# CONFIG GENERATION (copied from grid_search_segmentation.py)
# ============================================================

def generate_configs(cfg: dict) -> List[dict]:
    """Generate list of parameter configs from YAML grid definition."""
    base = dict(cfg["base_config"])
    grid = cfg["grid"]
    mode = cfg.get("search_mode", "one_at_a_time")

    if mode == "full_grid":
        keys = sorted(grid.keys())
        values = [grid[k] for k in keys]
        configs = []
        for combo in itertools.product(*values):
            c = dict(zip(keys, combo))
            configs.append(c)
        return configs

    # one_at_a_time: sweep each param independently, others at base
    configs_set = set()
    configs = []

    # Always include the base config
    base_tuple = tuple(sorted(base.items()))
    configs_set.add(base_tuple)
    configs.append(dict(base))

    for param_name, param_values in sorted(grid.items()):
        for val in param_values:
            c = dict(base)
            c[param_name] = val
            c_tuple = tuple(sorted(c.items()))
            if c_tuple not in configs_set:
                configs_set.add(c_tuple)
                configs.append(c)

    return configs


def print_configs_summary(configs: List[dict]):
    """Print a brief summary of generated configs."""
    print(f"Total configs: {len(configs)}")
    if len(configs) <= 40:
        for i, c in enumerate(configs):
            parts = [f"{k}={v}" for k, v in sorted(c.items())]
            print(f"  [{i:4d}] {', '.join(parts)}")
    else:
        for i in [0, 1, 2]:
            parts = [f"{k}={v}" for k, v in sorted(configs[i].items())]
            print(f"  [{i:4d}] {', '.join(parts)}")
        print(f"  ... ({len(configs) - 6} more)")
        for i in [-3, -2, -1]:
            parts = [f"{k}={v}" for k, v in sorted(configs[i].items())]
            print(f"  [{len(configs)+i:4d}] {', '.join(parts)}")


# ============================================================
# V10 SEGMENTATION (inline, self-contained)
# ============================================================

def compute_vectorized_planar_segments_v10(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 18,
    device: str = "cuda",
    adaptive_frac: float = 0.75,
    min_valid_neighbors: int = 3,
    min_segment_pixels: int = 50,
) -> Tuple[np.ndarray, int]:
    """
    Planar segmentation v10: adaptive threshold + small-segment filter.

    Uses pairwise normal dot-product comparison, symmetric relative depth,
    5x5 F.unfold neighborhoods, and scipy.ndimage.label for 8-connected CC.

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Relative depth threshold (fraction, e.g. 0.025 = 2.5%)
        neighbor_match_count_thresh: Ignored when adaptive_frac > 0 (API compat)
        device: Torch device
        adaptive_frac: Fraction of valid neighbors required to match (0.75 = 75%)
        min_valid_neighbors: Minimum valid neighbor count to consider a pixel
        min_segment_pixels: Segments smaller than this are removed (0 to disable)

    Returns:
        labels: (H,W) int32 array with segment IDs (0 = background)
        num_components: Number of segments found
    """
    H, W = planarity_mask.shape

    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )

    normal_t = F.normalize(normal_t, dim=-1, eps=1e-6)

    kernel_size = 5
    pad = kernel_size // 2
    center_idx = kernel_size ** 2 // 2  # = 12
    neighbor_indices = [i for i in range(kernel_size ** 2) if i != center_idx]

    # 1. Pairwise normal comparison
    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode="constant", value=0)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size)
    normal_patches = normal_patches.view(3, kernel_size ** 2, H, W)

    neighbor_normals = normal_patches[:, neighbor_indices]
    neighbor_normals = F.normalize(neighbor_normals, dim=0, eps=1e-6)

    center_normal = normal_t.permute(2, 0, 1)

    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0)
    dot = torch.clamp(dot, -1.0, 1.0)
    cos_thresh = torch.cos(torch.tensor(normal_threshold_rad, device=device))
    normal_similar = dot > cos_thresh

    # 2. Unfold depth and planarity mask
    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode="constant", value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size)
    neighbor_depths = depth_patches.view(kernel_size ** 2, H, W)[neighbor_indices]

    mask_padded = F.pad(
        planarity_t[None, None].float(), (pad, pad, pad, pad), mode="constant", value=0
    )
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size)
    neighbor_masks = mask_patches.view(kernel_size ** 2, H, W)[neighbor_indices].bool()

    # 3. Symmetric relative depth comparison
    center_depth = depth_t.unsqueeze(0)
    depth_diff = torch.abs(center_depth - neighbor_depths)
    avg_depth = ((center_depth + neighbor_depths) * 0.5).clamp(min=0.1)
    depth_close = depth_diff < (depth_threshold * avg_depth)

    # 4. Valid pairs and matching
    depth_valid = (center_depth > 0) & (neighbor_depths > 0)
    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks & depth_valid

    matches = valid_pair & normal_similar & depth_close
    match_count = matches.sum(dim=0)
    valid_count = valid_pair.sum(dim=0)

    # 5. Adaptive threshold (v10)
    adaptive_thresh = (valid_count.float() * adaptive_frac).clamp(min=float(min_valid_neighbors))
    connected = (match_count >= adaptive_thresh) & (valid_count >= min_valid_neighbors)

    # 6. Connected components on CPU
    structure = np.ones((3, 3), dtype=bool)
    labels, num_components = label(connected.detach().cpu().numpy(), structure=structure)
    labels = labels.astype(np.int32)

    # 7. Small segment filter (v10)
    if min_segment_pixels > 0:
        flat = labels.ravel()
        counts = np.bincount(flat, minlength=num_components + 1)
        small_labels = np.where(counts < min_segment_pixels)[0]
        small_labels = small_labels[small_labels > 0]
        if len(small_labels) > 0:
            lut = np.arange(num_components + 1, dtype=np.int32)
            lut[small_labels] = 0
            labels = lut[flat].reshape(H, W)
            num_components = len(np.unique(labels)) - 1

    return labels, num_components


# ============================================================
# MODE: CACHE (GPU inference)
# ============================================================

def run_cache(cfg: dict):
    """Run MoGe inference on selected scenes and cache raw outputs."""
    from planamono.inference.planarity.moge_inference_v1 import MoGePlanarityInference

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Copy config for reproducibility
    config_copy_path = os.path.join(output_dir, "grid_search_config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Load dataset
    dataset_dir = scannetpp_rend_plane_path
    dataset = ScanNetPPPlaneDataset(
        rgb_root=os.path.join(scannetpp_path, "data"),
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=dataset_dir,
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split=cfg["split"],
        max_scenes=None,
    )
    print(f"[DATA] Full {cfg['split']} set: {len(dataset)} frames")

    # Identify all scenes
    scene_to_indices = defaultdict(list)
    for idx, pair in enumerate(dataset.valid_pairs):
        rgb_path = pair[0]
        scene_id = rgb_path.split("/")[-4]
        scene_to_indices[scene_id].append(idx)

    all_scenes = sorted(scene_to_indices.keys())
    print(f"[DATA] Total scenes in {cfg['split']}: {len(all_scenes)}")

    # Select scenes: null means all, integer means random sample
    n_scenes = cfg.get("n_scenes", None)
    if n_scenes is None or n_scenes >= len(all_scenes):
        selected_scenes = all_scenes
    else:
        random.seed(42)
        selected_scenes = sorted(random.sample(all_scenes, n_scenes))

    # Save selected scenes
    scenes_path = os.path.join(output_dir, "selected_scenes.txt")
    with open(scenes_path, "w") as f:
        for s in selected_scenes:
            f.write(s + "\n")
    print(f"[DATA] Selected {len(selected_scenes)} scenes: {selected_scenes[:5]}{'...' if len(selected_scenes) > 5 else ''}")

    # Collect indices for selected scenes
    selected_indices = []
    for scene_id in selected_scenes:
        selected_indices.extend(scene_to_indices[scene_id])
    selected_indices.sort()
    print(f"[DATA] Total frames to cache: {len(selected_indices)}")

    # Load model
    model_path = cfg["model_path"]
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] Loading from {model_path}")
    inference_model = MoGePlanarityInference(model_path, device=str(device))
    inference_model.model.encoder.use_memory_efficient_attention = False
    torch.set_grad_enabled(False)
    inference_model.model.eval()

    use_metric_depth = cfg.get("use_metric_depth", False)
    print(f"[CONFIG] use_metric_depth={use_metric_depth}")

    # Process scene-by-scene to keep memory bounded
    BATCH_SIZE = 16
    timer = Timer()
    cache_dir = os.path.join(output_dir, "cached_scenes")
    os.makedirs(cache_dir, exist_ok=True)
    total_frames = 0

    for scene_id in selected_scenes:
        scene_indices = sorted(scene_to_indices[scene_id])
        print(f"\n[SCENE] {scene_id}: {len(scene_indices)} frames")
        scene_frames = []

        for batch_start in tqdm(
            range(0, len(scene_indices), BATCH_SIZE),
            desc=f"  {scene_id}",
        ):
            batch_indices = scene_indices[batch_start:batch_start + BATCH_SIZE]

            # Collect batch data from dataset
            rgb_paths = []
            scene_ids_batch = []
            frame_idxs = []
            gt_segs = []
            depths = []
            Ks = []
            c2ws = []

            for idx in batch_indices:
                sample = dataset[idx]
                rgb_paths.append(sample["rgb_path"])
                scene_ids_batch.append(sample["scene_id"])
                frame_idxs.append(sample["frame_idx"])
                gt_segs.append(sample["plane"])
                depths.append(sample["depth"])
                Ks.append(sample["K"])
                c2ws.append(sample["c2w"])

            # GPU inference
            with timer("gpu_inference"):
                if use_metric_depth:
                    results = inference_model.predict_batch_fast_metric(
                        rgb_paths,
                        num_tokens=1024,
                        return_all_heads=True,
                    )
                else:
                    results = inference_model.predict_batch_fast(
                        rgb_paths,
                        num_tokens=1024,
                        return_all_heads=True,
                    )

            # Extract per-frame data
            with timer("extract"):
                for i, (res, rgb_path, sid, frame_idx) in enumerate(
                    zip(results, rgb_paths, scene_ids_batch, frame_idxs)
                ):
                    img = Image.open(rgb_path).convert("RGB")
                    img_np = np.array(img)
                    H_rgb, W_rgb = img_np.shape[:2]

                    planarity = res["planarity_probability"]

                    # Depth extraction depends on metric vs affine mode
                    if use_metric_depth:
                        depth_moge = res["depth"]  # (H, W) metric z-depth in meters
                    else:
                        depth_moge = res["points"][:, :, 2]  # affine z-coordinate

                    normal = res["normal"]

                    planarity_rgb = cv2.resize(planarity, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
                    depth_moge_rgb = cv2.resize(depth_moge, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)
                    normal_rgb = cv2.resize(normal, (W_rgb, H_rgb), interpolation=cv2.INTER_LINEAR)

                    gt_seg = gt_segs[i]
                    if gt_seg.ndim == 3:
                        gt_seg = gt_seg[0]
                    gt_seg_np = gt_seg.numpy().astype(np.int32)

                    depth_gt = depths[i]
                    depth_np = depth_gt[0].numpy() if depth_gt.ndim == 3 else depth_gt.numpy()

                    K_np = Ks[i].numpy()
                    c2w_np = c2ws[i].numpy()

                    scene_frames.append({
                        "frame_idx": frame_idx,
                        "planarity": planarity_rgb.astype(np.float16),
                        "depth_moge": depth_moge_rgb.astype(np.float16),
                        "normal": normal_rgb.astype(np.float16),
                        "gt_seg": gt_seg_np,
                        "gt_depth": depth_np.astype(np.float32),
                        "K": K_np.astype(np.float64),
                        "c2w": c2w_np.astype(np.float64),
                    })

        # Save this scene's cache as H5 (gzip-compressed)
        h5_path = os.path.join(cache_dir, f"{scene_id}.h5")
        N = len(scene_frames)
        with timer("save_h5"):
            with h5py.File(h5_path, "w") as hf:
                H, W = scene_frames[0]["planarity"].shape
                H_gt, W_gt = scene_frames[0]["gt_seg"].shape
                gzip_opts = dict(compression="gzip", compression_opts=4)

                hf.create_dataset("planarity",  data=np.stack([f["planarity"]  for f in scene_frames]), **gzip_opts)
                hf.create_dataset("depth_moge", data=np.stack([f["depth_moge"] for f in scene_frames]), **gzip_opts)
                hf.create_dataset("normal",     data=np.stack([f["normal"]     for f in scene_frames]), **gzip_opts)
                hf.create_dataset("gt_seg",     data=np.stack([f["gt_seg"]     for f in scene_frames]), **gzip_opts)
                hf.create_dataset("gt_depth",   data=np.stack([f["gt_depth"]   for f in scene_frames]), **gzip_opts)
                hf.create_dataset("K",          data=np.stack([f["K"]          for f in scene_frames]))
                hf.create_dataset("c2w",        data=np.stack([f["c2w"]        for f in scene_frames]))

                frame_idxs_enc = [f["frame_idx"].encode("utf-8") for f in scene_frames]
                hf.create_dataset("frame_idxs", data=frame_idxs_enc)
                hf.attrs["scene_id"] = scene_id

        h5_mb = os.path.getsize(h5_path) / (1024 * 1024)
        total_frames += N
        print(f"[CACHE] {scene_id}: {N} frames, {h5_mb:.1f} MB")
        del scene_frames
        torch.cuda.empty_cache()

    timer.print_summary(num_frames=total_frames)
    print(f"[DONE] Cached {total_frames} frames from {len(selected_scenes)} scenes -> {cache_dir}")


# ============================================================
# MODE: EVAL (CPU segmentation + evaluation)
# ============================================================

def run_eval_single(cfg: dict, config_index: int, configs: List[dict], scene_h5s: List[Path]):
    """Run v10 segmentation + evaluation for a single parameter config.

    Returns the config output directory path.
    """
    output_dir = cfg["output_dir"]

    seg_params = configs[config_index]
    print(f"\n[CONFIG] Index {config_index}/{len(configs)-1}: {seg_params}")

    # Create output directory for this config
    config_dir_out = os.path.join(output_dir, "configs", f"config_{config_index:04d}")
    os.makedirs(config_dir_out, exist_ok=True)

    # Save this config for traceability
    config_path = os.path.join(config_dir_out, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(seg_params, f, default_flow_style=False)

    # Evaluation constants from YAML
    thresholds = tuple(cfg.get("thresholds", [0.001, 0.005, 0.01]))
    ransac_iterations = cfg.get("ransac_iterations", 200)
    inlier_ratio_gate = cfg.get("inlier_ratio_gate", 0.9)

    # Segmentation parameters
    threshold_planarity = seg_params["threshold_planarity"]
    normal_threshold_rad = np.deg2rad(seg_params["normal_threshold_deg"])
    depth_threshold = seg_params["depth_threshold"]
    neighbor_match_count_thresh = int(seg_params["neighbor_match_count_thresh"])

    # v10-specific parameters
    adaptive_frac = seg_params.get("adaptive_frac", 0.75)
    min_valid_neighbors = int(seg_params.get("min_valid_neighbors", 3))
    min_segment_pixels = int(seg_params.get("min_segment_pixels", 50))

    # Use GPU for segmentation if available (unfold + normalize are much faster)
    seg_device = "cuda" if torch.cuda.is_available() else "cpu"

    timer = Timer()
    results = {}

    total_frames_count = 0
    pbar = tqdm(desc=f"Eval config {config_index}")
    for h5_path in scene_h5s:
        with h5py.File(h5_path, "r") as hf:
            scene_id = hf.attrs["scene_id"]
            N = hf["planarity"].shape[0]
            frame_idxs = [s.decode("utf-8") for s in hf["frame_idxs"][:]]

            for fi in range(N):
                frame_idx = frame_idxs[fi]
                planarity = hf["planarity"][fi].astype(np.float32)
                depth_moge = hf["depth_moge"][fi].astype(np.float32)
                normal = hf["normal"][fi].astype(np.float32)
                gt_seg = hf["gt_seg"][fi]
                gt_depth = hf["gt_depth"][fi]
                K = hf["K"][fi]
                c2w = hf["c2w"][fi]

                # Apply planarity threshold
                with timer("threshold"):
                    planarity_mask = (planarity > threshold_planarity).astype(np.int32)

                # Run v10 segmentation
                with timer("segmentation"):
                    labels, _ = compute_vectorized_planar_segments_v10(
                        planarity_mask,
                        normal,
                        depth_moge,
                        normal_threshold_rad,
                        depth_threshold,
                        neighbor_match_count_thresh=neighbor_match_count_thresh,
                        device=seg_device,
                        adaptive_frac=adaptive_frac,
                        min_valid_neighbors=min_valid_neighbors,
                        min_segment_pixels=min_segment_pixels,
                    )
                    labels, _ = remap_labels(labels)

                # Resize labels to GT resolution for evaluation
                H_gt, W_gt = gt_seg.shape
                H_pred, W_pred = labels.shape
                if (H_pred, W_pred) != (H_gt, W_gt):
                    with timer("resize_labels"):
                        labels = cv2.resize(labels, (W_gt, H_gt), interpolation=cv2.INTER_NEAREST)

                # Evaluate
                with timer("evaluate"):
                    metrics, _ = evaluate_single_frame(
                        scene_id,
                        frame_idx,
                        gt_depth,
                        gt_seg,
                        K,
                        c2w,
                        labels,
                        thresholds,
                        compute_plane_metrics_flag=True,
                        ransac_iterations=ransac_iterations,
                        inlier_ratio_gate=inlier_ratio_gate,
                    )

                results[(scene_id, frame_idx)] = metrics
                pbar.update(1)

            total_frames_count += N

    pbar.close()

    # Save results
    save_results_csv(results, config_dir_out)

    timer.print_summary(num_frames=len(results))
    print(f"[DONE] Config {config_index}: {len(results)} frames -> {config_dir_out}")
    return config_dir_out


def run_eval(cfg: dict, config_start: int, config_end: int):
    """Run v10 segmentation + evaluation for configs [config_start, config_end)."""
    output_dir = cfg["output_dir"]
    cache_dir = os.path.join(output_dir, "cached_scenes")

    if not os.path.isdir(cache_dir):
        print(f"[ERROR] Cache not found: {cache_dir}")
        print("Run --mode cache first.")
        sys.exit(1)

    configs = generate_configs(cfg)
    config_end = min(config_end, len(configs))

    if config_start < 0 or config_start >= len(configs):
        print(f"[ERROR] config-start {config_start} out of range [0, {len(configs)-1}]")
        sys.exit(1)

    n_configs = config_end - config_start
    print(f"[EVAL] Running configs [{config_start}, {config_end}) — {n_configs} configs")

    # Discover cached scene H5 files (shared across all configs)
    scene_h5s = sorted(Path(cache_dir).glob("*.h5"))
    print(f"[CACHE] Found {len(scene_h5s)} scene H5 files in {cache_dir}")

    seg_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SEG] v10 segmentation on device={seg_device}")

    for ci in range(config_start, config_end):
        run_eval_single(cfg, ci, configs, scene_h5s)


# ============================================================
# MODE: AGGREGATE
# ============================================================

def run_aggregate(cfg: dict):
    """Aggregate results from all configs into a summary CSV."""
    output_dir = cfg["output_dir"]
    configs_dir = os.path.join(output_dir, "configs")

    if not os.path.isdir(configs_dir):
        print(f"[ERROR] Configs directory not found: {configs_dir}")
        sys.exit(1)

    # Generate expected configs for reference
    configs = generate_configs(cfg)

    rows = []
    missing = []

    for i, expected_params in enumerate(configs):
        config_dir = os.path.join(configs_dir, f"config_{i:04d}")
        dataset_csv = os.path.join(config_dir, "results_dataset.csv")
        config_yaml = os.path.join(config_dir, "config.yaml")

        if not os.path.isfile(dataset_csv):
            missing.append(i)
            continue

        # Load config params
        if os.path.isfile(config_yaml):
            with open(config_yaml) as f:
                params = yaml.safe_load(f)
        else:
            params = expected_params

        # Load dataset-level metrics
        df = pd.read_csv(dataset_csv)
        if len(df) == 0:
            missing.append(i)
            continue

        row = {"config_index": i}
        row.update(params)

        # Extract all metric columns
        for col in df.columns:
            if col not in ("Unnamed: 0",):
                row[col] = df[col].iloc[0]

        rows.append(row)

    if missing:
        print(f"[WARN] Missing results for {len(missing)} configs: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    if not rows:
        print("[ERROR] No results found to aggregate.")
        sys.exit(1)

    summary_df = pd.DataFrame(rows)

    # Sort by config_index
    summary_df = summary_df.sort_values("config_index").reset_index(drop=True)

    # Save summary
    summary_path = os.path.join(output_dir, "grid_search_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[AGGREGATE] Saved {len(rows)}/{len(configs)} configs to {summary_path}")

    # Print best configs for key metrics
    key_metrics = [
        ("sc_mean", "Segmentation Covering", True),
        ("rand_index_mean", "Rand Index", True),
        ("voi_mean", "Variation of Information", False),
    ]
    # Add precision/recall at all thresholds
    thresholds = cfg.get("thresholds", [0.001, 0.005, 0.01])
    for thr in thresholds:
        thr_str = f"{thr*100:.1f}cm"
        key_metrics.append((f"prec@{thr_str}_mean", f"Precision@{thr_str}", True))
        key_metrics.append((f"rec@{thr_str}_mean", f"Recall@{thr_str}", True))

    # Add binary planarity metrics
    key_metrics.append(("bp_f1_mean", "Binary Planarity F1", True))
    key_metrics.append(("bp_iou_mean", "Binary Planarity IoU", True))

    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS PER METRIC")
    print("=" * 80)

    param_cols = [
        "threshold_planarity", "normal_threshold_deg", "depth_threshold",
        "neighbor_match_count_thresh", "adaptive_frac", "min_valid_neighbors",
        "min_segment_pixels",
    ]

    for metric_col, metric_name, higher_is_better in key_metrics:
        if metric_col not in summary_df.columns:
            continue

        if higher_is_better:
            best_idx = summary_df[metric_col].idxmax()
        else:
            best_idx = summary_df[metric_col].idxmin()

        best_row = summary_df.iloc[best_idx]
        params_str = ", ".join(f"{p}={best_row[p]}" for p in param_cols if p in best_row)
        print(f"\n{metric_name} ({'higher' if higher_is_better else 'lower'} is better):")
        print(f"  Best: {best_row[metric_col]:.4f}  (config {int(best_row['config_index'])})")
        print(f"  Params: {params_str}")

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Grid search over v10 segmentation parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--yaml", type=str, required=True,
                        help="Path to grid search config YAML")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["cache", "eval", "aggregate"],
                        help="Execution mode")
    parser.add_argument("--config-start", type=int, default=0,
                        help="First config index (inclusive) for eval mode")
    parser.add_argument("--config-end", type=int, default=None,
                        help="Last config index (exclusive) for eval mode. "
                             "Defaults to config-start+1 (single config).")

    args = parser.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    print(f"[MODE] {args.mode}")
    print(f"[YAML] {args.yaml}")
    print(f"[OUTPUT] {cfg['output_dir']}")

    # Print config summary
    configs = generate_configs(cfg)
    print_configs_summary(configs)

    if args.mode == "cache":
        run_cache(cfg)
    elif args.mode == "eval":
        config_end = args.config_end if args.config_end is not None else args.config_start + 1
        run_eval(cfg, args.config_start, config_end)
    elif args.mode == "aggregate":
        run_aggregate(cfg)


if __name__ == "__main__":
    main()
