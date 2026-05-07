#!/usr/bin/env python3
"""
Unified visualization + PDF/PPTX pipeline for ScanNet++ and Hypersim.

Generates:
1. PNG visualizations with 3-row layout (Inliers, Segmentation, Diff vs GT)
2. H5 file with structured results (inlier masks, stats, diff stats)
3. Summary CSV
4. Optional PDF/PPTX from generated PNGs

Usage:
    python visualize_and_generate_pdf.py --dataset scannetpp --methods ours zeroplane --n-samples 20
    python visualize_and_generate_pdf.py --dataset hypersim --methods moge_ours moge_mixed_bce --n-samples 10
    python visualize_and_generate_pdf.py --dataset scannetpp --specific-frames "scene1:frame1,scene2:frame2"
    python visualize_and_generate_pdf.py --dataset hypersim --n-samples 5 --format pdf --quality 60
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch

from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path
from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
    compute_inliers_at_threshold_with_indices,
)
from planamono.shared.utils import visualize_top_components_v2


# ============================================================
# DATASET CONFIGURATION
# ============================================================

def get_dataset_config(dataset_name: str) -> Dict:
    """
    Return dataset-specific configuration.

    Returns dict with:
        methods, thresholds, paths, dataset_factory, vis_root,
        h5_root, gt_root, ransac_iterations, inlier_ratio_gate, exp_ver
    """
    if dataset_name == "scannetpp":
        from planamono.evaluation.quantitative.evaluate_all_baselines import (
            THRESHOLDS as SCANNETPP_THRESHOLDS,
            METHODS as SCANNETPP_EVAL_METHODS,
            INLIER_RATIO_GATE as SCANNETPP_INLIER_RATIO_GATE,
            RANSAC_ITERATIONS as SCANNETPP_RANSAC_ITERATIONS,
            EXP_VER as SCANNETPP_EXP_VER,
        )

        methods = {
            "GT": {
                "h5_folder": None,
                "display_name": "GT",
                "label_offset": 0,
                "uses_rendered_h5": True,
                "nonplanar_label": None,
            },
        }
        for key, cfg in SCANNETPP_EVAL_METHODS.items():
            methods[key] = {
                "h5_folder": cfg["h5_folder"],
                "display_name": cfg["display_name"],
                "label_offset": cfg["label_offset"],
                "uses_rendered_h5": cfg.get("uses_gt_h5", False),
                "nonplanar_label": cfg.get("nonplanar_label"),
            }

        return {
            "name": "scannetpp",
            "methods": methods,
            "thresholds": SCANNETPP_THRESHOLDS,
            "vis_root": Path("/cluster/scratch/aoezkan/planeseg/scannetpp/visualizations/inliers"),
            "h5_root": Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference"),
            "gt_root": Path(scannetpp_rend_plane_path),
            "dataset_dir": scannetpp_rend_plane_path,
            "ransac_iterations": SCANNETPP_RANSAC_ITERATIONS,
            "inlier_ratio_gate": SCANNETPP_INLIER_RATIO_GATE,
            "exp_ver": SCANNETPP_EXP_VER,
        }

    elif dataset_name == "hypersim":
        from planamono.evaluation.quantitative.evaluate_hypersim_all_baselines import (
            THRESHOLDS as HYPERSIM_THRESHOLDS,
            METHODS as HYPERSIM_EVAL_METHODS,
            INLIER_RATIO_GATE as HYPERSIM_INLIER_RATIO_GATE,
            RANSAC_ITERATIONS as HYPERSIM_RANSAC_ITERATIONS,
            EXP_VER as HYPERSIM_EXP_VER,
            HYPERSIM_ROOT,
            PLANE_LABEL_ROOT,
            PARAMS_ROOT,
        )

        methods = {
            "GT": {
                "h5_folder": None,
                "display_name": "GT",
                "label_offset": 0,
                "uses_rendered_h5": True,
                "nonplanar_label": None,
            },
        }
        for key, cfg in HYPERSIM_EVAL_METHODS.items():
            methods[key] = {
                "h5_folder": cfg["h5_folder"],
                "display_name": cfg["display_name"],
                "label_offset": cfg["label_offset"],
                "uses_rendered_h5": cfg.get("uses_gt_h5", False),
                "nonplanar_label": cfg.get("nonplanar_label"),
            }

        return {
            "name": "hypersim",
            "methods": methods,
            "thresholds": HYPERSIM_THRESHOLDS,
            "vis_root": Path("/cluster/scratch/aoezkan/planeseg/hypersim/visualizations/inliers"),
            "h5_root": Path("/cluster/scratch/aoezkan/planeseg/hypersim/inference"),
            "gt_root": None,  # Hypersim GT comes from dataset directly
            "dataset_dir": None,
            "hypersim_root": HYPERSIM_ROOT,
            "plane_label_root": PLANE_LABEL_ROOT,
            "params_root": PARAMS_ROOT,
            "ransac_iterations": HYPERSIM_RANSAC_ITERATIONS,
            "inlier_ratio_gate": HYPERSIM_INLIER_RATIO_GATE,
            "exp_ver": HYPERSIM_EXP_VER,
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'scannetpp' or 'hypersim'.")


def load_dataset(config: Dict, split: str = "test", max_scenes: int = None):
    """Load the appropriate dataset based on config."""
    if config["name"] == "scannetpp":
        from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
        return ScanNetPPPlaneDataset(
            rgb_root=os.path.join(scannetpp_path, "data"),
            plane_label_root=scannetpp_rend_plane_path,
            sem_label_root=os.path.join(config["dataset_dir"], ""),
            depth_label_root=scannetpp_rend_plane_path,
            split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
            split=split,
            max_scenes=max_scenes,
        )
    elif config["name"] == "hypersim":
        from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
        return HypersimPlaneDataset(
            hypersim_root=config["hypersim_root"],
            plane_label_root=config["plane_label_root"],
            params_root=config["params_root"],
            split_txt_dir=os.path.join(repo_path, "splits", "hypersim"),
            split=split,
            image_height=512,
            image_width=768,
            max_scenes=max_scenes,
        )


# ============================================================
# H5 LOADING HELPERS
# ============================================================

def load_plane_from_h5(h5_path: str, frame_idx: str) -> Optional[np.ndarray]:
    """Load plane segmentation from H5 file (ScanNet++ format: planes.h5 or rendered.h5)."""
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        planes = f["planes"][:]
        frame_ids = [fid.decode() if isinstance(fid, bytes) else fid
                     for fid in f["frame_ids"][:]]
    if frame_idx not in frame_ids:
        return None
    idx = frame_ids.index(frame_idx)
    return planes[idx]


def load_plane_from_h5_hypersim(h5_path: str, frame_idx: str) -> Optional[np.ndarray]:
    """Load plane segmentation from per-camera H5 file (Hypersim format: planes_cam_XX.h5)."""
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        planes = f["planes"][:]
        frame_ids = [fid.decode() if isinstance(fid, bytes) else fid
                     for fid in f["frame_ids"][:]]
    if frame_idx not in frame_ids:
        return None
    idx = frame_ids.index(frame_idx)
    return planes[idx]


def resize_plane_nn(plane: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize plane segmentation with INTER_NEAREST."""
    Ht, Wt = target_hw
    return cv2.resize(plane.astype(np.float32), (Wt, Ht),
                      interpolation=cv2.INTER_NEAREST).astype(np.int32)


# ============================================================
# PREDICTION LOADING (dataset-specific)
# ============================================================

def load_prediction(
    method_key: str,
    method_config: Dict,
    scene_id: str,
    frame_idx: str,
    dataset_config: Dict,
    cam_name: Optional[str] = None,
    gt_plane_np: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Load plane segmentation for a method.

    For GT on Hypersim, returns gt_plane_np directly.
    For predictions, loads from H5 files using dataset-appropriate paths.
    """
    dataset_name = dataset_config["name"]

    if method_config["uses_rendered_h5"]:
        # GT method
        if dataset_name == "scannetpp":
            h5_path = dataset_config["gt_root"] / scene_id / "rendered.h5"
            plane_seg = load_plane_from_h5(str(h5_path), frame_idx)
        elif dataset_name == "hypersim":
            # Hypersim GT comes from dataset __getitem__ directly
            plane_seg = gt_plane_np
        else:
            return None
    else:
        # Prediction method
        h5_folder = method_config["h5_folder"]
        h5_root = dataset_config["h5_root"]

        if dataset_name == "scannetpp":
            h5_path = h5_root / h5_folder / scene_id / "planes.h5"
            plane_seg = load_plane_from_h5(str(h5_path), frame_idx)
        elif dataset_name == "hypersim":
            if cam_name is None:
                return None
            h5_path = h5_root / h5_folder / scene_id / f"planes_{cam_name}.h5"
            plane_seg = load_plane_from_h5_hypersim(str(h5_path), frame_idx)
        else:
            return None

    if plane_seg is None:
        return None

    # Handle non-planar label remapping (e.g., ZeroPlane's 20 -> 0)
    nonplanar_label = method_config.get("nonplanar_label")
    if nonplanar_label is not None:
        plane_seg = np.where(plane_seg == nonplanar_label, 0, plane_seg)

    # Apply label offset
    label_offset = method_config.get("label_offset", 0)
    if label_offset != 0:
        plane_seg = np.where(plane_seg > 0, plane_seg + label_offset, 0)

    return plane_seg.astype(np.int32)


# ============================================================
# INLIER COMPUTATION
# ============================================================

def compute_inlier_mask_and_stats(
    plane_seg: np.ndarray,
    depth_np: np.ndarray,
    K_np: np.ndarray,
    c2w_np: np.ndarray,
    distance_threshold: float = 0.02,
    inlier_ratio_threshold: float = 0.5,
    ransac_iterations: int = 200
) -> Tuple[np.ndarray, Dict]:
    """
    Compute inlier mask and stats using shared evaluation functions.

    Uses the SAME logic as evaluate_all_baselines.py:
    - fit_planes_per_label_v1 for RANSAC plane fitting
    - compute_inliers_at_threshold_with_indices for metric computation
    """
    H, W = depth_np.shape

    pts_world, labels, valid_idx = backproject(depth_np, K_np, c2w_np, plane_seg)

    if pts_world.shape[0] == 0:
        return np.zeros((H, W), dtype=bool), {
            "precision": 0.0, "recall": 0.0, "num_inliers": 0,
            "num_planes": 0, "total_predicted_points": 0
        }

    results, df = fit_planes_per_label_v1(
        pts_world, labels, ignore_labels=(0,),
        distance_threshold=distance_threshold,
        num_iterations=ransac_iterations,
        min_support=100
    )

    if df is None or len(df) == 0:
        return np.zeros((H, W), dtype=bool), {
            "precision": 0.0, "recall": 0.0, "num_inliers": 0,
            "num_planes": 0, "total_predicted_points": 0
        }

    plane_params = {}
    for pid, data in results.items():
        if "plane_model_refined" in data:
            plane_params[pid] = data["plane_model_refined"]

    if not plane_params:
        return np.zeros((H, W), dtype=bool), {
            "precision": 0.0, "recall": 0.0, "num_inliers": 0,
            "num_planes": 0, "total_predicted_points": 0
        }

    metrics = compute_inliers_at_threshold_with_indices(
        pts_world, labels, plane_params, distance_threshold, inlier_ratio_threshold
    )

    inlier_mask = np.zeros(H * W, dtype=bool)
    if len(metrics["inlier_indices"]) > 0:
        inlier_image_idx = valid_idx[metrics["inlier_indices"]]
        inlier_mask[inlier_image_idx] = True

    inlier_mask = inlier_mask.reshape(H, W)

    stats = {
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "num_inliers": int(metrics["num_inliers"]),
        "num_planes": int(metrics["num_valid_planes"]),
        "total_predicted_points": int(metrics["total_predicted_points"])
    }

    return inlier_mask, stats


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def overlay_inliers_green(rgb: np.ndarray, inlier_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay inlier mask on RGB image with green color."""
    rgb_f = rgb.astype(np.float32) / 255.0 if rgb.max() > 1 else rgb.astype(np.float32)
    out = rgb_f.copy()
    green = np.array([0.0, 1.0, 0.0])
    out[inlier_mask] = (1 - alpha) * rgb_f[inlier_mask] + alpha * green
    return np.clip(out, 0, 1)


def overlay_inlier_difference(
    rgb: np.ndarray,
    gt_inliers: np.ndarray,
    method_inliers: np.ndarray,
    alpha: float = 0.7
) -> Tuple[np.ndarray, Dict]:
    """
    Overlay inlier difference on RGB image:
    - GREEN: Common inliers (TP)
    - RED: GT inliers missed by method (FN)
    - BLUE: Method inliers not in GT (FP)
    """
    rgb_f = rgb.astype(np.float32) / 255.0 if rgb.max() > 1 else rgb.astype(np.float32)
    out = rgb_f.copy()

    tp_mask = gt_inliers & method_inliers
    fn_mask = gt_inliers & ~method_inliers
    fp_mask = ~gt_inliers & method_inliers

    green = np.array([0.0, 1.0, 0.0])
    red = np.array([1.0, 0.0, 0.0])
    blue = np.array([0.0, 0.5, 1.0])

    out[tp_mask] = (1 - alpha) * rgb_f[tp_mask] + alpha * green
    out[fn_mask] = (1 - alpha) * rgb_f[fn_mask] + alpha * red
    out[fp_mask] = (1 - alpha) * rgb_f[fp_mask] + alpha * blue

    stats = {
        "TP": int(tp_mask.sum()),
        "FN": int(fn_mask.sum()),
        "FP": int(fp_mask.sum()),
        "IoU": float(tp_mask.sum() / (tp_mask.sum() + fn_mask.sum() + fp_mask.sum() + 1e-8))
    }

    return np.clip(out, 0, 1), stats


# ============================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================

def visualize_frame(
    scene_id: str,
    frame_idx: str,
    image_np: np.ndarray,
    depth_np: np.ndarray,
    gt_plane_np: np.ndarray,
    K_np: np.ndarray,
    c2w_np: np.ndarray,
    output_dir: Path,
    distance_threshold: float,
    dataset_config: Dict,
    methods_to_visualize: List[str],
    cam_name: Optional[str] = None,
) -> Dict:
    """
    Visualize inliers for a single frame across all methods.

    Returns:
        frame_results: dict with all computed stats and metadata
    """
    H, W = depth_np.shape
    all_methods = dataset_config["methods"]
    inlier_ratio_threshold = dataset_config["inlier_ratio_gate"]
    ransac_iterations = dataset_config["ransac_iterations"]

    method_results = {}

    for method_key in methods_to_visualize:
        if method_key not in all_methods:
            print(f"  [SKIP] Unknown method: {method_key}")
            continue

        method_config = all_methods[method_key]
        display_name = method_config["display_name"]

        # Load prediction
        plane_seg = load_prediction(
            method_key, method_config,
            scene_id, frame_idx, dataset_config,
            cam_name=cam_name,
            gt_plane_np=gt_plane_np,
        )

        if plane_seg is None:
            print(f"  [SKIP] {display_name}: predictions not found")
            continue

        # Resize if needed
        if plane_seg.shape != (H, W):
            plane_seg = resize_plane_nn(plane_seg, (H, W))

        # Compute inlier mask
        inlier_mask, stats = compute_inlier_mask_and_stats(
            plane_seg, depth_np, K_np, c2w_np,
            distance_threshold=distance_threshold,
            inlier_ratio_threshold=inlier_ratio_threshold,
            ransac_iterations=ransac_iterations
        )

        method_results[method_key] = {
            "display_name": display_name,
            "inlier_mask": inlier_mask,
            "stats": stats,
            "plane_seg": plane_seg
        }

        print(f"  [{display_name}] P={stats['precision']:.3f} R={stats['recall']:.3f} "
              f"Inliers={stats['num_inliers']} Planes={stats['num_planes']} "
              f"TotalPts={stats['total_predicted_points']}")

    if len(method_results) == 0:
        print(f"  [ERROR] No methods loaded successfully for {scene_id}/{frame_idx}")
        return {}

    if "GT" not in method_results:
        print(f"  [ERROR] GT not loaded, cannot compute differences")
        return {}

    gt_inliers = method_results["GT"]["inlier_mask"]

    # Compute diff stats for each method
    diff_stats_all = {}
    for method_key, data in method_results.items():
        if method_key == "GT":
            diff_stats_all[method_key] = {
                "TP": data["stats"]["num_inliers"], "FN": 0, "FP": 0, "IoU": 1.0
            }
        else:
            _, diff_stats = overlay_inlier_difference(image_np, gt_inliers, data["inlier_mask"])
            diff_stats_all[method_key] = diff_stats

    # ============================================================
    # CREATE VISUALIZATION (3 ROWS)
    # ============================================================

    n_methods = len(method_results)
    fig, axes = plt.subplots(3, n_methods + 2, figsize=(4 * (n_methods + 2), 12))

    row_titles = ["Inliers (GREEN)", "Segmentation", "Diff vs GT"]

    # Column 0: RGB
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")
    axes[0, 0].set_ylabel(row_titles[0], fontsize=12, rotation=90, labelpad=10)

    gt_seg_vis = visualize_top_components_v2(gt_plane_np, k=20, return_colors=True, ignore_label=0)
    axes[1, 0].imshow(gt_seg_vis)
    axes[1, 0].set_title("Plane GT")
    axes[1, 0].axis("off")
    axes[1, 0].set_ylabel(row_titles[1], fontsize=12, rotation=90, labelpad=10)

    axes[2, 0].imshow(image_np)
    axes[2, 0].set_title("")
    axes[2, 0].axis("off")
    axes[2, 0].set_ylabel(row_titles[2], fontsize=12, rotation=90, labelpad=10)

    # Column 1: Depth
    axes[0, 1].imshow(depth_np, cmap='viridis')
    axes[0, 1].set_title("Depth (GT)")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(depth_np, cmap='viridis')
    axes[1, 1].set_title("")
    axes[1, 1].axis("off")

    axes[2, 1].imshow(depth_np, cmap='viridis')
    axes[2, 1].set_title("")
    axes[2, 1].axis("off")

    # Each method (columns 2+)
    for idx, (method_key, data) in enumerate(method_results.items()):
        col = idx + 2
        display_name = data["display_name"]

        # Row 0: Inlier overlay
        inlier_vis = overlay_inliers_green(image_np, data["inlier_mask"], alpha=0.6)
        axes[0, col].imshow(inlier_vis)
        axes[0, col].set_title(f"{display_name}\nP={data['stats']['precision']:.2f} R={data['stats']['recall']:.2f}")
        axes[0, col].axis("off")

        # Row 1: Segmentation
        seg_vis = visualize_top_components_v2(data["plane_seg"], k=20, return_colors=True, ignore_label=0)
        axes[1, col].imshow(seg_vis)
        axes[1, col].set_title(f"{data['stats']['num_planes']} planes")
        axes[1, col].axis("off")

        # Row 2: Difference vs GT
        if method_key == "GT":
            axes[2, col].imshow(inlier_vis)
            axes[2, col].set_title("(Reference)")
        else:
            diff_vis, _ = overlay_inlier_difference(image_np, gt_inliers, data["inlier_mask"], alpha=0.7)
            d = diff_stats_all[method_key]
            axes[2, col].imshow(diff_vis)
            axes[2, col].set_title(f"TP={d['TP']} FN={d['FN']} FP={d['FP']}\nIoU={d['IoU']:.3f}")
        axes[2, col].axis("off")

    dataset_label = dataset_config["name"].upper()
    legend_text = "GREEN=Common(TP)  RED=GT missed(FN)  BLUE=Method extra(FP)"
    frame_label = f"{scene_id} / {frame_idx}"
    if cam_name:
        frame_label = f"{scene_id} / {cam_name} / {frame_idx}"
    plt.suptitle(
        f"[{dataset_label}] RANSAC Inliers @ {distance_threshold*100:.1f}cm | {frame_label}\n{legend_text}",
        fontsize=12
    )
    plt.tight_layout()

    # Save PNG
    png_name = f"{scene_id}_{frame_idx}.png"
    if cam_name:
        png_name = f"{scene_id}_{cam_name}_{frame_idx}.png"
    png_path = output_dir / png_name
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {png_path}")

    # Prepare results for H5
    frame_results = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        "cam_name": cam_name or "",
        "methods": {}
    }

    for method_key, data in method_results.items():
        frame_results["methods"][method_key] = {
            "display_name": data["display_name"],
            "stats": data["stats"],
            "diff_stats": diff_stats_all[method_key],
            "inlier_mask": data["inlier_mask"],
        }

    return frame_results


# ============================================================
# RESULTS SAVING
# ============================================================

def save_results_h5(all_results: List[Dict], output_path: Path,
                    distance_threshold: float, inlier_ratio_threshold: float):
    """Save all visualization results to a structured H5 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.attrs["distance_threshold"] = distance_threshold
        f.attrs["inlier_ratio_threshold"] = inlier_ratio_threshold
        f.attrs["num_frames"] = len(all_results)
        f.attrs["version"] = "unified"

        for i, frame_result in enumerate(all_results):
            if not frame_result:
                continue

            grp = f.create_group(f"frame_{i:04d}")
            grp.attrs["scene_id"] = frame_result["scene_id"]
            grp.attrs["frame_idx"] = frame_result["frame_idx"]
            grp.attrs["cam_name"] = frame_result.get("cam_name", "")

            for method_key, method_data in frame_result["methods"].items():
                m_grp = grp.create_group(method_key)
                m_grp.attrs["display_name"] = method_data["display_name"]

                for stat_key, stat_val in method_data["stats"].items():
                    m_grp.attrs[f"stat_{stat_key}"] = stat_val

                for diff_key, diff_val in method_data["diff_stats"].items():
                    m_grp.attrs[f"diff_{diff_key}"] = diff_val

                m_grp.create_dataset(
                    "inlier_mask",
                    data=method_data["inlier_mask"].astype(np.uint8),
                    compression="gzip",
                    compression_opts=4
                )

    print(f"Saved H5 results: {output_path}")


def save_summary_csv(all_results: List[Dict], output_path: Path):
    """Save summary statistics to CSV."""
    rows = []
    for frame_result in all_results:
        if not frame_result:
            continue
        for method_key, method_data in frame_result["methods"].items():
            row = {
                "scene_id": frame_result["scene_id"],
                "frame_idx": frame_result["frame_idx"],
                "cam_name": frame_result.get("cam_name", ""),
                "method": method_data["display_name"],
                **method_data["stats"],
                **{f"diff_{k}": v for k, v in method_data["diff_stats"].items()}
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved summary CSV: {output_path}")


# ============================================================
# FRAME INDEX BUILDING (dataset-specific)
# ============================================================

def build_frame_specs_scannetpp(dataset, n_samples: int, random_seed: int,
                                specific_frames: str = None) -> List[Tuple]:
    """
    Build list of (scene_id, frame_idx) for ScanNet++.

    Returns list of (scene_id, frame_idx, cam_name) tuples.
    cam_name is always None for ScanNet++.
    """
    if specific_frames:
        specs = []
        for spec in specific_frames.split(","):
            scene_id, frame_idx = spec.strip().split(":")
            specs.append((scene_id, frame_idx, None))
        return specs

    np.random.seed(random_seed)
    n_total = len(dataset)
    indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    specs = []
    for idx in indices:
        sample = dataset[idx]
        specs.append((sample["scene_id"], sample["frame_idx"], None))
    return specs


def build_frame_specs_hypersim(dataset, n_samples: int, random_seed: int,
                               specific_frames: str = None) -> List[Tuple]:
    """
    Build list of (scene_id, frame_idx, cam_name) for Hypersim.
    """
    if specific_frames:
        specs = []
        for spec in specific_frames.split(","):
            parts = spec.strip().split(":")
            if len(parts) == 3:
                scene_id, cam_name, frame_idx = parts
            elif len(parts) == 2:
                scene_id, frame_idx = parts
                cam_name = None
            else:
                raise ValueError(f"Invalid frame spec: {spec}")
            specs.append((scene_id, frame_idx, cam_name))
        return specs

    np.random.seed(random_seed)
    n_total = len(dataset)
    indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    specs = []
    for idx in indices:
        sample = dataset[idx]
        scene_id = sample["scene_id"]
        frame_idx = sample["frame_idx"]
        # Extract cam_name from rgb_path (format: "scene_id/cam_name/frame_id")
        rgb_path = sample["rgb_path"]
        cam_name = rgb_path.split('/')[1] if '/' in rgb_path else "cam_00"
        specs.append((scene_id, frame_idx, cam_name))
    return specs


def build_frame_index_scannetpp(dataset) -> Dict[Tuple, int]:
    """Build (scene_id, frame_idx) -> dataset index for ScanNet++."""
    frame_index = {}
    for i, (rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K, c2w) in enumerate(dataset.valid_pairs):
        scene_id = rgb_path.split("/")[-4]
        fid = os.path.splitext(os.path.basename(rgb_path))[0]
        frame_index[(scene_id, fid, None)] = i
    return frame_index


def build_frame_index_hypersim(dataset) -> Dict[Tuple, int]:
    """Build (scene_id, frame_idx, cam_name) -> dataset index for Hypersim."""
    frame_index = {}
    for i, (scene_id, cam_name, frame_idx_int, fid, *rest) in enumerate(dataset.valid_pairs):
        frame_index[(scene_id, fid, cam_name)] = i
    return frame_index


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified visualization + PDF pipeline for ScanNet++ and Hypersim"
    )
    parser.add_argument("--dataset", type=str, required=True, choices=["scannetpp", "hypersim"],
                        help="Dataset to visualize")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of random samples to visualize")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--specific-frames", type=str, default=None,
                        help="Comma-separated scene:frame pairs (or scene:cam:frame for Hypersim)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (base dir, thresholds become subdirs)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum scenes to load (for testing)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to visualize (GT always included)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--format", type=str, choices=["pdf", "pptx", "both", "none"],
                        default="both", help="Output format for document generation")
    parser.add_argument("--quality", type=int, default=70,
                        help="JPEG quality for PDF/PPTX compression (1-100)")
    parser.add_argument("--max-width", type=int, default=1920,
                        help="Max image width for PDF/PPTX (0 = no resize)")
    args = parser.parse_args()

    # Load dataset config
    config = get_dataset_config(args.dataset)

    # Setup output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        exp_ver = config["exp_ver"]
        base_output_dir = (
            config["vis_root"]
            / f"baselines_n{args.n_samples}_seed{args.random_seed}_{exp_ver}"
        )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine methods to visualize (always include GT)
    all_method_keys = list(config["methods"].keys())
    if args.methods:
        methods_to_vis = list(args.methods)
        if "GT" not in methods_to_vis:
            methods_to_vis.insert(0, "GT")
        invalid = set(methods_to_vis) - set(all_method_keys)
        if invalid:
            print(f"[ERROR] Invalid methods: {invalid}")
            print(f"[INFO] Available methods: {all_method_keys}")
            return
    else:
        methods_to_vis = all_method_keys

    print(f"[CONFIG] Dataset: {args.dataset}")
    print(f"[CONFIG] N samples: {args.n_samples}")
    print(f"[CONFIG] Random seed: {args.random_seed}")
    print(f"[CONFIG] Base output dir: {base_output_dir}")
    print(f"[CONFIG] Thresholds: {[f'{t*100:.1f}cm' for t in config['thresholds']]}")
    print(f"[CONFIG] Inlier ratio gate: {config['inlier_ratio_gate']}")
    print(f"[CONFIG] RANSAC iterations: {config['ransac_iterations']}")
    print(f"[CONFIG] Methods: {methods_to_vis}")
    print(f"[CONFIG] Format: {args.format}")

    # Load dataset
    print("\n==> Loading dataset")
    dataset = load_dataset(config, split=args.split, max_scenes=args.max_scenes)
    print(f"[DATA] Loaded {len(dataset)} frames from {len(dataset.scene_ids)} scenes")

    # Build frame specs
    print("==> Building frame specs")
    if args.dataset == "scannetpp":
        frame_specs = build_frame_specs_scannetpp(
            dataset, args.n_samples, args.random_seed, args.specific_frames
        )
        frame_index = build_frame_index_scannetpp(dataset)
    else:
        frame_specs = build_frame_specs_hypersim(
            dataset, args.n_samples, args.random_seed, args.specific_frames
        )
        frame_index = build_frame_index_hypersim(dataset)
    print(f"[DATA] Selected {len(frame_specs)} frames")

    # Pre-load all frame data once (to avoid reloading for each threshold)
    print(f"\n==> Pre-loading {len(frame_specs)} frames")
    frame_data_cache = {}
    for scene_id, frame_idx, cam_name in tqdm(frame_specs, desc="Loading frames"):
        key = (scene_id, frame_idx, cam_name)
        if key not in frame_index:
            print(f"  [SKIP] {scene_id}/{frame_idx} (cam={cam_name}) not found in dataset")
            continue

        dataset_idx = frame_index[key]
        sample = dataset[dataset_idx]

        image_np = sample["image"].permute(1, 2, 0).numpy()
        depth_np = sample["depth"][0].numpy()
        gt_plane_np = sample["plane"][0].numpy().astype(np.int32)
        K_np = sample["K"].numpy()
        c2w_np = sample["c2w"].numpy()

        frame_data_cache[key] = {
            "image_np": image_np,
            "depth_np": depth_np,
            "gt_plane_np": gt_plane_np,
            "K_np": K_np,
            "c2w_np": c2w_np,
        }
    print(f"[DATA] Loaded {len(frame_data_cache)} frames into cache")

    # Loop over each threshold
    for distance_threshold in config["thresholds"]:
        thresh_str = f"{distance_threshold*100:.1f}cm"
        output_dir = base_output_dir / thresh_str
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"THRESHOLD: {thresh_str}")
        print(f"Output dir: {output_dir}")
        print(f"{'='*60}")

        all_results = []

        for scene_id, frame_idx, cam_name in tqdm(frame_specs, desc=f"Visualizing @ {thresh_str}"):
            key = (scene_id, frame_idx, cam_name)
            if key not in frame_data_cache:
                continue

            data = frame_data_cache[key]
            print(f"\n[Frame] {scene_id} / {frame_idx}" +
                  (f" / {cam_name}" if cam_name else ""))

            frame_result = visualize_frame(
                scene_id=scene_id,
                frame_idx=frame_idx,
                image_np=data["image_np"],
                depth_np=data["depth_np"],
                gt_plane_np=data["gt_plane_np"],
                K_np=data["K_np"],
                c2w_np=data["c2w_np"],
                output_dir=output_dir,
                distance_threshold=distance_threshold,
                dataset_config=config,
                methods_to_visualize=methods_to_vis,
                cam_name=cam_name,
            )

            if frame_result:
                all_results.append(frame_result)

        # Save results for this threshold
        print(f"\n==> Saving results for {thresh_str}")
        if all_results:
            save_results_h5(
                all_results, output_dir / "results.h5",
                distance_threshold, config["inlier_ratio_gate"]
            )
            save_summary_csv(all_results, output_dir / "summary.csv")

        print(f"[DONE] Threshold {thresh_str}: Visualized {len(all_results)} frames")

    # Generate PDF/PPTX
    if args.format != "none":
        print(f"\n{'='*60}")
        print("Generating PDF/PPTX")
        print(f"{'='*60}")

        from planamono.evaluation.quantitative.generate_inliers_pdf import (
            generate_all,
            PPTX_WIDESCREEN_16_9,
        )

        formats = ['pdf', 'pptx'] if args.format == "both" else [args.format]
        max_width = args.max_width if args.max_width > 0 else None

        generate_all(
            str(base_output_dir),
            page_size=PPTX_WIDESCREEN_16_9,
            aspect="16:9",
            max_width=max_width,
            quality=args.quality,
            formats=formats,
        )

    print(f"\n{'='*60}")
    print(f"[DONE] All thresholds complete!")
    print(f"[DONE] Results saved to: {base_output_dir}")
    print(f"[DONE] Subdirectories: {[f'{t*100:.1f}cm' for t in config['thresholds']]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
