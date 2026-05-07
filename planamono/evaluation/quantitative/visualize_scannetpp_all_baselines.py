"""
Visualization script for RANSAC inlier comparison across all baseline methods.

Generates:
1. PNG visualizations with 3-row layout (Inliers, Segmentation, Diff vs GT)
2. H5 file with structured results (inlier masks, stats, diff stats)

Usage:
    python visualize_scannetpp_all_baselines.py --n-samples 10
    python visualize_scannetpp_all_baselines.py --n-samples 5 --random-seed 42
    python visualize_scannetpp_all_baselines.py --specific-frames "scene1:frame1,scene2:frame2"
"""

import os
import argparse
import numpy as np
import cv2
import h5py
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label_v1,
    mark_planes_below_threshold_as_outliers,
)
from planamono.shared.utils import visualize_top_components_v1
from planamono.evaluation.quantitative.evaluate_all_baselines import (
    THRESHOLDS,
    METHODS as EVAL_METHODS,
    INLIER_RATIO_GATE,
    RANSAC_ITERATIONS,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Visualization parameters (imported from evaluate_all_baselines.py for consistency)
INLIER_RATIO_THRESHOLD = INLIER_RATIO_GATE  # Use same threshold as evaluation

# Output paths
VIS_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/visualizations/inliers")
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference")
GT_ROOT = Path(scannetpp_rend_plane_path)
DATASET_DIR = scannetpp_rend_plane_path

# Build METHODS from evaluate_all_baselines.py, adding visualization-specific keys
# "uses_rendered_h5" indicates whether to use rendered.h5 (GT) or planes.h5 (predictions)
METHODS = {
    # GT method (only used in visualization, not in evaluation)
    "GT": {
        "h5_folder": None,
        "display_name": "GT",
        "label_offset": 0,
        "uses_rendered_h5": True,
    },
}

# Add all methods from evaluate_all_baselines.py
for method_key, method_config in EVAL_METHODS.items():
    METHODS[method_key] = {
        "h5_folder": method_config["h5_folder"],
        "display_name": method_config["display_name"],
        "label_offset": method_config["label_offset"],
        "uses_rendered_h5": method_config.get("uses_gt_h5", False),
    }


# ============================================================
# H5 LOADING HELPERS
# ============================================================

def load_plane_from_h5(h5_path: str, frame_idx: str) -> Optional[np.ndarray]:
    """Load plane segmentation from H5 file."""
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
# INLIER COMPUTATION
# ============================================================

def compute_inlier_mask(
    plane_seg: np.ndarray,
    depth_np: np.ndarray,
    K_np: np.ndarray,
    c2w_np: np.ndarray,
    distance_threshold: float = 0.02,
    inlier_ratio_threshold: float = 0.5
) -> Tuple[np.ndarray, Dict]:
    """
    Compute a binary mask of RANSAC inliers.

    Returns:
        inlier_mask: (H, W) bool array, True for inlier pixels
        stats: dict with precision, recall, num_inliers, num_planes
    """
    H, W = depth_np.shape

    # Backproject to 3D
    pts_world, labels, valid_idx = backproject(depth_np, K_np, c2w_np, plane_seg)

    if pts_world.shape[0] == 0:
        return np.zeros((H, W), dtype=bool), {
            "precision": 0, "recall": 0, "num_inliers": 0, "num_planes": 0
        }

    # Fit planes per label
    results, df = fit_planes_per_label_v1(
        pts_world,
        labels,
        ignore_labels=(0,),
        distance_threshold=distance_threshold,
        num_iterations=RANSAC_ITERATIONS,
        min_support=100
    )

    if df is None or len(df) == 0:
        return np.zeros((H, W), dtype=bool), {
            "precision": 0, "recall": 0, "num_inliers": 0, "num_planes": 0
        }

    # Mark low-quality planes as outliers
    results, df = mark_planes_below_threshold_as_outliers(
        results, df, inlier_ratio_threshold
    )

    # Collect all inlier indices
    all_inlier_global_idx = []
    for pid, data in results.items():
        if "inliers_all" in data and len(data["inliers_all"]) > 0:
            all_inlier_global_idx.extend(data["inliers_all"].tolist())

    # Create inlier mask in image space
    inlier_mask = np.zeros(H * W, dtype=bool)
    if len(all_inlier_global_idx) > 0:
        inlier_image_idx = valid_idx[all_inlier_global_idx]
        inlier_mask[inlier_image_idx] = True

    inlier_mask = inlier_mask.reshape(H, W)

    # Compute stats
    total_predicted = df["num_points"].sum() if "num_points" in df.columns else 0
    total_inliers = df["refined_inlier_num_points"].sum() if "refined_inlier_num_points" in df.columns else 0
    precision = total_inliers / total_predicted if total_predicted > 0 else 0
    recall = total_inliers / pts_world.shape[0] if pts_world.shape[0] > 0 else 0

    stats = {
        "precision": float(precision),
        "recall": float(recall),
        "num_inliers": int(total_inliers),
        "num_planes": len([r for r in results.values() if r.get("refined_inlier_num_points", 0) > 0])
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
    Overlay inlier difference on RGB image with color coding:
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
    methods_to_visualize: List[str] = None,
) -> Dict:
    """
    Visualize inliers for a single frame across all methods.

    Args:
        distance_threshold: Distance threshold in meters for RANSAC inliers

    Returns:
        frame_results: dict with all computed stats and metadata
    """
    H, W = depth_np.shape

    if methods_to_visualize is None:
        methods_to_visualize = list(METHODS.keys())

    # Compute inliers for each method
    method_results = {}

    for method_key in methods_to_visualize:
        method_config = METHODS[method_key]
        display_name = method_config["display_name"]

        # Construct H5 path
        if method_config["uses_rendered_h5"]:
            h5_path = GT_ROOT / scene_id / "rendered.h5"
        else:
            h5_path = H5_ROOT / method_config["h5_folder"] / scene_id / "planes.h5"

        # Load plane segmentation
        plane_seg = load_plane_from_h5(str(h5_path), frame_idx)
        if plane_seg is None:
            print(f"  [SKIP] {display_name}: predictions not found")
            continue

        # Resize if needed
        if plane_seg.shape != (H, W):
            plane_seg = resize_plane_nn(plane_seg, (H, W))

        # Apply label offset
        plane_seg = plane_seg + method_config["label_offset"]

        # Compute inlier mask
        inlier_mask, stats = compute_inlier_mask(
            plane_seg, depth_np, K_np, c2w_np,
            distance_threshold=distance_threshold,
            inlier_ratio_threshold=INLIER_RATIO_THRESHOLD
        )

        method_results[method_key] = {
            "display_name": display_name,
            "inlier_mask": inlier_mask,
            "stats": stats,
            "plane_seg": plane_seg
        }

        print(f"  [{display_name}] P={stats['precision']:.3f} R={stats['recall']:.3f} "
              f"Inliers={stats['num_inliers']} Planes={stats['num_planes']}")

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

    gt_seg_vis = visualize_top_components_v1(gt_plane_np, k=10, return_colors=True, ignore_label=0)
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
        seg_vis = visualize_top_components_v1(data["plane_seg"], k=10, return_colors=True, ignore_label=0)
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

    legend_text = "GREEN=Common(TP)  RED=GT missed(FN)  BLUE=Method extra(FP)"
    plt.suptitle(f"RANSAC Inliers @ {distance_threshold*100:.1f}cm | {scene_id} / {frame_idx}\n{legend_text}", fontsize=12)
    plt.tight_layout()

    # Save PNG
    png_path = output_dir / f"{scene_id}_{frame_idx}.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {png_path}")

    # Prepare results for H5
    frame_results = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
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


def save_results_h5(all_results: List[Dict], output_path: Path, distance_threshold: float):
    """Save all visualization results to a structured H5 file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Metadata
        f.attrs["distance_threshold"] = distance_threshold
        f.attrs["inlier_ratio_threshold"] = INLIER_RATIO_THRESHOLD
        f.attrs["num_frames"] = len(all_results)

        # Store method names
        method_names = list(METHODS.keys())
        f.create_dataset("method_keys", data=np.array(method_names, dtype="S"))

        # Per-frame groups
        for i, frame_result in enumerate(all_results):
            if not frame_result:
                continue

            scene_id = frame_result["scene_id"]
            frame_idx = frame_result["frame_idx"]
            grp = f.create_group(f"frame_{i:04d}")
            grp.attrs["scene_id"] = scene_id
            grp.attrs["frame_idx"] = frame_idx

            # Per-method data
            for method_key, method_data in frame_result["methods"].items():
                m_grp = grp.create_group(method_key)
                m_grp.attrs["display_name"] = method_data["display_name"]

                # Stats
                for stat_key, stat_val in method_data["stats"].items():
                    m_grp.attrs[f"stat_{stat_key}"] = stat_val

                # Diff stats
                for diff_key, diff_val in method_data["diff_stats"].items():
                    m_grp.attrs[f"diff_{diff_key}"] = diff_val

                # Inlier mask (compressed)
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
                "method": method_data["display_name"],
                **method_data["stats"],
                **{f"diff_{k}": v for k, v in method_data["diff_stats"].items()}
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved summary CSV: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize RANSAC inliers across all baselines")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of random samples to visualize")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--specific-frames", type=str, default=None,
                        help="Comma-separated list of scene:frame pairs to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for visualizations (base dir, thresholds will be subdirs)")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum scenes to load (for testing)")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Methods to visualize. Options: {list(METHODS.keys())}")
    args = parser.parse_args()

    # Setup base output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        # EXP_VER = "v4"
        EXP_VER = "v2"
        base_output_dir = VIS_ROOT / f"baselines_n{args.n_samples}_seed{args.random_seed}_{EXP_VER}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] N samples: {args.n_samples}")
    print(f"[CONFIG] Random seed: {args.random_seed}")
    print(f"[CONFIG] Base output dir: {base_output_dir}")
    print(f"[CONFIG] Thresholds: {[f'{t*100:.1f}cm' for t in THRESHOLDS]}")

    # Determine methods to visualize
    if args.methods:
        methods_to_vis = args.methods
        invalid = set(methods_to_vis) - set(METHODS.keys())
        if invalid:
            print(f"[ERROR] Invalid methods: {invalid}")
            return
    else:
        methods_to_vis = list(METHODS.keys())
    print(f"[CONFIG] Methods: {methods_to_vis}")

    # Load dataset
    print("\n==> Loading dataset")
    val_dataset = ScanNetPPPlaneDataset(
        rgb_root=os.path.join(scannetpp_path, "data"),
        plane_label_root=scannetpp_rend_plane_path,
        sem_label_root=os.path.join(DATASET_DIR, ""),
        depth_label_root=scannetpp_rend_plane_path,
        split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
        split="test",
        max_scenes=args.max_scenes,
    )
    print(f"[DATA] Loaded {len(val_dataset)} frames from {len(val_dataset.scene_ids)} scenes")

    # Determine frames to visualize
    if args.specific_frames:
        # Parse specific frames
        frame_specs = []
        for spec in args.specific_frames.split(","):
            scene_id, frame_idx = spec.strip().split(":")
            frame_specs.append((scene_id, frame_idx))
        print(f"[DATA] Visualizing {len(frame_specs)} specific frames")
    else:
        # Random sampling
        np.random.seed(args.random_seed)
        n_total = len(val_dataset)
        indices = np.random.choice(n_total, size=min(args.n_samples, n_total), replace=False)
        frame_specs = []
        for idx in indices:
            sample = val_dataset[idx]
            frame_specs.append((sample["scene_id"], sample["frame_idx"]))
        print(f"[DATA] Randomly sampled {len(frame_specs)} frames")

    # Build frame index for quick lookup
    print("==> Building frame index")
    frame_index = {}
    for i, (rgb_path, plane_h5, sem_h5, depth_h5, frame_idx, K, c2w) in enumerate(val_dataset.valid_pairs):
        scene_id = rgb_path.split("/")[-4]
        fid = os.path.splitext(os.path.basename(rgb_path))[0]
        frame_index[(scene_id, fid)] = i

    # Pre-load all frame data once (to avoid reloading for each threshold)
    print(f"\n==> Pre-loading {len(frame_specs)} frames")
    frame_data_cache = {}
    for scene_id, frame_idx in tqdm(frame_specs, desc="Loading frames"):
        key = (scene_id, frame_idx)
        if key not in frame_index:
            print(f"  [SKIP] {scene_id}/{frame_idx} not found in dataset")
            continue

        dataset_idx = frame_index[key]
        sample = val_dataset[dataset_idx]

        frame_data_cache[key] = {
            "image_np": sample["image"].permute(1, 2, 0).numpy(),
            "depth_np": sample["depth"][0].numpy(),
            "gt_plane_np": sample["plane"][0].numpy().astype(np.int32),
            "K_np": sample["K"].numpy(),
            "c2w_np": sample["c2w"].numpy(),
        }
    print(f"[DATA] Loaded {len(frame_data_cache)} frames into cache")

    # Loop over each threshold
    for distance_threshold in THRESHOLDS:
        thresh_str = f"{distance_threshold*100:.1f}cm"
        output_dir = base_output_dir / thresh_str
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"THRESHOLD: {thresh_str}")
        print(f"Output dir: {output_dir}")
        print(f"{'='*60}")

        all_results = []

        for scene_id, frame_idx in tqdm(frame_specs, desc=f"Visualizing @ {thresh_str}"):
            key = (scene_id, frame_idx)
            if key not in frame_data_cache:
                continue

            data = frame_data_cache[key]
            print(f"\n[Frame] {scene_id} / {frame_idx}")

            # Visualize
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
                methods_to_visualize=methods_to_vis,
            )

            if frame_result:
                all_results.append(frame_result)

        # Save results for this threshold
        print(f"\n==> Saving results for {thresh_str}")
        if all_results:
            save_results_h5(all_results, output_dir / "results.h5", distance_threshold)
            save_summary_csv(all_results, output_dir / "summary.csv")

        print(f"[DONE] Threshold {thresh_str}: Visualized {len(all_results)} frames")

    print(f"\n{'='*60}")
    print(f"[DONE] All thresholds complete!")
    print(f"[DONE] Results saved to: {base_output_dir}")
    print(f"[DONE] Subdirectories: {[f'{t*100:.1f}cm' for t in THRESHOLDS]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
