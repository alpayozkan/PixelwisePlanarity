#!/usr/bin/env python3
"""
Unified evaluation script for all baseline methods on Hypersim dataset.

Evaluates all methods from H5 prediction folders and generates:
1. Per-method CSV results (results.csv, results_per_scene.csv, results_dataset.csv)
2. Aggregated baseline comparison tables

Usage:
    python evaluate_hypersim_all_baselines.py                    # Evaluate all methods
    python evaluate_hypersim_all_baselines.py --methods ours_mixed zeroplane  # Evaluate specific methods
    python evaluate_hypersim_all_baselines.py --aggregate-only   # Only aggregate existing results
"""

import os
import argparse
from torch.utils.data import DataLoader
import numpy as np
import cv2
import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple

from joblib import Parallel, delayed

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.paths import repo_path

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame,
)


# ============================================================
# CONFIGURATION
# ============================================================

# Evaluation parameters
COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
THRESHOLDS = (0.01, 0.02, 0.05)
BATCH_SIZE = 32
N_JOBS = min(16, os.cpu_count())

# Paths
EVAL_ROOT = Path("/cluster/scratch/aoezkan/planeseg/hypersim/eval")
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/hypersim/inference")
HYPERSIM_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_merged"
PLANE_LABEL_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_rendered"
PARAMS_ROOT = "/cluster/scratch/ayavuz/dataset/Hypersim_params"

# Experiment version (used in exp_name for all methods)
EXP_VER = "v1"

# Method definitions: {method_key: {h5_folder, display_name, label_offset, nonplanar_label, uses_gt_h5}}
METHODS = {
    "gt": {
        "h5_folder": None,  # Use GT labels directly
        "exp_name": f"gt_{EXP_VER}",
        "display_name": "GT (upper bound)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": True,
    },
    "ours": {
        "h5_folder": "moge_ours_h5",
        "exp_name": f"moge_ours_{EXP_VER}",
        "display_name": "Ours (Hypersim-trained)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "ours_mixed": {
        "h5_folder": "moge_mixed_bce_h5",
        "exp_name": f"moge_mixed_bce_{EXP_VER}",
        "display_name": "Ours (Mixed BCE)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
    "zeroplane": {
        "h5_folder": "zeroplane_h5",
        "exp_name": f"zeroplane_{EXP_VER}",
        "display_name": "ZeroPlane",
        "label_offset": 0,
        "nonplanar_label": 20,  # ZeroPlane uses 20 for non-planar regions
        "uses_gt_h5": False,
    },
}


# ============================================================
# H5 LOADING
# ============================================================

class LazyH5SceneLoader:
    """
    Memory-efficient loader that only keeps one camera in memory at a time.
    Adapted for Hypersim's per-camera H5 structure: scene_id/planes_cam_XX.h5
    """
    def __init__(self, h5_root: str, label_offset: int = 0, nonplanar_label: Optional[int] = None):
        self.h5_root = h5_root
        self.label_offset = label_offset
        self.nonplanar_label = nonplanar_label
        self._current_scene_id = None
        self._current_cam_name = None
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

    def _load_camera(self, scene_id: str, cam_name: str) -> bool:
        """Load a camera's predictions into memory, clearing previous."""
        if scene_id == self._current_scene_id and cam_name == self._current_cam_name:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, f"planes_{cam_name}.h5")
        if not os.path.exists(h5_path):
            return False

        # Clear previous camera to free memory
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

        with h5py.File(h5_path, "r") as f:
            self._current_planes = f["planes"][:]
            self._current_frame_ids = [
                fid.decode() if isinstance(fid, bytes) else fid
                for fid in f["frame_ids"][:]
            ]

        # Build index for O(1) lookup
        self._frame_id_to_idx = {
            fid: idx for idx, fid in enumerate(self._current_frame_ids)
        }
        self._current_scene_id = scene_id
        self._current_cam_name = cam_name

        return True

    def get_pred_seg(self, scene_id: str, cam_name: str, frame_id: str) -> Optional[np.ndarray]:
        """Get predicted segmentation for a frame."""
        if not self._load_camera(scene_id, cam_name):
            return None

        # Frame ID is stored without cam_name prefix in per-camera H5 files
        if frame_id not in self._frame_id_to_idx:
            return None

        idx = self._frame_id_to_idx[frame_id]
        labels = self._current_planes[idx].astype(np.int32)

        # Handle non-planar label remapping (e.g., ZeroPlane's 20 → 0)
        if self.nonplanar_label is not None:
            labels = np.where(labels == self.nonplanar_label, 0, labels)

        # Apply label offset
        if self.label_offset != 0:
            # Shift all non-zero labels
            labels = np.where(labels > 0, labels + self.label_offset, 0)

        return labels


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(method_key: str, method_config: Dict, args, split: str = "val"):
    """Evaluate a single method."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_config['display_name']}")
    print(f"{'='*60}")

    # Handle GT method (no H5 folder)
    if method_config["h5_folder"] is not None:
        h5_root = H5_ROOT / method_config["h5_folder"]
        print(f"H5 root:    {h5_root}")
    else:
        h5_root = None
        print(f"H5 root:    N/A (using GT labels)")

    exp_name = method_config["exp_name"]
    output_dir = EVAL_ROOT / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output:     {output_dir}")

    # Load dataset
    timer = Timer()
    print("\n[INFO] Loading dataset...")

    dataset = HypersimPlaneDataset(
        hypersim_root=HYPERSIM_ROOT,
        plane_label_root=PLANE_LABEL_ROOT,
        params_root=PARAMS_ROOT,
        split_txt_dir=os.path.join(repo_path, "splits", "hypersim"),
        split=split,
        image_height=512,
        image_width=768,
        max_scenes=None,
    )

    print(f"[INFO] Dataset size: {len(dataset)} frames")

    # Initialize H5 loader (skip for GT method)
    h5_loader = None
    if not method_config.get("uses_gt_h5", False) and h5_root is not None:
        h5_loader = LazyH5SceneLoader(
            str(h5_root),
            label_offset=method_config["label_offset"],
            nonplanar_label=method_config.get("nonplanar_label"),
        )

    # Evaluation function
    def eval_frame(idx):
        sample = dataset[idx]
        scene_id = sample["scene_id"]
        frame_id = sample["frame_idx"]

        # Extract cam_name from rgb_path (format: "scene_id/cam_name/frame_id")
        rgb_path = sample["rgb_path"]
        cam_name = rgb_path.split('/')[1] if '/' in rgb_path else "cam_00"

        # Full frame identifier with camera name
        full_frame_id = f"{cam_name}/{frame_id}"

        # Load prediction or use GT
        if method_config.get("uses_gt_h5", False):
            # Use GT labels directly (upper bound)
            gt_seg = sample["plane"].numpy().astype(np.int32)
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            labels = gt_seg.copy()  # Use GT as prediction
        else:
            # Load prediction from H5
            labels = h5_loader.get_pred_seg(scene_id, cam_name, frame_id)
            if labels is None:
                return None

        # Get ground truth
        gt_seg = sample["plane"].numpy().astype(np.int32)
        if gt_seg.ndim == 3:
            gt_seg = gt_seg[0]

        depth = sample["depth"].numpy()
        if depth.ndim == 3:
            depth = depth[0]

        K = sample["K"].numpy()
        c2w = sample["c2w"].numpy()

        # Resize prediction to match GT if needed
        if labels.shape != gt_seg.shape:
            labels = cv2.resize(
                labels.astype(np.uint16),
                (gt_seg.shape[1], gt_seg.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        # Evaluate (use full_frame_id so it's stored in metrics dict)
        metrics, _ = evaluate_single_frame(
            scene_id,
            full_frame_id,
            depth,
            gt_seg,
            K,
            c2w,
            labels,
            THRESHOLDS,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE,
        )

        return (scene_id, full_frame_id), metrics

    # Run evaluation in parallel
    print("\n[INFO] Running evaluation...")
    with timer("evaluation"):
        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_frame)(idx) for idx in tqdm(range(len(dataset)), desc="Evaluating")
        )

    # Collect results
    results = {}
    skipped = 0
    for output in outputs:
        if output is None:
            skipped += 1
            continue
        (scene_id, frame_id), metrics = output
        results[(scene_id, frame_id)] = metrics

    print(f"\n[INFO] Processed {len(results)} frames ({skipped} skipped)")

    # Save results
    print("\n[INFO] Saving results...")
    save_results_csv(results, str(output_dir))
    save_runtime(timer, str(output_dir))

    timer.print_summary(num_frames=len(results))
    print(f"\n[DONE] Results saved to: {output_dir}")

    return results


def aggregate_results(methods: list):
    """Aggregate results from multiple methods into comparison tables."""
    print(f"\n{'='*60}")
    print("Aggregating Results")
    print(f"{'='*60}")

    all_results = {}
    for method_key in methods:
        if method_key not in METHODS:
            print(f"[WARN] Unknown method: {method_key}")
            continue

        method_config = METHODS[method_key]
        exp_name = method_config["exp_name"]
        csv_path = EVAL_ROOT / exp_name / "results_dataset.csv"

        if not csv_path.exists():
            print(f"[WARN] Missing results for {method_key}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        all_results[method_config["display_name"]] = df

    if not all_results:
        print("[ERROR] No results to aggregate")
        return

    # Create comparison table
    comparison = pd.DataFrame({
        name: df.set_index("metric")["value"]
        for name, df in all_results.items()
    })

    # Save
    output_path = EVAL_ROOT / f"comparison_{EXP_VER}.csv"
    comparison.to_csv(output_path)
    print(f"\n[DONE] Comparison saved to: {output_path}")
    print("\n" + str(comparison))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Hypersim predictions for all baseline methods"
    )

    parser.add_argument("--methods", nargs="+", default=None,
                       help="Methods to evaluate (default: all)")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing results, skip evaluation")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                       help="Dataset split to evaluate")

    args = parser.parse_args()

    # Determine which methods to evaluate
    methods_to_eval = args.methods if args.methods else list(METHODS.keys())

    print(f"Methods to evaluate: {methods_to_eval}")

    # Run evaluation
    if not args.aggregate_only:
        for method_key in methods_to_eval:
            if method_key not in METHODS:
                print(f"[ERROR] Unknown method: {method_key}")
                continue

            evaluate_method(method_key, METHODS[method_key], args, split=args.split)

    # Aggregate results
    aggregate_results(methods_to_eval)


if __name__ == "__main__":
    main()
