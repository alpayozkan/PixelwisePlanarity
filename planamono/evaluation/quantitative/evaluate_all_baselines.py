"""
Unified evaluation script for all baseline methods.

Evaluates all methods from H5 prediction folders and generates:
1. Per-method CSV results (results.csv, results_per_scene.csv, results_dataset.csv)
2. Aggregated baseline comparison tables

Usage:
    python evaluate_all_baselines.py                    # Evaluate all methods
    python evaluate_all_baselines.py --methods ours zeroplane  # Evaluate specific methods
    python evaluate_all_baselines.py --aggregate-only   # Only aggregate existing results
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

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_rend_plane_path

from eval_utils import (
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
EVAL_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval")
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference")
DATASET_DIR = "/cluster/scratch/aoezkan/planeseg/dataset/scannetpp"

# Method definitions: {method_key: {h5_folder, display_name, label_offset, uses_gt_h5}}
METHODS = {
    "ours": {
        "h5_folder": "moge_ours_v2_h5",
        "exp_name": "moge_ours_v2",
        "display_name": "Ours (full)",
        "label_offset": 0,
        "uses_gt_h5": False,
    },
    "zeroplane": {
        "h5_folder": "zeroplane_h5",
        "exp_name": "zeroplane_v2",
        "display_name": "ZeroPlane",
        "label_offset": 1,  # ZeroPlane doesn't have background label 0
        "uses_gt_h5": False,
    },
    "gtseg": {
        "h5_folder": "gtseg_v1_h5",
        "exp_name": "gtseg_v2",
        "display_name": "GT Seg (upper bound)",
        "label_offset": 0,
        "uses_gt_h5": False,
    },
    "gtplanarity_ourseg": {
        "h5_folder": "gtplanarity_ourseg_h5",
        "exp_name": "gtplanarity_ourseg_v2",
        "display_name": "GT Planarity + Our Seg",
        "label_offset": 0,
        "uses_gt_h5": False,
    },
    "ourplanarity_gtseg": {
        "h5_folder": "ourplanarity_gtseg_v1_h5",
        "exp_name": "ourplanarity_gtseg_v2",
        "display_name": "Our Planarity + GT Seg",
        "label_offset": 0,
        "uses_gt_h5": False,
    },
}


# ============================================================
# H5 LOADING
# ============================================================

class LazyH5SceneLoader:
    """
    Memory-efficient loader that only keeps one scene in memory at a time.
    """
    def __init__(self, h5_root: str, label_offset: int = 0):
        self.h5_root = h5_root
        self.label_offset = label_offset
        self._current_scene_id = None
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

    def _load_scene(self, scene_id: str) -> bool:
        """Load a scene's predictions into memory, clearing previous."""
        if scene_id == self._current_scene_id:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, "planes.h5")
        if not os.path.exists(h5_path):
            return False

        # Clear previous scene to free memory
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
        self._frame_id_to_idx = {fid: i for i, fid in enumerate(self._current_frame_ids)}
        self._current_scene_id = scene_id
        return True

    def get_prediction(self, scene_id: str, frame_idx: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Get prediction for a specific frame, loading scene if needed.

        Returns:
            labels: (H, W) plane labels, or None if not found
        """
        if not self._load_scene(scene_id):
            return None

        if frame_idx not in self._frame_id_to_idx:
            return None

        idx = self._frame_id_to_idx[frame_idx]
        pred = self._current_planes[idx]

        # Resize to target shape if needed
        if pred.shape != target_shape:
            pred = cv2.resize(
                pred.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        # Apply label offset (e.g., ZeroPlane doesn't have 0 label)
        pred = pred + self.label_offset
        return pred.astype(np.int32)

    def has_scene(self, scene_id: str) -> bool:
        """Check if a scene exists without loading it."""
        h5_path = os.path.join(self.h5_root, scene_id, "planes.h5")
        return os.path.exists(h5_path)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(
    method_key: str,
    method_config: Dict,
    val_dataset: ScanNetPPPlaneDataset,
    val_loader: DataLoader,
) -> Dict:
    """
    Evaluate a single method and save results.

    Returns:
        results: {(scene_id, frame_id): metrics_dict}
    """
    h5_root = H5_ROOT / method_config["h5_folder"]
    exp_name = method_config["exp_name"]
    label_offset = method_config["label_offset"]
    csv_out_dir = EVAL_ROOT / exp_name

    print(f"\n{'='*60}")
    print(f"Evaluating: {method_config['display_name']} ({method_key})")
    print(f"H5 root: {h5_root}")
    print(f"Output: {csv_out_dir}")
    print(f"{'='*60}")

    if not h5_root.exists():
        print(f"[ERROR] H5 root does not exist: {h5_root}")
        return {}

    timer = Timer()

    # Initialize lazy loader
    loader = LazyH5SceneLoader(str(h5_root), label_offset=label_offset)

    # Check available scenes
    scene_ids = val_dataset.scene_ids
    available_scenes = [s for s in scene_ids if loader.has_scene(s)]
    missing_scenes = set(scene_ids) - set(available_scenes)
    if missing_scenes:
        print(f"[WARN] Missing predictions for {len(missing_scenes)} scenes")
    print(f"[DATA] Found predictions for {len(available_scenes)}/{len(scene_ids)} scenes")

    if len(available_scenes) == 0:
        print(f"[ERROR] No predictions found for {method_key}")
        return {}

    # Evaluation wrapper
    def eval_frame_wrapper(scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, labels, thresholds):
        return evaluate_single_frame(
            scene_id,
            frame_idx,
            depth_np,
            gt_seg_np,
            K_np,
            c2w_np,
            labels,
            thresholds,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE
        )

    results = {}
    skipped_frames = 0

    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc=f"Evaluating {method_key}"):
            scene_ids_batch = batch["scene_id"]
            frame_ids = batch["frame_idx"]
            gt_planes = batch["plane"]
            depths = batch["depth"]
            Ks = batch["K"]
            c2ws = batch["c2w"]

            B = len(scene_ids_batch)

            # Prepare batch data
            batch_items = []
            for i in range(B):
                scene_id = scene_ids_batch[i]
                frame_idx = frame_ids[i]

                # Get GT
                gt_seg = gt_planes[i]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
                H, W = gt_seg_np.shape

                depth = depths[i]
                depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

                # Get prediction
                labels = loader.get_prediction(scene_id, frame_idx, (H, W))

                if labels is None:
                    skipped_frames += 1
                    continue

                batch_items.append({
                    "scene_id": scene_id,
                    "frame_idx": frame_idx,
                    "depth_np": depth_np,
                    "gt_seg_np": gt_seg_np,
                    "K_np": Ks[i].numpy(),
                    "c2w_np": c2ws[i].numpy(),
                    "labels": labels,
                })

            if not batch_items:
                continue

            # Parallel evaluation
            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(eval_frame_wrapper)(
                    item["scene_id"],
                    item["frame_idx"],
                    item["depth_np"],
                    item["gt_seg_np"],
                    item["K_np"],
                    item["c2w_np"],
                    item["labels"],
                    THRESHOLDS
                )
                for item in batch_items
            )

            for (metrics, labels), item in zip(outputs, batch_items):
                scene_id = item["scene_id"]
                frame_id = item["frame_idx"]
                results[(scene_id, frame_id)] = metrics

    print(f"[PIPELINE] Evaluated {len(results)} frames (skipped {skipped_frames})")

    # Save results
    if results:
        print("==> Saving results")
        save_results_csv(results, str(csv_out_dir))
        save_runtime(timer, str(csv_out_dir))
        timer.print_summary(num_frames=len(results))

    return results


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_results(output_dir: Path = None):
    """
    Aggregate results from all methods into summary tables.
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    # Collect results for all methods
    all_results = []

    for method_key, method_config in METHODS.items():
        exp_name = method_config["exp_name"]
        display_name = method_config["display_name"]
        csv_path = EVAL_ROOT / exp_name / "results_dataset.csv"

        if not csv_path.exists():
            print(f"[WARN] Missing results for {method_key}: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path).iloc[0]
            row = {
                "Method": display_name,
                "method_key": method_key,
                "num_scenes": int(df["num_scenes"]),
                "num_frames": int(df["num_frames_total"]),
            }

            # Segmentation metrics
            for col, display in [("rand_index_mean", "RI"),
                                  ("voi_mean", "VOI"),
                                  ("sc_mean", "SC")]:
                if col in df.index:
                    row[display] = df[col]

            # Precision/recall metrics
            for thresh in ["1cm", "2cm", "5cm"]:
                prec_col = f"prec@{thresh}_mean"
                rec_col = f"rec@{thresh}_mean"
                if prec_col in df.index:
                    row[f"P@{thresh}"] = df[prec_col]
                if rec_col in df.index:
                    row[f"R@{thresh}"] = df[rec_col]

            all_results.append(row)
            print(f"[OK] Loaded results for {display_name}")

        except Exception as e:
            print(f"[ERROR] Could not read results for {method_key}: {e}")

    if not all_results:
        print("[ERROR] No results to aggregate")
        return

    # Create DataFrames
    df_all = pd.DataFrame(all_results)

    # Table 1: Precision/Recall
    prec_rec_cols = ["Method", "num_scenes", "num_frames"]
    for thresh in ["1cm", "2cm", "5cm"]:
        prec_rec_cols.extend([f"P@{thresh}", f"R@{thresh}"])
    df_pr = df_all[[c for c in prec_rec_cols if c in df_all.columns]]
    out_path = output_dir / "table_precision_recall_baselines.csv"
    df_pr.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Table 2: Segmentation
    seg_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    df_seg = df_all[[c for c in seg_cols if c in df_all.columns]]
    out_path = output_dir / "table_segmentation_baselines.csv"
    df_seg.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Table 3: Combined summary
    combined_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC", "P@2cm", "R@2cm"]
    df_combined = df_all[[c for c in combined_cols if c in df_all.columns]]
    out_path = output_dir / "table_combined_baselines.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df_combined.to_string(index=False))
    print("=" * 100)

    return df_all


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate all baseline methods")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Methods to evaluate (default: all). Options: {list(METHODS.keys())}")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, skip evaluation")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum number of scenes to evaluate (for testing)")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory to save aggregated tables")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    args = parser.parse_args()

    # Determine which methods to evaluate
    if args.methods is None:
        methods_to_eval = list(METHODS.keys())
    else:
        methods_to_eval = args.methods
        invalid = set(methods_to_eval) - set(METHODS.keys())
        if invalid:
            print(f"[ERROR] Invalid methods: {invalid}")
            print(f"[ERROR] Valid options: {list(METHODS.keys())}")
            return

    print(f"[CONFIG] Methods to evaluate: {methods_to_eval}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")

    if not args.aggregate_only:
        # Load dataset once
        print("\n==> Loading dataset")
        val_dataset = ScanNetPPPlaneDataset(
            rgb_root="/cluster/project/cvg/Shared_datasets/scannet++/data",
            plane_label_root=scannetpp_rend_plane_path,
            sem_label_root=os.path.join(DATASET_DIR, ""),
            depth_label_root=scannetpp_rend_plane_path,
            split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
            split="val",
            max_scenes=args.max_scenes,
        )
        print(f"[DATA] Validation set: {len(val_dataset)} frames")

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Evaluate each method
        for method_key in methods_to_eval:
            method_config = METHODS[method_key]
            evaluate_method(method_key, method_config, val_dataset, val_loader)

    # Aggregate results
    aggregate_results(Path(args.output_dir))

    print("\n[DONE] All evaluations complete!")


if __name__ == "__main__":
    main()
