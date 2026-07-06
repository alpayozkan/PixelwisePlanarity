#!/usr/bin/env python3
"""
Unified evaluation script for all baseline methods on VKITTI2 dataset.

Evaluates all methods from H5 prediction folders and generates:
1. Per-method CSV results (results.csv, results_per_scene.csv, results_dataset.csv)
2. Aggregated baseline comparison tables

Uses standard pinhole backprojection (evaluate_single_frame) since VKITTI2 has
proper camera intrinsics and poses.

Usage:
    python evaluate_vkitti2_all_baselines.py                    # Evaluate all methods
    python evaluate_vkitti2_all_baselines.py --methods ours zeroplane  # Evaluate specific methods
    python evaluate_vkitti2_all_baselines.py --aggregate-only   # Only aggregate existing results
"""

import os
import sys
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

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pxwplanar.shared.datasets.vkitti2_plane_dataset import VKITTI2PlaneDataset
from pxwplanar.paths import repo_path, vkitti2_path, vkitti2_eval_root as _eval_root, vkitti2_h5_root as _h5_root

from pxwplanar.evaluation.quantitative.eval_utils import (
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
THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
BATCH_SIZE = 32
N_JOBS = min(16, os.cpu_count())

# Paths
EVAL_ROOT = Path(_eval_root)
H5_ROOT = Path(_h5_root)
VKITTI2_ROOT = vkitti2_path

# Experiment version
METHODS = {
    "gt": {
        "h5_folder": None,  # Use GT labels directly
        "exp_name": "gt",
        "display_name": "GT (upper bound)",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": True,
    },
    "ours": {
        # planes.h5 from the signals->planes pipeline with the 4-head MoGe
        # checkpoint (moge_HIRES_4datasets, epoch 1).
        "h5_folder": "moge_hires_4ds_ep1_h5",
        "exp_name": "moge_ours_ep1",
        "display_name": "Ours",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
}


# ============================================================
# H5 LOADING (ScanNet++-style: one planes.h5 per scene_id)
# ============================================================

class LazyH5SceneLoader:
    """
    Memory-efficient loader that only keeps one scene in memory at a time.
    VKITTI2 scene_id is "scene/variant" (e.g., "Scene06/clone").

    Supports ordinal-based frame matching via `dataset_root`: when the H5 stores
    sequential frame IDs (0000, 0001, ...) instead of actual frame numbers
    (00000, 00005, ...), pass dataset_root to build the mapping from dataset
    frame IDs to pred ordinal indices.
    """
    def __init__(self, h5_root: str, label_offset: int = 0,
                 nonplanar_label: Optional[int] = None, h5_filename: str = "planes.h5",
                 dataset_root: Optional[str] = None, dataset_h5_filename: str = "scene_data.h5"):
        self.h5_root = h5_root
        self.label_offset = label_offset
        self.nonplanar_label = nonplanar_label
        self.h5_filename = h5_filename
        self.dataset_root = dataset_root
        self.dataset_h5_filename = dataset_h5_filename
        self._current_scene_id = None
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}
        self._ordinal_map = {}  # dataset_frame_id -> pred_ordinal_idx (for sequential ID H5s)

    def _load_scene(self, scene_id: str) -> bool:
        """Load a scene's predictions into memory, clearing previous."""
        if scene_id == self._current_scene_id:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
        if not os.path.exists(h5_path):
            return False

        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}
        self._ordinal_map = {}

        with h5py.File(h5_path, "r") as f:
            self._current_planes = f["planes"][:]
            self._current_frame_ids = [
                fid.decode() if isinstance(fid, bytes) else fid
                for fid in f["frame_ids"][:]
            ]

        # Index by both the raw string and integer-normalized form to handle
        # padding mismatches (e.g. H5 stores '0000' but dataset returns '00000')
        self._frame_id_to_idx = {}
        for i, fid in enumerate(self._current_frame_ids):
            self._frame_id_to_idx[fid] = i
            try:
                self._frame_id_to_idx[str(int(fid))] = i
            except ValueError:
                pass

        # Build ordinal map: dataset frame IDs -> pred ordinal index.
        # Used when the H5 uses sequential IDs (0, 1, 2, ...) instead of actual
        # frame numbers. E.g. pseudo_planamono stores '0000','0001',... while the
        # dataset has '00000','00005','00010',... The i-th dataset frame maps to
        # pred index i regardless of the actual frame number.
        if self.dataset_root is not None:
            ds_h5_path = os.path.join(self.dataset_root, scene_id, self.dataset_h5_filename)
            if os.path.exists(ds_h5_path):
                with h5py.File(ds_h5_path, "r") as f:
                    ds_frame_ids = [
                        fid.decode() if isinstance(fid, bytes) else str(fid)
                        for fid in f["frame_ids"][:]
                    ]
                for ordinal, ds_fid in enumerate(ds_frame_ids):
                    self._ordinal_map[ds_fid] = ordinal
                    try:
                        self._ordinal_map[str(int(ds_fid))] = ordinal
                    except ValueError:
                        pass

        self._current_scene_id = scene_id
        return True

    def _apply_postproc(self, pred: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize, remap non-planar label, apply offset."""
        if pred.shape != target_shape:
            pred = cv2.resize(
                pred.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
        if self.nonplanar_label is not None:
            pred[pred == self.nonplanar_label] = 0
        if self.label_offset != 0:
            pred = pred + self.label_offset
        return pred.astype(np.int32)

    def get_prediction(self, scene_id: str, frame_idx: str,
                       target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Get prediction for a specific frame, loading scene if needed."""
        if not self._load_scene(scene_id):
            return None

        # Normalize frame_idx to handle padding mismatches ('00000' vs '0000')
        lookup = frame_idx
        if lookup not in self._frame_id_to_idx:
            try:
                lookup = str(int(frame_idx))
            except ValueError:
                pass

        if lookup in self._frame_id_to_idx:
            idx = self._frame_id_to_idx[lookup]
            return self._apply_postproc(self._current_planes[idx].copy(), target_shape)

        # Fallback: ordinal map (for sequential-ID H5s like pseudo_planamono).
        # The i-th frame in the dataset corresponds to pred index i.
        if self._ordinal_map:
            ordinal = self._ordinal_map.get(frame_idx)
            if ordinal is None:
                try:
                    ordinal = self._ordinal_map.get(str(int(frame_idx)))
                except ValueError:
                    pass
            if ordinal is not None and ordinal < len(self._current_frame_ids):
                return self._apply_postproc(self._current_planes[ordinal].copy(), target_shape)

        return None

    def has_scene(self, scene_id: str) -> bool:
        """Check if a scene exists without loading it."""
        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
        return os.path.exists(h5_path)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(
    method_key: str,
    method_config: Dict,
    val_dataset: VKITTI2PlaneDataset,
    val_loader: DataLoader,
    shard_id: int = None,
) -> Dict:
    """Evaluate a single method and save results."""
    uses_gt_h5 = method_config.get("uses_gt_h5", False)
    exp_name = method_config["exp_name"]
    label_offset = method_config["label_offset"]
    csv_out_dir = EVAL_ROOT / exp_name

    if uses_gt_h5:
        h5_root = None
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_config['display_name']} ({method_key})")
        print(f"H5 root: N/A (using GT labels)")
        print(f"Output: {csv_out_dir}")
        print(f"{'='*60}")
    else:
        if "h5_root_override" in method_config:
            h5_root = Path(method_config["h5_root_override"])
        else:
            h5_root = H5_ROOT / method_config["h5_folder"]
        print(f"\n{'='*60}")
        print(f"Evaluating: {method_config['display_name']} ({method_key})")
        print(f"H5 root: {h5_root}")
        print(f"Output: {csv_out_dir}")
        print(f"{'='*60}")

        if not h5_root.exists():
            print(f"[ERROR] H5 root does not exist: {h5_root}")
            return {}

    timer = Timer()

    # Initialize lazy loader (skip for GT method)
    loader = None
    if not uses_gt_h5:
        nonplanar_label = method_config.get("nonplanar_label", None)
        h5_filename = method_config.get("h5_filename", "planes.h5")
        dataset_root = method_config.get("dataset_root", None)
        dataset_h5_filename = method_config.get("dataset_h5_filename", "scene_data.h5")
        loader = LazyH5SceneLoader(
            str(h5_root), label_offset=label_offset,
            nonplanar_label=nonplanar_label, h5_filename=h5_filename,
            dataset_root=dataset_root, dataset_h5_filename=dataset_h5_filename,
        )

        # Check available scenes
        scene_ids = val_dataset.scene_ids
        # VKITTI2 scene_ids are base scene names; actual H5 scene_ids include variant
        # We need to check all scene_id/variant combinations
        available_count = 0
        for pair in val_dataset.valid_pairs:
            sid = f"{pair[2]}/{pair[3]}"  # scene/variant
            if loader.has_scene(sid):
                available_count += 1
                break  # just checking existence
        print(f"[DATA] Checking predictions availability...")

        if available_count == 0 and len(val_dataset.valid_pairs) > 0:
            # Check first few scene_ids
            sample_scene_ids = set()
            for pair in val_dataset.valid_pairs[:10]:
                sample_scene_ids.add(f"{pair[2]}/{pair[3]}")
            has_any = any(loader.has_scene(s) for s in sample_scene_ids)
            if not has_any:
                print(f"[ERROR] No predictions found for {method_key}")
                print(f"[DEBUG] Checked: {sample_scene_ids}")
                return {}
    else:
        print(f"[DATA] Using GT labels as predictions for {len(val_dataset)} frames")

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
                if uses_gt_h5:
                    labels = gt_seg_np.copy()
                else:
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
        if shard_id is not None:
            # Save as shard file (for distributed eval)
            csv_out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame.from_records(list(results.values()))
            shard_path = csv_out_dir / f"results_shard_{shard_id}.csv"
            df.to_csv(shard_path, index=False)
            print(f"[CSV] Saved shard {shard_id} ({len(results)} frames) to {shard_path}")
        else:
            print("==> Saving results")
            save_results_csv(results, str(csv_out_dir))
        save_runtime(timer, str(csv_out_dir))
        timer.print_summary(num_frames=len(results))

    return results


# ============================================================
# SHARD MERGING
# ============================================================

def _merge_shards(exp_dir: Path):
    """Merge shard CSV files into results.csv, then produce per-scene and dataset CSVs."""
    import glob as glob_mod
    shard_files = sorted(glob_mod.glob(str(exp_dir / "results_shard_*.csv")))
    if not shard_files:
        return False

    print(f"[MERGE] Found {len(shard_files)} shard files in {exp_dir}")
    dfs = [pd.read_csv(f) for f in shard_files]
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[MERGE] Total frames: {len(df_all)}")

    # Reconstruct results dict for save_results_csv
    results = {}
    for _, row in df_all.iterrows():
        key = (row["scene_id"], row["frame_idx"])
        results[key] = row.to_dict()
    save_results_csv(results, str(exp_dir))
    print(f"[MERGE] Saved merged results to {exp_dir}")
    return True


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_results(methods: list, output_dir: Path = None):
    """Aggregate results from specified methods into summary tables."""
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    # Merge shards for each method if needed
    for method_key in methods:
        if method_key not in METHODS:
            continue
        exp_dir = EVAL_ROOT / METHODS[method_key]["exp_name"]
        if exp_dir.exists():
            _merge_shards(exp_dir)

    all_results = []

    for method_key in methods:
        if method_key not in METHODS:
            print(f"[WARN] Unknown method: {method_key}")
            continue
        method_config = METHODS[method_key]
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

            # Precision/recall/F1 metrics
            for thr in THRESHOLDS:
                thresh_str = f"{thr*100:.1f}cm"
                prec_col = f"prec@{thresh_str}_mean"
                rec_col = f"rec@{thresh_str}_mean"
                if prec_col in df.index:
                    row[f"P@{thresh_str}"] = df[prec_col]
                if rec_col in df.index:
                    row[f"R@{thresh_str}"] = df[rec_col]
                p = row.get(f"P@{thresh_str}", 0)
                r = row.get(f"R@{thresh_str}", 0)
                row[f"F1@{thresh_str}"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            # Binary planarity metrics
            for bp_col in ["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"]:
                mean_col = f"{bp_col}_mean"
                if mean_col in df.index:
                    row[bp_col] = df[mean_col]

            all_results.append(row)
            print(f"[OK] Loaded results for {display_name}")

        except Exception as e:
            print(f"[ERROR] Could not read results for {method_key}: {e}")

    if not all_results:
        print("[ERROR] No results to aggregate")
        return

    df_all = pd.DataFrame(all_results)

    # Table 1: Precision/Recall
    prec_rec_cols = ["Method", "num_scenes", "num_frames"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        prec_rec_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}", f"F1@{thresh_str}"])
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
    combined_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        combined_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}", f"F1@{thresh_str}"])
    combined_cols.extend(["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"])
    df_combined = df_all[[c for c in combined_cols if c in df_all.columns]]
    out_path = output_dir / "table_combined_baselines.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("VKITTI2 BASELINE RESULTS SUMMARY")
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
    parser = argparse.ArgumentParser(description="Evaluate all baseline methods on VKITTI2")
    parser.add_argument("--methods", nargs="+", default=None,
                        help=f"Methods to evaluate (default: all). Options: {list(METHODS.keys())}")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, skip evaluation")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum number of scenes to evaluate (for testing)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Variants to include (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save aggregated tables (default: EVAL_ROOT)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--scene-start", type=int, default=None,
                        help="Start scene index (for distributed eval across SLURM array jobs)")
    parser.add_argument("--scene-end", type=int, default=None,
                        help="End scene index exclusive (for distributed eval)")
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
    print(f"[CONFIG] Split: {args.split}")
    print(f"[CONFIG] Variants: {args.variants or 'all'}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] Compute plane metrics: {COMPUTE_PLANE_METRICS}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")

    if not args.aggregate_only:
        # Load dataset once
        print("\n==> Loading dataset")
        val_dataset = VKITTI2PlaneDataset(
            data_root=VKITTI2_ROOT,
            split_txt_dir=os.path.join(repo_path, "splits", "vkitti2"),
            split=args.split,
            variants=args.variants,
            max_scenes=args.max_scenes,
        )
        print(f"[DATA] {args.split} set: {len(val_dataset)} frames")

        # Scene range slicing for distributed eval
        # VKITTI2 scene_ids are base names (Scene18, Scene20) but we shard on
        # scene/variant combos for better parallelism (e.g. 20 combos for 2 scenes x 10 variants)
        if args.scene_start is not None or args.scene_end is not None:
            all_combos = sorted(set(f"{p[2]}/{p[3]}" for p in val_dataset.valid_pairs))
            s = args.scene_start or 0
            e = args.scene_end or len(all_combos)
            subset_combos = set(all_combos[s:e])
            val_dataset.valid_pairs = [
                p for p in val_dataset.valid_pairs
                if f"{p[2]}/{p[3]}" in subset_combos
            ]
            # Update scene_ids to only include base scenes that still have pairs
            remaining_base = sorted(set(p[2] for p in val_dataset.valid_pairs))
            val_dataset.scene_ids = remaining_base
            print(f"[DATA] Scene range [{s}:{e}] -> {len(subset_combos)} scene/variant combos, {len(val_dataset)} frames")

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Derive shard_id from scene_start (for distributed eval)
        shard_id = None
        if args.scene_start is not None:
            shard_id = args.scene_start

        # Evaluate each method
        for method_key in methods_to_eval:
            method_config = METHODS[method_key]
            evaluate_method(method_key, method_config, val_dataset, val_loader, shard_id=shard_id)

    # Aggregate results (merge shards if needed)
    # Skip aggregation when running as a shard job — let the dedicated --aggregate-only job handle it
    if args.scene_start is None:
        output_dir = Path(args.output_dir) if args.output_dir else EVAL_ROOT
        aggregate_results(methods_to_eval, output_dir)

    print("\n[DONE] All evaluations complete!")


if __name__ == "__main__":
    main()
