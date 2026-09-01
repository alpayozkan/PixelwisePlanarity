"""
Unified evaluation script for all baseline methods.

Properly handles different label conventions across methods:
- Standard methods (ours, gt): use label 0 for non-planar/background
- Methods with a different non-planar label (e.g. ZeroPlane uses 20) are
  remapped via nonplanar_label

Evaluates all methods from H5 prediction folders and generates:
1. Per-method CSV results (results.csv, results_per_scene.csv,
   results_dataset.csv)
2. Aggregated baseline comparison tables

Usage:
    # Evaluate all methods
    python evaluate_all_baselines.py
    # Evaluate specific methods
    python evaluate_all_baselines.py --methods ours
    # Only aggregate existing results
    python evaluate_all_baselines.py --aggregate-only
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pxwplanar.evaluation.quantitative.eval_utils import (
    Timer,
    evaluate_single_frame,
    save_results_csv,
    save_runtime,
)
from pxwplanar.paths import (
    eval_root as _default_eval_root,
)
from pxwplanar.paths import (
    inference_h5_root as _default_h5_root,
)
from pxwplanar.paths import (
    ours_planes_root,
    repo_path,
    scannetpp_path,
    scannetpp_rend_plane_path,
)
from pxwplanar.shared.datasets.scannetpp import ScanNetPPPlaneDataset

# ============================================================
# CONFIGURATION
# ============================================================

# Evaluation parameters
COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
# RANSAC RNG seed for reproducible 3D metrics. Open3D's segment_plane draws
# random triplets; seeding makes precision/recall@tau repeatable run-to-run
# (residual cross-machine variance stays ~1e-3 at the tightest threshold).
# 0 = reproducible (default); --ransac-seed -1 = legacy non-deterministic.
RANSAC_SEED = 0
THRESHOLDS = (0.001, 0.005, 0.01)
BATCH_SIZE = 32
N_JOBS = min(16, os.cpu_count())

# Paths
EVAL_ROOT = Path(_default_eval_root)
H5_ROOT = Path(_default_h5_root)
DATASET_DIR = scannetpp_rend_plane_path

# Method definitions:
#   {method_key: {h5_folder, display_name, label_offset, nonplanar_label,
#                 uses_gt_h5}}
# - h5_folder: per-scene planes.h5 folder under H5_ROOT
#   (paths.inference_h5_root);
#   use h5_root_override instead for an absolute root.
# - nonplanar_label: label meaning "non-planar" in a method's H5s
#   (remapped to 0);
#   None if the method already uses 0.
# Add new baselines here.
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
        # planes.h5 produced by the release pipeline
        # (save_moge_signals_planarity.py -> segment_signals_to_planes.py) with
        # the 4-head MoGe checkpoint (moge_HIRES_4datasets, epoch 1) at
        # 1440x1920, canonical segmentation parameters (planarity>0.3, n<5deg,
        # d_rel<0.025, >=8 neighbors).
        "h5_folder": None,
        "h5_root_override": ours_planes_root,
        "exp_name": "moge_ours_ep1",
        "display_name": "Ours",
        "label_offset": 0,
        "nonplanar_label": None,
        "uses_gt_h5": False,
    },
}


# ============================================================
# H5 LOADING
# ============================================================


class LazyH5SceneLoader:
    """
    Memory-efficient loader that only keeps one scene in memory at a time.

    Handles non-planar region remapping for methods like ZeroPlane.
    """

    def __init__(
        self,
        h5_root: str,
        label_offset: int = 0,
        nonplanar_label: int | None = None,
        h5_filename: str = "planes.h5",
    ):
        self.h5_root = h5_root
        self.label_offset = label_offset
        self.nonplanar_label = (
            nonplanar_label  # Label to remap to 0 (e.g., 20 for ZeroPlane)
        )
        self.h5_filename = h5_filename
        self._current_scene_id = None
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

    def _load_scene(self, scene_id: str) -> bool:
        """Load a scene's predictions into memory, clearing previous."""
        if scene_id == self._current_scene_id:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
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
        self._frame_id_to_idx = {
            fid: i for i, fid in enumerate(self._current_frame_ids)
        }
        self._current_scene_id = scene_id
        return True

    def get_prediction(
        self, scene_id: str, frame_idx: str, target_shape: tuple[int, int]
    ) -> np.ndarray | None:
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
        pred = self._current_planes[
            idx
        ].copy()  # Copy to avoid modifying cached data

        # Resize to target shape if needed
        if pred.shape != target_shape:
            pred = cv2.resize(
                pred.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        # Remap non-planar label to 0 WITHOUT colliding with plane-id 0.
        # ZeroPlane planes are argmax over channels 0..19 with
        # non-planar = 20, so a plain 20->0 would merge the real plane-0
        # into the background (present in every frame). Shift planes up by 1
        # first, then send non-planar to 0.
        if self.nonplanar_label is not None:
            nonplanar_mask = pred == self.nonplanar_label
            pred = pred + 1
            pred[nonplanar_mask] = 0

        # Apply label offset (usually 0 after remapping)
        if self.label_offset != 0:
            pred = pred + self.label_offset

        return pred.astype(np.int32)

    def has_scene(self, scene_id: str) -> bool:
        """Check if a scene exists without loading it."""
        h5_path = os.path.join(self.h5_root, scene_id, self.h5_filename)
        return os.path.exists(h5_path)


# ============================================================
# EVALUATION
# ============================================================


def evaluate_method(
    method_key: str,
    method_config: dict,
    val_dataset: ScanNetPPPlaneDataset,
    val_loader: DataLoader,
    shard_id: int = None,
) -> dict:
    """
    Evaluate a single method and save results.

    Args:
        shard_id: If set, save results as results_shard_{shard_id}.csv
            instead of results.csv

    Returns:
        results: {(scene_id, frame_id): metrics_dict}
    """
    uses_gt_h5 = method_config.get("uses_gt_h5", False)
    exp_name = method_config["exp_name"]
    label_offset = method_config["label_offset"]
    csv_out_dir = EVAL_ROOT / exp_name

    if uses_gt_h5:
        h5_root = None
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {method_config['display_name']} ({method_key})")
        print("H5 root: N/A (using GT labels)")
        print(f"Output: {csv_out_dir}")
        print(f"{'=' * 60}")
    else:
        if "h5_root_override" in method_config:
            h5_root = Path(method_config["h5_root_override"])
        else:
            h5_root = H5_ROOT / method_config["h5_folder"]
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {method_config['display_name']} ({method_key})")
        print(f"H5 root: {h5_root}")
        print(f"Output: {csv_out_dir}")
        print(f"{'=' * 60}")

        if not h5_root.exists():
            print(f"[ERROR] H5 root does not exist: {h5_root}")
            return {}

    timer = Timer()

    # Initialize lazy loader (skip for GT method)
    loader = None
    if not uses_gt_h5:
        nonplanar_label = method_config.get("nonplanar_label")
        h5_filename = method_config.get("h5_filename", "planes.h5")
        loader = LazyH5SceneLoader(
            str(h5_root),
            label_offset=label_offset,
            nonplanar_label=nonplanar_label,
            h5_filename=h5_filename,
        )

        # Check available scenes
        scene_ids = val_dataset.scene_ids
        available_scenes = [s for s in scene_ids if loader.has_scene(s)]
        missing_scenes = set(scene_ids) - set(available_scenes)
        if missing_scenes:
            print(
                f"[WARN] Missing predictions for {len(missing_scenes)} scenes"
            )
        print(
            f"[DATA] Found predictions for "
            f"{len(available_scenes)}/{len(scene_ids)} scenes"
        )

        if len(available_scenes) == 0:
            print(f"[ERROR] No predictions found for {method_key}")
            return {}
    else:
        print(
            f"[DATA] Using GT labels as predictions for "
            f"{len(val_dataset)} frames"
        )

    # Evaluation wrapper (uses threshold-consistent RANSAC).
    # Capture RANSAC_SEED into a closure local so the value (incl. any
    # --ransac-seed override) travels to loky workers — workers re-import the
    # module and would otherwise see the unmodified module-level default.
    _ransac_seed = RANSAC_SEED

    def eval_frame_wrapper(
        scene_id,
        frame_idx,
        depth_np,
        gt_seg_np,
        K_np,
        c2w_np,
        labels,
        thresholds,
    ):
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
            inlier_ratio_gate=INLIER_RATIO_GATE,
            ransac_seed=_ransac_seed,
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
                depth_np = (
                    depth[0].cpu().numpy()
                    if depth.ndim == 3
                    else depth.cpu().numpy()
                )

                # Get prediction
                if uses_gt_h5:
                    # Use GT labels directly as prediction (upper bound)
                    labels = gt_seg_np.copy()
                else:
                    labels = loader.get_prediction(scene_id, frame_idx, (H, W))

                    if labels is None:
                        skipped_frames += 1
                        continue

                batch_items.append(
                    {
                        "scene_id": scene_id,
                        "frame_idx": frame_idx,
                        "depth_np": depth_np,
                        "gt_seg_np": gt_seg_np,
                        "K_np": Ks[i].numpy(),
                        "c2w_np": c2ws[i].numpy(),
                        "labels": labels,
                    }
                )

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
                    THRESHOLDS,
                )
                for item in batch_items
            )

            for (metrics, _labels), item in zip(
                outputs, batch_items, strict=False
            ):
                scene_id = item["scene_id"]
                frame_id = item["frame_idx"]
                results[(scene_id, frame_id)] = metrics

    print(
        f"[PIPELINE] Evaluated {len(results)} frames (skipped {skipped_frames})"
    )

    # Save results
    if results:
        if shard_id is not None:
            # Save as shard file (for distributed eval)
            csv_out_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame.from_records(list(results.values()))
            shard_path = csv_out_dir / f"results_shard_{shard_id}.csv"
            df.to_csv(shard_path, index=False)
            print(
                f"[CSV] Saved shard {shard_id} ({len(results)} frames) "
                f"to {shard_path}"
            )
        else:
            print("==> Saving results")
            save_results_csv(results, str(csv_out_dir))
        save_runtime(timer, str(csv_out_dir))
        timer.print_summary(num_frames=len(results))

    return results


# ============================================================
# AGGREGATION
# ============================================================


def _merge_shards(exp_dir: Path):
    """Merge shard CSV files into results.csv, then produce per-scene and
    dataset CSVs.

    Looks for results_shard_*.csv in exp_dir, concatenates them, and calls
    save_results_csv() to produce the standard results.csv /
    results_per_scene.csv / results_dataset.csv files.
    """
    import glob as glob_mod

    shard_files = sorted(glob_mod.glob(str(exp_dir / "results_shard_*.csv")))
    if not shard_files:
        return False

    # Do not let stale shards from an older sharded run clobber a newer full run
    results_csv = exp_dir / "results.csv"
    if results_csv.exists():
        newest_shard = max(os.path.getmtime(f) for f in shard_files)
        if os.path.getmtime(results_csv) > newest_shard:
            print(
                f"[MERGE] Skipping {len(shard_files)} shard files older than "
                f"results.csv in {exp_dir} (delete them to force a re-merge)"
            )
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


def aggregate_results(methods: list, output_dir: Path = None):
    """
    Aggregate results from specified methods into summary tables.
    Merges shard files first if present.
    """
    if output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("AGGREGATING RESULTS")
    print(f"{'=' * 60}")

    # Merge shards for each method if needed
    for method_key in methods:
        if method_key not in METHODS:
            continue
        exp_dir = EVAL_ROOT / METHODS[method_key]["exp_name"]
        if exp_dir.exists():
            _merge_shards(exp_dir)

    # Collect results for specified methods
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
            for col, display in [
                ("rand_index_mean", "RI"),
                ("voi_mean", "VOI"),
                ("sc_mean", "SC"),
            ]:
                if col in df.index:
                    row[display] = df[col]

            # Precision/recall/F1 metrics (dynamically from THRESHOLDS)
            for thr in THRESHOLDS:
                thresh_str = f"{thr * 100:.1f}cm"
                prec_col = f"prec@{thresh_str}_mean"
                rec_col = f"rec@{thresh_str}_mean"
                if prec_col in df.index:
                    row[f"P@{thresh_str}"] = df[prec_col]
                if rec_col in df.index:
                    row[f"R@{thresh_str}"] = df[rec_col]
                p = row.get(f"P@{thresh_str}", 0)
                r = row.get(f"R@{thresh_str}", 0)
                row[f"F1@{thresh_str}"] = (
                    2 * p * r / (p + r) if (p + r) > 0 else 0.0
                )

            # Binary planarity metrics
            for bp_col in [
                "bp_accuracy",
                "bp_precision",
                "bp_recall",
                "bp_f1",
                "bp_iou",
            ]:
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

    # Create DataFrames
    df_all = pd.DataFrame(all_results)

    # Table 1: Precision/Recall
    prec_rec_cols = ["Method", "num_scenes", "num_frames"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr * 100:.1f}cm"
        prec_rec_cols.extend(
            [f"P@{thresh_str}", f"R@{thresh_str}", f"F1@{thresh_str}"]
        )
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

    # Table 3: Combined summary (all thresholds for P/R + bp metrics)
    combined_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr * 100:.1f}cm"
        combined_cols.extend(
            [f"P@{thresh_str}", f"R@{thresh_str}", f"F1@{thresh_str}"]
        )
    combined_cols.extend(
        ["bp_accuracy", "bp_precision", "bp_recall", "bp_f1", "bp_iou"]
    )
    df_combined = df_all[[c for c in combined_cols if c in df_all.columns]]
    out_path = output_dir / "table_combined_baselines.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary
    print("\n" + "=" * 100)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 100)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df_combined.to_string(index=False))
    print("=" * 100)

    return df_all


# ============================================================
# MAIN
# ============================================================


def main():
    global RANSAC_SEED, EVAL_ROOT
    parser = argparse.ArgumentParser(
        description="Evaluate all baseline methods"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Methods to evaluate (default: all). "
            f"Options: {list(METHODS.keys())}"
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing results, skip evaluation",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Maximum number of scenes to evaluate (for testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save aggregated tables",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--scene-start",
        type=int,
        default=None,
        help="Start scene index (for distributed eval across SLURM array jobs)",
    )
    parser.add_argument(
        "--scene-end",
        type=int,
        default=None,
        help="End scene index exclusive (for distributed eval)",
    )
    parser.add_argument(
        "--ransac-seed",
        type=int,
        default=RANSAC_SEED,
        help="Seed for RANSAC plane-fitting RNG (default 0 = "
        "reproducible). Pass -1 to disable seeding (legacy "
        "non-deterministic behaviour).",
    )
    parser.add_argument(
        "--eval-root",
        type=str,
        default=None,
        help="Override EVAL_ROOT, the directory where per-method "
        "results (<eval-root>/<exp_name>/) are written. Use a "
        "unique value per run to avoid overwriting results "
        "(e.g. seed/repeat sweeps).",
    )
    args = parser.parse_args()

    # Apply RANSAC seed override (-1 => None => non-deterministic).
    RANSAC_SEED = (
        None
        if args.ransac_seed is not None and args.ransac_seed < 0
        else args.ransac_seed
    )

    # Apply EVAL_ROOT override (keeps the hardcoded default when not supplied).
    if args.eval_root is not None:
        EVAL_ROOT = Path(args.eval_root)
        print(f"[CONFIG] EVAL_ROOT override: {EVAL_ROOT}")

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
    print(f"[CONFIG] RANSAC seed: {RANSAC_SEED}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")

    if not args.aggregate_only:
        # Load dataset once
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
        print(f"[DATA] Test set: {len(val_dataset)} frames")

        # Scene range slicing for distributed eval
        if args.scene_start is not None or args.scene_end is not None:
            all_scenes = val_dataset.scene_ids
            s = args.scene_start or 0
            e = args.scene_end or len(all_scenes)
            subset_scenes = set(all_scenes[s:e])
            val_dataset.valid_pairs = [
                p
                for p in val_dataset.valid_pairs
                if p[0].split("/")[-4] in subset_scenes
            ]
            val_dataset.scene_ids = [
                sid for sid in all_scenes if sid in subset_scenes
            ]
            print(
                f"[DATA] Scene range [{s}:{e}] → "
                f"{len(subset_scenes)} scenes, {len(val_dataset)} frames"
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Derive shard_id for distributed eval — any scene-range slicing makes
        # this a partial run that must not overwrite the full results.csv
        shard_id = None
        if args.scene_start is not None or args.scene_end is not None:
            shard_id = args.scene_start or 0

        # Evaluate each method
        for method_key in methods_to_eval:
            method_config = METHODS[method_key]
            evaluate_method(
                method_key,
                method_config,
                val_dataset,
                val_loader,
                shard_id=shard_id,
            )

    # Aggregate results (merge shards if needed)
    # Skip aggregation when running as a shard job — let the dedicated
    # --aggregate-only job handle it
    if args.scene_start is None and args.scene_end is None:
        aggregate_results(methods_to_eval, Path(args.output_dir))

    print("\n[DONE] All evaluations complete!")


if __name__ == "__main__":
    main()
