#!/usr/bin/env python3
"""
Split-based evaluation for Hypersim dataset.

Evaluates a subset of scenes (from --scene-list file) and saves results to
a part-specific output directory. After all parts finish, use --merge to
combine partial results into the final format.

Usage:
    # Worker mode: evaluate scenes from a scene list file
    python evaluate_hypersim_gt_split.py --methods gt --scene-list /tmp/scenes_part01.txt --part-id 01

    # Merge mode: combine all partial results
    python evaluate_hypersim_gt_split.py --methods gt --merge --num-parts 15

    # Multi-gate variant
    python evaluate_hypersim_gt_split.py --methods gt --scene-list /tmp/scenes_part01.txt --part-id 01 --inlier-gates 0.5 0.7 0.8 0.9
    python evaluate_hypersim_gt_split.py --methods gt --merge --num-parts 15 --inlier-gates 0.5 0.7 0.8 0.9
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm

from planamono.shared.datasets.hypersim_plane_dataset import HypersimPlaneDataset
from planamono.paths import repo_path

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame_hypersim,
    evaluate_single_frame_hypersim_multigates,
)

# Import shared config from the main evaluation script
from evaluate_hypersim_all_baselines import (
    COMPUTE_PLANE_METRICS,
    RANSAC_ITERATIONS,
    INLIER_RATIO_GATE,
    THRESHOLDS,
    N_JOBS,
    EVAL_ROOT,
    H5_ROOT,
    HYPERSIM_ROOT,
    PLANE_LABEL_ROOT,
    PARAMS_ROOT,
    EXP_VER,
    METHODS,
    LazyH5SceneLoader,
)


# ============================================================
# WORKER: EVALUATE A SCENE SUBSET
# ============================================================

def evaluate_method_split(
    method_key: str,
    method_config: Dict,
    scene_list: list,
    part_id: str,
    split: str = "test",
    inlier_gates: Optional[Tuple[float, ...]] = None,
):
    """Evaluate a single method on a subset of scenes."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {method_config['display_name']} (part {part_id})")
    print(f"Scenes: {len(scene_list)} ({scene_list[0]}...{scene_list[-1]})")
    if inlier_gates:
        print(f"Inlier gates: {inlier_gates}")
    print(f"{'='*60}")

    # Part-specific output directory
    exp_name = method_config["exp_name"]
    if inlier_gates:
        output_dir = EVAL_ROOT / f"{exp_name}_multigate_part{part_id}"
    else:
        output_dir = EVAL_ROOT / f"{exp_name}_part{part_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load full dataset
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
        use_raycasted_depth="euclidean",
    )

    # Build index: which dataset indices belong to our scene subset
    scene_set = set(scene_list)
    my_indices = []
    for idx in range(len(dataset)):
        pair = dataset.valid_pairs[idx]
        scene_id = pair[0]  # First element is scene_id
        if scene_id in scene_set:
            my_indices.append(idx)

    print(f"[INFO] Dataset: {len(dataset)} total, {len(my_indices)} in this part ({len(scene_set)} scenes)")

    if len(my_indices) == 0:
        print("[WARN] No frames found for the given scene list. Exiting.")
        return {}

    # Initialize H5 loader
    h5_loader = None
    if not method_config.get("uses_gt_h5", False) and method_config["h5_folder"] is not None:
        h5_root = H5_ROOT / method_config["h5_folder"]
        h5_loader = LazyH5SceneLoader(
            str(h5_root),
            label_offset=method_config["label_offset"],
            nonplanar_label=method_config.get("nonplanar_label"),
        )

    def eval_frame(idx):
        sample = dataset[idx]
        scene_id = sample["scene_id"]
        frame_id = sample["frame_idx"]

        rgb_path = sample["rgb_path"]
        cam_name = rgb_path.split('/')[1] if '/' in rgb_path else "cam_00"
        full_frame_id = f"{cam_name}/{frame_id}"

        # Load prediction or use GT
        if method_config.get("uses_gt_h5", False):
            gt_seg = sample["plane"].numpy().astype(np.int32)
            if gt_seg.ndim == 3:
                gt_seg = gt_seg[0]
            labels = gt_seg.copy()
        else:
            labels = h5_loader.get_pred_seg(scene_id, cam_name, frame_id)
            if labels is None:
                return None

        gt_seg = sample["plane"].numpy().astype(np.int32)
        if gt_seg.ndim == 3:
            gt_seg = gt_seg[0]

        depth_euc = sample["depth"].numpy()
        if depth_euc.ndim == 3:
            depth_euc = depth_euc[0]

        c2w = sample["c2w"].numpy()

        M_cam = dataset._get_M_cam_from_uv(scene_id)
        native_wh = dataset.valid_pairs[idx][-1]

        if labels.shape != gt_seg.shape:
            labels = cv2.resize(
                labels.astype(np.uint16),
                (gt_seg.shape[1], gt_seg.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        if inlier_gates:
            metrics, _ = evaluate_single_frame_hypersim_multigates(
                scene_id, full_frame_id, depth_euc, gt_seg,
                M_cam, native_wh, c2w, labels, THRESHOLDS,
                inlier_ratio_gates=inlier_gates,
                compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
                ransac_iterations=RANSAC_ITERATIONS,
            )
        else:
            metrics, _ = evaluate_single_frame_hypersim(
                scene_id, full_frame_id, depth_euc, gt_seg,
                M_cam, native_wh, c2w, labels, THRESHOLDS,
                compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
                ransac_iterations=RANSAC_ITERATIONS,
                inlier_ratio_gate=INLIER_RATIO_GATE,
            )

        return (scene_id, full_frame_id), metrics

    # Run evaluation in parallel
    print("\n[INFO] Running evaluation...")
    with timer("evaluation"):
        outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(eval_frame)(idx) for idx in tqdm(my_indices, desc=f"Part {part_id}")
        )

    results = {}
    skipped = 0
    for output in outputs:
        if output is None:
            skipped += 1
            continue
        (scene_id, frame_id), metrics = output
        results[(scene_id, frame_id)] = metrics

    print(f"\n[INFO] Part {part_id}: {len(results)} frames ({skipped} skipped)")

    # Save partial results
    print("\n[INFO] Saving partial results...")
    save_results_csv(results, str(output_dir))
    save_runtime(timer, str(output_dir))

    timer.print_summary(num_frames=len(results))
    print(f"\n[DONE] Part {part_id} saved to: {output_dir}")

    return results


# ============================================================
# MERGE: COMBINE PARTIAL RESULTS
# ============================================================

def merge_partial_results(
    method_key: str,
    method_config: Dict,
    num_parts: int,
    inlier_gates: Optional[Tuple[float, ...]] = None,
):
    """Merge partial results from all parts into the final format."""
    exp_name = method_config["exp_name"]
    suffix = "_multigate" if inlier_gates else ""

    print(f"\n{'='*60}")
    print(f"Merging: {method_config['display_name']} ({num_parts} parts)")
    print(f"{'='*60}")

    # Collect all partial results.csv files
    all_frames = []
    found_parts = 0
    for part_idx in range(1, num_parts + 1):
        part_id = f"{part_idx:02d}"
        part_dir = EVAL_ROOT / f"{exp_name}{suffix}_part{part_id}"
        csv_path = part_dir / "results.csv"

        if not csv_path.exists():
            print(f"[WARN] Missing part {part_id}: {csv_path}")
            continue

        df_part = pd.read_csv(csv_path)
        all_frames.append(df_part)
        found_parts += 1
        print(f"[OK] Part {part_id}: {len(df_part)} frames")

    if not all_frames:
        print("[ERROR] No partial results found!")
        return

    print(f"\n[INFO] Found {found_parts}/{num_parts} parts")

    # Concatenate all frames
    df_all = pd.concat(all_frames, ignore_index=True)

    # Check for duplicates
    dup_mask = df_all.duplicated(subset=["scene_id", "frame_idx"], keep="first")
    if dup_mask.any():
        n_dup = dup_mask.sum()
        print(f"[WARN] Removing {n_dup} duplicate frames")
        df_all = df_all[~dup_mask]

    print(f"[INFO] Total: {len(df_all)} frames")

    # Save merged results
    if inlier_gates:
        output_dir = EVAL_ROOT / f"{exp_name}_multigate"
    else:
        output_dir = EVAL_ROOT / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-frame results
    df_all.to_csv(output_dir / "results.csv", index=False)
    print(f"[CSV] Saved per-frame: {output_dir / 'results.csv'}")

    # 2. Per-scene aggregation
    scene_group = df_all.groupby("scene_id")
    df_scene = scene_group.mean(numeric_only=True)
    df_scene["num_frames"] = scene_group.size()
    cols = ["num_frames"] + [c for c in df_scene.columns if c != "num_frames"]
    df_scene = df_scene[cols]
    df_scene.to_csv(output_dir / "results_per_scene.csv")
    print(f"[CSV] Saved per-scene: {output_dir / 'results_per_scene.csv'}")

    # 3. Dataset-level summary
    dataset_stats = {
        "num_scenes": len(df_scene),
        "num_frames_total": int(df_scene["num_frames"].sum()),
    }
    numeric_cols = df_scene.select_dtypes(include="number").columns
    metric_cols = [c for c in numeric_cols if c != "num_frames"]
    for c in metric_cols:
        dataset_stats[f"{c}_mean"] = df_scene[c].mean()
        dataset_stats[f"{c}_std"] = df_scene[c].std()

    df_dataset = pd.DataFrame([dataset_stats])
    df_dataset.to_csv(output_dir / "results_dataset.csv", index=False)
    print(f"[CSV] Saved dataset summary: {output_dir / 'results_dataset.csv'}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"MERGED RESULTS: {method_config['display_name']}")
    print(f"{'='*60}")
    print(f"Scenes: {dataset_stats['num_scenes']}, Frames: {dataset_stats['num_frames_total']}")

    if inlier_gates:
        for thr in THRESHOLDS:
            thresh_str = f"{thr*100:.1f}cm"
            for gate in inlier_gates:
                p_col = f"prec@{thresh_str}_gate{gate}_mean"
                r_col = f"rec@{thresh_str}_gate{gate}_mean"
                if p_col in dataset_stats:
                    print(f"  @ {thresh_str} gate={gate}: P={dataset_stats[p_col]:.4f}  R={dataset_stats[r_col]:.4f}")
    else:
        for thr in THRESHOLDS:
            thresh_str = f"{thr*100:.1f}cm"
            p_col = f"prec@{thresh_str}_mean"
            r_col = f"rec@{thresh_str}_mean"
            if p_col in dataset_stats:
                print(f"  @ {thresh_str}: P={dataset_stats[p_col]:.4f}  R={dataset_stats[r_col]:.4f}")

    for col in ["sc_mean", "rand_index_mean", "voi_mean"]:
        if col in dataset_stats:
            print(f"  {col}: {dataset_stats[col]:.4f}")

    print(f"\n[DONE] Merged results saved to: {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split-based Hypersim evaluation (worker + merge)"
    )

    parser.add_argument("--methods", nargs="+", required=True,
                        help="Methods to evaluate (e.g., gt moge_ours)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")

    # Worker mode
    parser.add_argument("--scene-list", type=str, default=None,
                        help="Path to file with scene IDs (one per line)")
    parser.add_argument("--part-id", type=str, default="01",
                        help="Part identifier (e.g., 01, 02, ...)")

    # Merge mode
    parser.add_argument("--merge", action="store_true",
                        help="Merge partial results instead of evaluating")
    parser.add_argument("--num-parts", type=int, default=15,
                        help="Number of parts to merge (default: 15)")

    # Multi-gate support
    parser.add_argument("--inlier-gates", nargs="+", type=float, default=None,
                        help="Evaluate at multiple inlier ratio gates")

    args = parser.parse_args()

    inlier_gates = tuple(sorted(args.inlier_gates)) if args.inlier_gates else None

    if args.merge:
        # Merge mode
        for method_key in args.methods:
            if method_key not in METHODS:
                print(f"[ERROR] Unknown method: {method_key}")
                continue
            merge_partial_results(
                method_key, METHODS[method_key],
                args.num_parts, inlier_gates
            )
    else:
        # Worker mode
        if args.scene_list is None:
            parser.error("--scene-list is required in worker mode (use --merge for merge mode)")

        # Read scene list
        with open(args.scene_list) as f:
            scene_list = [line.strip() for line in f if line.strip()]

        if not scene_list:
            print("[ERROR] Empty scene list!")
            return

        print(f"[CONFIG] Part ID: {args.part_id}")
        print(f"[CONFIG] Scene list: {args.scene_list} ({len(scene_list)} scenes)")

        for method_key in args.methods:
            if method_key not in METHODS:
                print(f"[ERROR] Unknown method: {method_key}")
                continue
            evaluate_method_split(
                method_key, METHODS[method_key],
                scene_list, args.part_id,
                split=args.split,
                inlier_gates=inlier_gates,
            )


if __name__ == "__main__":
    main()
