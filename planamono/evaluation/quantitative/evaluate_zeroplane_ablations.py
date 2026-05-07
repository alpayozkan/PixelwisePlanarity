"""
Evaluation script for ZeroPlane ablation experiments on ScanNet++.

Auto-discovers ablation variants from the nested H5 directory structure:
    H5_ROOT/{exp_dir}/model_{ckpt:07d}/{thresh_label}/{scene_id}/planes.h5

Evaluates each variant using the same pipeline as evaluate_all_baselines.py
(threshold-consistent RANSAC via evaluate_single_frame) and produces:
1. Per-variant CSV results (results.csv, results_per_scene.csv, results_dataset.csv)
2. Per-experiment summary tables
3. Combined table across all experiments

Usage:
    python evaluate_zeroplane_ablations.py                          # Evaluate all discovered variants
    python evaluate_zeroplane_ablations.py --experiments mixed_dust3r  # Specific experiments only
    python evaluate_zeroplane_ablations.py --variant mixed_dust3r/model_0024999/thresh_default  # Single variant (for SLURM jobs)
    python evaluate_zeroplane_ablations.py --aggregate-only          # Only aggregate existing results
    python evaluate_zeroplane_ablations.py --max-scenes 2            # Limit scenes (for testing)
"""

import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame,
)

# Import LazyH5SceneLoader from evaluate_all_baselines
from planamono.evaluation.quantitative.evaluate_all_baselines import LazyH5SceneLoader


# ============================================================
# CONFIGURATION
# ============================================================

COMPUTE_PLANE_METRICS = True
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9
THRESHOLDS = (0.001, 0.005, 0.01)
BATCH_SIZE = 32
N_JOBS = min(16, os.cpu_count())

# Paths
EVAL_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval/zp_ablations")
H5_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference")
DATASET_DIR = scannetpp_rend_plane_path

# ZeroPlane uses label 20 for non-planar regions
NONPLANAR_LABEL = 20


# ============================================================
# ABLATION VARIANT DISCOVERY
# ============================================================

@dataclass
class AblationVariant:
    """Represents a single ablation variant (experiment + checkpoint + threshold)."""
    exp_dir: str
    model_name: str
    thresh_label: str
    h5_path: Path  # Full path to the thresh directory containing scene subdirs

    @property
    def display_name(self) -> str:
        return f"{self.exp_dir}/{self.model_name}/{self.thresh_label}"

    @property
    def eval_subdir(self) -> str:
        """Relative path under EVAL_ROOT for results."""
        return os.path.join(self.exp_dir, self.model_name, self.thresh_label)


def discover_variants(
    h5_root: Path,
    experiments: Optional[List[str]] = None,
) -> List[AblationVariant]:
    """
    Auto-discover ablation variants by scanning H5_ROOT for nested directories.

    Looks for pattern: {exp_dir}/model_*/thresh_*/ containing scene subdirs with planes.h5.

    Args:
        h5_root: Root directory to scan
        experiments: If specified, only discover variants for these experiment names

    Returns:
        List of AblationVariant sorted by (exp_dir, model_name, thresh_label)
    """
    variants = []

    for exp_dir in sorted(h5_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        # Skip flat baseline directories (no model_* subdirs)
        if not any(d.name.startswith("model_") for d in exp_dir.iterdir() if d.is_dir()):
            continue

        exp_name = exp_dir.name
        if experiments and exp_name not in experiments:
            continue

        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir() or not model_dir.name.startswith("model_"):
                continue

            for thresh_dir in sorted(model_dir.iterdir()):
                if not thresh_dir.is_dir() or not thresh_dir.name.startswith("thresh_"):
                    continue

                # Verify at least one scene has planes.h5
                has_scenes = any(
                    (thresh_dir / scene / "planes.h5").exists()
                    for scene in os.listdir(thresh_dir)
                    if (thresh_dir / scene).is_dir()
                )
                if not has_scenes:
                    continue

                variants.append(AblationVariant(
                    exp_dir=exp_name,
                    model_name=model_dir.name,
                    thresh_label=thresh_dir.name,
                    h5_path=thresh_dir,
                ))

    return variants


# ============================================================
# EVALUATION
# ============================================================

def evaluate_variant(
    variant: AblationVariant,
    val_dataset: ScanNetPPPlaneDataset,
    val_loader: DataLoader,
) -> Dict:
    """
    Evaluate a single ablation variant.

    Returns:
        results: {(scene_id, frame_id): metrics_dict}
    """
    csv_out_dir = EVAL_ROOT / variant.eval_subdir

    print(f"\n{'='*60}")
    print(f"Evaluating: {variant.display_name}")
    print(f"H5 path:    {variant.h5_path}")
    print(f"Output:     {csv_out_dir}")
    print(f"{'='*60}")

    # Initialize lazy loader
    loader = LazyH5SceneLoader(
        str(variant.h5_path),
        label_offset=0,
        nonplanar_label=NONPLANAR_LABEL,
    )

    # Check available scenes
    scene_ids = val_dataset.scene_ids
    available_scenes = [s for s in scene_ids if loader.has_scene(s)]
    missing_scenes = set(scene_ids) - set(available_scenes)
    if missing_scenes:
        print(f"[WARN] Missing predictions for {len(missing_scenes)} scenes: {sorted(missing_scenes)}")
    print(f"[DATA] Found predictions for {len(available_scenes)}/{len(scene_ids)} scenes")

    if len(available_scenes) == 0:
        print(f"[ERROR] No predictions found for {variant.display_name}")
        return {}

    timer = Timer()

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
            inlier_ratio_gate=INLIER_RATIO_GATE,
        )

    results = {}
    skipped_frames = 0

    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc=f"Eval {variant.display_name}"):
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
                    THRESHOLDS,
                )
                for item in batch_items
            )

            for (metrics, _labels), item in zip(outputs, batch_items):
                results[(item["scene_id"], item["frame_idx"])] = metrics

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

def load_variant_results(variant: AblationVariant) -> Optional[pd.Series]:
    """Load dataset-level results for a variant, or None if not available."""
    csv_path = EVAL_ROOT / variant.eval_subdir / "results_dataset.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path).iloc[0]
    except Exception as e:
        print(f"[ERROR] Could not read {csv_path}: {e}")
        return None


def build_row(variant: AblationVariant, df_row: pd.Series) -> Dict:
    """Build a summary row dict from variant metadata and dataset results."""
    row = {
        "Experiment": variant.exp_dir,
        "Checkpoint": variant.model_name,
        "Threshold": variant.thresh_label,
        "num_scenes": int(df_row["num_scenes"]),
        "num_frames": int(df_row["num_frames_total"]),
    }

    # Segmentation metrics
    for col, display in [("rand_index_mean", "RI"),
                          ("voi_mean", "VOI"),
                          ("sc_mean", "SC")]:
        if col in df_row.index:
            row[display] = df_row[col]

    # Precision/recall metrics
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        prec_col = f"prec@{thresh_str}_mean"
        rec_col = f"rec@{thresh_str}_mean"
        if prec_col in df_row.index:
            row[f"P@{thresh_str}"] = df_row[prec_col]
        if rec_col in df_row.index:
            row[f"R@{thresh_str}"] = df_row[rec_col]

    return row


def aggregate_results(variants: List[AblationVariant]):
    """
    Aggregate results from all evaluated variants into summary tables.

    Produces:
    1. Per-experiment tables: table_{exp_dir}.csv
    2. Combined table: table_all_ablations.csv
    """
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    all_rows = []
    for variant in variants:
        df_row = load_variant_results(variant)
        if df_row is None:
            print(f"[WARN] Missing results for {variant.display_name}")
            continue
        all_rows.append(build_row(variant, df_row))
        print(f"[OK] Loaded results for {variant.display_name}")

    if not all_rows:
        print("[ERROR] No results to aggregate")
        return

    df_all = pd.DataFrame(all_rows)

    # Define column order
    metric_cols = ["RI", "VOI", "SC"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        metric_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}"])
    all_cols = ["Experiment", "Checkpoint", "Threshold", "num_scenes", "num_frames"] + metric_cols
    all_cols = [c for c in all_cols if c in df_all.columns]

    # Per-experiment tables
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    for exp_name, exp_df in df_all.groupby("Experiment"):
        exp_df_sorted = exp_df.sort_values(["Checkpoint", "Threshold"])
        exp_cols = [c for c in all_cols if c != "Experiment"]
        exp_table = exp_df_sorted[[c for c in exp_cols if c in exp_df_sorted.columns]]

        out_path = EVAL_ROOT / exp_name / f"table_{exp_name}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exp_table.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        # Print per-experiment table
        print(f"\n--- {exp_name} ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(exp_table.to_string(index=False))

    # Combined table
    df_combined = df_all[all_cols].sort_values(["Experiment", "Checkpoint", "Threshold"])
    out_csv = EVAL_ROOT / "table_all_ablations.csv"
    df_combined.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Also save as Excel if openpyxl is available
    try:
        out_xlsx = EVAL_ROOT / "table_all_ablations.xlsx"
        df_combined.to_excel(out_xlsx, index=False)
        print(f"Saved: {out_xlsx}")
    except ImportError:
        pass

    # Print combined summary
    print(f"\n{'='*100}")
    print("ALL ABLATION RESULTS")
    print(f"{'='*100}")
    print(df_combined.to_string(index=False))
    print(f"{'='*100}")

    return df_combined


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ZeroPlane ablation experiments")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Only evaluate these experiments (default: all discovered)")
    parser.add_argument("--variant", type=str, default=None,
                        help="Evaluate a single variant: 'exp_dir/model_NNNNNNN/thresh_X'")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, skip evaluation")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum number of scenes to evaluate (for testing)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    args = parser.parse_args()

    # Discover all variants (needed for aggregation and filtering)
    print("[DISCOVERY] Scanning for ablation variants...")
    all_variants = discover_variants(H5_ROOT, args.experiments)

    if not all_variants:
        print("[ERROR] No ablation variants found!")
        print(f"  Looked in: {H5_ROOT}")
        print(f"  Expected pattern: {{exp_dir}}/model_*/thresh_*/{{scene_id}}/planes.h5")
        return

    # Filter to single variant if --variant specified
    if args.variant:
        parts = args.variant.strip("/").split("/")
        if len(parts) != 3:
            print(f"[ERROR] --variant must be 'exp_dir/model_name/thresh_label', got: {args.variant}")
            return
        exp_dir, model_name, thresh_label = parts
        variants = [
            v for v in all_variants
            if v.exp_dir == exp_dir and v.model_name == model_name and v.thresh_label == thresh_label
        ]
        if not variants:
            print(f"[ERROR] Variant not found: {args.variant}")
            print(f"  Available: {[v.display_name for v in all_variants]}")
            return
    else:
        variants = all_variants

    print(f"[DISCOVERY] Found {len(all_variants)} total variants, evaluating {len(variants)}:")
    for v in variants:
        print(f"  - {v.display_name}")

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
        print(f"[DATA] Validation set: {len(val_dataset)} frames")

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Evaluate each variant
        for variant in variants:
            evaluate_variant(variant, val_dataset, val_loader)

    # Aggregate all discovered variants (not just the one we evaluated)
    aggregate_results(all_variants)

    print("\n[DONE] All ablation evaluations complete!")


if __name__ == "__main__":
    main()
