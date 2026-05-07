"""
Evaluate ZeroPlane variants on ScanNet++ validation split.

Auto-discovers all {method}/{checkpoint}/thresh_0.5/ combos under inference_val/
and evaluates each. All methods use ZeroPlane convention (nonplanar_label=20).

Results are saved to eval_val/{method}__{checkpoint}/.

Usage:
    python evaluate_scannetpp_val.py                                          # All methods
    python evaluate_scannetpp_val.py --methods default_dust3r_released/model_0000000
    python evaluate_scannetpp_val.py --max-scenes 2                           # Quick test
    python evaluate_scannetpp_val.py --aggregate-only                         # Only aggregate
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
from typing import Dict, Optional, Tuple, List

from joblib import Parallel, delayed

from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset
from planamono.paths import repo_path, scannetpp_path, scannetpp_rend_plane_path

from planamono.evaluation.quantitative.eval_utils import (
    Timer,
    save_results_csv,
    save_runtime,
    evaluate_single_frame,
)


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
INFERENCE_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/inference_val")
EVAL_ROOT = Path("/cluster/scratch/aoezkan/planeseg/scannetpp/eval_val")
DATASET_DIR = scannetpp_rend_plane_path

EXP_VER = "v5"


# ============================================================
# AUTO-DISCOVERY
# ============================================================

def discover_methods(inference_root: Path) -> List[Dict]:
    """
    Auto-discover all {method}/{checkpoint}/thresh_0.5/ combos.

    Returns list of dicts with keys: method, checkpoint, h5_root, display_name, exp_name
    """
    methods = []
    for method_dir in sorted(inference_root.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
        for ckpt_dir in sorted(method_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            checkpoint = ckpt_dir.name
            thresh_dir = ckpt_dir / "thresh_0.5"
            if not thresh_dir.is_dir():
                continue
            # Check it has at least one scene
            scene_dirs = [d for d in thresh_dir.iterdir() if d.is_dir()]
            if not scene_dirs:
                continue

            key = f"{method_name}/{checkpoint}"
            methods.append({
                "key": key,
                "method": method_name,
                "checkpoint": checkpoint,
                "h5_root": str(thresh_dir),
                "display_name": f"{method_name} ({checkpoint})",
                "exp_name": f"{method_name}__{checkpoint}_{EXP_VER}",
            })

    return methods


# ============================================================
# H5 LOADING (reused from evaluate_all_baselines.py)
# ============================================================

class LazyH5SceneLoader:
    """Memory-efficient loader that only keeps one scene in memory at a time."""

    def __init__(self, h5_root: str, nonplanar_label: Optional[int] = None):
        self.h5_root = h5_root
        self.nonplanar_label = nonplanar_label
        self._current_scene_id = None
        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

    def _load_scene(self, scene_id: str) -> bool:
        if scene_id == self._current_scene_id:
            return True

        h5_path = os.path.join(self.h5_root, scene_id, "planes.h5")
        if not os.path.exists(h5_path):
            return False

        self._current_planes = None
        self._current_frame_ids = None
        self._frame_id_to_idx = {}

        with h5py.File(h5_path, "r") as f:
            self._current_planes = f["planes"][:]
            self._current_frame_ids = [
                fid.decode() if isinstance(fid, bytes) else fid
                for fid in f["frame_ids"][:]
            ]

        self._frame_id_to_idx = {fid: i for i, fid in enumerate(self._current_frame_ids)}
        self._current_scene_id = scene_id
        return True

    def get_prediction(self, scene_id: str, frame_idx: str, target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self._load_scene(scene_id):
            return None

        if frame_idx not in self._frame_id_to_idx:
            return None

        idx = self._frame_id_to_idx[frame_idx]
        pred = self._current_planes[idx].copy()

        if pred.shape != target_shape:
            pred = cv2.resize(
                pred.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        if self.nonplanar_label is not None:
            pred[pred == self.nonplanar_label] = 0

        return pred.astype(np.int32)

    def has_scene(self, scene_id: str) -> bool:
        h5_path = os.path.join(self.h5_root, scene_id, "planes.h5")
        return os.path.exists(h5_path)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(
    method_info: Dict,
    val_dataset: ScanNetPPPlaneDataset,
    val_loader: DataLoader,
) -> Dict:
    """Evaluate a single method and save results."""
    h5_root = method_info["h5_root"]
    exp_name = method_info["exp_name"]
    csv_out_dir = EVAL_ROOT / exp_name

    print(f"\n{'='*60}")
    print(f"Evaluating: {method_info['display_name']}")
    print(f"H5 root: {h5_root}")
    print(f"Output: {csv_out_dir}")
    print(f"{'='*60}")

    timer = Timer()

    # All ZeroPlane methods use label 20 for non-planar
    loader = LazyH5SceneLoader(h5_root, nonplanar_label=20)

    # Check available scenes
    scene_ids = val_dataset.scene_ids
    available_scenes = [s for s in scene_ids if loader.has_scene(s)]
    missing_scenes = set(scene_ids) - set(available_scenes)
    if missing_scenes:
        print(f"[WARN] Missing predictions for {len(missing_scenes)} scenes: {sorted(missing_scenes)}")
    print(f"[DATA] Found predictions for {len(available_scenes)}/{len(scene_ids)} scenes")

    if len(available_scenes) == 0:
        print(f"[ERROR] No predictions found")
        return {}

    def eval_frame_wrapper(scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, labels, thresholds):
        return evaluate_single_frame(
            scene_id, frame_idx, depth_np, gt_seg_np, K_np, c2w_np, labels, thresholds,
            compute_plane_metrics_flag=COMPUTE_PLANE_METRICS,
            ransac_iterations=RANSAC_ITERATIONS,
            inlier_ratio_gate=INLIER_RATIO_GATE
        )

    results = {}
    skipped_frames = 0

    with timer("evaluation_pipeline"):
        for batch in tqdm(val_loader, desc=f"Evaluating {method_info['key']}"):
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

                gt_seg = gt_planes[i]
                if gt_seg.ndim == 3:
                    gt_seg = gt_seg[0]
                gt_seg_np = gt_seg.cpu().numpy().astype(np.int32)
                H, W = gt_seg_np.shape

                depth = depths[i]
                depth_np = depth[0].cpu().numpy() if depth.ndim == 3 else depth.cpu().numpy()

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

            outputs = Parallel(n_jobs=N_JOBS, backend="loky")(
                delayed(eval_frame_wrapper)(
                    item["scene_id"], item["frame_idx"],
                    item["depth_np"], item["gt_seg_np"],
                    item["K_np"], item["c2w_np"],
                    item["labels"], THRESHOLDS
                )
                for item in batch_items
            )

            for (metrics, _labels), item in zip(outputs, batch_items):
                results[(item["scene_id"], item["frame_idx"])] = metrics

    print(f"[PIPELINE] Evaluated {len(results)} frames (skipped {skipped_frames})")

    if results:
        print("==> Saving results")
        save_results_csv(results, str(csv_out_dir))
        save_runtime(timer, str(csv_out_dir))
        timer.print_summary(num_frames=len(results))

    return results


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_results(method_infos: List[Dict], output_dir: Path = None):
    """Aggregate results from all methods into summary tables."""
    if output_dir is None:
        output_dir = EVAL_ROOT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    all_results = []
    for info in method_infos:
        exp_name = info["exp_name"]
        csv_path = EVAL_ROOT / exp_name / "results_dataset.csv"

        if not csv_path.exists():
            print(f"[WARN] Missing results for {info['key']}: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path).iloc[0]
            row = {
                "Method": info["display_name"],
                "method_key": info["key"],
                "num_scenes": int(df["num_scenes"]),
                "num_frames": int(df["num_frames_total"]),
            }

            for col, display in [("rand_index_mean", "RI"),
                                  ("voi_mean", "VOI"),
                                  ("sc_mean", "SC")]:
                if col in df.index:
                    row[display] = df[col]

            for thr in THRESHOLDS:
                thresh_str = f"{thr*100:.1f}cm"
                prec_col = f"prec@{thresh_str}_mean"
                rec_col = f"rec@{thresh_str}_mean"
                if prec_col in df.index:
                    row[f"P@{thresh_str}"] = df[prec_col]
                if rec_col in df.index:
                    row[f"R@{thresh_str}"] = df[rec_col]

            all_results.append(row)
            print(f"[OK] Loaded results for {info['display_name']}")

        except Exception as e:
            print(f"[ERROR] Could not read results for {info['key']}: {e}")

    if not all_results:
        print("[ERROR] No results to aggregate")
        return

    df_all = pd.DataFrame(all_results)

    # Combined summary table
    combined_cols = ["Method", "num_scenes", "num_frames", "RI", "VOI", "SC"]
    for thr in THRESHOLDS:
        thresh_str = f"{thr*100:.1f}cm"
        combined_cols.extend([f"P@{thresh_str}", f"R@{thresh_str}"])
    df_combined = df_all[[c for c in combined_cols if c in df_all.columns]]

    out_path = output_dir / "table_combined_val.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print summary
    print("\n" + "=" * 120)
    print("VAL RESULTS SUMMARY")
    print("=" * 120)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 40)
    print(df_combined.to_string(index=False))
    print("=" * 120)

    return df_all


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ZeroPlane variants on ScanNet++ val split")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Specific method/checkpoint combos (e.g. default_dust3r_released/model_0000000)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only aggregate existing results, skip evaluation")
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum number of scenes to evaluate (for testing)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of DataLoader workers")
    args = parser.parse_args()

    # Discover all methods
    all_methods = discover_methods(INFERENCE_ROOT)
    print(f"[DISCOVERY] Found {len(all_methods)} method+checkpoint combos:")
    for m in all_methods:
        print(f"  - {m['key']}")

    # Filter to requested methods
    if args.methods is not None:
        filtered = [m for m in all_methods if m["key"] in args.methods]
        invalid = set(args.methods) - {m["key"] for m in filtered}
        if invalid:
            print(f"[ERROR] Not found: {invalid}")
            print(f"[ERROR] Available: {[m['key'] for m in all_methods]}")
            return
        methods_to_eval = filtered
    else:
        methods_to_eval = all_methods

    print(f"\n[CONFIG] Methods to evaluate: {[m['key'] for m in methods_to_eval]}")
    print(f"[CONFIG] Max scenes: {args.max_scenes}")
    print(f"[CONFIG] RANSAC iterations: {RANSAC_ITERATIONS}")
    print(f"[CONFIG] Inlier ratio gate: {INLIER_RATIO_GATE}")
    print(f"[CONFIG] Thresholds: {THRESHOLDS}")

    if not args.aggregate_only:
        # Load dataset once (val split)
        print("\n==> Loading dataset (val split)")
        val_dataset = ScanNetPPPlaneDataset(
            rgb_root=os.path.join(scannetpp_path, "data"),
            plane_label_root=scannetpp_rend_plane_path,
            sem_label_root=os.path.join(DATASET_DIR, ""),
            depth_label_root=scannetpp_rend_plane_path,
            split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
            split="val",
            max_scenes=args.max_scenes,
        )
        print(f"[DATA] Val set: {len(val_dataset)} frames from {len(val_dataset.scene_ids)} scenes")

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        for method_info in methods_to_eval:
            evaluate_method(method_info, val_dataset, val_loader)

    # Aggregate
    aggregate_results(methods_to_eval)

    print("\n[DONE] All evaluations complete!")


if __name__ == "__main__":
    main()
