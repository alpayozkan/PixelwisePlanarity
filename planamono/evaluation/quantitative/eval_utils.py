"""
Shared utilities for evaluation scripts.

Provides:
- Timing infrastructure for profiling
- Result saving utilities (CSV, H5)
- Common evaluation pipeline components
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


# ============================================================
# TIMING INFRASTRUCTURE
# ============================================================

class Timer:
    """
    Global timing infrastructure for profiling evaluation pipelines.

    Usage:
        timer = Timer()
        with timer("gpu_inference"):
            model.predict(...)
        timer.print_summary()
    """

    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.start_time = time.perf_counter()

    @contextmanager
    def __call__(self, name: str, verbose: bool = False):
        """Context manager for timing a code block."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.timings[name] += elapsed
        self.counts[name] += 1
        if verbose and not name.startswith("_"):
            print(f"[TIMER] {name:30s} {self.format_time(elapsed)}")

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def total_elapsed(self) -> float:
        """Get total wall time since timer creation."""
        return time.perf_counter() - self.start_time

    def print_summary(self, num_frames: int = 0):
        """Print timing summary."""
        total = self.total_elapsed()
        print("\n" + "=" * 60)
        print("RUNTIME BREAKDOWN (aggregated)")
        print("=" * 60)
        for k, v in sorted(self.timings.items(), key=lambda x: -x[1]):
            count = self.counts[k]
            avg = v / count if count > 0 else 0
            print(f"{k:25s} {self.format_time(v):>15s} ({count:>6d} calls, {avg*1000:>8.2f}ms avg)")
        print("-" * 60)
        print(f"{'TOTAL WALL TIME':25s} {self.format_time(total):>15s}")
        if num_frames > 0:
            print(f"{'Throughput':25s} {num_frames / total:>15.2f} fps")
        print("=" * 60)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert timings to DataFrame for saving."""
        rows = []
        for name, seconds in self.timings.items():
            rows.append({
                "stage": name,
                "time_seconds": seconds,
                "time_hms": self.format_time(seconds),
                "calls": self.counts[name],
                "avg_ms": (seconds / self.counts[name] * 1000) if self.counts[name] > 0 else 0
            })
        return pd.DataFrame(rows).sort_values(by="time_seconds", ascending=False)


# ============================================================
# RESULT SAVING UTILITIES
# ============================================================

def save_results_csv(
    results: Dict[Tuple[str, str], Dict],
    csv_out_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Save evaluation results to CSV files.

    Creates three files:
    - results.csv: Per-frame results
    - results_per_scene.csv: Per-scene aggregated results
    - results_dataset.csv: Dataset-level summary

    Args:
        results: {(scene_id, frame_idx): metrics_dict}
        csv_out_dir: Output directory

    Returns:
        (df_frames, df_scenes, df_dataset) DataFrames
    """
    os.makedirs(csv_out_dir, exist_ok=True)

    # Per-frame results
    df = pd.DataFrame.from_records(list(results.values()))
    df = df.set_index(["scene_id", "frame_idx"])

    out_path = os.path.join(csv_out_dir, 'results.csv')
    df.to_csv(out_path)
    print(f"[CSV] Saved per-frame results to {out_path}")

    # Per-scene results
    df_reset = df.reset_index()
    scene_group = df_reset.groupby("scene_id")
    df_scene = scene_group.mean(numeric_only=True)
    df_scene["num_frames"] = scene_group.size()
    cols = ["num_frames"] + [c for c in df_scene.columns if c != "num_frames"]
    df_scene = df_scene[cols]

    scene_csv = os.path.join(csv_out_dir, "results_per_scene.csv")
    df_scene.to_csv(scene_csv)
    print(f"[CSV] Saved per-scene results to {scene_csv}")

    # Dataset stats
    dataset_stats = {
        "num_scenes": len(df_scene),
        "num_frames_total": int(df_scene["num_frames"].sum())
    }
    numeric_cols = df_scene.select_dtypes(include="number").columns
    metric_cols = [c for c in numeric_cols if c != "num_frames"]
    for c in metric_cols:
        dataset_stats[f"{c}_mean"] = df_scene[c].mean()
        dataset_stats[f"{c}_std"] = df_scene[c].std()

    df_dataset = pd.DataFrame([dataset_stats])
    dataset_csv = os.path.join(csv_out_dir, "results_dataset.csv")
    df_dataset.to_csv(dataset_csv, index=False)
    print(f"[CSV] Saved dataset stats to {dataset_csv}")

    return df, df_scene, df_dataset


def save_predictions_h5(
    scene_predictions: Dict[str, List[Tuple[str, np.ndarray]]],
    h5_root: str
):
    """
    Save predictions to H5 files (one per scene).

    Args:
        scene_predictions: {scene_id: [(frame_id, labels), ...]}
        h5_root: Root directory for H5 files
    """
    os.makedirs(h5_root, exist_ok=True)

    for scene_id, frame_data in tqdm(scene_predictions.items(), desc="Writing H5"):
        frame_data.sort(key=lambda x: x[0])
        frame_ids_list = [fd[0] for fd in frame_data]
        planes = np.stack([fd[1] for fd in frame_data], axis=0).astype(np.uint16)

        scene_h5_dir = os.path.join(h5_root, scene_id)
        os.makedirs(scene_h5_dir, exist_ok=True)

        h5_path = os.path.join(scene_h5_dir, "planes.h5")
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("planes", data=planes, compression="gzip", compression_opts=4)
            f.create_dataset("frame_ids", data=np.array(frame_ids_list, dtype="S"))

    print(f"[H5] Written {len(scene_predictions)} scene files to {h5_root}")


def save_runtime(timer: Timer, csv_out_dir: str):
    """Save runtime breakdown to CSV."""
    df_runtime = timer.to_dataframe()
    runtime_path = os.path.join(csv_out_dir, "runtime_breakdown.csv")
    df_runtime.to_csv(runtime_path, index=False)
    print(f"[CSV] Saved runtime breakdown to {runtime_path}")


# ============================================================
# FRAME EVALUATION
# ============================================================

def evaluate_single_frame_old(
    scene_id: str,
    frame_idx: str,
    depth_np: np.ndarray,
    gt_seg_np: np.ndarray,
    K_np: np.ndarray,
    c2w_np: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    compute_plane_metrics_flag: bool = True,
    ransac_iterations: int = 200,
    inlier_ratio_gate: float = 0.5
) -> Tuple[Dict, np.ndarray]:
    """
    [DEPRECATED] Use evaluate_single_frame() instead (v1 version).

    Old version with fixed 2cm RANSAC. Kept for backward compatibility.

    Evaluate a single frame with pre-computed segmentation labels.

    Computes:
    - Clustering metrics: rand_index, voi, sc (image-to-image)
    - Plane fitting metrics: precision/recall at multiple thresholds

    Args:
        scene_id: Scene identifier
        frame_idx: Frame identifier
        depth_np: (H, W) depth map in meters
        gt_seg_np: (H, W) ground truth segmentation
        K_np: (3, 3) or (4, 4) camera intrinsics
        c2w_np: (4, 4) camera-to-world pose
        labels: (H, W) predicted segmentation labels
        thresholds: Tuple of distance thresholds for plane metrics (meters)
        compute_plane_metrics_flag: Whether to compute RANSAC plane metrics
        ransac_iterations: Number of RANSAC iterations
        inlier_ratio_gate: Minimum inlier ratio to count a segment as valid (default 0.5)

    Returns:
        (metrics_dict, labels) where metrics_dict contains all computed metrics
    """
    from planamono.shared.plane_fitting import backproject_v1 as backproject

    metric_thr = {}

    if compute_plane_metrics_flag:
        pts_world, pt_labels, _ = backproject(depth_np, K_np, c2w_np, labels)

        if pts_world.shape[0] == 0:
            metric_thr = {f"prec@{thr*100:.1f}cm": 0.0 for thr in thresholds}
            metric_thr.update({f"rec@{thr*100:.1f}cm": 0.0 for thr in thresholds})
        else:
            metric_thr = compute_plane_metrics_old(
                pts_world, pt_labels, thresholds,
                num_iterations=ransac_iterations,
                inlier_ratio_gate=inlier_ratio_gate
            )
    else:
        for thr in thresholds:
            metric_thr[f"prec@{thr*100:.1f}cm"] = np.nan
            metric_thr[f"rec@{thr*100:.1f}cm"] = np.nan

    # Clustering metrics (pure img-to-img)
    clustering = compute_clustering_metrics(gt_seg_np, labels)

    metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        **clustering,
        **metric_thr
    }

    return metrics, labels


# ============================================================
# EVALUATION HELPERS
# ============================================================

def compute_clustering_metrics(
    gt_seg: np.ndarray,
    pred_seg: np.ndarray
) -> Dict[str, float]:
    """
    Compute clustering metrics between GT and predicted segmentation.

    Metrics computed:
    - rand_index: Rand Index (higher is better)
    - voi: Variation of Information (lower is better)
    - sc: Segmentation Covering (higher is better)

    Note: These are pure image-to-image metrics. Pre-filter backgrounds
    before calling if needed.

    Args:
        gt_seg: (H, W) ground truth segmentation
        pred_seg: (H, W) predicted segmentation

    Returns:
        {"rand_index": float, "voi": float, "sc": float}
    """
    from sklearn.metrics import rand_score
    from skimage.metrics import variation_of_information
    from planamono.shared.plane_fitting import segmentation_covering_fast

    # Rand Index
    ri = rand_score(gt_seg.flatten(), pred_seg.flatten())

    # Variation of Information
    Hs, Hm = variation_of_information(gt_seg, pred_seg)
    voi = Hs + Hm

    # Segmentation Covering
    sc = segmentation_covering_fast(gt_seg, pred_seg)

    return {"rand_index": ri, "voi": voi, "sc": sc}


def compute_plane_metrics_old(
    pts_world: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    num_iterations: int = 200,
    min_support: int = 100,
    inlier_ratio_gate: float = 0.5
) -> Dict[str, float]:
    """
    [DEPRECATED] Use compute_plane_metrics() instead (v1 version).

    Old version with fixed 2cm RANSAC. Kept for backward compatibility.

    Compute plane fitting metrics at multiple thresholds.

    Args:
        pts_world: (N, 3) world coordinates
        labels: (N,) segment labels
        thresholds: Tuple of distance thresholds (meters)
        num_iterations: RANSAC iterations
        min_support: Minimum points for RANSAC
        inlier_ratio_gate: Minimum inlier ratio to count a segment as valid (default 0.5)

    Returns:
        {"prec@Xcm": float, "rec@Xcm": float} for each threshold X
    """
    from planamono.shared.plane_fitting import fit_planes_and_evaluate_multi_threshold

    multi_metrics = fit_planes_and_evaluate_multi_threshold(
        pts_world,
        labels,
        thresholds,
        base_threshold=0.02,
        num_iterations=num_iterations,
        min_support=min_support,
        inlier_ratio_gate=inlier_ratio_gate
    )

    result = {}
    for thr in thresholds:
        result[f"prec@{thr*100:.1f}cm"] = multi_metrics[thr]["precision"]
        result[f"rec@{thr*100:.1f}cm"] = multi_metrics[thr]["recall"]

    return result


def compute_plane_metrics(
    pts_world: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    num_iterations: int = 200,
    min_support: int = 100,
    inlier_ratio_gate: float = 0.5
) -> Dict[str, float]:
    """
    Compute plane fitting metrics at multiple thresholds (threshold-consistent).

    This is the RECOMMENDED version (formerly compute_plane_metrics_v1).

    DIFFERENCE FROM compute_plane_metrics_old:
    - compute_plane_metrics_old: Uses fixed base_threshold=0.02 (2cm) for RANSAC fitting,
      then evaluates inliers at each threshold. This means the same plane equation
      is used for all thresholds.
    - compute_plane_metrics (current): Uses the evaluation threshold as the RANSAC threshold.
      This means each threshold gets its own plane fit, and precision/recall reflect
      how well planes can be fit AND evaluated at that specific tolerance.

    This version measures:
    "What is the precision/recall when planes are fit at threshold X?"

    The old version measured:
    "Given a robustly-fit plane (at 2cm), how precise is it at stricter thresholds?"

    Args:
        pts_world: (N, 3) world coordinates
        labels: (N,) segment labels
        thresholds: Tuple of distance thresholds (meters)
        num_iterations: RANSAC iterations
        min_support: Minimum points for RANSAC
        inlier_ratio_gate: Minimum inlier ratio to count a segment as valid (default 0.5)

    Returns:
        {"prec@Xcm": float, "rec@Xcm": float} for each threshold X
    """
    from planamono.shared.plane_fitting import (
        fit_planes_per_label_v1,
        compute_inliers_at_threshold,
    )

    result = {}
    for thr in thresholds:
        # Fit planes with RANSAC at this threshold
        fit_results, df = fit_planes_per_label_v1(
            pts_world,
            labels,
            ignore_labels=(0,),
            distance_threshold=thr,  # Use evaluation threshold for RANSAC
            num_iterations=num_iterations,
            min_support=min_support
        )

        if df is None or len(df) == 0:
            result[f"prec@{thr*100:.1f}cm"] = 0.0
            result[f"rec@{thr*100:.1f}cm"] = 0.0
            continue

        # Extract fitted plane parameters
        plane_params = {}
        for pid, data in fit_results.items():
            if "plane_model_refined" in data:
                plane_params[pid] = data["plane_model_refined"]

        if not plane_params:
            result[f"prec@{thr*100:.1f}cm"] = 0.0
            result[f"rec@{thr*100:.1f}cm"] = 0.0
            continue

        # Evaluate at the same threshold
        metrics = compute_inliers_at_threshold(
            pts_world, labels, plane_params, thr, inlier_ratio_gate
        )

        result[f"prec@{thr*100:.1f}cm"] = metrics["precision"]
        result[f"rec@{thr*100:.1f}cm"] = metrics["recall"]

    return result


def compute_plane_metrics_multigates(
    pts_world: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    inlier_ratio_gates: Tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
    num_iterations: int = 200,
    min_support: int = 100,
) -> Dict[str, float]:
    """
    Compute plane fitting metrics at multiple thresholds AND multiple inlier ratio gates.

    Fits RANSAC once per threshold, then evaluates at each gate — the gate only
    affects which segments count as valid planes, so the plane fits are reused.

    Args:
        pts_world: (N, 3) world coordinates
        labels: (N,) segment labels
        thresholds: Tuple of distance thresholds (meters)
        inlier_ratio_gates: Tuple of inlier ratio gates to evaluate
        num_iterations: RANSAC iterations
        min_support: Minimum points for RANSAC

    Returns:
        {"prec@Xcm_gateY": float, "rec@Xcm_gateY": float} for each (threshold, gate)
    """
    from planamono.shared.plane_fitting import (
        fit_planes_per_label_v1,
        compute_inliers_at_threshold,
    )

    result = {}
    for thr in thresholds:
        thresh_str = f"{thr*100:.1f}cm"

        # Fit planes once per threshold
        fit_results, df = fit_planes_per_label_v1(
            pts_world,
            labels,
            ignore_labels=(0,),
            distance_threshold=thr,
            num_iterations=num_iterations,
            min_support=min_support
        )

        if df is None or len(df) == 0:
            for gate in inlier_ratio_gates:
                result[f"prec@{thresh_str}_gate{gate}"] = 0.0
                result[f"rec@{thresh_str}_gate{gate}"] = 0.0
            continue

        plane_params = {}
        for pid, data in fit_results.items():
            if "plane_model_refined" in data:
                plane_params[pid] = data["plane_model_refined"]

        if not plane_params:
            for gate in inlier_ratio_gates:
                result[f"prec@{thresh_str}_gate{gate}"] = 0.0
                result[f"rec@{thresh_str}_gate{gate}"] = 0.0
            continue

        # Evaluate at each gate (cheap — reuses plane fits)
        for gate in inlier_ratio_gates:
            metrics = compute_inliers_at_threshold(
                pts_world, labels, plane_params, thr, gate
            )
            result[f"prec@{thresh_str}_gate{gate}"] = metrics["precision"]
            result[f"rec@{thresh_str}_gate{gate}"] = metrics["recall"]

    return result


def evaluate_single_frame_multigates(
    scene_id: str,
    frame_idx: str,
    depth_np: np.ndarray,
    gt_seg_np: np.ndarray,
    K_np: np.ndarray,
    c2w_np: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    inlier_ratio_gates: Tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
    compute_plane_metrics_flag: bool = True,
    ransac_iterations: int = 200,
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate a single frame at multiple inlier ratio gates.

    Same as evaluate_single_frame but fits RANSAC once and evaluates at all gates.
    Produces columns like prec@0.1cm_gate0.5, prec@0.1cm_gate0.9, etc.

    Args:
        scene_id: Scene identifier
        frame_idx: Frame identifier
        depth_np: (H, W) depth map in meters
        gt_seg_np: (H, W) ground truth segmentation
        K_np: (3, 3) or (4, 4) camera intrinsics
        c2w_np: (4, 4) camera-to-world pose
        labels: (H, W) predicted segmentation labels
        thresholds: Tuple of distance thresholds for plane metrics (meters)
        inlier_ratio_gates: Tuple of inlier ratio gates to evaluate
        compute_plane_metrics_flag: Whether to compute RANSAC plane metrics
        ransac_iterations: Number of RANSAC iterations

    Returns:
        (metrics_dict, labels) where metrics_dict contains all computed metrics
    """
    from planamono.shared.plane_fitting import backproject_v1 as backproject

    metric_thr = {}

    if compute_plane_metrics_flag:
        pts_world, pt_labels, _ = backproject(depth_np, K_np, c2w_np, labels)

        if pts_world.shape[0] == 0:
            for thr in thresholds:
                thresh_str = f"{thr*100:.1f}cm"
                for gate in inlier_ratio_gates:
                    metric_thr[f"prec@{thresh_str}_gate{gate}"] = 0.0
                    metric_thr[f"rec@{thresh_str}_gate{gate}"] = 0.0
        else:
            metric_thr = compute_plane_metrics_multigates(
                pts_world, pt_labels, thresholds,
                inlier_ratio_gates=inlier_ratio_gates,
                num_iterations=ransac_iterations,
            )
    else:
        for thr in thresholds:
            thresh_str = f"{thr*100:.1f}cm"
            for gate in inlier_ratio_gates:
                metric_thr[f"prec@{thresh_str}_gate{gate}"] = np.nan
                metric_thr[f"rec@{thresh_str}_gate{gate}"] = np.nan

    # Clustering metrics (pure img-to-img, gate-independent)
    clustering = compute_clustering_metrics(gt_seg_np, labels)

    metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        **clustering,
        **metric_thr
    }

    return metrics, labels


def evaluate_single_frame(
    scene_id: str,
    frame_idx: str,
    depth_np: np.ndarray,
    gt_seg_np: np.ndarray,
    K_np: np.ndarray,
    c2w_np: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    compute_plane_metrics_flag: bool = True,
    ransac_iterations: int = 200,
    inlier_ratio_gate: float = 0.5
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate a single frame with pre-computed segmentation labels (threshold-consistent).

    This is the RECOMMENDED version (formerly evaluate_single_frame_v1).

    Uses compute_plane_metrics() which runs RANSAC at each evaluation threshold
    (not fixed 2cm). This ensures visualization and evaluation use identical plane fits.

    For the deprecated old version (fixed 2cm RANSAC), use evaluate_single_frame_old().

    Args:
        scene_id: Scene identifier
        frame_idx: Frame identifier
        depth_np: (H, W) depth map in meters
        gt_seg_np: (H, W) ground truth segmentation
        K_np: (3, 3) or (4, 4) camera intrinsics
        c2w_np: (4, 4) camera-to-world pose
        labels: (H, W) predicted segmentation labels
        thresholds: Tuple of distance thresholds for plane metrics (meters)
        compute_plane_metrics_flag: Whether to compute RANSAC plane metrics
        ransac_iterations: Number of RANSAC iterations
        inlier_ratio_gate: Minimum inlier ratio to count a segment as valid (default 0.5)

    Returns:
        (metrics_dict, labels) where metrics_dict contains all computed metrics
    """
    from planamono.shared.plane_fitting import backproject_v1 as backproject

    metric_thr = {}

    if compute_plane_metrics_flag:
        pts_world, pt_labels, _ = backproject(depth_np, K_np, c2w_np, labels)

        if pts_world.shape[0] == 0:
            metric_thr = {f"prec@{thr*100:.1f}cm": 0.0 for thr in thresholds}
            metric_thr.update({f"rec@{thr*100:.1f}cm": 0.0 for thr in thresholds})
        else:
            metric_thr = compute_plane_metrics(
                pts_world, pt_labels, thresholds,
                num_iterations=ransac_iterations,
                inlier_ratio_gate=inlier_ratio_gate
            )
    else:
        for thr in thresholds:
            metric_thr[f"prec@{thr*100:.1f}cm"] = np.nan
            metric_thr[f"rec@{thr*100:.1f}cm"] = np.nan

    # Clustering metrics (pure img-to-img)
    clustering = compute_clustering_metrics(gt_seg_np, labels)

    metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        **clustering,
        **metric_thr
    }

    return metrics, labels


# ============================================================
# HYPERSIM-SPECIFIC EVALUATION (backproject_mcam)
# ============================================================

def evaluate_single_frame_hypersim(
    scene_id: str,
    frame_idx: str,
    depth_euc_np: np.ndarray,
    gt_seg_np: np.ndarray,
    M_cam_from_uv: np.ndarray,
    native_wh: Tuple[int, int],
    c2w_np: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    compute_plane_metrics_flag: bool = True,
    ransac_iterations: int = 200,
    inlier_ratio_gate: float = 0.5,
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate a single Hypersim frame using backproject_mcam (exact V-Ray ray directions).

    Uses raycasted Euclidean depth from planes.ply + M_cam_from_uv for backprojection,
    which is geometrically consistent with the plane labels and avoids pinhole
    approximation errors.

    Args:
        scene_id: Scene identifier
        frame_idx: Frame identifier
        depth_euc_np: (H, W) Euclidean ray distance in meters (raycasted from planes.ply)
        gt_seg_np: (H, W) ground truth segmentation
        M_cam_from_uv: (3, 3) V-Ray camera matrix from metadata CSV
        native_wh: (native_w, native_h) native render resolution
        c2w_np: (4, 4) camera-to-world pose
        labels: (H, W) predicted segmentation labels
        thresholds: Tuple of distance thresholds for plane metrics (meters)
        compute_plane_metrics_flag: Whether to compute RANSAC plane metrics
        ransac_iterations: Number of RANSAC iterations
        inlier_ratio_gate: Minimum inlier ratio to count a segment as valid

    Returns:
        (metrics_dict, labels) where metrics_dict contains all computed metrics
    """
    from planamono.shared.plane_fitting import backproject_mcam

    metric_thr = {}

    if compute_plane_metrics_flag:
        pts_world, pt_labels, _ = backproject_mcam(
            depth_euc_np, M_cam_from_uv, native_wh[0], native_wh[1], c2w_np, labels
        )

        if pts_world.shape[0] == 0:
            metric_thr = {f"prec@{thr*100:.1f}cm": 0.0 for thr in thresholds}
            metric_thr.update({f"rec@{thr*100:.1f}cm": 0.0 for thr in thresholds})
        else:
            metric_thr = compute_plane_metrics(
                pts_world, pt_labels, thresholds,
                num_iterations=ransac_iterations,
                inlier_ratio_gate=inlier_ratio_gate
            )
    else:
        for thr in thresholds:
            metric_thr[f"prec@{thr*100:.1f}cm"] = np.nan
            metric_thr[f"rec@{thr*100:.1f}cm"] = np.nan

    clustering = compute_clustering_metrics(gt_seg_np, labels)

    metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        **clustering,
        **metric_thr
    }

    return metrics, labels


def evaluate_single_frame_hypersim_multigates(
    scene_id: str,
    frame_idx: str,
    depth_euc_np: np.ndarray,
    gt_seg_np: np.ndarray,
    M_cam_from_uv: np.ndarray,
    native_wh: Tuple[int, int],
    c2w_np: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    inlier_ratio_gates: Tuple[float, ...] = (0.5, 0.7, 0.8, 0.9),
    compute_plane_metrics_flag: bool = True,
    ransac_iterations: int = 200,
) -> Tuple[Dict, np.ndarray]:
    """
    Evaluate a single Hypersim frame at multiple inlier ratio gates using backproject_mcam.

    Same as evaluate_single_frame_hypersim but fits RANSAC once and evaluates at all gates.

    Args:
        scene_id: Scene identifier
        frame_idx: Frame identifier
        depth_euc_np: (H, W) Euclidean ray distance in meters (raycasted from planes.ply)
        gt_seg_np: (H, W) ground truth segmentation
        M_cam_from_uv: (3, 3) V-Ray camera matrix from metadata CSV
        native_wh: (native_w, native_h) native render resolution
        c2w_np: (4, 4) camera-to-world pose
        labels: (H, W) predicted segmentation labels
        thresholds: Tuple of distance thresholds for plane metrics (meters)
        inlier_ratio_gates: Tuple of inlier ratio gates to evaluate
        compute_plane_metrics_flag: Whether to compute RANSAC plane metrics
        ransac_iterations: Number of RANSAC iterations

    Returns:
        (metrics_dict, labels) where metrics_dict contains all computed metrics
    """
    from planamono.shared.plane_fitting import backproject_mcam

    metric_thr = {}

    if compute_plane_metrics_flag:
        pts_world, pt_labels, _ = backproject_mcam(
            depth_euc_np, M_cam_from_uv, native_wh[0], native_wh[1], c2w_np, labels
        )

        if pts_world.shape[0] == 0:
            for thr in thresholds:
                thresh_str = f"{thr*100:.1f}cm"
                for gate in inlier_ratio_gates:
                    metric_thr[f"prec@{thresh_str}_gate{gate}"] = 0.0
                    metric_thr[f"rec@{thresh_str}_gate{gate}"] = 0.0
        else:
            metric_thr = compute_plane_metrics_multigates(
                pts_world, pt_labels, thresholds,
                inlier_ratio_gates=inlier_ratio_gates,
                num_iterations=ransac_iterations,
            )
    else:
        for thr in thresholds:
            thresh_str = f"{thr*100:.1f}cm"
            for gate in inlier_ratio_gates:
                metric_thr[f"prec@{thresh_str}_gate{gate}"] = np.nan
                metric_thr[f"rec@{thresh_str}_gate{gate}"] = np.nan

    clustering = compute_clustering_metrics(gt_seg_np, labels)

    metrics = {
        "scene_id": scene_id,
        "frame_idx": frame_idx,
        **clustering,
        **metric_thr
    }

    return metrics, labels
