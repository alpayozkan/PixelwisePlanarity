"""
Metrics computation for plane fitting and segmentation evaluation.

This module provides:
- Precision/recall for fitted planes based on inlier ratios
- Segmentation covering (SC) metric
- Multi-threshold plane evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from .planefit import fit_planes_per_label_v1


# ============================================================
# SEGMENTATION METRICS (image-to-image)
# ============================================================

def segmentation_covering_fast(gt_seg: np.ndarray, pred_seg: np.ndarray) -> float:
    """
    Compute Segmentation Covering (SC) metric between two segmentation maps.

    SC measures how well the predicted segmentation covers the ground truth.
    For each GT segment, we find the best matching predicted segment (by IoU)
    and weight by segment area.

    This is a pure image-to-image metric - no labels are ignored.
    Both inputs should be pre-processed (e.g., background filtered) before calling.

    Args:
        gt_seg: (H, W) ground truth segmentation labels (int)
        pred_seg: (H, W) predicted segmentation labels (int)

    Returns:
        SC score in [0, 1]. Higher is better.
    """
    gt = gt_seg.ravel().astype(np.int64)
    pr = pred_seg.ravel().astype(np.int64)

    if gt.size == 0:
        return 0.0

    gt_labels, gt_inv = np.unique(gt, return_inverse=True)
    pr_labels, pr_inv = np.unique(pr, return_inverse=True)

    n_gt = gt_labels.size
    n_pr = pr_labels.size

    # Build contingency matrix via bincount
    combined = gt_inv * n_pr + pr_inv
    counts = np.bincount(combined, minlength=n_gt * n_pr).astype(np.int64)
    contingency = counts.reshape((n_gt, n_pr))

    gt_areas = contingency.sum(axis=1)
    pr_areas = contingency.sum(axis=0)

    # IoU = intersection / union
    union = gt_areas[:, None] + pr_areas[None, :] - contingency

    with np.errstate(divide='ignore', invalid='ignore'):
        iou_matrix = np.where(union > 0, contingency / union, 0.0)

    # For each GT segment, find best IoU with any predicted segment
    best_iou = iou_matrix.max(axis=1)

    # Weighted average by GT segment area
    total_area = gt_areas.sum()
    sc = (best_iou * gt_areas).sum() / total_area if total_area > 0 else 0.0

    return float(sc)


# ============================================================
# PLANE FITTING METRICS
# ============================================================

def compute_inliers_at_threshold(
    pts_world: np.ndarray,
    labels: np.ndarray,
    plane_params: Dict[int, Tuple[float, float, float, float]],
    threshold: float,
    inlier_ratio_gate: float = 0.5
) -> Dict[str, float]:
    """
    Count inliers at a given distance threshold using pre-fitted plane parameters.

    For each plane segment, computes the number of points within `threshold`
    distance of the fitted plane. Only segments with inlier ratio >= gate
    are counted (quality filtering).

    Args:
        pts_world: (N, 3) world coordinates
        labels: (N,) segment labels for each point
        plane_params: {segment_id: (a, b, c, d)} plane parameters (ax + by + cz + d = 0)
        threshold: Distance threshold in meters
        inlier_ratio_gate: Minimum inlier ratio to count a segment (default 0.5)

    Returns:
        {"precision": float, "recall": float}
        - precision: inliers / total_points (in valid segments only)
        - recall: inliers / all_scene_points
    """
    total_inliers = 0
    total_points = 0

    for pid, params in plane_params.items():
        mask = (labels == pid)
        pts_plane = pts_world[mask]
        n_pts = pts_plane.shape[0]

        if n_pts == 0:
            continue

        a, b, c, d = params
        distances = np.abs(pts_plane @ np.array([a, b, c]) + d)
        n_inliers = np.sum(distances < threshold)

        # Quality gate: only count segments with sufficient inlier ratio
        if n_inliers / n_pts >= inlier_ratio_gate:
            total_inliers += n_inliers
            total_points += n_pts

    precision = total_inliers / total_points if total_points > 0 else 0.0
    recall = total_inliers / len(labels) if len(labels) > 0 else 0.0

    return {"precision": precision, "recall": recall}


def fit_planes_and_evaluate_multi_threshold(
    pts_world: np.ndarray,
    labels: np.ndarray,
    thresholds: Tuple[float, ...],
    base_threshold: float = 0.02,
    num_iterations: int = 200,
    min_support: int = 100,
    inlier_ratio_gate: float = 0.5
) -> Dict[float, Dict[str, float]]:
    """
    Fit planes ONCE with RANSAC, then evaluate at multiple distance thresholds.

    This is ~3x faster than running RANSAC separately for each threshold.

    Args:
        pts_world: (N, 3) world coordinates
        labels: (N,) segment labels for each point
        thresholds: Tuple of distance thresholds to evaluate (in meters)
        base_threshold: RANSAC distance threshold for fitting
        num_iterations: RANSAC iterations
        min_support: Minimum points for RANSAC
        inlier_ratio_gate: Quality gate for segment filtering

    Returns:
        {threshold: {"precision": float, "recall": float}} for each threshold
    """
    results, df = fit_planes_per_label_v1(
        pts_world,
        labels,
        ignore_labels=(0,),
        distance_threshold=base_threshold,
        num_iterations=num_iterations,
        min_support=min_support
    )

    if df is None or len(df) == 0:
        return {thr: {"precision": 0.0, "recall": 0.0} for thr in thresholds}

    # Extract fitted plane parameters
    plane_params = {}
    for pid, data in results.items():
        if "plane_model_refined" in data:
            plane_params[pid] = data["plane_model_refined"]

    if not plane_params:
        return {thr: {"precision": 0.0, "recall": 0.0} for thr in thresholds}

    # Evaluate at each threshold
    metrics = {}
    for thr in thresholds:
        metrics[thr] = compute_inliers_at_threshold(
            pts_world, labels, plane_params, thr, inlier_ratio_gate
        )

    return metrics


# ============================================================
# PRECISION/RECALL FROM DATAFRAME
# ============================================================

def compute_precision_recall_v1(
    df: pd.DataFrame,
    total_scene_points: Optional[int] = None
) -> Dict:
    """
    Compute per-plane and global precision/recall from plane fitting results.

    Precision measures what fraction of predicted plane points are actual inliers.
    Recall measures what fraction of all scene points are explained by planes.

    Args:
        df: DataFrame with columns:
            - num_points: Total predicted points for each plane
            - refined_inlier_num_points: Inlier points after LS refinement
        total_scene_points: Total number of 3D points in scene (planar + non-planar).
                          If None, recall is not computed.

    Returns:
        Dictionary with:
            - df_with_metrics: DataFrame with added precision and recall columns
            - global_precision: Overall precision across all planes
            - global_recall: Overall recall (None if total_scene_points not provided)
    """
    df = df.copy()

    # Handle empty or missing columns
    if df.empty or "refined_inlier_num_points" not in df.columns:
        return {
            "df_with_metrics": df,
            "global_precision": 0.0,
            "global_recall": 0.0 if total_scene_points else None,
        }

    # Per-plane precision: fraction of predicted points that are inliers
    df["precision"] = df["refined_inlier_num_points"] / df["num_points"]

    # Per-plane recall: fraction of all scene points explained by this plane
    if total_scene_points is not None and total_scene_points > 0:
        df["recall"] = df["refined_inlier_num_points"] / total_scene_points
    else:
        df["recall"] = None

    # Global metrics
    total_inliers = df["refined_inlier_num_points"].sum()
    total_predicted = df["num_points"].sum()

    global_precision = total_inliers / total_predicted if total_predicted > 0 else 0.0
    global_recall = (total_inliers / total_scene_points) if total_scene_points else None

    return {
        "df_with_metrics": df,
        "global_precision": global_precision,
        "global_recall": global_recall,
    }
