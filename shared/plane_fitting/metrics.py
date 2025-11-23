"""
Metrics computation for plane fitting evaluation.

This module computes precision and recall for fitted planes based on
inlier ratios and point counts.
"""

import pandas as pd
from typing import Dict, Optional


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
