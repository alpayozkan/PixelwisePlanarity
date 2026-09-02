"""
Plane fitting utilities for RANSAC-based plane extraction and evaluation.
"""

from .metrics import (
    compute_inliers_at_threshold,
    compute_inliers_at_threshold_with_indices,
    compute_precision_recall,
    fit_planes_and_evaluate_multi_threshold,
    segmentation_covering_fast,
)
from .planefit import (
    backproject,
    backproject_mcam,
    filter_planes_by_inlier_ratio,
    fit_planes_per_label,
    mark_planes_below_threshold_as_outliers,
    refine_plane_least_squares,
    set_ransac_seed,
)
from .projection import project_labels_to_image, project_points_to_image

__all__ = [
    "set_ransac_seed",
    "backproject",
    "backproject_mcam",
    "refine_plane_least_squares",
    "fit_planes_per_label",
    "filter_planes_by_inlier_ratio",
    "mark_planes_below_threshold_as_outliers",
    "compute_precision_recall",
    "segmentation_covering_fast",
    "compute_inliers_at_threshold",
    "compute_inliers_at_threshold_with_indices",
    "fit_planes_and_evaluate_multi_threshold",
    "project_points_to_image",
    "project_labels_to_image",
]
