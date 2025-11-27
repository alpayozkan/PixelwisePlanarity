"""
Planar segmentation algorithms using normal and depth consistency.

This module provides vectorized algorithms for segmenting planar regions
based on surface normal similarity and depth proximity.
"""

import numpy as np
from scipy.ndimage import label
from typing import Tuple


def compute_vectorized_planar_segments_v1(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 8
) -> Tuple[np.ndarray, int]:
    """
    Compute planar segments using 8-connected neighbor matching.

    This is the recommended version with neighbor match count thresholding
    for robustness against noise.

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Depth difference threshold in meters
        neighbor_match_count_thresh: Minimum number of matching neighbors (default 1)

    Returns:
        labeled: (H,W) int array with segment IDs (0-indexed)
        num_components: Number of segments found
    """
    H, W = planarity_mask.shape
    mask = planarity_mask.astype(bool)

    # Pad arrays for neighbor access
    pad = ((1, 1), (1, 1))
    padded_mask = np.pad(mask, pad, constant_values=0)
    padded_depth = np.pad(depth, pad, constant_values=0)
    padded_normal = np.pad(normal, pad + ((0, 0),), mode='constant')

    # 8-connected neighbor offsets
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

    def get_shifted(arr, shift):
        dy, dx = shift
        return arr[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]

    # Stack shifted versions of neighbors
    neighbor_masks = np.stack([get_shifted(padded_mask, s) for s in shifts], axis=2)  # (H, W, 8)
    neighbor_depths = np.stack([get_shifted(padded_depth, s) for s in shifts], axis=2)  # (H, W, 8)
    neighbor_normals = np.stack([get_shifted(padded_normal, s) for s in shifts], axis=3)  # (H, W, 3, 8)

    # Prepare center values
    center_depth = depth[:, :, None]  # (H, W, 1)
    center_normal = normal[:, :, :, None]  # (H, W, 3, 1)

    # Valid planar neighbor pairs
    valid_pair = mask[:, :, None] & neighbor_masks

    # Normal similarity check (angular difference)
    dot = np.sum(center_normal * neighbor_normals, axis=2)  # (H, W, 8)
    norm_center = np.linalg.norm(center_normal, axis=2, keepdims=True)  # (H, W, 1, 1)
    norm_center = norm_center[..., 0]  # (H, W, 1)
    norm_neighbor = np.linalg.norm(neighbor_normals, axis=2)  # (H, W, 8)

    cos_angle = np.clip(dot / (norm_center * norm_neighbor + 1e-8), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    normal_similar = angle < normal_threshold_rad  # (H, W, 8)

    # Depth closeness check
    depth_diff = np.abs(center_depth - neighbor_depths)
    depth_close = depth_diff < depth_threshold  # (H, W, 8)

    # Count matching neighbors
    neighbor_match_count = np.sum(valid_pair & normal_similar & depth_close, axis=2)
    connected = neighbor_match_count >= neighbor_match_count_thresh  # (H, W)

    # Connected component labeling with 8-connectivity
    structure = np.ones((3, 3), dtype=bool)
    labeled, num_components = label(connected & mask, structure=structure)

    return labeled, num_components


def filter_small_segments(
    segmentation: np.ndarray,
    min_size: int = 50
) -> np.ndarray:
    """
    Remove small segments and relabel remaining ones.

    Args:
        segmentation: (H,W) segment labels (>=0 for valid, <0 for background)
        min_size: Minimum number of pixels for a valid segment

    Returns:
        new_seg: (H,W) filtered and relabeled segmentation
    """
    seg = segmentation.copy()
    unique_labels = np.unique(seg[seg >= 0])
    new_seg = np.full_like(seg, -1)
    next_label = 0

    for label_val in unique_labels:
        mask = seg == label_val
        if np.sum(mask) < min_size:
            continue  # Skip small regions
        new_seg[mask] = next_label
        next_label += 1

    return new_seg
