"""
3D to 2D projection utilities for plane visualization.

This module provides functions for projecting 3D points back to 2D image coordinates
for visualization and label mapping.
"""

import numpy as np
from typing import Tuple


def project_points_to_image_v1(
    pts_world: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    H: int,
    W: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world points into 2D pixel coordinates.

    Args:
        pts_world: (N,3) points in world coordinates
        K: (3,3) camera intrinsic matrix
        T_cw: (4,4) camera-to-world transformation matrix
        H: Image height
        W: Image width

    Returns:
        uv_inside: (M,2) pixel coordinates of visible points
        visible_idx: (M,) indices into pts_world of visible points
    """
    # Convert world → camera frame
    pts_h = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)  # (N,4)
    T_wc = np.linalg.inv(T_cw)  # World-to-camera
    pts_cam = (T_wc @ pts_h.T).T[:, :3]  # (N,3)

    # Keep only points in front of camera (positive Z)
    mask_front = pts_cam[:, 2] > 0
    pts_cam_front = pts_cam[mask_front]

    if len(pts_cam_front) == 0:
        return np.empty((0, 2), dtype=np.int32), np.array([], dtype=np.int64)

    # Project to pixel coordinates
    uv = (K @ pts_cam_front.T).T  # (M,3)
    uv = uv[:, :2] / uv[:, 2:3]  # Normalize by depth

    # Keep only points within image bounds
    mask_inside = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    uv_inside = uv[mask_inside]

    # Compute visible indices in the original pts_world array
    idx_front = np.nonzero(mask_front)[0]
    visible_idx = idx_front[mask_inside]

    return uv_inside.astype(np.int32), visible_idx


def project_labels_to_image(
    pts_world: np.ndarray,
    labels: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    H: int,
    W: int
) -> np.ndarray:
    """
    Project 3D points with labels into a 2D label image.

    Args:
        pts_world: (N,3) 3D points in world coordinates
        labels: (N,) plane labels for each point
        K: (3,3) camera intrinsic matrix
        T_cw: (4,4) camera-to-world transformation
        H: Image height
        W: Image width

    Returns:
        label_img: (H,W) int32 array with projected labels
    """
    # Project points to pixel coordinates
    uv, visible_idx = project_points_to_image_v1(pts_world, K, T_cw, H, W)

    # Create label image
    label_img = np.zeros((H, W), dtype=np.int32)

    if len(uv) > 0:
        visible_labels = labels[visible_idx]
        label_img[uv[:, 1], uv[:, 0]] = visible_labels

    return label_img
