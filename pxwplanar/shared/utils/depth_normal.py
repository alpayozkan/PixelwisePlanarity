"""
Depth and surface normal processing utilities.

This module provides functions for:
- Converting depth maps to surface normals
- Transforming between Euclidean and Z-depth
- 3D point cloud generation from depth
"""

import numpy as np
import cv2


def depth_to_normal_remi(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """
    Convert depth image to surface normals using gradient-based method.

    This uses cross product of spatial gradients to estimate surface normals.

    Args:
        depth: (H,W) depth map in meters
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x (pixels)
        cy: Principal point y (pixels)

    Returns:
        normal: (H,W,3) surface normal vectors (unnormalized, need normalization)
    """
    # Convert depth to 3D points
    ans = depth_to_3d(depth, fx, fy, cx, cy)

    # Compute spatial vectors using convolution
    h_kernel = np.array([[1], [-1]])  # Vertical gradient
    v_kernel = np.array([[1, -1]])    # Horizontal gradient

    h_map = cv2.filter2D(ans, -1, h_kernel)
    v_map = cv2.filter2D(ans, -1, v_kernel)

    # Cross product gives normal
    normal = np.cross(h_map, v_map)

    # Normalize
    normal = vector_normalization(normal)

    return normal


def depth_to_3d(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud in camera coordinates.

    Args:
        depth: (H,W) depth map in meters
        fx: Focal length in x (pixels)
        fy: Focal length in y (pixels)
        cx: Principal point x (pixels)
        cy: Principal point y (pixels)

    Returns:
        points: (H,W,3) 3D points with (x, y, z) coordinates
    """
    h, w = depth.shape
    ans = np.zeros((h, w, 3))

    # Z coordinate is depth
    ans[:, :, 2] = depth

    # Create pixel grid (1-indexed in original code)
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - cx
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - cy

    # Backproject to 3D
    ans[:, :, 0] = u_map / fx * depth
    ans[:, :, 1] = v_map / fy * depth

    return ans


def extract_zdepth(depth_map: np.ndarray, K: np.ndarray = None) -> np.ndarray:
    """
    Convert Euclidean distance depth to Z-depth (optical axis depth).

    Args:
        depth_map: (H,W) depth map in Euclidean distance (meters)
        K: (3,3) camera intrinsic matrix. If None, uses Hypersim defaults.

    Returns:
        Z: (H,W) Z-depth projected onto camera optical axis
    """
    # Default Hypersim camera parameters
    if K is None:
        fx = 886.810
        fy = 886.810
        cx = 512.0
        cy = 384.0
    else:
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

    H, W = depth_map.shape

    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Normalized image coordinates
    x_n = (u - cx) / fx
    y_n = (v - cy) / fy

    # Compute per-pixel ray direction length
    ray_length = np.sqrt(x_n**2 + y_n**2 + 1)

    # Project Euclidean distance onto optical axis
    Z = depth_map / ray_length

    return Z


def vector_normalization(normal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize normal vectors to unit length.

    Args:
        normal: (H,W,3) normal vectors
        eps: Small epsilon to avoid division by zero

    Returns:
        normal: (H,W,3) unit normal vectors
    """
    mag = np.linalg.norm(normal, axis=2)
    normal = normal / (np.expand_dims(mag, axis=2) + eps)
    return normal
