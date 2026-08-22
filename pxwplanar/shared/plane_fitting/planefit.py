"""
Plane fitting utilities using RANSAC and least-squares refinement.

This module provides functions for:
- Backprojecting depth images to 3D point clouds
- Fitting planes to segmented regions using RANSAC
- Refining plane parameters with least squares
- Filtering planes by inlier ratio
"""

import numpy as np
import open3d as o3d
import pandas as pd
import copy
from typing import Tuple, Dict, Optional, List


# ============================================================
# RANSAC SEEDING / REPRODUCIBILITY
# ------------------------------------------------------------
# Open3D's ``PointCloud.segment_plane`` uses a process-global Mersenne-Twister
# RNG (``open3d.utility.random``) that is seeded from ``std::random_device`` at
# first use, i.e. non-deterministically. Two mechanisms make fitting
# reproducible:
#   1. ``set_ransac_seed(seed)`` pins the global RNG (works on open3d >= 0.16,
#      including 0.17 which has no per-call ``seed`` argument). Call this once
#      per frame, inside the worker process, before any fitting.
#   2. ``segment_plane(..., seed=...)`` (open3d >= 0.18) makes a single call
#      deterministic regardless of fit order/threads. ``_segment_plane()``
#      passes it only when the installed open3d supports it.
# ``seed=None`` everywhere restores the legacy non-deterministic behaviour.


def _detect_segment_plane_seed_support() -> bool:
    """Return True if this open3d's ``segment_plane`` accepts a ``seed`` kwarg.

    open3d's ``segment_plane`` is a pybind11 builtin with no inspectable
    signature, so we probe it with a tiny throw-away fit (use float64 points —
    float32 can segfault ``Vector3dVector``).
    """
    try:
        _p = o3d.geometry.PointCloud()
        _p.points = o3d.utility.Vector3dVector(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64)
        )
        _p.segment_plane(distance_threshold=0.01, ransac_n=3,
                         num_iterations=1, seed=0)
        return True
    except TypeError:
        return False
    except Exception:
        # Any other failure: assume unsupported and fall back to global seeding.
        return False


_SEGMENT_PLANE_HAS_SEED = _detect_segment_plane_seed_support()


def set_ransac_seed(seed: Optional[int]) -> None:
    """Pin open3d's process-global RANSAC RNG. No-op when ``seed`` is None.

    Must be called *inside* each worker process (e.g. at the top of every
    per-frame evaluation) — loky workers are fresh processes and do not inherit
    a seed set in the parent.
    """
    if seed is None:
        return
    try:
        o3d.utility.random.seed(int(seed))
    except Exception:
        # Older/newer open3d without utility.random — global seeding unavailable;
        # per-call seed (if supported) still applies.
        pass


def _segment_plane(pcd, distance_threshold: float, ransac_n: int,
                   num_iterations: int, seed: Optional[int] = None):
    """``pcd.segment_plane`` wrapper that forwards ``seed`` only when supported."""
    if seed is not None and _SEGMENT_PLANE_HAS_SEED:
        return pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            seed=int(seed),
        )
    return pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
    T_cw: np.ndarray,
    plane_seg: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Backproject depth image to 3D world coordinates.


    Args:
        depth: (H,W) depth map in meters
        K: (3,3) or (4,4) camera intrinsic matrix
        T_cw: (4,4) camera-to-world transformation matrix
        plane_seg: (H,W) optional plane segmentation labels

    Returns:
        pts_world: (N,3) 3D points in world coordinates
        labels: (N,) plane labels for valid points (None if plane_seg is None)
        valid_idx: (N,) indices into flattened H*W image that were valid
    """
    H, W = depth.shape

    # Ensure intrinsics are 3x3
    K = np.asarray(K, dtype=np.float64)[:3, :3]
    Kinv = np.linalg.inv(K)

    # Create pixel grid (u = x, v = y)
    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)  # (HW,3)

    # Valid depth mask
    z = depth.reshape(-1).astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    valid_idx = np.nonzero(valid)[0]

    # Rays in camera coordinates, then scale by depth
    rays = (Kinv @ uv1.T).T  # (HW,3)
    pts_cam = rays[valid] * z[valid, None]  # (N,3)

    # Transform to world coordinates
    pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])  # (N,4)
    T_cw = np.asarray(T_cw, dtype=np.float64)
    pts_world = (pts_cam_h @ T_cw.T)[:, :3]  # (N,3)

    # Extract plane labels for valid points
    labels = None
    if plane_seg is not None:
        plane_seg = np.asarray(plane_seg, dtype=np.int32)
        labels = plane_seg.reshape(-1)[valid]  # (N,)

    return pts_world, labels, valid_idx



def backproject_mcam(
    depth_euc: np.ndarray,
    M_cam_from_uv: np.ndarray,
    native_w: int,
    native_h: int,
    T_cw: np.ndarray,
    plane_seg: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Backproject Euclidean depth to 3D using V-Ray's M_cam_from_uv camera model.

    Unlike backproject/v2 which use a pinhole K approximation, this function
    uses the actual V-Ray camera matrix to compute per-pixel ray directions.
    This eliminates the systematic position errors (radial artifacts) that grow
    toward image edges with the pinhole approximation.

    The 3D point for each pixel is:  P = depth_euc * normalize(M_cam_from_uv @ [u_ndc, v_ndc, 1])

    Args:
        depth_euc: (H, W) Euclidean ray distance in meters (from depth_meters.hdf5)
        M_cam_from_uv: (3, 3) V-Ray camera matrix from metadata CSV
        native_w: Native render width (for NDC grid computation)
        native_h: Native render height (for NDC grid computation)
        T_cw: (4, 4) camera-to-world transformation matrix
        plane_seg: (H, W) optional plane segmentation labels

    Returns:
        pts_world: (N, 3) 3D points in world coordinates
        labels: (N,) plane labels for valid points (None if plane_seg is None)
        valid_idx: (N,) indices into flattened H*W image that were valid
    """
    H, W = depth_euc.shape
    M = np.asarray(M_cam_from_uv, dtype=np.float64)

    # Build NDC grid (same convention as V-Ray / render.py)
    u_px = np.arange(W, dtype=np.float64)
    v_px = np.arange(H, dtype=np.float64)

    # If depth has been resized, map pixel coords through native resolution
    u_native = u_px * (native_w / W)
    v_native = v_px * (native_h / H)
    u_ndc = (2.0 * u_native + 1.0) / native_w - 1.0
    v_ndc = 1.0 - (2.0 * v_native + 1.0) / native_h

    uu, vv = np.meshgrid(u_ndc, v_ndc)
    uvs = np.stack([uu, vv, np.ones_like(uu)], axis=-1)  # (H, W, 3)

    # Camera-space ray directions via M_cam_from_uv
    dirs_cam = uvs @ M.T  # (H, W, 3)

    # Normalize to unit vectors
    ray_lengths = np.linalg.norm(dirs_cam, axis=-1, keepdims=True)  # (H, W, 1)
    ray_lengths = np.maximum(ray_lengths, 1e-12)
    dirs_unit = dirs_cam / ray_lengths  # (H, W, 3)

    # Valid depth mask
    depth_flat = depth_euc.ravel().astype(np.float64)
    valid = np.isfinite(depth_flat) & (depth_flat > 0)
    valid_idx = np.flatnonzero(valid)

    if valid_idx.size == 0:
        labels = np.array([], dtype=np.int32) if plane_seg is not None else None
        return np.zeros((0, 3), dtype=np.float64), labels, valid_idx

    # 3D points in camera space: P_cam = depth_euc * ray_unit_direction
    dirs_flat = dirs_unit.reshape(-1, 3)
    pts_cam = dirs_flat[valid] * depth_flat[valid, None]  # (N, 3)

    # Transform to world coordinates
    T_cw = np.asarray(T_cw, dtype=np.float64)
    pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    pts_world = (pts_cam_h @ T_cw.T)[:, :3]

    # Extract plane labels for valid points
    labels = None
    if plane_seg is not None:
        labels = np.asarray(plane_seg, dtype=np.int32).ravel()[valid]

    return pts_world, labels, valid_idx


def refine_plane_least_squares(P: np.ndarray) -> np.ndarray:
    """
    Refine plane parameters using PCA (least squares).

    Fits plane to points using SVD: normal is the smallest eigenvector,
    and distance d = -n · centroid.

    Args:
        P: (N,3) array of 3D points

    Returns:
        (4,) array [a, b, c, d] where plane equation is ax + by + cz + d = 0
        and ||[a,b,c]|| = 1
    """
    c = P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]  # Normal is smallest singular vector
    d = -float(n @ c)

    # Normalize to unit normal
    norm = np.linalg.norm(n)
    n = n / norm
    d = d / norm

    return np.r_[n, d]


def fit_planes_per_label(
    pts: np.ndarray,
    labels: np.ndarray,
    ignore_labels: Tuple[int, ...] = (0,),
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    min_support: int = 50,
    verbose: bool = False,
    ransac_seed: Optional[int] = 0
) -> Tuple[Dict, pd.DataFrame]:
    """
    Fit planes to each labeled segment using RANSAC + least-squares refinement.

    Args:
        pts: (N,3) 3D points
        labels: (N,) plane labels for each point
        ignore_labels: Labels to skip (e.g., background)
        distance_threshold: RANSAC inlier threshold in same units as pts (meters)
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        min_support: Minimum number of points required for a valid plane
        verbose: Print progress messages
        ransac_seed: Seed for reproducible RANSAC (default 0). Pins the global
            open3d RNG and (on open3d >= 0.18) also passes a per-call seed.
            None = legacy non-deterministic behaviour.

    Returns:
        results: Dict mapping plane_id to dict with:
            - plane_model: (4,) RANSAC plane parameters [a,b,c,d]
            - plane_model_refined: (4,) LS-refined plane parameters
            - inliers_local: Local indices of RANSAC inliers
            - inliers_global: Global indices of RANSAC inliers
            - inliers_all: Global indices of all inliers to refined plane
            - indices_all: Global indices of all points with this label
            - support: Number of RANSAC inliers
            - rms: RMS error of inliers to refined plane
            - num_points: Total points with this label
            - ransac_inlier_num_points: Number of RANSAC inliers
            - refined_inlier_num_points: Number of inliers to refined plane
            - ransac_inlier_ratio: Fraction of RANSAC inliers
            - refined_inlier_ratio: Fraction of refined inliers
        plane_df: DataFrame with per-plane statistics
    """
    # Reproducibility: pin the global RNG so direct callers that don't seed it
    # themselves are deterministic on open3d 0.17 (no per-call seed) too. The
    # per-call seed below additionally covers open3d >= 0.18. ransac_seed=None
    # restores legacy non-deterministic behaviour.
    set_ransac_seed(ransac_seed)

    # Flatten and filter invalid points
    pts = np.asarray(pts)
    labels_arr = np.asarray(labels).reshape(-1)
    pts = pts.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts_valid = pts[valid]
    labels_valid = labels_arr[valid]
    idx_valid = np.nonzero(valid)[0]

    unique_labels = [l for l in np.unique(labels_valid) if l not in ignore_labels]
    results = {}

    for pid in unique_labels:
        mask = (labels_valid == pid)
        idx_global = idx_valid[mask]

        if idx_global.size < max(min_support, ransac_n):
            if verbose:
                print(f'[Plane {pid}] Not enough points: {idx_global.size} < {min_support}')
            continue

        # Build point cloud for this label
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[idx_global])

        # RANSAC plane fitting
        plane_model, inliers_local = _segment_plane(
            pcd,
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
            seed=ransac_seed,
        )
        inliers_local = np.asarray(inliers_local, dtype=np.int64)

        if inliers_local.size < min_support:
            if verbose:
                print(f'[Plane {pid}] RANSAC inliers too few: {inliers_local.size}')
            continue

        # Map back to global indices
        inliers_global = idx_global[inliers_local]
        P_in = pts[inliers_global]
        ransac_inlier_ratio = len(inliers_global) / max(1, len(idx_global))

        # Refine plane parameters using least squares on inliers
        plane_refined = refine_plane_least_squares(P_in)

        # Compute RMS error to refined plane
        a, b, c, d = plane_refined
        dist = np.abs(P_in @ np.array([a, b, c]) + d)  # Distance to plane
        rms = float(np.sqrt(np.mean(dist**2)))

        # Find all inliers to the refined plane
        P_all = pts[idx_global]
        dist_all = np.abs(P_all @ np.array([a, b, c]) + d)
        inliers_all_mask = dist_all < distance_threshold
        inliers_all = idx_global[inliers_all_mask]
        refined_inlier_ratio = len(inliers_all) / max(1, len(idx_global))

        results[pid] = {
            'plane_model': np.array(plane_model, dtype=float),
            'plane_model_refined': plane_refined,
            'inliers_local': inliers_local,
            'inliers_global': inliers_global,
            'inliers_all': inliers_all,
            'indices_all': idx_global,
            'support': int(inliers_local.size),
            'rms': rms,
            'num_points': len(idx_global),
            'ransac_inlier_num_points': len(inliers_global),
            'refined_inlier_num_points': len(inliers_all),
            'ransac_inlier_ratio': ransac_inlier_ratio,
            'refined_inlier_ratio': refined_inlier_ratio,
        }

    # Create summary DataFrame
    rows = []
    for pid, data in results.items():
        rows.append({
            "plane_id": pid,
            "num_points": data["num_points"],
            "ransac_inlier_num_points": data["ransac_inlier_num_points"],
            "refined_inlier_num_points": data["refined_inlier_num_points"],
            "inlier_ratio_ransac": data["ransac_inlier_ratio"],
            "inlier_ratio_refined": data["refined_inlier_ratio"],
        })

    plane_df = pd.DataFrame(rows)
    return results, plane_df


def filter_planes_by_inlier_ratio(
    results: Dict,
    plane_df: pd.DataFrame,
    inlier_ratio_threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[Dict, pd.DataFrame]:
    """
    Remove planes with low inlier ratios.

    Args:
        results: Output from fit_planes_per_label
        plane_df: DataFrame with per-plane statistics
        inlier_ratio_threshold: Minimum acceptable refined inlier ratio
        verbose: Print filtering statistics

    Returns:
        filtered_results: Results with only planes meeting threshold
        filtered_df: Corresponding filtered DataFrame
    """
    # No fitted planes: the empty DataFrame has no columns to index
    if plane_df.empty:
        return results, plane_df

    # Find planes that meet the threshold
    valid_plane_ids = plane_df[
        plane_df["inlier_ratio_refined"] >= inlier_ratio_threshold
    ]["plane_id"].tolist()

    # Filter dictionary and DataFrame
    filtered_results = {pid: data for pid, data in results.items()
                       if pid in valid_plane_ids}
    filtered_df = plane_df[plane_df["plane_id"].isin(valid_plane_ids)].reset_index(drop=True)

    if verbose:
        n_removed = len(results) - len(filtered_results)
        print(f"Filtered out {n_removed} planes (kept {len(filtered_results)} / {len(results)})")

    return filtered_results, filtered_df


def mark_planes_below_threshold_as_outliers(
    results: Dict,
    plane_df: pd.DataFrame,
    inlier_ratio_threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[Dict, pd.DataFrame]:
    """
    Mark low-quality planes as all-outliers instead of removing them.

    This keeps the plane IDs but marks all their points as outliers.

    Args:
        results: Output from fit_planes_per_label
        plane_df: DataFrame with per-plane statistics
        inlier_ratio_threshold: Minimum acceptable refined inlier ratio
        verbose: Print marking statistics

    Returns:
        updated_results: Results with low-quality planes marked as outliers
        updated_df: Updated DataFrame with is_below_threshold column
    """
    # No fitted planes: the empty DataFrame has no columns to index
    if plane_df.empty:
        return copy.deepcopy(results), plane_df.copy()

    updated_results = copy.deepcopy(results)
    updated_df = plane_df.copy()

    # Identify planes below threshold
    below_ids = updated_df[
        updated_df["inlier_ratio_refined"] < inlier_ratio_threshold
    ]["plane_id"].tolist()

    updated_df["is_below_threshold"] = updated_df["plane_id"].isin(below_ids)

    # Mark as outliers
    for pid in below_ids:
        if pid in updated_results:
            data = updated_results[pid]

            # Clear inlier arrays
            data["inliers_all"] = np.array([], dtype=int)
            data["inliers_global"] = np.array([], dtype=int)
            data["inliers_local"] = np.array([], dtype=int)

            # Reset statistics
            data["refined_inlier_num_points"] = 0
            data["refined_inlier_ratio"] = 0.0
            data["support"] = 0
            data["rms"] = None
            data["is_all_outlier"] = True

            # Update DataFrame
            mask = updated_df["plane_id"] == pid
            updated_df.loc[mask, [
                "refined_inlier_num_points",
                "inlier_ratio_refined",
                "ransac_inlier_num_points",
                "inlier_ratio_ransac"
            ]] = 0

    if verbose:
        print(f"Marked {len(below_ids)} planes as all-outliers "
              f"(below {inlier_ratio_threshold:.2f})")

    return updated_results, updated_df
