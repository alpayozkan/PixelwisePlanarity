"""
Post-processing utilities for plane segmentations.

This module provides functions for cleaning and refining segmentation results.
"""

import cv2
import numpy as np
from scipy import ndimage


def remove_small_components(label_map: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove small connected components from segmentation.

    Args:
        label_map: (H,W) labeled segmentation map
        min_size: Minimum number of pixels for a component to be kept

    Returns:
        cleaned_label_map: (H,W) cleaned segmentation with small components
            removed
    """
    cleaned_label_map = np.zeros_like(label_map)
    current_label = 1

    unique_labels = np.unique(label_map)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background

        mask = label_map == label
        labeled_mask, num_features = ndimage.label(mask)
        component_sizes = np.bincount(labeled_mask.ravel())

        # Keep only components >= min_size
        for i in range(1, num_features + 1):
            if component_sizes[i] >= min_size:
                cleaned_label_map[labeled_mask == i] = current_label
                current_label += 1

    return cleaned_label_map


def plane_merge(labels, moge_signals):
    class UnionFind:
        def __init__(self):
            self.parent = {}

        def find(self, x):
            if self.parent.setdefault(x, x) != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, a, b):
            self.parent[self.find(a)] = self.find(b)

    def fit_plane_svd(points):
        """
        Fit plane n·x + d = 0
        points: (N,3)
        """
        centroid = points.mean(axis=0)
        _, _, Vt = np.linalg.svd(points - centroid)
        n = Vt[-1]
        n /= np.linalg.norm(n)
        d = -np.dot(n, centroid)
        return n, d, centroid

    def robust_plane_distance(
        points,  # (N,3) points from segment A
        plane_normal,  # (3,)
        plane_d,  # scalar
        max_samples=500,
        percentile=20,
    ):
        """
        Robust point-to-plane distance using subsampling + percentile.
        """
        if points.shape[0] > max_samples:
            idx = np.random.choice(points.shape[0], max_samples, replace=False)
            points = points[idx]

        dists = np.abs(points @ plane_normal + plane_d)
        return np.percentile(dists, percentile)

    def merge_planar_segments_moge(
        labels,
        points,  # (H,W,3) from MoGe
        normals,  # (3,H,W) or (H,W,3)
        normal_merge_thresh_deg=5.0,
        plane_offset_thresh=0.02,  # meters
        min_points=200,
        # centroid_dist_thresh=0.5,   # meters
    ):
        """
        Merge planar segments using MoGe geometry only.

        Args:
            labels: (H,W) int, 0 = non-planar
            points: (H,W,3) metric XYZ
            normals: (3,H,W) or (H,W,3)
        Returns:
            merged_labels: (H,W)
        """

        if normals.shape[0] == 3:
            normals = np.transpose(normals, (1, 2, 0))

        H, W = labels.shape
        plane_info = {}

        # --- 1. Fit plane per segment ---
        MAX_PLANE_POINTS = 5000

        for lbl in np.unique(labels):
            if lbl == 0:
                continue

            ys, xs = np.where(labels == lbl)
            if len(xs) < min_points:
                continue

            pts = points[ys, xs]
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]

            if pts.shape[0] < min_points:
                continue

            # 🔥 critical: subsample
            if pts.shape[0] > MAX_PLANE_POINTS:
                idx = np.random.choice(
                    pts.shape[0], MAX_PLANE_POINTS, replace=False
                )
                pts = pts[idx]

            try:
                n, d, c = fit_plane_svd(pts)
            except np.linalg.LinAlgError:
                continue

            # plane_info[lbl] = {
            #     "normal": n,
            #     "d": d,
            #     "centroid": c
            # }
            plane_info[lbl] = {
                "normal": n,
                "d": d,
                "points": pts,  # 🔥 store sampled points
            }

        kept_labels = list(plane_info.keys())
        uf = UnionFind()
        ang_thresh = np.deg2rad(normal_merge_thresh_deg)

        # --- 2. Pairwise plane merge ---
        for i in range(len(kept_labels)):
            li = kept_labels[i]
            pi = plane_info[li]

            for j in range(i + 1, len(kept_labels)):
                lj = kept_labels[j]
                pj = plane_info[lj]

                # (1) normal similarity
                cosang = abs(np.dot(pi["normal"], pj["normal"]))
                ang = np.arccos(np.clip(cosang, -1.0, 1.0))
                if ang > ang_thresh:
                    print("surface normal rejects")
                    continue

                # (2) plane offset
                if abs(pi["d"] - pj["d"]) > plane_offset_thresh:
                    print("plane offset rejects")
                    continue

                # (3) spatial proximity
                # if (np.linalg.norm(pi["centroid"] - pj["centroid"])
                #         > centroid_dist_thresh):
                #     print('spatial proximity rejects')
                #     continue
                dist_ij = robust_plane_distance(
                    plane_info[li]["points"],
                    pj["normal"],
                    pj["d"],
                    max_samples=300,
                    percentile=20,
                )

                dist_ji = robust_plane_distance(
                    plane_info[lj]["points"],
                    pi["normal"],
                    pi["d"],
                    max_samples=300,
                    percentile=20,
                )

                if max(dist_ij, dist_ji) > plane_offset_thresh * 2.0:
                    continue

                uf.union(li, lj)

        # --- 3. Relabel ---
        merged_labels = labels.copy()
        root_to_new = {}
        next_id = 1

        for lbl in kept_labels:
            root = uf.find(lbl)
            if root not in root_to_new:
                root_to_new[root] = next_id
                next_id += 1
            merged_labels[labels == lbl] = root_to_new[root]

        return merged_labels

    Hm, Wm = moge_signals["points"].shape[:2]
    labels_resized = cv2.resize(
        labels.astype(np.int32), (Wm, Hm), interpolation=cv2.INTER_NEAREST
    )

    labels_merged = merge_planar_segments_moge(
        labels_resized,
        moge_signals["points"],
        np.transpose(moge_signals["normal"], (2, 0, 1)),
        normal_merge_thresh_deg=5.0,
        plane_offset_thresh=0.02,
        # centroid_dist_thresh=0.5
    )

    labels_merged = cv2.resize(
        labels_merged.astype(np.int32),
        (labels.shape[1], labels.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return labels_merged


# ──────────────────────────────────────────────────────────────────
# v11 post-merge: dilation-based gap bridging + mean-normal/depth check
# ──────────────────────────────────────────────────────────────────


def postmerge_adjacent_segments(
    labels: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    merge_normal_deg: float = 10.0,
    merge_offset_m: float = 0.02,
    merge_min_pixels: int = 50,
    merge_gap_px: int = 5,
) -> np.ndarray:
    """Merge nearby segments whose mean normals and depths are similar.

    Uses binary dilation to bridge non-planar gaps between segments.
    Iterates greedily (largest pairs first) until convergence.

    Args:
        labels: (H, W) int32 segment labels (0 = background)
        normal: (H, W, 3) surface normals
        depth: (H, W) depth in meters
        merge_normal_deg: max angle between mean normals to merge
        merge_offset_m: max mean-depth difference for merge (meters)
        merge_min_pixels: ignore segments smaller than this
        merge_gap_px: dilation radius to bridge non-planar gaps (pixels)

    Returns:
        labels_merged: (H, W) int32 relabeled 1..N
    """
    from scipy.ndimage import binary_dilation

    labels = labels.copy()
    cos_thresh = np.cos(np.deg2rad(merge_normal_deg))

    # Disk structuring element for dilation
    sz = 2 * merge_gap_px + 1
    yy, xx = np.mgrid[:sz, :sz]
    struct = (
        (yy - merge_gap_px) ** 2 + (xx - merge_gap_px) ** 2
    ) <= merge_gap_px**2

    def _plane_params(labels, normal, depth):
        """label -> (mean_normal, mean_depth, pixel_count)"""
        params = {}
        for lab in np.unique(labels):
            if lab == 0:
                continue
            mask = labels == lab
            count = int(mask.sum())
            if count < merge_min_pixels:
                continue
            n_mean = normal[mask].mean(axis=0)
            n_norm = np.linalg.norm(n_mean)
            if n_norm < 1e-6:
                continue
            params[lab] = (n_mean / n_norm, float(depth[mask].mean()), count)
        return params

    def _nearby_pairs(labels, params, struct):
        """Find segment pairs within dilation radius."""
        adj = set()
        for lab in params:
            mask = labels == lab
            dilated = binary_dilation(mask, structure=struct)
            for other in np.unique(labels[dilated & ~mask]):
                if other > 0 and other != lab and other in params:
                    adj.add((min(lab, other), max(lab, other)))
        return adj

    for _ in range(20):
        params = _plane_params(labels, normal, depth)
        if len(params) < 2:
            break

        candidates = []
        for a, b in _nearby_pairs(labels, params, struct):
            n_a, d_a, cnt_a = params[a]
            n_b, d_b, cnt_b = params[b]
            if abs(np.dot(n_a, n_b)) < cos_thresh:
                continue
            if abs(d_a - d_b) > merge_offset_m:
                continue
            candidates.append((cnt_a + cnt_b, a, b))

        if not candidates:
            break

        candidates.sort(reverse=True)
        used = set()
        n_merged = 0
        for _, a, b in candidates:
            if a in used or b in used:
                continue
            labels[labels == b] = a
            used.add(b)
            n_merged += 1

        if n_merged == 0:
            break

    # Relabel contiguously
    unique = np.unique(labels)
    unique = unique[unique > 0]
    if len(unique) > 0:
        lut = np.zeros(labels.max() + 1, dtype=np.int32)
        for new_id, old_id in enumerate(unique, start=1):
            lut[old_id] = new_id
        labels = lut[labels.ravel()].reshape(labels.shape)

    return labels
