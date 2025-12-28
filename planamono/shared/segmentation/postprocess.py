"""
Post-processing utilities for plane segmentations.

This module provides functions for cleaning and refining segmentation results.
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import label
from typing import Tuple

import torch
import torch.nn.functional as F
import cc3d
import cv2

def remove_small_components(
    label_map: np.ndarray,
    min_size: int
) -> np.ndarray:
    """
    Remove small connected components from segmentation.

    Args:
        label_map: (H,W) labeled segmentation map
        min_size: Minimum number of pixels for a component to be kept

    Returns:
        cleaned_label_map: (H,W) cleaned segmentation with small components removed
    """
    cleaned_label_map = np.zeros_like(label_map)
    current_label = 1

    unique_labels = np.unique(label_map)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background

        mask = (label_map == label)
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
        points,        # (N,3) points from segment A
        plane_normal,  # (3,)
        plane_d,       # scalar
        max_samples=500,
        percentile=20
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
        points,     # (H,W,3) from MoGe
        normals,    # (3,H,W) or (H,W,3)
        normal_merge_thresh_deg=5.0,
        plane_offset_thresh=0.02,   # meters
        min_points=200
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
                idx = np.random.choice(pts.shape[0], MAX_PLANE_POINTS, replace=False)
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
                "points": pts   # 🔥 store sampled points
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
                    print('surface normal rejects')
                    continue

                # (2) plane offset
                if abs(pi["d"] - pj["d"]) > plane_offset_thresh:
                    print('plane offset rejects')
                    continue

                # (3) spatial proximity
                # if np.linalg.norm(pi["centroid"] - pj["centroid"]) > centroid_dist_thresh:
                #     print('spatial proximity rejects')
                #     continue
                dist_ij = robust_plane_distance(
                    plane_info[li]["points"], pj["normal"], pj["d"],
                    max_samples=300, percentile=20
                )
                
                dist_ji = robust_plane_distance(
                    plane_info[lj]["points"], pi["normal"], pi["d"],
                    max_samples=300, percentile=20
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


    Hm, Wm = moge_signals['points'].shape[:2]
    labels_resized = cv2.resize(
        labels.astype(np.int32),
        (Wm, Hm),
        interpolation=cv2.INTER_NEAREST
    )

    labels_merged = merge_planar_segments_moge(
        labels_resized,
        moge_signals['points'],
        np.transpose(moge_signals["normal"], (2, 0, 1)),
        normal_merge_thresh_deg=5.0,
        plane_offset_thresh=0.02,
        # centroid_dist_thresh=0.5
    )

    labels_merged = cv2.resize(
        labels_merged.astype(np.int32),
        (labels.shape[1], labels.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return labels_merged