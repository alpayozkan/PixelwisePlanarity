"""
Image-based LO-RANSAC plane extraction for outdoor synthetic datasets (SYNTHIA, VKITTI2).

Key differences from shared/plane_fitting/ransac_image.py (used by ScanNet++/Hypersim):
  - Depth-adaptive distance threshold (scales with depth beyond z_ref)
  - Degenerate normal handling (skips normal check when norm < 0.5)
  - More LO iterations (10 vs 5)
  - Lower min_inliers default (50 vs 500) for smaller planes
  - Absolute merge distance floor (0.1m)
  - Stricter defaults: dist_thresh=0.01, normal_cos=0.985
"""

import numpy as np
from collections import defaultdict


def depth_to_xyz(depth, fx, fy, cx, cy):
    """Convert depth map to 3D point cloud using camera intrinsics."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    return np.stack([X, Y, depth], axis=-1).astype(np.float32)


def compute_normals(xyz, valid):
    """Compute surface normals from point cloud via finite differences."""
    dx = np.zeros_like(xyz)
    dy = np.zeros_like(xyz)
    dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dy[1:-1, :] = xyz[2:, :] - xyz[:-2, :]
    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)
    flip = normals[:, :, 2] > 0
    normals[flip] = -normals[flip]
    normals[~valid] = 0
    return normals


def fit_plane_svd(points):
    """Fit plane via SVD. Returns (n, d) where n.x = d."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n) + 1e-12
    d = float(n @ centroid)
    if d < 0:
        n, d = -n, -d
    return n, d


def _inlier_mask(dists, dots, nrm, dist_thresh, normal_cos, depths, z_ref=10.0):
    """Depth-adaptive distance gate + normal gate (skip for degenerate normals)."""
    adaptive = dist_thresh * np.maximum(1.0, depths / z_ref)
    nrm_len = np.linalg.norm(nrm, axis=-1)
    degenerate = nrm_len < 0.5
    return (dists < adaptive) & ((dots > normal_cos) | degenerate)


def local_optimize(n, d, points, nrm, dist_thresh, normal_cos, lo_iters=10):
    """Local optimization of RANSAC plane fit."""
    depths = points[:, 2]
    for _ in range(lo_iters):
        dists = np.abs(points @ n - d)
        dots = np.abs(nrm @ n)
        inlier = _inlier_mask(dists, dots, nrm, dist_thresh, normal_cos, depths)
        if inlier.sum() < 3:
            return n, d, inlier
        n, d = fit_plane_svd(points[inlier])
    dists = np.abs(points @ n - d)
    dots = np.abs(nrm @ n)
    inlier = _inlier_mask(dists, dots, nrm, dist_thresh, normal_cos, depths)
    return n, d, inlier


def ransac_with_removal(points, nrm, indices,
                        ransac_iters=500, dist_thresh=0.05,
                        normal_cos=0.94, min_inliers=50,
                        max_planes=50, lo_iters=10):
    """
    LO-RANSAC with sequential plane removal.

    Args:
        points: (N, 3) 3D points
        nrm: (N, 3) surface normals
        indices: list of (y, x) pixel coordinates
        ransac_iters: number of RANSAC iterations per plane
        dist_thresh: inlier distance threshold (meters)
        normal_cos: minimum cosine similarity for normal consistency
        min_inliers: minimum inlier count to accept a plane
        max_planes: maximum number of planes to extract
        lo_iters: local optimization iterations

    Returns:
        list of plane dicts with keys: n, d, pixel_indices, num_pixels, p95
    """
    alive = np.ones(len(points), dtype=bool)
    planes = []

    for _ in range(max_planes):
        pool = np.where(alive)[0]
        if len(pool) < min_inliers:
            break

        pts = points[pool]
        nrm_pool = nrm[pool]

        best_count = 0
        best_n = None
        best_d = None
        best_inlier = None

        for _ in range(ransac_iters):
            idx3 = np.random.choice(len(pts), 3, replace=False)
            p0, p1, p2 = pts[idx3]

            n = np.cross(p1 - p0, p2 - p0)
            nlen = np.linalg.norm(n)
            if nlen < 1e-8:
                continue
            n /= nlen
            d_val = float(n @ p0)

            dists = np.abs(pts @ n - d_val)
            dots = np.abs(nrm_pool @ n)
            inlier = _inlier_mask(dists, dots, nrm_pool, dist_thresh, normal_cos, pts[:, 2])
            count = inlier.sum()

            if count > best_count:
                n_lo, d_lo, inlier_lo = local_optimize(
                    n, d_val, pts, nrm_pool, dist_thresh, normal_cos, lo_iters)
                count_lo = inlier_lo.sum()

                if count_lo > best_count:
                    best_count = count_lo
                    best_n = n_lo
                    best_d = d_lo
                    best_inlier = inlier_lo

        if best_count < min_inliers:
            break

        inlier_global = pool[best_inlier]
        residuals = np.abs(points[inlier_global] @ best_n - best_d)
        p95 = float(np.percentile(residuals, 95))

        planes.append({
            'n': best_n, 'd': best_d,
            'pixel_indices': [indices[i] for i in inlier_global],
            'num_pixels': int(len(inlier_global)),
            'p95': p95,
        })

        alive[inlier_global] = False

    return planes


def classes_can_merge(c1, c2, merge_compatible):
    """Check if two semantic classes are allowed to merge."""
    if c1 == c2:
        return True
    for group in merge_compatible:
        if c1 in group and c2 in group:
            return True
    return False


def merge_planes(planes, xyz, merge_compatible=None,
                 normal_thresh=0.985, rel_dist_thresh=0.02,
                 abs_merge_dist=0.1):
    """
    Merge planes with similar normals and close distances using union-find.

    Args:
        planes: list of plane dicts
        xyz: (H, W, 3) point cloud
        merge_compatible: list of sets defining which class IDs can merge
        normal_thresh: cosine threshold for normal similarity
        rel_dist_thresh: relative distance threshold for merging
        abs_merge_dist: absolute floor on merge distance (meters)
    """
    if merge_compatible is None:
        merge_compatible = []

    if len(planes) < 2:
        return planes

    n_planes = len(planes)
    parent = list(range(n_planes))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            pi, pj = planes[i], planes[j]

            if not classes_can_merge(pi.get('class_id', -1),
                                     pj.get('class_id', -2),
                                     merge_compatible):
                continue

            if abs(float(pi['n'] @ pj['n'])) < normal_thresh:
                continue

            mean_d = (pi['d'] + pj['d']) / 2.0
            thresh = max(abs_merge_dist, rel_dist_thresh * mean_d)

            pix_i = pi['pixel_indices']
            pix_j = pj['pixel_indices']
            sample_i = pix_i[::max(1, len(pix_i) // 500)]
            sample_j = pix_j[::max(1, len(pix_j) // 500)]

            pts_i = np.array([xyz[y, x] for y, x in sample_i])
            pts_j = np.array([xyz[y, x] for y, x in sample_j])

            d_i_to_j = np.median(np.abs(pts_i @ pj['n'] - pj['d']))
            d_j_to_i = np.median(np.abs(pts_j @ pi['n'] - pi['d']))

            if max(d_i_to_j, d_j_to_i) < thresh:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n_planes):
        groups[find(i)].append(i)

    merged = []
    for root, members in groups.items():
        all_pixels = []
        for m in members:
            all_pixels.extend(planes[m]['pixel_indices'])

        pts = np.array([xyz[y, x] for y, x in all_pixels])
        n, d = fit_plane_svd(pts)
        residuals = np.abs(pts @ n - d)

        largest = max(members, key=lambda m: len(planes[m]['pixel_indices']))
        merged.append({
            'n': n, 'd': d,
            'pixel_indices': all_pixels,
            'num_pixels': len(all_pixels),
            'p95': float(np.percentile(residuals, 95)),
            'class_id': planes[largest].get('class_id', -1),
            'class_name': planes[largest].get('class_name', ''),
        })

    return merged


def extract_planes_from_frame(depth, class_ids, valid, xyz, normals,
                              planar_classes, class_names,
                              merge_compatible=None,
                              ransac_iters=500, dist_thresh=0.01,
                              normal_cos=0.985, min_inliers=50,
                              max_planes=50, merge_normal=0.985,
                              merge_rel_dist=0.02,
                              abs_merge_dist=0.1):
    """
    Full plane extraction pipeline for a single frame.

    Args:
        depth: (H, W) depth map in meters
        class_ids: (H, W) integer semantic class IDs
        valid: (H, W) boolean mask of valid depth pixels
        xyz: (H, W, 3) 3D point cloud
        normals: (H, W, 3) surface normals
        planar_classes: set of class IDs considered planar
        class_names: dict mapping class_id -> name string
        merge_compatible: list of sets for merge-compatible classes
        ransac_iters, dist_thresh, normal_cos, min_inliers, max_planes:
            RANSAC hyperparameters
        merge_normal, merge_rel_dist: merge thresholds

    Returns:
        plane_map: (H, W) int32 array, 0 = non-planar, 1+ = plane ID
        planes_info: list of plane dicts with metadata
    """
    H, W = depth.shape

    all_planes = []
    for label in sorted(planar_classes):
        mask = valid & (class_ids == label)
        yx = np.argwhere(mask)
        if len(yx) < min_inliers:
            continue

        pts = xyz[yx[:, 0], yx[:, 1]]
        nrm = normals[yx[:, 0], yx[:, 1]]
        indices = [(int(y), int(x)) for y, x in yx]

        name = class_names.get(label, f'Class_{label}')

        planes = ransac_with_removal(
            pts, nrm, indices,
            ransac_iters=ransac_iters,
            dist_thresh=dist_thresh,
            normal_cos=normal_cos,
            min_inliers=min_inliers,
            max_planes=max_planes,
        )
        for p in planes:
            p['class_id'] = int(label)
            p['class_name'] = name

        all_planes.extend(planes)

    # Merge
    if merge_compatible is None:
        merge_compatible = []
    all_planes = merge_planes(all_planes, xyz,
                              merge_compatible=merge_compatible,
                              normal_thresh=merge_normal,
                              rel_dist_thresh=merge_rel_dist,
                              abs_merge_dist=abs_merge_dist)

    # Build pixel map (0 = non-planar, 1+ = plane ID)
    plane_map = np.zeros((H, W), dtype=np.int32)
    for pid, p in enumerate(all_planes):
        for y, x in p['pixel_indices']:
            plane_map[y, x] = pid + 1

    return plane_map, all_planes
