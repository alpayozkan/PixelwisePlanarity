"""
Merge algorithms for plane segmentation post-processing.

merge_v5: Top-K + Union-Find + nearest-neighbor 3D proximity.
  - Only considers the K largest segments for merging
  - Uses Union-Find for transitive closure of candidate pairs
  - Uses KD-tree nearest-neighbor distance for 3D adjacency
  - EDT-based 2D adjacency for nearby segments
"""

import time
import numpy as np
from typing import Tuple
from collections import defaultdict

from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from planamono.shared.plane_fitting import backproject_v1


# ============================================================
# SHARED HELPERS
# ============================================================

def _fit_plane_svd(points):
    """Fit plane n·x + d = 0 via SVD. Returns (normal, offset, centroid)."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    d = -np.dot(n, centroid)
    return n, d, centroid


def _cross_plane_distance(pts, normal, offset, percentile=20):
    """Robust point-to-plane distance (p-th percentile of |n·p + d|)."""
    dists = np.abs(pts @ normal + offset)
    return np.percentile(dists, percentile)


def _build_point_map(depth, K, c2w, labels):
    """Backproject depth -> 3D world points. Vectorized (no Python loop)."""
    H, W = depth.shape
    pts_world, pt_labels, valid_idx = backproject_v1(depth, K, c2w, labels)
    pts_map = np.full((H, W, 3), np.nan, dtype=np.float32)
    rows = valid_idx // W
    cols = valid_idx % W
    pts_map[rows, cols] = pts_world
    return pts_map


def _fit_segment(pts_map, labels, lab, min_pixels,
                 max_fit_pts=5000, max_sample_pts=500):
    """Fit SVD plane for one segment. Returns dict or None."""
    mask = labels == lab
    count = int(mask.sum())
    if count < min_pixels:
        return None
    pts = pts_map[mask]
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if pts.shape[0] < min_pixels:
        return None

    pts_fit = pts
    if pts.shape[0] > max_fit_pts:
        pts_fit = pts[np.random.choice(pts.shape[0], max_fit_pts, replace=False)]
    try:
        n, d, centroid = _fit_plane_svd(pts_fit)
    except np.linalg.LinAlgError:
        return None

    pts_sample = pts
    if pts.shape[0] > max_sample_pts:
        pts_sample = pts[np.random.choice(pts.shape[0], max_sample_pts, replace=False)]

    return dict(normal=n, offset=d, centroid=centroid,
                count=count, pts_sample=pts_sample)


def _relabel(labels):
    """Relabel to contiguous 1..N."""
    unique = np.unique(labels)
    unique = unique[unique > 0]
    if len(unique) == 0:
        return labels
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique, start=1):
        lut[old_id] = new_id
    return lut[labels.ravel()].reshape(labels.shape)


def _check_coplanarity(pa, pb, cos_thresh, merge_offset_m, cross_dist_factor=2.0):
    """Check if two segments are coplanar. Returns (ok, angle, offset, xdist, decision)."""
    n_a, d_a = pa["normal"], pa["offset"]
    n_b, d_b = pb["normal"], pb["offset"]

    dot_ab = np.dot(n_a, n_b)
    angle = np.degrees(np.arccos(np.clip(abs(dot_ab), -1.0, 1.0)))

    if abs(dot_ab) < cos_thresh:
        return False, angle, None, None, "REJECT_NORMAL"

    # Sign-flip aware offset
    offset = abs(d_a + d_b) if dot_ab < 0 else abs(d_a - d_b)
    if offset > merge_offset_m:
        return False, angle, offset, None, "REJECT_OFFSET"

    # Cross-distance (expensive -- evaluated last)
    d_ab = _cross_plane_distance(pa["pts_sample"], n_b, d_b)
    d_ba = _cross_plane_distance(pb["pts_sample"], n_a, d_a)
    xdist = max(d_ab, d_ba)
    if xdist > merge_offset_m * cross_dist_factor:
        return False, angle, offset, xdist, "REJECT_CROSSDIST"

    return True, angle, offset, xdist, "CANDIDATE"


def _log_entry(iteration, a, b, pa, pb, angle, offset, xdist, decision, **extra):
    """Build a merge log entry dict."""
    entry = dict(
        iteration=iteration, seg_a=a, seg_b=b,
        cnt_a=pa["count"], cnt_b=pb["count"],
        angle_deg=angle, offset_diff=offset, cross_dist=xdist,
        decision=decision,
        n_a=pa["normal"].copy(), d_a=pa["offset"],
        n_b=pb["normal"].copy(), d_b=pb["offset"],
    )
    entry.update(extra)
    return entry


class _UnionFind:
    """Simple Union-Find with path compression."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if self.parent.setdefault(x, x) != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _unionfind_merge(labels, params, candidates, pts_map, min_pixels, verbose, merge_log, iteration):
    """Union-Find merge: transitive closure of all candidate pairs."""
    if not candidates:
        return 0

    uf = _UnionFind()
    for _, a, b in candidates:
        uf.union(a, b)

    groups = defaultdict(set)
    for _, a, b in candidates:
        groups[uf.find(a)].add(a)
        groups[uf.find(a)].add(b)

    n_merged = 0
    refit_labs = set()

    for root, members in groups.items():
        if len(members) < 2:
            continue

        rep = max(members, key=lambda l: params[l]["count"] if l in params else 0)
        others = members - {rep}

        for b in others:
            labels[labels == b] = rep
            params.pop(b, None)
            n_merged += 1
            if verbose:
                merge_log.append(dict(
                    iteration=iteration, seg_a=rep, seg_b=b,
                    decision="MERGED_UF", group_size=len(members)))

        refit_labs.add(rep)

    for lab in refit_labs:
        result = _fit_segment(pts_map, labels, lab, min_pixels)
        if result is not None:
            params[lab] = result
        else:
            params.pop(lab, None)

    return n_merged


# ============================================================
# merge_v5: Top-K + Union-Find + nearest-neighbor 3D proximity
# ============================================================

def merge_v5(
    labels, depth, normal, K, c2w,
    merge_normal_deg=5.0,
    merge_offset_m=0.05,
    merge_min_pixels=100,
    merge_gap_px=20,
    nn_dist_m=0.2,
    topk=20,
    max_iterations=20,
    verbose=False,
):
    """Top-K + Union-Find + nearest-neighbor 3D proximity merge.

    Only considers the K largest segments. Uses KD-tree nearest-neighbor
    distance for 3D adjacency (catches same-plane segments whose centroids
    are far apart but edges touch). EDT-based 2D adjacency for nearby segments.

    Args:
        labels: (H, W) int32 segment IDs (0 = background)
        depth: (H, W) depth in meters
        normal: (H, W, 3) surface normals
        K: (3, 3) camera intrinsics
        c2w: (4, 4) camera-to-world pose
        merge_normal_deg: Max angle between normals for merge (degrees)
        merge_offset_m: Max plane offset difference (meters)
        merge_min_pixels: Minimum segment size for merge consideration
        merge_gap_px: EDT gap bridging distance (pixels)
        nn_dist_m: Max nearest-neighbor distance for 3D adjacency (meters)
        topk: Number of largest segments to consider
        max_iterations: Maximum merge iterations
        verbose: If True, return (labels, n_merged, merge_log, timing)

    Returns:
        labels: (H, W) int32 merged segment IDs
        n_merged: Total number of segments consumed
        (merge_log, timing): Only if verbose=True
    """
    labels = labels.copy()
    H, W = labels.shape
    cos_thresh = np.cos(np.deg2rad(merge_normal_deg))

    timing = dict(backproject=0.0, plane_fitting=0.0, adjacency=0.0,
                  pair_eval=0.0, relabel=0.0, total=0.0, n_iterations=0)
    t_total = time.time()

    # Identify top-K segments by pixel count
    unique_labs, counts = np.unique(labels, return_counts=True)
    fg = unique_labs > 0
    unique_labs, counts = unique_labs[fg], counts[fg]

    if len(unique_labs) == 0:
        timing["total"] = time.time() - t_total
        if verbose:
            return labels, 0, [], timing
        return labels, 0

    order = np.argsort(-counts)
    topk_set = set(unique_labs[order[:topk]].tolist())

    t0 = time.time()
    pts_map = _build_point_map(depth, K, c2w, labels)
    timing["backproject"] = time.time() - t0

    t0 = time.time()
    params = {}
    for lab in topk_set:
        result = _fit_segment(pts_map, labels, lab, merge_min_pixels)
        if result is not None:
            params[lab] = result
    timing["plane_fitting"] += time.time() - t0

    merge_log = []
    total_merged = 0

    for it in range(max_iterations):
        timing["n_iterations"] = it + 1
        if len(params) < 2:
            break

        # --- Adjacency: single EDT (2D) + nearest-neighbor (3D) ---
        t0 = time.time()

        bg_mask = labels == 0
        expanded = labels.copy()
        if bg_mask.any():
            dist, nearest_idx = distance_transform_edt(bg_mask, return_indices=True)
            bridge = bg_mask & (dist <= merge_gap_px)
            expanded[bridge] = labels[nearest_idx[0][bridge], nearest_idx[1][bridge]]

        edt_pairs = set()
        for arr_a, arr_b in [(expanded[:, :-1], expanded[:, 1:]),
                             (expanded[:-1, :], expanded[1:, :])]:
            mask = (arr_a > 0) & (arr_b > 0) & (arr_a != arr_b)
            if mask.any():
                pairs = np.sort(np.column_stack([arr_a[mask], arr_b[mask]]), axis=1)
                pairs = np.unique(pairs, axis=0)
                for a, b in pairs:
                    a, b = int(a), int(b)
                    if a in params and b in params:
                        edt_pairs.add((a, b))

        # 3D: KD-tree nearest-neighbor between sampled point clouds
        kept = list(params.keys())
        nn_pairs = set()
        if len(kept) > 1:
            trees = {}
            for lab in kept:
                trees[lab] = cKDTree(params[lab]["pts_sample"])

            for i in range(len(kept)):
                for j in range(i + 1, len(kept)):
                    a, b = kept[i], kept[j]
                    dists_ab, _ = trees[b].query(params[a]["pts_sample"], k=1)
                    min_dist = dists_ab.min()
                    if min_dist <= nn_dist_m:
                        nn_pairs.add((min(a, b), max(a, b)))

        all_pairs = edt_pairs | nn_pairs
        timing["adjacency"] += time.time() - t0

        if not all_pairs:
            break

        # --- Vectorized pair evaluation ---
        t0 = time.time()
        pair_list = list(all_pairs)
        n_pairs = len(pair_list)

        normals_a = np.array([params[a]["normal"] for a, _ in pair_list])
        normals_b = np.array([params[b]["normal"] for _, b in pair_list])
        offsets_a = np.array([params[a]["offset"] for a, _ in pair_list])
        offsets_b = np.array([params[b]["offset"] for _, b in pair_list])

        dots = np.einsum("ij,ij->i", normals_a, normals_b)
        angles = np.degrees(np.arccos(np.clip(np.abs(dots), -1.0, 1.0)))
        normal_ok = np.abs(dots) >= cos_thresh

        offset_diffs = np.where(dots < 0,
                                np.abs(offsets_a + offsets_b),
                                np.abs(offsets_a - offsets_b))
        offset_ok = offset_diffs <= merge_offset_m

        candidates = []
        for idx in range(n_pairs):
            a, b = pair_list[idx]

            if not normal_ok[idx]:
                if verbose:
                    merge_log.append(_log_entry(
                        it, a, b, params[a], params[b],
                        angles[idx], None, None, "REJECT_NORMAL"))
                continue

            if not offset_ok[idx]:
                if verbose:
                    merge_log.append(_log_entry(
                        it, a, b, params[a], params[b],
                        angles[idx], offset_diffs[idx], None, "REJECT_OFFSET"))
                continue

            d_ab = _cross_plane_distance(params[a]["pts_sample"],
                                         params[b]["normal"], params[b]["offset"])
            d_ba = _cross_plane_distance(params[b]["pts_sample"],
                                         params[a]["normal"], params[a]["offset"])
            xdist = max(d_ab, d_ba)

            if xdist > merge_offset_m * 2.0:
                if verbose:
                    merge_log.append(_log_entry(
                        it, a, b, params[a], params[b],
                        angles[idx], offset_diffs[idx], xdist, "REJECT_CROSSDIST"))
                continue

            candidates.append((params[a]["count"] + params[b]["count"], a, b))
            if verbose:
                merge_log.append(_log_entry(
                    it, a, b, params[a], params[b],
                    angles[idx], offset_diffs[idx], xdist, "CANDIDATE"))

        timing["pair_eval"] += time.time() - t0

        if not candidates:
            break

        # --- Union-Find transitive merge ---
        t0 = time.time()
        n = _unionfind_merge(labels, params, candidates, pts_map,
                             merge_min_pixels, verbose, merge_log, it)
        timing["plane_fitting"] += time.time() - t0
        total_merged += n
        if n == 0:
            break

    t0 = time.time()
    labels = _relabel(labels)
    timing["relabel"] = time.time() - t0
    timing["total"] = time.time() - t_total

    if verbose:
        return labels, total_merged, merge_log, timing
    return labels, total_merged
