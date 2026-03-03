"""
Merge algorithms for plane segmentation post-processing.

Three versions with different adjacency strategies and speed trade-offs:

- merge_v0: Pixel adjacency via per-segment binary dilation (original)
    + Only merges physically adjacent segments
    - Slow: O(N_segments * H * W * kernel²) for adjacency
    - Fails when non-planar gaps exceed merge_gap_px

- merge_v1: All-pairs with 3D centroid distance filter
    + Handles arbitrary 2D gaps (no pixel adjacency)
    + Simple, no dilation at all
    - May be noisy with many small segments (N² pairs)
    - centroid_dist_m needs tuning per scene depth range

- merge_v2: Fast merge — single EDT adjacency + 3D centroid fallback + vectorized filtering
    + Single EDT: O(H*W) instead of O(N * H*W) for adjacency
    + Vectorized normal/offset checks (batch dot products)
    + Incremental plane fitting (only refit merged segments)
    + Combines 2D adjacency (EDT) and 3D proximity (centroid) for coverage

- merge_v3: Top-K merge — same as v2 but only considers the K largest segments
    + Skips small segments entirely (no plane fitting, no pair evaluation)
    + Much faster when there are many small fragments
    + topk parameter controls how many segments to consider

- merge_v4: Top-K + Union-Find transitive merge
    + Same as v3 but uses Union-Find instead of greedy merge
    + If A~B and B~C, all three merge into one group (transitive closure)
    + Greedy (v0-v3) would merge A←B but skip B←C because B is consumed
    + Single-pass UF per iteration, then refit merged groups

- merge_v5: Top-K + Union-Find + nearest-neighbor 3D proximity
    + Same as v4 but replaces centroid distance with KD-tree nearest-neighbor
    + Two segments are 3D-adjacent if min point-to-point distance <= nn_dist_m
    + Catches same-plane segments whose centroids are far apart but edges touch
    + Slightly slower than centroid (KD-tree build + query) but more accurate
"""

import time
import numpy as np
from typing import Tuple, Dict, Optional

from scipy.ndimage import binary_dilation, distance_transform_edt
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
    """Backproject depth → 3D world points. Vectorized (no Python loop)."""
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


def _fit_all_segments(pts_map, labels, min_pixels):
    """Fit planes for every segment. Returns {label: params_dict}."""
    params = {}
    for lab in np.unique(labels):
        if lab == 0:
            continue
        result = _fit_segment(pts_map, labels, lab, min_pixels)
        if result is not None:
            params[lab] = result
    return params


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

    # Cross-distance (expensive — evaluated last)
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


def _greedy_merge(labels, params, candidates, pts_map, min_pixels, verbose, merge_log, iteration):
    """Greedy merge largest-first. Returns (n_merged, labels, params, merge_log)."""
    candidates.sort(reverse=True)
    used = set()
    merged_into = []
    n_merged = 0

    for _, a, b in candidates:
        if a in used or b in used:
            continue
        labels[labels == b] = a
        used.add(b)
        params.pop(b, None)
        merged_into.append(a)
        n_merged += 1
        if verbose:
            merge_log.append(dict(iteration=iteration, seg_a=a, seg_b=b, decision="MERGED"))

    # Incremental refit only merged segments
    for lab in set(merged_into):
        result = _fit_segment(pts_map, labels, lab, min_pixels)
        if result is not None:
            params[lab] = result
        else:
            params.pop(lab, None)

    return n_merged


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
    """Union-Find merge: transitive closure of all candidate pairs.

    If A~B and B~C are both candidates, all three merge into one group.
    The representative is the largest segment in each group.
    Returns n_merged (number of segments consumed).
    """
    if not candidates:
        return 0

    uf = _UnionFind()
    for _, a, b in candidates:
        uf.union(a, b)

    # Group segments by their UF root
    from collections import defaultdict
    groups = defaultdict(set)
    for _, a, b in candidates:
        groups[uf.find(a)].add(a)
        groups[uf.find(a)].add(b)

    n_merged = 0
    refit_labs = set()

    for root, members in groups.items():
        if len(members) < 2:
            continue

        # Pick the largest segment as the representative
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

    # Refit merged segments
    for lab in refit_labs:
        result = _fit_segment(pts_map, labels, lab, min_pixels)
        if result is not None:
            params[lab] = result
        else:
            params.pop(lab, None)

    return n_merged


# ============================================================
# v0: Pixel adjacency via per-segment binary dilation
# ============================================================

def merge_v0(
    labels, depth, normal, K, c2w,
    merge_normal_deg=10.0,
    merge_offset_m=0.05,
    merge_min_pixels=100,
    merge_gap_px=20,
    max_iterations=20,
    verbose=False,
):
    """v0: Pixel adjacency merge (per-segment binary dilation).

    For each segment, dilates its binary mask by merge_gap_px pixels and checks
    which other segments overlap with the dilated region. Only those pairs are
    evaluated for coplanarity.

    Slow for large merge_gap_px or many segments: O(N_seg * H * W * kernel²).
    """
    labels = labels.copy()
    cos_thresh = np.cos(np.deg2rad(merge_normal_deg))

    timing = dict(backproject=0.0, plane_fitting=0.0, adjacency=0.0,
                  pair_eval=0.0, relabel=0.0, total=0.0, n_iterations=0)
    t_total = time.time()

    # Dilation kernel (disk)
    sz = 2 * merge_gap_px + 1
    yy, xx = np.mgrid[:sz, :sz]
    struct = ((yy - merge_gap_px)**2 + (xx - merge_gap_px)**2) <= merge_gap_px**2

    t0 = time.time()
    pts_map = _build_point_map(depth, K, c2w, labels)
    timing["backproject"] = time.time() - t0

    t0 = time.time()
    params = _fit_all_segments(pts_map, labels, merge_min_pixels)
    timing["plane_fitting"] += time.time() - t0

    merge_log = []
    total_merged = 0

    for it in range(max_iterations):
        timing["n_iterations"] = it + 1
        if len(params) < 2:
            break

        # Per-segment dilation adjacency
        t0 = time.time()
        adj = set()
        for lab in params:
            mask = labels == lab
            dilated = binary_dilation(mask, structure=struct)
            for other in np.unique(labels[dilated & ~mask]):
                if other > 0 and other != lab and other in params:
                    adj.add((min(lab, other), max(lab, other)))
        timing["adjacency"] += time.time() - t0

        # Evaluate pairs
        t0 = time.time()
        candidates = []
        for a, b in adj:
            ok, angle, offset, xdist, decision = _check_coplanarity(
                params[a], params[b], cos_thresh, merge_offset_m)
            if verbose:
                merge_log.append(_log_entry(it, a, b, params[a], params[b],
                                            angle, offset, xdist, decision))
            if ok:
                candidates.append((params[a]["count"] + params[b]["count"], a, b))
        timing["pair_eval"] += time.time() - t0

        if not candidates:
            break

        t0 = time.time()
        n = _greedy_merge(labels, params, candidates, pts_map,
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


# ============================================================
# v1: All-pairs with 3D centroid distance filter
# ============================================================

def merge_v1(
    labels, depth, normal, K, c2w,
    merge_normal_deg=10.0,
    merge_offset_m=0.05,
    merge_min_pixels=100,
    centroid_dist_m=1.0,
    max_iterations=20,
    verbose=False,
):
    """v1: All-pairs merge with 3D centroid distance filter.

    No pixel adjacency — considers every pair of segments whose 3D centroids
    are within centroid_dist_m meters. Handles arbitrary 2D gaps.

    Complexity: O(N² * S) per iteration where S = sample size for cross-distance.
    Fast enough for N < ~100 segments.
    """
    labels = labels.copy()
    cos_thresh = np.cos(np.deg2rad(merge_normal_deg))

    timing = dict(backproject=0.0, plane_fitting=0.0, adjacency=0.0,
                  pair_eval=0.0, relabel=0.0, total=0.0, n_iterations=0)
    t_total = time.time()

    t0 = time.time()
    pts_map = _build_point_map(depth, K, c2w, labels)
    timing["backproject"] = time.time() - t0

    t0 = time.time()
    params = _fit_all_segments(pts_map, labels, merge_min_pixels)
    timing["plane_fitting"] += time.time() - t0

    merge_log = []
    total_merged = 0

    for it in range(max_iterations):
        timing["n_iterations"] = it + 1
        if len(params) < 2:
            break

        kept = list(params.keys())

        # Vectorized centroid distances
        t0 = time.time()
        centroids = np.array([params[l]["centroid"] for l in kept])
        diff = centroids[:, None, :] - centroids[None, :, :]
        cdist_matrix = np.linalg.norm(diff, axis=2)
        timing["adjacency"] += time.time() - t0

        # Evaluate pairs
        t0 = time.time()
        candidates = []
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a, b = kept[i], kept[j]
                cdist = cdist_matrix[i, j]

                if cdist > centroid_dist_m:
                    continue

                ok, angle, offset, xdist, decision = _check_coplanarity(
                    params[a], params[b], cos_thresh, merge_offset_m)
                if verbose:
                    merge_log.append(_log_entry(
                        it, a, b, params[a], params[b],
                        angle, offset, xdist, decision,
                        centroid_dist=cdist))
                if ok:
                    candidates.append((params[a]["count"] + params[b]["count"], a, b))
        timing["pair_eval"] += time.time() - t0

        if not candidates:
            break

        t0 = time.time()
        n = _greedy_merge(labels, params, candidates, pts_map,
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


# ============================================================
# v2: Fast merge — single EDT + centroid fallback + vectorized
# ============================================================

def merge_v2(
    labels, depth, normal, K, c2w,
    merge_normal_deg=10.0,
    merge_offset_m=0.05,
    merge_min_pixels=100,
    merge_gap_px=20,
    centroid_dist_m=1.0,
    max_iterations=20,
    verbose=False,
):
    """v2: Fast merge with EDT adjacency + 3D centroid fallback.

    Optimizations over v0 and v1:
      1. Single EDT for 2D adjacency: O(H*W) instead of O(N * H*W)
      2. 3D centroid distance as fallback for large gaps (vectorized)
      3. Batch normal/offset filtering (numpy vectorized dot products)
      4. Cross-distance only for pairs passing fast filters
      5. Incremental plane fitting (only refit merged segments)

    Combines EDT (catches nearby segments cheaply) with centroid distance
    (catches segments separated by large non-planar regions).
    """
    labels = labels.copy()
    H, W = labels.shape
    cos_thresh = np.cos(np.deg2rad(merge_normal_deg))

    timing = dict(backproject=0.0, plane_fitting=0.0, adjacency=0.0,
                  pair_eval=0.0, relabel=0.0, total=0.0, n_iterations=0)
    t_total = time.time()

    t0 = time.time()
    pts_map = _build_point_map(depth, K, c2w, labels)
    timing["backproject"] = time.time() - t0

    t0 = time.time()
    params = _fit_all_segments(pts_map, labels, merge_min_pixels)
    timing["plane_fitting"] += time.time() - t0

    merge_log = []
    total_merged = 0

    for it in range(max_iterations):
        timing["n_iterations"] = it + 1
        if len(params) < 2:
            break

        # --- Adjacency: single EDT (2D) + centroid distance (3D) ---
        t0 = time.time()

        # 2D: expand background pixels to nearest segment within gap_px
        bg_mask = labels == 0
        expanded = labels.copy()
        if bg_mask.any():
            dist, nearest_idx = distance_transform_edt(bg_mask, return_indices=True)
            bridge = bg_mask & (dist <= merge_gap_px)
            expanded[bridge] = labels[nearest_idx[0][bridge], nearest_idx[1][bridge]]

        # Find label transitions in expanded map (horizontal + vertical)
        edt_pairs = set()
        for arr_a, arr_b in [(expanded[:, :-1], expanded[:, 1:]),
                             (expanded[:-1, :], expanded[1:, :])]:
            mask = (arr_a > 0) & (arr_b > 0) & (arr_a != arr_b)
            if mask.any():
                pairs = np.sort(np.column_stack([arr_a[mask], arr_b[mask]]), axis=1)
                # Deduplicate via unique rows
                pairs = np.unique(pairs, axis=0)
                for a, b in pairs:
                    a, b = int(a), int(b)
                    if a in params and b in params:
                        edt_pairs.add((a, b))

        # 3D: vectorized centroid distances
        kept = list(params.keys())
        centroid_pairs = set()
        if len(kept) > 1:
            centroids = np.array([params[l]["centroid"] for l in kept])
            diff = centroids[:, None, :] - centroids[None, :, :]
            cdists = np.linalg.norm(diff, axis=2)
            ii, jj = np.triu_indices(len(kept), k=1)
            close_mask = cdists[ii, jj] <= centroid_dist_m
            for idx in np.where(close_mask)[0]:
                a, b = kept[ii[idx]], kept[jj[idx]]
                centroid_pairs.add((min(a, b), max(a, b)))

        all_pairs = edt_pairs | centroid_pairs
        timing["adjacency"] += time.time() - t0

        if not all_pairs:
            break

        # --- Vectorized pair evaluation ---
        t0 = time.time()
        pair_list = list(all_pairs)
        n_pairs = len(pair_list)

        # Batch normals and offsets
        normals_a = np.array([params[a]["normal"] for a, _ in pair_list])
        normals_b = np.array([params[b]["normal"] for _, b in pair_list])
        offsets_a = np.array([params[a]["offset"] for a, _ in pair_list])
        offsets_b = np.array([params[b]["offset"] for _, b in pair_list])

        # Batch dot products → angle check
        dots = np.einsum("ij,ij->i", normals_a, normals_b)
        angles = np.degrees(np.arccos(np.clip(np.abs(dots), -1.0, 1.0)))
        normal_ok = np.abs(dots) >= cos_thresh

        # Batch offset check (sign-flip aware)
        offset_diffs = np.where(dots < 0,
                                np.abs(offsets_a + offsets_b),
                                np.abs(offsets_a - offsets_b))
        offset_ok = offset_diffs <= merge_offset_m

        # Only compute cross-distance for pairs passing both fast filters
        pass_fast = normal_ok & offset_ok

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

            # Cross-distance (most expensive check — only for survivors)
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

        t0 = time.time()
        n = _greedy_merge(labels, params, candidates, pts_map,
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



# ============================================================
# v3: Top-K merge — only consider the K largest segments
# ============================================================

def merge_v3(
    labels, depth, normal, K, c2w,
    merge_normal_deg=10.0,
    merge_offset_m=0.05,
    merge_min_pixels=100,
    merge_gap_px=20,
    centroid_dist_m=1.0,
    topk=20,
    max_iterations=20,
    verbose=False,
):
    """v3: Top-K merge — v2 logic but only considers the K largest segments.

    Segments outside the top-K are left untouched (not merged, not removed).
    This reduces plane fitting from O(N) to O(K), pair evaluation from O(N²)
    to O(K²), and EDT/centroid filtering only checks top-K labels.

    Args:
        topk: Number of largest segments to consider for merging.
              Smaller segments are preserved as-is.
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

    if verbose:
        n_total = len(unique_labs)
        print(f"  v3: {n_total} segments, keeping top-{min(topk, n_total)} "
              f"(smallest kept: {counts[order[min(topk, n_total)-1]]} px)")

    t0 = time.time()
    pts_map = _build_point_map(depth, K, c2w, labels)
    timing["backproject"] = time.time() - t0

    # Only fit planes for top-K segments
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

        # --- Adjacency: single EDT (2D) + centroid distance (3D) ---
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
                    # Only pairs where BOTH are in top-K params
                    if a in params and b in params:
                        edt_pairs.add((a, b))

        kept = list(params.keys())
        centroid_pairs = set()
        if len(kept) > 1:
            centroids = np.array([params[l]["centroid"] for l in kept])
            diff = centroids[:, None, :] - centroids[None, :, :]
            cdists = np.linalg.norm(diff, axis=2)
            ii, jj = np.triu_indices(len(kept), k=1)
            close_mask = cdists[ii, jj] <= centroid_dist_m
            for idx in np.where(close_mask)[0]:
                a, b = kept[ii[idx]], kept[jj[idx]]
                centroid_pairs.add((min(a, b), max(a, b)))

        all_pairs = edt_pairs | centroid_pairs
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

        t0 = time.time()
        n = _greedy_merge(labels, params, candidates, pts_map,
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



# ============================================================
# v4: Top-K + Union-Find transitive merge
# ============================================================

def merge_v4(
    labels, depth, normal, K, c2w,
    merge_normal_deg=10.0,
    merge_offset_m=0.05,
    merge_min_pixels=100,
    merge_gap_px=20,
    centroid_dist_m=1.0,
    topk=20,
    max_iterations=20,
    verbose=False,
):
    """v4: Top-K + Union-Find transitive merge.

    Same adjacency and filtering as v3, but instead of greedy merge
    (which can break chains like A~B, B~C), uses Union-Find to merge
    all transitively connected candidates in one shot.

    Example: if A~B and B~C are both candidates, v3 greedy might merge
    A←B but skip B←C (B consumed). v4 merges {A,B,C} into one group.
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

    if verbose:
        n_total = len(unique_labs)
        print(f"  v4: {n_total} segments, keeping top-{min(topk, n_total)} "
              f"(smallest kept: {counts[order[min(topk, n_total)-1]]} px)")

    t0 = time.time()
    pts_map = _build_point_map(depth, K, c2w, labels)
    timing["backproject"] = time.time() - t0

    # Only fit planes for top-K segments
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

        # --- Adjacency: single EDT (2D) + centroid distance (3D) ---
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

        kept = list(params.keys())
        centroid_pairs = set()
        if len(kept) > 1:
            centroids = np.array([params[l]["centroid"] for l in kept])
            diff = centroids[:, None, :] - centroids[None, :, :]
            cdists = np.linalg.norm(diff, axis=2)
            ii, jj = np.triu_indices(len(kept), k=1)
            close_mask = cdists[ii, jj] <= centroid_dist_m
            for idx in np.where(close_mask)[0]:
                a, b = kept[ii[idx]], kept[jj[idx]]
                centroid_pairs.add((min(a, b), max(a, b)))

        all_pairs = edt_pairs | centroid_pairs
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

        # --- Union-Find transitive merge (instead of greedy) ---
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


# ============================================================
# v5: Top-K + Union-Find + nearest-neighbor 3D proximity
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
    """v5: Top-K + Union-Find + nearest-neighbor 3D proximity.

    Same as v4 but replaces centroid distance with KD-tree nearest-neighbor
    distance for 3D adjacency. Two segments are considered 3D-adjacent if
    the minimum point-to-point distance between their sampled point clouds
    is <= nn_dist_m.

    This catches same-plane segments whose centroids are far apart but whose
    edges are close in 3D (e.g., a large wall split by an occluding object).

    Args:
        nn_dist_m: Maximum nearest-neighbor distance (meters) between two
                   segments' point clouds to consider them 3D-adjacent.
                   Replaces centroid_dist_m from v4.
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

    if verbose:
        n_total = len(unique_labs)
        print(f"  v5: {n_total} segments, keeping top-{min(topk, n_total)} "
              f"(smallest kept: {counts[order[min(topk, n_total)-1]]} px)")

    t0 = time.time()
    pts_map = _build_point_map(depth, K, c2w, labels)
    timing["backproject"] = time.time() - t0

    # Only fit planes for top-K segments
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
