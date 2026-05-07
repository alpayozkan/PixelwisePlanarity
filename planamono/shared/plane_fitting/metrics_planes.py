"""
Pixel-aligned plane-segmentation evaluation metrics.

Bundles three families:
  1. Segmentation:   RI, VOI, SC                          (`compute_segmentation_metrics`)
  2. Plane recall:   per-GT-plane recall @ depth / normal (`plane_recall_at_depth`,
                                                          `plane_recall_at_normal`)
  3. Direct error:   per-plane normal-angle and offset    (`per_plane_error_stats`)

Plus helpers:
  - `compute_gt_normals_from_depth_labels` — SVD per-plane fit when only
                                              depth_gt + labels_gt + K are available.
  - `match_planes_by_overlap`  — argmax-overlap GT→pred matching.
  - `aggregate_plane_normals` / `aggregate_plane_depths` — per-plane reductions.

Top-level convenience: `evaluate_plane_predictions(...)` runs everything.

Inputs throughout are pixel-aligned `(H, W)` / `(H, W, 3)` arrays — directly
compatible with the per-pixel outputs from this repo's inference scripts
(`labels`, `depth`, `normal`). Plane labels follow the repo convention
(`0` = background, positive ints = plane instances), and plane parameter
dicts (e.g. from `compute_plane_params`) can be converted to per-pixel maps
trivially: assign `(a, b, c)` and the rendered depth to every pixel of
the corresponding mask.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .metrics import segmentation_covering_fast


__all__ = [
    "compute_gt_normals_from_depth_labels",
    "compute_segmentation_metrics",
    "match_planes_by_overlap",
    "aggregate_plane_normals",
    "aggregate_plane_depths",
    "plane_recall_at_depth",
    "plane_recall_at_normal",
    "per_plane_error_stats",
    "evaluate_plane_predictions",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ignore_set(labels: Iterable[int]) -> set:
    return {int(x) for x in labels}


def compute_gt_normals_from_depth_labels(
    depth_gt: np.ndarray,
    labels_gt: np.ndarray,
    K: np.ndarray,
    ignore_labels: Iterable[int] = (0,),
    orient_positive_z: bool = True,
    min_pixels: int = 3,
) -> np.ndarray:
    """Fit a single plane normal per GT label via SVD on backprojected points,
    then broadcast that normal to every pixel of the label.

    Args:
        depth_gt:       (H, W) GT depth in meters
        labels_gt:      (H, W) integer label map
        K:              (3, 3) camera intrinsics
        ignore_labels:  label IDs to skip (default (0,) = background)
        orient_positive_z: flip per-plane normal so n[2] > 0 (matches the
                        spec doc; toggle off if your convention differs)
        min_pixels:     minimum valid-depth pixels per plane to attempt SVD

    Returns:
        (H, W, 3) array. Pixels outside fitted planes are zeros.
    """
    H, W = depth_gt.shape
    out = np.zeros((H, W, 3), dtype=np.float64)

    K = np.asarray(K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_gt.astype(np.float64)
    valid_z = np.isfinite(Z) & (Z > 0)
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1)

    ignore = _ignore_set(ignore_labels)
    for pid in np.unique(labels_gt):
        if int(pid) in ignore:
            continue
        m = (labels_gt == pid) & valid_z
        if int(m.sum()) < min_pixels:
            continue
        plane_pts = pts[m]
        c = plane_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(plane_pts - c, full_matrices=False)
        n = Vt[-1]
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-12:
            continue
        n = n / nrm
        if orient_positive_z and n[2] < 0:
            n = -n
        out[m] = n

    return out


# --------------------------------------------------------------------------- #
# 1. Segmentation metrics: RI, VOI, SC
# --------------------------------------------------------------------------- #

def compute_segmentation_metrics(
    labels_gt: np.ndarray,
    labels_pred: np.ndarray,
) -> Dict[str, float]:
    """Rand Index (sklearn), Variation of Information (skimage Hs+Hm),
    Segmentation Covering (this repo).

    Both inputs must be the same `(H, W)` shape. Background-aware filtering
    is the caller's responsibility — see `evaluate_plane_predictions` for an
    example that does no filtering (matching the user spec).
    """
    from sklearn.metrics import rand_score
    from skimage.metrics import variation_of_information

    gt = np.asarray(labels_gt)
    pr = np.asarray(labels_pred)
    ri = float(rand_score(gt.ravel(), pr.ravel()))
    Hs, Hm = variation_of_information(gt, pr)
    voi = float(Hs + Hm)
    sc = segmentation_covering_fast(gt, pr)
    return {"rand_index": ri, "voi": voi, "sc": sc}


# --------------------------------------------------------------------------- #
# 2. Plane matching (argmax overlap by GT plane)
# --------------------------------------------------------------------------- #

def match_planes_by_overlap(
    labels_gt: np.ndarray,
    labels_pred: np.ndarray,
    ignore_labels_gt: Iterable[int] = (0,),
    ignore_labels_pred: Iterable[int] = (0,),
) -> Dict[int, Tuple[int, int]]:
    """For each GT plane id, return `{gt_pid: (pred_pid, overlap_pixels)}`
    where pred_pid is the predicted label with the largest pixel overlap on
    that GT plane (excluding ignored predicted labels).

    GT planes whose only overlapping pred labels are all in
    `ignore_labels_pred` are dropped from the result.
    """
    ignore_g = _ignore_set(ignore_labels_gt)
    ignore_p = _ignore_set(ignore_labels_pred)

    out: Dict[int, Tuple[int, int]] = {}
    gt_unique = np.unique(labels_gt)
    for gpid in gt_unique:
        gpid_int = int(gpid)
        if gpid_int in ignore_g:
            continue
        m = (labels_gt == gpid)
        if not m.any():
            continue
        pred_in_gt = labels_pred[m]
        pids, counts = np.unique(pred_in_gt, return_counts=True)
        keep = np.array([int(p) not in ignore_p for p in pids])
        if not keep.any():
            continue
        pids = pids[keep]
        counts = counts[keep]
        idx = int(counts.argmax())
        out[gpid_int] = (int(pids[idx]), int(counts[idx]))
    return out


# --------------------------------------------------------------------------- #
# 3. Per-plane reductions
# --------------------------------------------------------------------------- #

def aggregate_plane_normals(
    normals: np.ndarray,
    labels: np.ndarray,
    ignore_labels: Iterable[int] = (0,),
) -> Dict[int, np.ndarray]:
    """Mean unit normal per label. Pixels with zero-norm normals are excluded."""
    ignore = _ignore_set(ignore_labels)
    n_arr = np.asarray(normals, dtype=np.float64)
    valid_n = np.linalg.norm(n_arr, axis=-1) > 1e-9

    out: Dict[int, np.ndarray] = {}
    for pid in np.unique(labels):
        if int(pid) in ignore:
            continue
        m = (labels == pid) & valid_n
        if not m.any():
            continue
        n = n_arr[m].mean(axis=0)
        nrm = float(np.linalg.norm(n))
        if nrm < 1e-12:
            continue
        out[int(pid)] = n / nrm
    return out


def aggregate_plane_depths(
    depth: np.ndarray,
    labels: np.ndarray,
    ignore_labels: Iterable[int] = (0,),
) -> Dict[int, float]:
    """Mean valid depth per label (positive, finite)."""
    ignore = _ignore_set(ignore_labels)
    z = np.asarray(depth, dtype=np.float64)
    valid_z = np.isfinite(z) & (z > 0)
    out: Dict[int, float] = {}
    for pid in np.unique(labels):
        if int(pid) in ignore:
            continue
        m = (labels == pid) & valid_z
        if not m.any():
            continue
        out[int(pid)] = float(z[m].mean())
    return out


# --------------------------------------------------------------------------- #
# 4. Plane recall metrics
# --------------------------------------------------------------------------- #

def plane_recall_at_depth(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    labels_pred: np.ndarray,
    labels_gt: np.ndarray,
    threshold: float,
    matches: Optional[Dict[int, Tuple[int, int]]] = None,
    ignore_labels_gt: Iterable[int] = (0,),
    ignore_labels_pred: Iterable[int] = (0,),
) -> Dict[str, float]:
    """Fraction of GT planes whose matched pred plane has |Δmean_depth| < threshold.

    `n_total`   = number of GT planes (after filtering ignore_labels_gt).
    `n_matched` = GT planes that found any pred match.
    `n_within`  = matched GT planes within threshold (numerator of recall).
    `recall`    = n_within / n_total.
    """
    if matches is None:
        matches = match_planes_by_overlap(
            labels_gt, labels_pred, ignore_labels_gt, ignore_labels_pred
        )
    pred_d = aggregate_plane_depths(depth_pred, labels_pred, ignore_labels_pred)
    gt_d = aggregate_plane_depths(depth_gt, labels_gt, ignore_labels_gt)

    n_total = len(gt_d)
    n_matched = 0
    n_within = 0
    for gpid, dg in gt_d.items():
        if gpid not in matches:
            continue
        ppid, _ = matches[gpid]
        if ppid not in pred_d:
            continue
        n_matched += 1
        if abs(pred_d[ppid] - dg) < threshold:
            n_within += 1

    recall = n_within / n_total if n_total > 0 else 0.0
    return {
        "recall": recall,
        "n_total": n_total,
        "n_matched": n_matched,
        "n_within": n_within,
    }


def plane_recall_at_normal(
    normals_pred: np.ndarray,
    normals_gt: np.ndarray,
    labels_pred: np.ndarray,
    labels_gt: np.ndarray,
    threshold_deg: float,
    matches: Optional[Dict[int, Tuple[int, int]]] = None,
    ignore_labels_gt: Iterable[int] = (0,),
    ignore_labels_pred: Iterable[int] = (0,),
) -> Dict[str, float]:
    """Fraction of GT planes whose matched pred mean-normal is within
    `threshold_deg` of the GT mean-normal (sign-agnostic via |dot|)."""
    if matches is None:
        matches = match_planes_by_overlap(
            labels_gt, labels_pred, ignore_labels_gt, ignore_labels_pred
        )
    pred_n = aggregate_plane_normals(normals_pred, labels_pred, ignore_labels_pred)
    gt_n = aggregate_plane_normals(normals_gt, labels_gt, ignore_labels_gt)

    n_total = len(gt_n)
    n_matched = 0
    n_within = 0
    cos_thr = float(np.cos(np.deg2rad(threshold_deg)))
    for gpid, ng in gt_n.items():
        if gpid not in matches:
            continue
        ppid, _ = matches[gpid]
        if ppid not in pred_n:
            continue
        n_matched += 1
        c = float(abs(np.dot(pred_n[ppid], ng)))
        if c > 1.0:
            c = 1.0
        if c >= cos_thr:
            n_within += 1

    recall = n_within / n_total if n_total > 0 else 0.0
    return {
        "recall": recall,
        "n_total": n_total,
        "n_matched": n_matched,
        "n_within": n_within,
    }


# --------------------------------------------------------------------------- #
# 5. Direct per-plane error stats
# --------------------------------------------------------------------------- #

def _summarize(values: List[float], prefix: str) -> Dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean":   float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std":    float("nan"),
            f"{prefix}_n":      0,
        }
    a = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean":   float(a.mean()),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_std":    float(a.std()),
        f"{prefix}_n":      int(a.size),
    }


def per_plane_error_stats(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    normals_pred: np.ndarray,
    normals_gt: np.ndarray,
    labels_pred: np.ndarray,
    labels_gt: np.ndarray,
    matches: Optional[Dict[int, Tuple[int, int]]] = None,
    ignore_labels_gt: Iterable[int] = (0,),
    ignore_labels_pred: Iterable[int] = (0,),
) -> Dict[str, float]:
    """Per-plane normal-angle (degrees) and depth-offset (meters) errors,
    summarized as mean / median / std / n."""
    if matches is None:
        matches = match_planes_by_overlap(
            labels_gt, labels_pred, ignore_labels_gt, ignore_labels_pred
        )
    pred_n = aggregate_plane_normals(normals_pred, labels_pred, ignore_labels_pred)
    gt_n = aggregate_plane_normals(normals_gt, labels_gt, ignore_labels_gt)
    pred_d = aggregate_plane_depths(depth_pred, labels_pred, ignore_labels_pred)
    gt_d = aggregate_plane_depths(depth_gt, labels_gt, ignore_labels_gt)

    n_errs: List[float] = []
    d_errs: List[float] = []
    for gpid in matches.keys():
        ppid, _ = matches[gpid]
        if gpid in gt_n and ppid in pred_n:
            c = float(abs(np.dot(pred_n[ppid], gt_n[gpid])))
            if c > 1.0:
                c = 1.0
            n_errs.append(float(np.degrees(np.arccos(c))))
        if gpid in gt_d and ppid in pred_d:
            d_errs.append(float(abs(pred_d[ppid] - gt_d[gpid])))

    out: Dict[str, float] = {}
    out.update(_summarize(n_errs, "normal_err_deg"))
    out.update(_summarize(d_errs, "offset_err_m"))
    return out


# --------------------------------------------------------------------------- #
# 6. Top-level driver
# --------------------------------------------------------------------------- #

def evaluate_plane_predictions(
    depth_pred: np.ndarray,
    normals_pred: np.ndarray,
    labels_pred: np.ndarray,
    depth_gt: np.ndarray,
    labels_gt: np.ndarray,
    normals_gt: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    depth_thresholds_m: Tuple[float, ...] = (0.05, 0.1, 0.2),
    normal_thresholds_deg: Tuple[float, ...] = (5.0, 10.0, 20.0),
    ignore_labels_gt: Iterable[int] = (0,),
    ignore_labels_pred: Iterable[int] = (0,),
) -> Dict[str, float]:
    """End-to-end evaluator. Computes RI / VOI / SC, plane-recall @ each
    depth and normal threshold, and per-plane normal/offset error stats.

    If `normals_gt` is None, it is computed via SVD from
    `depth_gt + labels_gt + K`.

    Output keys (flat dict, ready for CSV rows):
        rand_index, voi, sc
        plane_recall_d_<mm>mm                    for each depth threshold
        plane_recall_n_<deg>deg                  for each normal threshold
        normal_err_deg_{mean,median,std,n}
        offset_err_m_{mean,median,std,n}
    """
    if normals_gt is None:
        if K is None:
            raise ValueError(
                "normals_gt is None — pass K to compute via SVD, or supply normals_gt"
            )
        normals_gt = compute_gt_normals_from_depth_labels(
            depth_gt, labels_gt, K, ignore_labels=ignore_labels_gt
        )

    matches = match_planes_by_overlap(
        labels_gt, labels_pred, ignore_labels_gt, ignore_labels_pred
    )

    out: Dict[str, float] = {}
    out.update(compute_segmentation_metrics(labels_gt, labels_pred))

    for thr in depth_thresholds_m:
        r = plane_recall_at_depth(
            depth_pred, depth_gt, labels_pred, labels_gt, thr,
            matches=matches,
            ignore_labels_gt=ignore_labels_gt,
            ignore_labels_pred=ignore_labels_pred,
        )
        out[f"plane_recall_d_{int(round(thr * 1000))}mm"] = r["recall"]

    for thr_deg in normal_thresholds_deg:
        r = plane_recall_at_normal(
            normals_pred, normals_gt, labels_pred, labels_gt, thr_deg,
            matches=matches,
            ignore_labels_gt=ignore_labels_gt,
            ignore_labels_pred=ignore_labels_pred,
        )
        out[f"plane_recall_n_{int(round(thr_deg))}deg"] = r["recall"]

    out.update(per_plane_error_stats(
        depth_pred, depth_gt, normals_pred, normals_gt,
        labels_pred, labels_gt,
        matches=matches,
        ignore_labels_gt=ignore_labels_gt,
        ignore_labels_pred=ignore_labels_pred,
    ))

    out["n_gt_planes"] = int(len(np.unique(labels_gt)) - sum(
        1 for x in ignore_labels_gt if x in np.unique(labels_gt)
    ))
    out["n_pred_planes"] = int(len(np.unique(labels_pred)) - sum(
        1 for x in ignore_labels_pred if x in np.unique(labels_pred)
    ))
    return out
