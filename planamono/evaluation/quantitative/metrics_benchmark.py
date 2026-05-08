"""
Benchmark metric kernels combining ZeroPlane + PlaneRCNN + PlaneRecTR.

The goal is a single self-contained module that produces every plane-eval
metric used across the three reference repositories. All kernels are pure
numpy / torch and take **planamono-native** inputs (labels with 0 =
non-planar, plane_params dict in Hessian normal form, depth/normal as
ndarrays). Internal adapters remap to the conventions each source repo
expects.

Source files merged here:
- ZeroPlane    /utils/metrics.py             evaluateMasks, eval_plane_recall_depth/normal
- ZeroPlane    /utils/metrics_de.py          evaluateDepths
- ZeroPlane    /utils/metrics_onlyparams.py  eval_plane_bestmatch_normal_offset
- PlaneRecTR   /utils/metrics.py             eval_plane_recall_offset (ZeroPlane has it
                                              but never wires it in)
- PlaneRCNN    /evaluate_utils.py            evaluatePlanesTensor (AP @ τ)
                                             evaluatePlaneDepth (param L2 from depth)

Top-level entry point: ``compute_benchmark_metrics()``. Returns a flat
``Dict[str, float]`` ready to be written to a per-frame CSV row.

Conventions
-----------
- ``seg`` arrays use 0 for non-planar; positive ints for plane instances
  (not necessarily contiguous). Internally densified to 0..N-1 with
  non-planar mapped to ``NONPLANAR_IDX = 20`` before the ZP-style kernels
  are called.
- ``plane_params_dict``: ``{label_id: (a, b, c, d)}`` in standard Hessian
  normal form ``aX + bY + cZ + d = 0`` with ``‖(a,b,c)‖ = 1``. Internally
  converted to n/d form ``a'X + b'Y + c'Z = 1`` with
  ``‖(a',b',c')‖ = 1/|d|`` for ZP-style kernels.
- AP scoring (PlaneRCNN) needs per-prediction confidence scores. planamono
  predictions don't carry scores, so we use **plane area (pixel count) as
  proxy** — predictions sorted descending. This is documented in
  ``compute_benchmark_metrics``.

Densification details
---------------------
``_densify_labels`` produces a per-frame **dynamic non-planar sentinel**:
the dense seg uses labels 0..N-1 for planes and N for non-planar, where N
is the number of distinct planes in that frame. Each metric kernel takes
``pred_plane_num`` / ``gt_plane_num`` explicitly so its inner ``range(N)``
loops scale to the actual plane count. This avoids the
``NONPLANAR_IDX=20`` over-cap that silently dropped MoGe predictions
(MoGe routinely produces 50+ planes per frame; ZeroPlane is bounded by its
20-query budget by construction).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.optimize
import scipy.spatial.distance
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ZeroPlane uses 20 (= num_queries) as the non-planar sentinel in its own H5
# output. The driver's `_load_zeroplane_scene` remaps to planamono's 0=non-
# planar convention before any metric is computed. Inside this module the
# non-planar sentinel is **dynamic per frame** — see `_densify_labels`.
ZP_NONPLANAR_IDX = 20

# ZeroPlane indoor depth recall: 13 thresholds @ 5 cm stride, 0..0.6 m
INDOOR_DEPTH_THRESHOLDS = np.arange(13) * 0.05               # m
# ZeroPlane outdoor depth recall: 13 thresholds @ 1 m stride, 0..12 m
OUTDOOR_DEPTH_THRESHOLDS = np.arange(13) * 1.0               # m
# Normal recall: 13 thresholds, 0..30°
NORMAL_THRESHOLDS_DEG = np.linspace(0.0, 30.0, 13)
# Offset recall: 13 thresholds, 0..300 mm
OFFSET_THRESHOLDS_MM = np.linspace(0.0, 300.0, 13)
# PlaneRCNN AP thresholds (m)
AP_DEPTH_THRESHOLDS = (0.2, 0.3, 0.6, 0.9)


# ---------------------------------------------------------------------------
# Adapters: planamono → ZP/PlaneRCNN convention
# ---------------------------------------------------------------------------

def _densify_labels(
    seg: np.ndarray,
    plane_params: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, int], int]:
    """Remap an arbitrary-int label map (0 = non-planar) to dense 0..N-1.

    The non-planar sentinel is **N itself** (one above the max plane index),
    so every kernel that iterates ``range(N)`` builds masks for real planes
    only. Callers receive ``n_planes`` directly and pass it to each kernel
    that needs an upper bound.

    Returns ``(dense_seg, params_array, original_to_dense, n_planes)``.
    ``params_array`` is shape ``(N, 3)`` in n/d form (a*x+b*y+c*z=1)
    ordered to match the dense plane indices, or ``None`` if no params were
    provided.
    """
    seg = np.asarray(seg)
    plane_ids = sorted(int(p) for p in np.unique(seg) if int(p) != 0)
    n_planes = len(plane_ids)
    nonplanar_idx = n_planes  # sentinel = N, dense plane labels are 0..N-1

    out = np.full_like(seg, fill_value=nonplanar_idx, dtype=np.int32)
    mapping: Dict[int, int] = {}
    for new_id, orig_id in enumerate(plane_ids):
        mapping[orig_id] = new_id
        out[seg == orig_id] = new_id

    params_arr: Optional[np.ndarray] = None
    if plane_params is not None:
        rows: List[np.ndarray] = []
        for orig_id in plane_ids:
            p = plane_params.get(orig_id)
            if p is None:
                # Missing param — emit a sentinel that won't match anything.
                rows.append(np.array([0.0, 0.0, 1e9], dtype=np.float64))
                continue
            p = np.asarray(p, dtype=np.float64)
            # Convert from Hessian (A,B,C,D) ‖n‖=1 to n/d (a,b,c) ‖·‖=1/|D|.
            #   AX + BY + CZ + D = 0  ⇔  (-A/D)X + (-B/D)Y + (-C/D)Z = 1
            if p.size == 4:
                A, B, C, D = float(p[0]), float(p[1]), float(p[2]), float(p[3])
                if abs(D) < 1e-12:
                    rows.append(np.array([0.0, 0.0, 1e9], dtype=np.float64))
                else:
                    rows.append(np.array([-A / D, -B / D, -C / D], dtype=np.float64))
            elif p.size == 3:
                # Already in n/d form.
                rows.append(p.astype(np.float64))
            else:
                raise ValueError(f"plane_params[{orig_id}] has unexpected shape {p.shape}")
        params_arr = np.stack(rows, axis=0) if rows else np.zeros((0, 3), dtype=np.float64)

    return out, params_arr, mapping, n_planes


# ---------------------------------------------------------------------------
# 1. RI / VI / SC  — ZeroPlane evaluateMasks (≡ PlaneRCNN evaluateMasksTensor)
# ---------------------------------------------------------------------------

def evaluate_masks(
    pred_seg_dense: np.ndarray,
    gt_seg_dense: np.ndarray,
    pred_non_plane_idx: int,
    gt_non_plane_idx: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """Rand Index / Variation of Information / Segmentation Covering.

    Faithful port of ZeroPlane ``evaluateMasks`` (= PlaneRCNN
    ``evaluateMasksTensor``). Both inputs must be **dense** label maps with
    non-planar at ``*_non_plane_idx``.

    The reduction is masked to GT-planar pixels, so non-planar background
    does not contribute to the numerator.

    Implementation: bincount-based ``(G+1, P+1)`` matrix instead of the
    reference ``(G+1, P+1, H, W)`` torch stack. Memory O(H·W + G·P).
    Mathematically equivalent for RI/VI/SC (extra zero rows/cols from
    densification contribute zero to every sum and zero to every max).
    """
    G = int(gt_non_plane_idx) + 1
    P = int(pred_non_plane_idx) + 1

    # Restrict to GT-planar pixels (matches ZP's `valid_mask = gt_masks.max(0)`).
    valid = gt_seg_dense < gt_non_plane_idx
    if not valid.any():
        return {"RI": float("nan"), "VI": float("nan"), "SC": float("nan")}

    # Cap any out-of-range labels to the non-planar bin (defensive — should
    # be a no-op for inputs from `_densify_labels`).
    gt = np.minimum(gt_seg_dense, gt_non_plane_idx).astype(np.int64)
    pred = np.minimum(pred_seg_dense, pred_non_plane_idx).astype(np.int64)

    g_flat = gt[valid]
    p_flat = pred[valid]
    lin = g_flat * P + p_flat
    intersection_np = np.bincount(lin, minlength=G * P).reshape(G, P).astype(np.float64)

    # |gt_g| over valid pixels; |pred_p| over valid pixels (both equal the
    # rows/columns of `intersection_np` summed).
    gt_areas = intersection_np.sum(axis=1)                          # (G+1,)
    pred_areas = intersection_np.sum(axis=0)                        # (P+1,)

    intersection_t = torch.from_numpy(intersection_np).to(device)
    gt_areas_t = torch.from_numpy(gt_areas).to(device)
    pred_areas_t = torch.from_numpy(pred_areas).to(device)

    N = intersection_t.sum()
    if N <= 1:
        return {"RI": float("nan"), "VI": float("nan"), "SC": float("nan")}

    RI = 1 - (
        (gt_areas_t.pow(2).sum() + pred_areas_t.pow(2).sum()) / 2
        - intersection_t.pow(2).sum()
    ) / (N * (N - 1) / 2)

    joint = intersection_t / N
    marg2 = joint.sum(0)
    marg1 = joint.sum(1)
    H_1 = (-marg1 * torch.log2(marg1 + (marg1 == 0).float())).sum()
    H_2 = (-marg2 * torch.log2(marg2 + (marg2 == 0).float())).sum()
    B = marg1.unsqueeze(-1) * marg2
    log2q = torch.log2(torch.clamp(joint, 1e-8) / torch.clamp(B, 1e-8)) * (
        torch.min(joint, B) > 1e-8
    ).float()
    MI = (joint * log2q).sum()
    voi = H_1 + H_2 - 2 * MI

    union = gt_areas_t.unsqueeze(1) + pred_areas_t.unsqueeze(0) - intersection_t
    IOU = intersection_t / torch.clamp(union, min=1)
    sc_a = (IOU.max(-1)[0] * torch.clamp(gt_areas_t, min=1e-4)).sum() / N
    sc_b = (IOU.max(0)[0] * torch.clamp(pred_areas_t, min=1e-4)).sum() / N
    SC = (sc_a + sc_b) / 2

    return {"RI": float(RI), "VI": float(voi), "SC": float(SC)}


# ---------------------------------------------------------------------------
# 2. Depth metrics — ZeroPlane evaluateDepths
# ---------------------------------------------------------------------------

def evaluate_depths(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    pred_seg_dense: np.ndarray,
    gt_seg_dense: np.ndarray,
    pred_nonplanar_idx: int,
    gt_nonplanar_idx: int,
    max_depth: float = 10.0,
    prefix: str = "DE",
) -> Dict[str, float]:
    """Faithful port of ZeroPlane ``evaluateDepths`` (≡ PlaneRecTR /
    PlaneRCNN with ZeroPlane's max_depth + valid-mask additions).

    Returns 8 metrics keyed ``<prefix>_rel/_rel_sqr/_log10/_rmse/_rmse_log
    /_accuracy_{1,2,3}``. ``prefix='DE'`` reproduces ZP's plane-depth eval;
    ``prefix='pixel_DE'`` reproduces ZP's per-pixel-depth eval.

    The two ``*_nonplanar_idx`` arguments are the dynamic non-planar sentinels
    of the densified pred / gt seg maps (= number of planes in each).
    """
    pred = pred_depth.astype(np.float64).copy()
    gt = gt_depth.astype(np.float64).copy()

    # Planar pixels are anything below the (per-side) sentinel.
    pred_planar = pred_seg_dense < pred_nonplanar_idx
    gt_planar = gt_seg_dense < gt_nonplanar_idx

    valid_gt = (gt > 1e-4) & (gt < max_depth)
    valid = valid_gt & pred_planar & gt_planar

    keys = [f"{prefix}_{k}" for k in
            ("rel", "rel_sqr", "log10", "rmse", "rmse_log",
             "accuracy_1", "accuracy_2", "accuracy_3")]

    if not valid.any():
        return {k: float("nan") for k in keys}

    gt_v = gt[valid]
    pred_v = pred[valid]
    pred_v = np.where(pred_v > max_depth, max_depth, pred_v)

    n = float(gt_v.size)
    rmse = np.sqrt(((pred_v - gt_v) ** 2).sum() / n)
    rmse_log = np.sqrt(
        (
            (np.log(np.maximum(pred_v, 1e-4)) - np.log(np.maximum(gt_v, 1e-4))) ** 2
        ).sum()
        / n
    )
    log10 = np.abs(
        np.log10(np.maximum(pred_v, 1e-4)) - np.log10(np.maximum(gt_v, 1e-4))
    ).sum() / n
    rel = (np.abs(pred_v - gt_v) / np.maximum(gt_v, 1e-4)).sum() / n
    rel_sqr = ((pred_v - gt_v) ** 2 / np.maximum(gt_v, 1e-4)).sum() / n
    deltas = np.maximum(
        pred_v / np.maximum(gt_v, 1e-4), gt_v / np.maximum(pred_v, 1e-4)
    )
    a1 = float((deltas < 1.25).sum() / n)
    a2 = float((deltas < 1.25 ** 2).sum() / n)
    a3 = float((deltas < 1.25 ** 3).sum() / n)

    return dict(zip(keys, [float(rel), float(rel_sqr), float(log10),
                            float(rmse), float(rmse_log), a1, a2, a3]))


# ---------------------------------------------------------------------------
# 3. Plane recall by per-pixel depth diff — ZeroPlane eval_plane_recall_depth
# ---------------------------------------------------------------------------

def eval_plane_recall_depth(
    pred_seg_dense: np.ndarray,
    gt_seg_dense: np.ndarray,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    pred_plane_num: int,
    gt_plane_num: int,
    threshold: float = 0.5,
    eval_indoor: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port. Returns (pixelRecalls[13], planeStatistics[13, 3]).

    planeStatistics columns: (n_recalled, gt_plane_num, pred_plane_num).
    Both plane counts are passed in explicitly (dynamic per-frame sentinel
    convention; see ``_densify_labels``).
    """
    if gt_plane_num == 0:
        out_stats = np.zeros((13, 3))
        return np.zeros(13), out_stats

    # Bincount-based (G, P) contraction. Memory O(H·W + G·P) instead of
    # O(H·W·G·P) — critical for MoGe predictions with 50+ planes per frame.
    valid = (gt_seg_dense < gt_plane_num) & (pred_seg_dense < pred_plane_num)
    g_flat = gt_seg_dense[valid].astype(np.int64)
    p_flat = pred_seg_dense[valid].astype(np.int64)
    diff_flat = np.abs(gt_depth - pred_depth)[valid].astype(np.float64)

    n_bins = gt_plane_num * pred_plane_num
    if g_flat.size == 0:
        intersection = np.zeros((gt_plane_num, pred_plane_num), dtype=np.float64)
        sum_diff = np.zeros((gt_plane_num, pred_plane_num), dtype=np.float64)
    else:
        lin = g_flat * pred_plane_num + p_flat
        intersection = np.bincount(lin, minlength=n_bins).reshape(
            gt_plane_num, pred_plane_num
        ).astype(np.float64)
        sum_diff = np.bincount(lin, weights=diff_flat, minlength=n_bins).reshape(
            gt_plane_num, pred_plane_num
        )

    plane_diffs = np.where(
        intersection >= 1e-4, sum_diff / np.maximum(intersection, 1e-4), 1.0
    )

    # Areas over the full image (matches the original reduction).
    plane_areas = np.bincount(
        gt_seg_dense.flatten(), minlength=gt_plane_num + 1
    )[:gt_plane_num].astype(np.float64)
    pred_areas = np.bincount(
        pred_seg_dense.flatten(), minlength=pred_plane_num + 1
    )[:pred_plane_num].astype(np.float64)
    union = plane_areas[:, None] + pred_areas[None, :] - intersection
    plane_IOUs = intersection / np.maximum(union, 1e-4)

    num_predictions = int((pred_areas > 0).sum())
    num_pixels = float(plane_areas.sum())

    iou_mask = (plane_IOUs > threshold).astype(np.float32)
    min_diff = np.min(plane_diffs * iou_mask + 1e6 * (1 - iou_mask), axis=1)

    if eval_indoor:
        thresholds = INDOOR_DEPTH_THRESHOLDS
    else:
        thresholds = OUTDOOR_DEPTH_THRESHOLDS

    pixel_recalls = np.zeros(13, dtype=np.float64)
    plane_stats = np.zeros((13, 3), dtype=np.float64)

    for k, diff in enumerate(thresholds):
        pixel_recalls[k] = (
            np.minimum(
                (intersection * (plane_diffs <= diff).astype(np.float32) * iou_mask).sum(1),
                plane_areas,
            ).sum()
            / max(num_pixels, 1.0)
        )
        plane_stats[k] = (
            float((min_diff <= diff).sum()),
            float(gt_plane_num),
            float(num_predictions),
        )
    return pixel_recalls, plane_stats


# ---------------------------------------------------------------------------
# 4 & 5. Plane recall by normal angle / offset — ZP & PlaneRecTR
# ---------------------------------------------------------------------------

def _eval_iou(annotation: np.ndarray, segmentation: np.ndarray) -> float:
    a = annotation.astype(bool)
    s = segmentation.astype(bool)
    if np.isclose(a.sum(), 0) and np.isclose(s.sum(), 0):
        return 1.0
    return float((a & s).sum()) / float(np.maximum((a | s).sum(), 1))


def eval_plane_recall_normal(
    pred_seg_dense: np.ndarray,
    gt_seg_dense: np.ndarray,
    pred_params_nd: np.ndarray,
    gt_params_nd: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plane normal-angle recall curve.

    Faithful port of ZeroPlane ``eval_plane_recall_normal`` /
    PlaneRecTR's identical version. ``*_params_nd`` are in n/d form
    (``aX + bY + cZ = 1``); the unit normal is ``param / ‖param‖``.
    """
    return _eval_recall_param(
        pred_seg_dense, gt_seg_dense, pred_params_nd, gt_params_nd,
        thresholds=NORMAL_THRESHOLDS_DEG,
        cost_fn=_normal_angle_cost,
        threshold=threshold,
    )


def eval_plane_recall_offset(
    pred_seg_dense: np.ndarray,
    gt_seg_dense: np.ndarray,
    pred_params_nd: np.ndarray,
    gt_params_nd: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plane offset (mm) recall curve.

    Faithful port of PlaneRecTR's ``eval_plane_recall_offset`` (function
    exists in ZeroPlane but is not wired into its evaluator).
    """
    return _eval_recall_param(
        pred_seg_dense, gt_seg_dense, pred_params_nd, gt_params_nd,
        thresholds=OFFSET_THRESHOLDS_MM,
        cost_fn=_offset_mm_cost,
        threshold=threshold,
    )


def _normal_angle_cost(p_pred: np.ndarray, p_gt: np.ndarray) -> float:
    nrm_pred = np.linalg.norm(p_pred) + 1e-12
    nrm_gt = np.linalg.norm(p_gt) + 1e-12
    n_pred = p_pred / nrm_pred
    n_gt = p_gt / nrm_gt
    angle = np.arccos(np.clip(np.dot(n_pred, n_gt), -1.0, 1.0))
    return float(np.degrees(angle))


def _offset_mm_cost(p_pred: np.ndarray, p_gt: np.ndarray) -> float:
    off_pred = 1.0 / (np.linalg.norm(p_pred) + 1e-12)
    off_gt = 1.0 / (np.linalg.norm(p_gt) + 1e-12)
    return float(abs(off_pred - off_gt) * 1000.0)


def _eval_recall_param(
    pred_seg_dense: np.ndarray,
    gt_seg_dense: np.ndarray,
    pred_params_nd: np.ndarray,
    gt_params_nd: np.ndarray,
    thresholds: np.ndarray,
    cost_fn,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    plane_num = len(pred_params_nd)
    gt_plane_num = len(gt_params_nd)
    if gt_plane_num == 0 or plane_num == 0:
        return np.zeros((13, 3)), np.zeros(13)

    plane_recall = np.zeros((gt_plane_num, len(thresholds)), dtype=np.float64)
    pixel_recall = np.zeros((gt_plane_num, len(thresholds)), dtype=np.float64)
    plane_area = 0.0

    for i in range(gt_plane_num):
        gt_plane = gt_seg_dense == i
        plane_area += float(gt_plane.sum())
        for j in range(plane_num):
            pred_plane = pred_seg_dense == j
            if _eval_iou(gt_plane, pred_plane) > threshold:
                err = cost_fn(pred_params_nd[j], gt_params_nd[i])
                hit = (err < thresholds).astype(np.float32)
                plane_recall[i] = hit
                pixel_recall[i] = hit * float((gt_plane & pred_plane).sum())
                break

    pixel_recall = pixel_recall.sum(axis=0) / max(plane_area, 1.0)
    plane_recall_new = np.zeros((len(thresholds), 3), dtype=np.float64)
    plane_recall_new[:, 0] = plane_recall.sum(axis=0)
    plane_recall_new[:, 1] = gt_plane_num
    plane_recall_new[:, 2] = plane_num
    return plane_recall_new, pixel_recall


# ---------------------------------------------------------------------------
# 6. Best-match (Hungarian) normal/offset error
# ---------------------------------------------------------------------------

def eval_plane_bestmatch_normal_offset(
    pred_params_nd: np.ndarray,
    gt_params_nd: np.ndarray,
) -> Dict[str, float]:
    """Hungarian match on L1 cost between n/d params; return mean+median
    angular error (deg) and offset error (m).

    Faithful port of ZeroPlane ``eval_plane_bestmatch_normal_offset``.
    Unlike ZeroPlane's evaluator (which discards offset), we keep both.
    """
    if len(pred_params_nd) == 0 or len(gt_params_nd) == 0:
        return {
            "mean_normal_error_deg": float("nan"),
            "median_normal_error_deg": float("nan"),
            "mean_offset_error_m": float("nan"),
            "median_offset_error_m": float("nan"),
        }
    C = scipy.spatial.distance.cdist(pred_params_nd, gt_params_nd, "minkowski", p=1)
    rows, cols = scipy.optimize.linear_sum_assignment(C)

    p_norm = np.linalg.norm(pred_params_nd, axis=1) + 1e-8
    g_norm = np.linalg.norm(gt_params_nd, axis=1) + 1e-8
    p_offset = 1.0 / p_norm
    g_offset = 1.0 / g_norm
    p_n = pred_params_nd / p_norm.reshape(-1, 1)
    g_n = gt_params_nd / g_norm.reshape(-1, 1)

    angle = np.arccos(np.clip(np.einsum("ij,kj->ik", p_n, g_n), -1.0, 1.0))
    deg = np.degrees(angle)
    matched_deg = deg[rows, cols]
    matched_off = np.abs(p_offset[rows] - g_offset[cols])

    return {
        "mean_normal_error_deg": float(np.mean(matched_deg)),
        "median_normal_error_deg": float(np.median(matched_deg)),
        "mean_offset_error_m": float(np.mean(matched_off)),
        "median_offset_error_m": float(np.median(matched_off)),
    }


# ---------------------------------------------------------------------------
# 7. AP @ τ — PlaneRCNN evaluatePlanesTensor
# ---------------------------------------------------------------------------

def evaluate_planes_ap(
    pred_seg_dense: np.ndarray,
    gt_seg_dense: np.ndarray,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    pred_plane_num: int,
    gt_plane_num: int,
    iou_threshold: float = 0.5,
    diff_thresholds: Tuple[float, ...] = AP_DEPTH_THRESHOLDS,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Faithful port of PlaneRCNN ``evaluatePlanesTensor``.

    AP @ each ``diff_threshold`` (meters). A prediction is "correct" w.r.t.
    a GT plane iff IoU > 0.5 AND mean intersection-depth-diff < threshold.
    Standard rank → precision/recall integration.

    Predictions are ranked by ``scores`` (high → low). If ``scores`` is
    ``None`` we use **plane area as proxy** (largest plane first).

    Both plane counts are passed in explicitly (dynamic per-frame sentinel
    convention; see ``_densify_labels``).
    """
    keys = [f"AP@{int(round(t * 100))}cm" for t in diff_thresholds]
    if gt_plane_num == 0 or pred_plane_num == 0:
        return {k: float("nan") for k in keys}

    # Bincount-based (G, P) contraction. Memory O(H·W + G·P) instead of
    # O(H·W·G·P).
    plane_areas = np.bincount(
        gt_seg_dense.flatten(), minlength=gt_plane_num + 1
    )[:gt_plane_num].astype(np.float64)
    pred_areas = np.bincount(
        pred_seg_dense.flatten(), minlength=pred_plane_num + 1
    )[:pred_plane_num].astype(np.float64)

    valid = (gt_seg_dense < gt_plane_num) & (pred_seg_dense < pred_plane_num)
    g_flat = gt_seg_dense[valid].astype(np.int64)
    p_flat = pred_seg_dense[valid].astype(np.int64)
    diff = np.abs(gt_depth - pred_depth)
    diff = np.where(gt_depth < 1e-4, 0.0, diff)
    diff_flat = diff[valid].astype(np.float64)

    n_bins = gt_plane_num * pred_plane_num
    if g_flat.size == 0:
        intersection = np.zeros((gt_plane_num, pred_plane_num), dtype=np.float64)
        sum_diff = np.zeros((gt_plane_num, pred_plane_num), dtype=np.float64)
    else:
        lin = g_flat * pred_plane_num + p_flat
        intersection = np.bincount(lin, minlength=n_bins).reshape(
            gt_plane_num, pred_plane_num
        ).astype(np.float64)
        sum_diff = np.bincount(lin, weights=diff_flat, minlength=n_bins).reshape(
            gt_plane_num, pred_plane_num
        )

    union = plane_areas[:, None] + pred_areas[None, :] - intersection
    plane_IOUs = intersection / np.maximum(union, 1e-4)
    depths_diff = np.where(
        intersection >= 1e-4, sum_diff / np.maximum(intersection, 1e-4), 1e6
    )

    # Score-rank predictions (descending). plane area as proxy if no scores.
    if scores is None:
        scores = pred_areas.copy()
    rank_order = np.argsort(-np.asarray(scores))                  # (P,)
    plane_IOUs = plane_IOUs[:, rank_order]
    depths_diff = depths_diff[:, rank_order]

    out: Dict[str, float] = {}
    for thr, key in zip(diff_thresholds, keys):
        correct = np.minimum(
            (depths_diff < thr).astype(np.float32),
            (plane_IOUs > iou_threshold).astype(np.float32),
        )                                                          # (G, P_ranked)
        # Per PlaneRCNN: accumulate "any rank ≤ r matched this GT plane" — match
        # is per-GT, not per-prediction.
        match = np.zeros(correct.shape[0], dtype=bool)
        recalls: List[float] = []
        precisions: List[float] = []
        num_targets = int((plane_areas > 0).sum())
        if num_targets == 0:
            out[key] = float("nan")
            continue
        for rank in range(correct.shape[1]):
            match = np.maximum(match, correct[:, rank].astype(bool))
            num_matches = int(match.sum())
            precisions.append(num_matches / (rank + 1))
            recalls.append(num_matches / num_targets)
        # Smoothed AP (PlaneRCNN style).
        AP = 0.0
        max_p = 0.0
        prev_r = 1.0
        for r, p in zip(recalls[::-1], precisions[::-1]):
            AP += (prev_r - r) * max_p
            max_p = max(max_p, p)
            prev_r = r
        AP += prev_r * max_p
        out[key] = float(AP)
    return out


# ---------------------------------------------------------------------------
# 8. Plane-parameter L2 from depth — PlaneRCNN evaluatePlaneDepth
# ---------------------------------------------------------------------------

def evaluate_plane_depth_param_l2(
    gt_seg_dense: np.ndarray,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    K: np.ndarray,
    gt_plane_num: int,
) -> Dict[str, float]:
    """Faithful port of PlaneRCNN ``evaluatePlaneDepth``.

    For each GT plane mask, fit n/d² · n parameters to backprojected XYZ
    from (depth_pred, depth_gt). Return area-weighted and unweighted L2
    diff means.

    Uses pinhole backprojection with the supplied K. PlaneRCNN's original
    implementation crops to ``[80:560]`` (640-wide images); we evaluate
    the full frame since we're not bound to NYU 640×640 inputs.
    """
    H, W = gt_depth.shape
    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rx = (us.astype(np.float64) - cx) / fx
    ry = (vs.astype(np.float64) - cy) / fy
    rays = np.stack([rx, ry, np.ones_like(rx)], axis=-1)              # (H, W, 3)

    if gt_plane_num == 0:
        return {
            "plane_param_L2_mean": float("nan"),
            "plane_param_L2_area_weighted": float("nan"),
        }

    plane_diff = np.full(gt_plane_num, np.nan, dtype=np.float64)
    plane_area = np.zeros(gt_plane_num, dtype=np.float64)

    for i in range(gt_plane_num):
        mask = gt_seg_dense == i
        plane_area[i] = float(mask.sum())
        if plane_area[i] < 3:
            continue
        params: List[np.ndarray] = []
        for d in (pred_depth, gt_depth):
            xyz = rays * d[..., None]
            A = xyz[mask]                                            # (M, 3)
            b = np.ones(A.shape[0], dtype=np.float64)                # ax + by + cz = 1
            try:
                p, *_ = np.linalg.lstsq(A, b, rcond=None)            # (3,)
            except np.linalg.LinAlgError:
                p = np.array([np.nan, np.nan, np.nan])
            # PlaneRCNN normalisation: p / max(‖p‖², 1e-4)
            offset_sq = max(float(np.dot(p, p)), 1e-4)
            params.append(p / offset_sq)
        plane_diff[i] = float(np.linalg.norm(params[0] - params[1]))

    valid = np.isfinite(plane_diff)
    if not valid.any():
        return {
            "plane_param_L2_mean": float("nan"),
            "plane_param_L2_area_weighted": float("nan"),
        }
    mean_diff = float(plane_diff[valid].mean())
    if plane_area[valid].sum() > 0:
        weighted = float(
            (plane_diff[valid] * plane_area[valid]).sum() / plane_area[valid].sum()
        )
    else:
        weighted = float("nan")
    return {
        "plane_param_L2_mean": mean_diff,
        "plane_param_L2_area_weighted": weighted,
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _expand_recall_curve(
    pixel_curve: np.ndarray,
    plane_curve: np.ndarray,
    keys_pixel: List[Tuple[int, str]],
    keys_plane: List[Tuple[int, str]],
    pixel_prefix: str,
    plane_prefix: str,
) -> Dict[str, float]:
    """Expand a 13-step recall curve into named columns. ``keys_*`` is a list
    of ``(curve_index, suffix)`` pairs."""
    out: Dict[str, float] = {}
    for idx, suffix in keys_pixel:
        out[f"{pixel_prefix}_{suffix}"] = (
            float(pixel_curve[idx]) if pixel_curve.size else float("nan")
        )
    for idx, suffix in keys_plane:
        if plane_curve.size and plane_curve[idx, 1] > 0:
            out[f"{plane_prefix}_{suffix}"] = float(plane_curve[idx, 0] / plane_curve[idx, 1])
        else:
            out[f"{plane_prefix}_{suffix}"] = float("nan")
    return out


def compute_benchmark_metrics(
    *,
    pred_seg: np.ndarray,
    gt_seg: np.ndarray,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    pred_plane_params: Dict[int, np.ndarray],
    gt_plane_params: Dict[int, np.ndarray],
    K: np.ndarray,
    pixel_depth_pred: Optional[np.ndarray] = None,
    eval_indoor: bool = True,
    max_depth: float = 10.0,
    scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Run all benchmark metrics on a single frame.

    Inputs are in **planamono native** format:
    - ``*_seg``: int32 ndarray (H, W), 0 = non-planar.
    - ``*_depth``: float32 ndarray (H, W), meters.
    - ``*_plane_params``: ``{label_id: (a, b, c, d)}`` Hessian normal form
      with ``‖(a,b,c)‖ = 1``. The plane equation is ``aX + bY + cZ + d = 0``.
    - ``K``: (3, 3) intrinsics matched to the depth-map resolution.
    - ``pixel_depth_pred``: optional separate per-pixel-depth-head output.
      If supplied, ZP's ``pixel_DE_*`` block is also populated.
    - ``scores``: optional per-prediction confidences for AP. If absent,
      plane area is used as a proxy.

    Returns a flat dict with all reported metric columns.
    """
    pred_dense, pred_params_arr, _, n_pred_planes = _densify_labels(
        pred_seg, pred_plane_params
    )
    gt_dense, gt_params_arr, _, n_gt_planes = _densify_labels(
        gt_seg, gt_plane_params
    )

    out: Dict[str, float] = {}

    # 1. RI / VI / SC
    out.update(evaluate_masks(
        pred_dense, gt_dense,
        pred_non_plane_idx=n_pred_planes,
        gt_non_plane_idx=n_gt_planes,
    ))

    # 2. Depth quality (plane-rendered depth)
    out.update(evaluate_depths(
        pred_depth, gt_depth, pred_dense, gt_dense,
        pred_nonplanar_idx=n_pred_planes,
        gt_nonplanar_idx=n_gt_planes,
        max_depth=max_depth, prefix="DE",
    ))

    # 2b. Optional per-pixel-depth-head quality
    if pixel_depth_pred is not None:
        out.update(evaluate_depths(
            pixel_depth_pred, gt_depth, pred_dense, gt_dense,
            pred_nonplanar_idx=n_pred_planes,
            gt_nonplanar_idx=n_gt_planes,
            max_depth=max_depth, prefix="pixel_DE",
        ))

    # 3. Plane recall by depth diff
    pix_d, pln_d = eval_plane_recall_depth(
        pred_dense, gt_dense, pred_depth, gt_depth,
        pred_plane_num=n_pred_planes,
        gt_plane_num=n_gt_planes,
        eval_indoor=eval_indoor,
    )
    if eval_indoor:
        # Indoor reports: pixel @ 0.10 m / 0.60 m; plane @ 0.05 / 0.10 / 0.60 m
        out.update(_expand_recall_curve(
            pix_d, pln_d,
            keys_pixel=[(2, "01"), (12, "06")],
            keys_plane=[(1, "005"), (2, "01"), (12, "06")],
            pixel_prefix="per_pixel_depth",
            plane_prefix="per_plane_depth",
        ))
    else:
        # Outdoor reports: pixel @ 1 / 10 m; plane @ 1 / 3 / 10 m
        out.update(_expand_recall_curve(
            pix_d, pln_d,
            keys_pixel=[(1, "1"), (10, "10")],
            keys_plane=[(1, "1"), (3, "3"), (10, "10")],
            pixel_prefix="per_pixel_depth",
            plane_prefix="per_plane_depth",
        ))

    # 4. Plane recall by normal angle
    pln_n, pix_n = eval_plane_recall_normal(
        pred_dense, gt_dense, pred_params_arr, gt_params_arr,
    )
    out.update(_expand_recall_curve(
        pix_n, pln_n,
        keys_pixel=[(2, "5"), (12, "30")],
        keys_plane=[(2, "5"), (4, "10"), (12, "30")],
        pixel_prefix="per_pixel_normal",
        plane_prefix="per_plane_normal",
    ))

    # 5. Plane recall by offset (PlaneRecTR; ZP dropped this)
    pln_o, pix_o = eval_plane_recall_offset(
        pred_dense, gt_dense, pred_params_arr, gt_params_arr,
    )
    # PlaneRecTR reports offsets at indices ~6 (≈150 mm) and full 300 mm; we
    # surface a reasonable spread: 50 mm (idx 2), 150 mm (idx 6), 300 mm (12).
    out.update(_expand_recall_curve(
        pix_o, pln_o,
        keys_pixel=[(2, "50mm"), (6, "150mm"), (12, "300mm")],
        keys_plane=[(2, "50mm"), (6, "150mm"), (12, "300mm")],
        pixel_prefix="per_pixel_offset",
        plane_prefix="per_plane_offset",
    ))

    # 6. Best-match (Hungarian) normal/offset error
    out.update(eval_plane_bestmatch_normal_offset(pred_params_arr, gt_params_arr))

    # 7. AP @ τ (PlaneRCNN)
    out.update(evaluate_planes_ap(
        pred_dense, gt_dense, pred_depth, gt_depth,
        pred_plane_num=n_pred_planes,
        gt_plane_num=n_gt_planes,
        scores=scores,
    ))

    # 8. PlaneRCNN plane-parameter L2 from depth
    out.update(evaluate_plane_depth_param_l2(
        gt_dense, pred_depth, gt_depth, K,
        gt_plane_num=n_gt_planes,
    ))

    return out
