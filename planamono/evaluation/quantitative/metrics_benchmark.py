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

NONPLANAR_IDX = 20      # ZeroPlane convention; matches num_queries=20

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
    nonplanar_idx: int = NONPLANAR_IDX,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, int]]:
    """Remap an arbitrary-int label map (0 = non-planar) to dense 0..N-1
    with non-planar set to ``nonplanar_idx``.

    Returns (dense_seg, params_array, original_to_dense). ``params_array``
    is shape ``(N, 3)`` in n/d form (a*x+b*y+c*z=1) ordered to match the
    dense plane indices, or ``None`` if no params were provided.
    """
    seg = np.asarray(seg)
    plane_ids = sorted(int(p) for p in np.unique(seg) if int(p) != 0)

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

    return out, params_arr, mapping


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

    The reduction is masked to GT-planar pixels (``valid_mask``), so
    non-planar background does not contribute to the numerator.
    """
    pred = torch.from_numpy(pred_seg_dense.astype(np.int64)).to(device)
    gt = torch.from_numpy(gt_seg_dense.astype(np.int64)).to(device)

    # Build per-plane mask stacks (drop empty masks, as ZP does).
    pred_masks = []
    for i in range(pred_non_plane_idx):
        m = (pred == i).float()
        if m.sum() > 0:
            pred_masks.append(m)
    if not pred_masks:
        return {"RI": float("nan"), "VI": float("nan"), "SC": float("nan")}
    pred_masks = torch.stack(pred_masks, dim=0)

    gt_masks = []
    for i in range(gt_non_plane_idx):
        m = (gt == i).float()
        if m.sum() > 0:
            gt_masks.append(m)
    if not gt_masks:
        return {"RI": float("nan"), "VI": float("nan"), "SC": float("nan")}
    gt_masks = torch.stack(gt_masks, dim=0)

    valid_mask = gt_masks.max(0)[0].unsqueeze(0)

    # Append non-planar pseudo-mask so rows/cols sum to full image.
    gt_masks = torch.cat(
        [gt_masks, torch.clamp(1 - gt_masks.sum(0, keepdim=True), min=0)], dim=0
    )
    pred_masks = torch.cat(
        [pred_masks, torch.clamp(1 - pred_masks.sum(0, keepdim=True), min=0)], dim=0
    )

    intersection = (gt_masks.unsqueeze(1) * pred_masks * valid_mask).sum(-1).sum(-1)
    union = (torch.max(gt_masks.unsqueeze(1), pred_masks) * valid_mask).sum(-1).sum(-1)

    N = intersection.sum()
    if N <= 1:
        return {"RI": float("nan"), "VI": float("nan"), "SC": float("nan")}

    RI = 1 - (
        (intersection.sum(0).pow(2).sum() + intersection.sum(1).pow(2).sum()) / 2
        - intersection.pow(2).sum()
    ) / (N * (N - 1) / 2)

    joint = intersection / N
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

    IOU = intersection / torch.clamp(union, min=1)
    sc_a = (
        IOU.max(-1)[0]
        * torch.clamp((gt_masks * valid_mask).sum(-1).sum(-1), min=1e-4)
    ).sum() / N
    sc_b = (
        IOU.max(0)[0]
        * torch.clamp((pred_masks * valid_mask).sum(-1).sum(-1), min=1e-4)
    ).sum() / N
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
    nonplanar_idx: int = NONPLANAR_IDX,
    max_depth: float = 10.0,
    prefix: str = "DE",
) -> Dict[str, float]:
    """Faithful port of ZeroPlane ``evaluateDepths`` (≡ PlaneRecTR /
    PlaneRCNN with ZeroPlane's max_depth + valid-mask additions).

    Returns 8 metrics keyed ``<prefix>_rel/_rel_sqr/_log10/_rmse/_rmse_log
    /_accuracy_{1,2,3}``. ``prefix='DE'`` reproduces ZP's plane-depth eval;
    ``prefix='pixel_DE'`` reproduces ZP's per-pixel-depth eval.
    """
    pred = pred_depth.astype(np.float64).copy()
    gt = gt_depth.astype(np.float64).copy()

    # ZP uses < 20 (i.e. < non_plane_idx) as the planar mask.
    pred_planar = pred_seg_dense < nonplanar_idx
    gt_planar = gt_seg_dense < nonplanar_idx

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
    threshold: float = 0.5,
    eval_indoor: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Faithful port. Returns (pixelRecalls[13], planeStatistics[13, 3]).

    planeStatistics columns: (n_recalled, gt_plane_num, pred_plane_num).
    """
    if NONPLANAR_IDX in np.unique(gt_seg_dense):
        gt_plane_num = len(np.unique(gt_seg_dense)) - 1
    else:
        gt_plane_num = len(np.unique(gt_seg_dense))

    if gt_plane_num == 0:
        out_stats = np.zeros((13, 3))
        return np.zeros(13), out_stats

    H, W = gt_seg_dense.shape
    gt_oh = (gt_seg_dense[..., None] == np.arange(gt_plane_num)).astype(np.float32)
    pred_oh = (pred_seg_dense[..., None] == np.arange(pred_plane_num)).astype(np.float32)

    plane_areas = gt_oh.sum(axis=(0, 1))
    intersection_mask = (gt_oh[..., None] * pred_oh[:, :, None, :]) > 0.5
    depth_diff = np.abs(gt_depth - pred_depth)[..., None, None]

    intersection = intersection_mask.astype(np.float32).sum(axis=(0, 1))
    plane_diffs = (depth_diff * intersection_mask).sum(axis=(0, 1)) / np.maximum(intersection, 1e-4)
    plane_diffs[intersection < 1e-4] = 1.0

    union = ((gt_oh[..., None] + pred_oh[:, :, None, :]) > 0.5).astype(np.float32).sum(axis=(0, 1))
    plane_IOUs = intersection / np.maximum(union, 1e-4)

    num_predictions = int(pred_oh.max(axis=(0, 1)).sum())
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
    nonplanar_idx: int = NONPLANAR_IDX,
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
    """
    if NONPLANAR_IDX in np.unique(gt_seg_dense):
        gt_plane_num = len(np.unique(gt_seg_dense)) - 1
    else:
        gt_plane_num = len(np.unique(gt_seg_dense))

    keys = [f"AP@{int(round(t * 100))}cm" for t in diff_thresholds]
    if gt_plane_num == 0 or pred_plane_num == 0:
        return {k: float("nan") for k in keys}

    H, W = gt_seg_dense.shape
    gt_masks = (gt_seg_dense[..., None] == np.arange(gt_plane_num)).astype(np.float32)
    pred_masks = (pred_seg_dense[..., None] == np.arange(pred_plane_num)).astype(np.float32)

    plane_areas = gt_masks.sum(axis=(0, 1))                       # (G,)
    pred_areas = pred_masks.sum(axis=(0, 1))                      # (P,)

    # Score-rank predictions (descending). plane area as proxy if no scores.
    if scores is None:
        scores = pred_areas.copy()
    rank_order = np.argsort(-np.asarray(scores))                  # (P,)
    pred_masks_ranked = pred_masks[..., rank_order]

    # (G, P) overlaps via einsum
    intersection = np.einsum("hwg,hwp->gp", gt_masks, pred_masks_ranked)
    union = (
        gt_masks.sum(axis=(0, 1))[:, None]
        + pred_masks_ranked.sum(axis=(0, 1))[None, :]
        - intersection
    )
    plane_IOUs = intersection / np.maximum(union, 1e-4)

    depth_diff = np.abs(gt_depth - pred_depth)
    depth_diff = np.where(gt_depth < 1e-4, 0.0, depth_diff)
    intersection_mask = (gt_masks[..., None] * pred_masks_ranked[:, :, None, :]) > 0.5
    depths_diff = (depth_diff[..., None, None] * intersection_mask).sum(axis=(0, 1)) / np.maximum(
        intersection, 1e-4
    )
    depths_diff = np.where(intersection < 1e-4, 1e6, depths_diff)

    out: Dict[str, float] = {}
    for thr, key in zip(diff_thresholds, keys):
        correct = np.minimum(
            (depths_diff < thr).astype(np.float32),
            (plane_IOUs > iou_threshold).astype(np.float32),
        )                                                          # (G, P_ranked)
        match = np.zeros(correct.shape[1], dtype=bool)             # which pred ranks were used
        recalls: List[float] = []
        precisions: List[float] = []
        num_targets = int((plane_areas > 0).sum())
        if num_targets == 0:
            out[key] = float("nan")
            continue
        # Iterate ranks in order (1..P).
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
    nonplanar_idx: int = NONPLANAR_IDX,
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

    if NONPLANAR_IDX in np.unique(gt_seg_dense):
        gt_plane_num = len(np.unique(gt_seg_dense)) - 1
    else:
        gt_plane_num = len(np.unique(gt_seg_dense))

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
    pred_dense, pred_params_arr, _ = _densify_labels(pred_seg, pred_plane_params)
    gt_dense, gt_params_arr, _ = _densify_labels(gt_seg, gt_plane_params)

    pred_plane_num = pred_dense.max() + 1 if (pred_dense != NONPLANAR_IDX).any() else 0
    pred_plane_num = int(pred_plane_num) if pred_plane_num != NONPLANAR_IDX else 0
    if pred_plane_num >= NONPLANAR_IDX:
        pred_plane_num = NONPLANAR_IDX

    out: Dict[str, float] = {}

    # 1. RI / VI / SC
    out.update(evaluate_masks(
        pred_dense, gt_dense,
        pred_non_plane_idx=NONPLANAR_IDX,
        gt_non_plane_idx=NONPLANAR_IDX,
    ))

    # 2. Depth quality (plane-rendered depth)
    out.update(evaluate_depths(
        pred_depth, gt_depth, pred_dense, gt_dense,
        max_depth=max_depth, prefix="DE",
    ))

    # 2b. Optional per-pixel-depth-head quality
    if pixel_depth_pred is not None:
        out.update(evaluate_depths(
            pixel_depth_pred, gt_depth, pred_dense, gt_dense,
            max_depth=max_depth, prefix="pixel_DE",
        ))

    # 3. Plane recall by depth diff
    pix_d, pln_d = eval_plane_recall_depth(
        pred_dense, gt_dense, pred_depth, gt_depth,
        pred_plane_num=NONPLANAR_IDX,
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
        pred_plane_num=NONPLANAR_IDX,
        scores=scores,
    ))

    # 8. PlaneRCNN plane-parameter L2 from depth
    out.update(evaluate_plane_depth_param_l2(
        gt_dense, pred_depth, gt_depth, K,
    ))

    return out
