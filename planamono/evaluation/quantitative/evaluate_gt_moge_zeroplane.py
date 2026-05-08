"""
Multi-dataset, 3-method plane-segmentation evaluation.

Methods:
    gt          GT labels — upper bound. Pred normals are computed from
                (depth_gt, labels_gt, K) via compute_gt_normals_from_depth_labels.
    moge        Reads moge_signals.h5 (planarity, normal, depth_metric) and runs
                compute_vectorized_planar_segments_v5_relative with the user-spec
                params: planarity>0.3, normal<5°, depth_rel<0.025, match≥8.
    zeroplane   Reads pre-computed planes.h5 (label 20 → 0 remap, normals (3,H,W)→(H,W,3)).

Datasets (all three by default):
    scannetpp   ScanNetPPPlaneDataset over the test split (per-scene folders).
    nyuv2       NYUv2PlaneDataset (single virtual scene "nyuv2").
    sevenscenes SevenScenesPlaneDataset (7 logical scenes from origin_img_path).

Per-frame metrics emitted (column names are stable):

    From eval_utils.evaluate_single_frame:
        sc, rand_index, voi                                (2D segmentation)
        prec@<τ>cm / rec@<τ>cm                              (3D plane RANSAC)
        bp_accuracy / bp_precision / bp_recall              (binary planarity)
        bp_f1 / bp_iou

    From metrics_planes (compute_segmentation_metrics intentionally skipped —
    sc/rand_index/voi above already cover it):
        plane_recall_d_<τ>mm  (+ _n_total / _n_matched)     plane_recall_at_depth
        plane_recall_n_<τ>deg (+ _n_total / _n_matched)     plane_recall_at_normal
        normal_err_deg_{mean, median, std, n}               per_plane_error_stats
        offset_err_m_{mean, median, std, n}                 per_plane_error_stats

Output layout:

    <eval_root>/<exp>/
        <method>/                      gt | moge | zeroplane
            <dataset>/                 scannetpp | nyuv2 | sevenscenes
                <scene>/
                    results.csv        one row per frame in this scene
                    summary.csv        one row: scene_id, num_frames,
                                       <metric>_mean, <metric>_std
                aggregate_results.csv  concat of every per-scene results.csv
                aggregate_per_scene.csv concat of every per-scene summary.csv
                aggregate_dataset.csv  one row: <metric>_mean / _std across
                                       scene-means + num_scenes / num_frames_total
                runtime.csv            wall_time + fps for this (method, dataset)
        summary.csv                    cross-(method, dataset) — one row per pair

Path conventions for prediction roots:
    <moge_signals_root>/<dataset>/<scene>/moge_signals.h5
    <zeroplane_h5_root>/<dataset>/<scene>/planes.h5

For NYU-v2 the dataset has a single virtual scene name "nyuv2", so the H5 lives
at ``<moge_signals_root>/nyuv2/nyuv2/moge_signals.h5``.

Example
-------
    python evaluate_gt_moge_zeroplane.py \\
        --exp gt_moge_zp_v1 \\
        --moge_signals_root  /cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1 \\
        --zeroplane_h5_root  /cluster/scratch/aoezkan/planeseg/inference/zeroplane_default_dust3r_released_h5 \\
        --eval_root          /cluster/scratch/aoezkan/planeseg/eval
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.paths import (                                                                # noqa: E402
    repo_path, scannetpp_path, scannetpp_rend_plane_path,
    nyuv2_path, sevenscenes_path,
)
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset                       # noqa: E402
from planamono.evaluation.quantitative.eval_utils import evaluate_single_frame               # noqa: E402
from planamono.shared.segmentation.plan2seg import compute_vectorized_planar_segments_v5_relative  # noqa: E402
from planamono.shared.segmentation.compute_plane_params import compute_plane_params           # noqa: E402
from planamono.shared.plane_fitting.metrics_planes import (                                  # noqa: E402
    compute_gt_normals_from_depth_labels,
    match_planes_by_overlap,
    plane_recall_at_depth,
    plane_recall_at_normal,
    per_plane_error_stats,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

# v5_relative segmentation params (locked to user spec — the same MoGe seg is
# applied across all datasets so cross-dataset numbers stay comparable).
SEG_THRESHOLD_PLANARITY = 0.3
SEG_NORMAL_THRESHOLD_DEG = 5.0
SEG_DEPTH_THRESHOLD_REL = 0.025
SEG_NEIGHBOR_MATCH_COUNT = 8

# Plane prec/rec — same defaults as evaluate_all_baselines.py (EXP_VER=v6).
RANSAC_THRESHOLDS = (0.001, 0.005, 0.01)            # meters
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9

# metrics_planes thresholds (same defaults as compare_plane_param_methods.py).
DEPTH_RECALL_THRESHOLDS = (0.05, 0.1, 0.6)          # meters
NORMAL_RECALL_THRESHOLDS_DEG = (5.0, 10.0, 30.0)    # degrees

ZP_NONPLANAR_LABEL = 20

DEFAULT_MOGE_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1"
DEFAULT_ZP_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/zeroplane_default_dust3r_released_h5"
DEFAULT_EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/eval"


# ---------------------------------------------------------------------------
# Frame-id normalization
# ---------------------------------------------------------------------------

def _norm_fid(fid) -> str:
    """Make frame ids comparable across conventions.

    ScanNet++ JPGs are stored as 'frame_000000' both in the dataset and H5s.
    NYUv2/7-Scenes loaders return plain ints ('0', '1', ...) while
    ``save_moge_signals_planarity.py`` writes 'frame_000000' to H5. Normalizing
    both forms to the bare int string ('0') lets us look up across either.
    """
    if isinstance(fid, (bytes, bytearray)):
        fid = fid.decode("utf-8")
    s = str(fid)
    if s.startswith("frame_"):
        s = s[len("frame_"):]
    try:
        return str(int(s))
    except ValueError:
        return s


# ---------------------------------------------------------------------------
# Plane-parameter computation + rendering  (matches compare_plane_param_methods.py)
# ---------------------------------------------------------------------------

# Method used by MoGe to fit per-plane params from (depth, normal, labels). One
# of the six estimators in compute_plane_params.py (see MoGe rationale in
# user-spec). SVD = orthogonal regression on backprojected points; cheap & clean.
MOGE_FIT_METHOD = "svd"


def _scale_K_to_hw(K: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Scale K so its principal point matches the centre of an (H, W) image.

    ScanNet++ stores intrinsics at the iPhone native resolution (~1920×1440)
    but the H5 depth/plane maps are downsampled to 480×640. The reference
    evaluation scripts pass the unscaled K to ``backproject_v1`` (a known
    pre-existing bug — 3D points come out at the wrong absolute scale).
    For the new plane-fitting + rendering step we DO scale K, since the rendered
    depth values must match raw GT depth in absolute meters for
    ``plane_recall_at_depth`` / ``per_plane_error_stats`` to be meaningful.

    Assumes the source K's principal point is image-centred (cx ≈ W_src/2).
    """
    H, W = hw
    K = np.asarray(K, dtype=np.float64)
    W_src = 2.0 * float(K[0, 2])
    H_src = 2.0 * float(K[1, 2])
    if W_src <= 0 or H_src <= 0:
        return K.copy()
    sx = W / W_src
    sy = H / H_src
    out = K.copy()
    out[0, 0] *= sx
    out[1, 1] *= sy
    out[0, 2] *= sx
    out[1, 2] *= sy
    return out


def _render_plane_params_to_maps(
    plane_params: Dict[int, np.ndarray],
    labels: np.ndarray,
    K: np.ndarray,
    H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-render per-pixel depth + normal maps from {label_id: (a,b,c,d)}.

    Same recipe as ``compare_plane_param_methods._render_plane_params_to_maps``:
    inside each label-mask, every pixel gets the plane's normal (a, b, c)
    and depth ``z = -d / (a·xn + b·yn + c)`` where (xn, yn) = ((u-cx)/fx, (v-cy)/fy).

    Pixels outside any plane stay at zero (will be ignored by aggregate_plane_*).
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    xn = (us.astype(np.float64) - cx) / fx
    yn = (vs.astype(np.float64) - cy) / fy

    depth_out = np.zeros((H, W), dtype=np.float32)
    normal_out = np.zeros((H, W, 3), dtype=np.float32)

    for pid, p in plane_params.items():
        a, b, c, d = float(p[0]), float(p[1]), float(p[2]), float(p[3])
        mask = (labels == pid)
        if not mask.any():
            continue
        denom = a * xn + b * yn + c
        valid = np.abs(denom) > 1e-9
        z = np.zeros_like(denom)
        z[valid] = -d / denom[valid]
        ok = mask & (z > 0) & np.isfinite(z)
        depth_out[ok] = z[ok].astype(np.float32)
        normal_out[mask] = (a, b, c)

    return depth_out, normal_out


def _zp_plane_params_to_dict(
    nd_arr: np.ndarray,
    label_offset: int = 1,
) -> Dict[int, np.ndarray]:
    """Convert ZeroPlane H5 ``plane_params`` (n/d form) to standard (a,b,c,d).

    ZeroPlane stores rows as (a, b, c) such that  a·x + b·y + c·z = 1  for any
    3D point on the plane. Standard form is  A·X + B·Y + C·Z + D = 0  with
    ‖(A, B, C)‖ = 1. Conversion:
        nrm = ‖(a, b, c)‖
        (A, B, C) = (a, b, c) / nrm   →  unit normal
        D = −1 / nrm                  →  signed offset

    Row index ``i`` corresponds to the ZeroPlane label ``i`` BEFORE the +1 shift
    that ``_load_zeroplane_scene`` applies. After the shift, row i belongs to
    post-shift label ``i + label_offset`` (= ``i + 1`` by default).
    """
    out: Dict[int, np.ndarray] = {}
    for i in range(nd_arr.shape[0]):
        nd = np.asarray(nd_arr[i], dtype=np.float64)
        nrm = float(np.linalg.norm(nd))
        if nrm < 1e-12:
            continue
        unit_n = nd / nrm
        D = -1.0 / nrm
        out[i + label_offset] = np.array([unit_n[0], unit_n[1], unit_n[2], D],
                                          dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Per-frame evaluation (worker)
# ---------------------------------------------------------------------------

def _eval_one_frame(
    scene_id: str,
    frame_id: str,
    depth_gt: np.ndarray,
    labels_gt: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    labels_pred: np.ndarray,
    depth_pred: np.ndarray,
    normals_pred: Optional[np.ndarray],
    ransac_thresholds: Tuple[float, ...],
    depth_recall_thresholds: Tuple[float, ...],
    normal_recall_thresholds_deg: Tuple[float, ...],
    ransac_iterations: int,
    inlier_ratio_gate: float,
) -> Dict[str, float]:
    """Compute the full metric block for one (scene, frame). Returns flat dict."""
    metrics, _ = evaluate_single_frame(
        scene_id, frame_id,
        depth_gt, labels_gt, K, c2w, labels_pred,
        thresholds=ransac_thresholds,
        compute_plane_metrics_flag=True,
        ransac_iterations=ransac_iterations,
        inlier_ratio_gate=inlier_ratio_gate,
    )

    normals_gt = compute_gt_normals_from_depth_labels(
        depth_gt, labels_gt, K, ignore_labels=(0,),
    ).astype(np.float64)

    if normals_pred is None:
        normals_pred = normals_gt
    else:
        normals_pred = np.asarray(normals_pred, dtype=np.float64)

    matches = match_planes_by_overlap(
        labels_gt, labels_pred,
        ignore_labels_gt=(0,), ignore_labels_pred=(0,),
    )

    for thr in depth_recall_thresholds:
        r = plane_recall_at_depth(
            depth_pred, depth_gt, labels_pred, labels_gt,
            threshold=thr, matches=matches,
        )
        key = f"plane_recall_d_{int(round(thr * 1000))}mm"
        metrics[key] = float(r["recall"])
        metrics[f"{key}_n_total"] = int(r["n_total"])
        metrics[f"{key}_n_matched"] = int(r["n_matched"])

    for thr_deg in normal_recall_thresholds_deg:
        r = plane_recall_at_normal(
            normals_pred, normals_gt, labels_pred, labels_gt,
            threshold_deg=thr_deg, matches=matches,
        )
        key = f"plane_recall_n_{int(round(thr_deg))}deg"
        metrics[key] = float(r["recall"])
        metrics[f"{key}_n_total"] = int(r["n_total"])
        metrics[f"{key}_n_matched"] = int(r["n_matched"])

    err_stats = per_plane_error_stats(
        depth_pred, depth_gt, normals_pred, normals_gt,
        labels_pred, labels_gt, matches=matches,
    )
    metrics.update(err_stats)

    return metrics


# ---------------------------------------------------------------------------
# Dataset GT adapters
# ---------------------------------------------------------------------------

def _build_gt_dataset(dataset_name: str, args):
    """Instantiate the right GT loader. Returns the dataset object.

    ``--scenes`` / ``--frames`` are applied post-hoc via
    ``_apply_caps_to_scene_index`` (after _build_scene_index runs) so all
    three datasets get identical semantics: --scenes caps distinct scene_ids,
    --frames caps frames per scene.
    """
    if dataset_name == "scannetpp":
        return ScanNetPPPlaneDataset(
            rgb_root=os.path.join(scannetpp_path, "data"),
            plane_label_root=scannetpp_rend_plane_path,
            sem_label_root=scannetpp_rend_plane_path,
            depth_label_root=scannetpp_rend_plane_path,
            split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
            split=args.split,
        )
    if dataset_name == "nyuv2":
        from planamono.shared.datasets.nyuv2_plane_dataset import NYUv2PlaneDataset
        return NYUv2PlaneDataset(data_root=nyuv2_path, split="test")
    if dataset_name == "sevenscenes":
        from planamono.shared.datasets.sevenscenes_plane_dataset import SevenScenesPlaneDataset
        return SevenScenesPlaneDataset(data_root=sevenscenes_path, split="val")
    raise ValueError(f"unknown dataset: {dataset_name}")


def _apply_scene_cap_to_index(
    idx: Dict[str, List[Tuple[int, str]]],
    dataset_name: str,
    scenes: Optional[int],
) -> Dict[str, List[Tuple[int, str]]]:
    """Cap scene_index to first ``scenes`` distinct scene_ids (preserves order).

    No-op for NYU-v2 (single virtual scene). Frame-per-scene capping is
    handled separately in ``_evaluate_method_dataset`` via the
    ``--max_frames_per_scene`` evenly-spaced subsampler.
    """
    if scenes is None:
        return idx
    if dataset_name == "nyuv2":
        return idx
    from collections import OrderedDict
    new_idx = OrderedDict(idx.items())
    new_idx = OrderedDict(list(new_idx.items())[:scenes])
    return dict(new_idx)


def _build_scene_index(dataset_name: str, ds) -> Dict[str, List[Tuple[int, str]]]:
    """Cheap scene_id → [(dataset_idx, frame_idx_str), ...] map. No sample I/O.

    Each dataset class exposes a ``valid_pairs`` list whose elements encode
    enough to recover (scene_id, frame_id_str) without calling ``__getitem__``;
    we exploit that so we can iterate sample-by-sample inside the per-scene
    loop and avoid holding tens of GB of GT in RAM.
    """
    out: Dict[str, List[Tuple[int, str]]] = {}
    if dataset_name == "scannetpp":
        # valid_pairs[i] = (rgb_path, plane_h5, sem_h5, depth_h5, idx, K, c2w)
        for i, vp in enumerate(ds.valid_pairs):
            rgb_path = vp[0]
            parts = rgb_path.split(os.sep)
            try:
                iph = parts.index("iphone")
                scene_id = parts[iph - 1]
            except (ValueError, IndexError):
                continue
            frame_id_str = os.path.splitext(os.path.basename(rgb_path))[0]
            out.setdefault(scene_id, []).append((i, frame_id_str))
        return out
    if dataset_name == "nyuv2":
        for i, vp in enumerate(ds.valid_pairs):
            # NYUv2PlaneDataset.valid_pairs[i] = (npz_path, sample_idx)
            sample_idx = int(vp[1])
            out.setdefault("nyuv2", []).append((i, str(sample_idx)))
        return out
    if dataset_name == "sevenscenes":
        for i, vp in enumerate(ds.valid_pairs):
            # SevenScenesPlaneDataset.valid_pairs[i] = (npz_path, sample_idx, scene_id, origin)
            sample_idx = int(vp[1])
            scene_id = vp[2]
            out.setdefault(scene_id, []).append((i, str(sample_idx)))
        return out
    raise ValueError(f"unknown dataset: {dataset_name}")


def _load_gt_sample(ds, ds_idx: int) -> Dict[str, np.ndarray]:
    """Load one sample's GT (depth, labels, K, c2w, frame_idx) — called lazily."""
    s = ds[ds_idx]
    depth = s["depth"][0].numpy() if hasattr(s["depth"], "numpy") else s["depth"][0]
    plane = s["plane"][0].numpy() if hasattr(s["plane"], "numpy") else s["plane"][0]
    K = s["K"].numpy() if hasattr(s["K"], "numpy") else s["K"]
    c2w = s["c2w"].numpy() if hasattr(s["c2w"], "numpy") else s["c2w"]
    return {
        "depth": depth.astype(np.float32),
        "labels": plane.astype(np.int32),
        "K": K.astype(np.float32),
        "c2w": c2w.astype(np.float32),
        "frame_idx": str(s["frame_idx"]),
        "scene_id": s["scene_id"],
    }


# ---------------------------------------------------------------------------
# Per-method, per-scene prediction loaders (key = normalized fid)
# ---------------------------------------------------------------------------

def _decode_frame_ids(arr) -> List[str]:
    return [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in np.asarray(arr).tolist()
    ]


def _load_moge_scene(scene_dir: str) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Read moge_signals.h5 and run v5_relative segmentation per frame.

    Returns frame_id (normalized) -> {labels, depth, normals} or None if missing.
    Segmentation runs sequentially in this process (CPU torch).
    """
    h5_path = os.path.join(scene_dir, "moge_signals.h5")
    if not os.path.isfile(h5_path):
        return None

    out: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:
        frame_ids = _decode_frame_ids(f["frame_ids"][:])
        for i, raw_fid in enumerate(frame_ids):
            planarity = f["planarity"][i].astype(np.float32)
            normal = f["normal"][i].astype(np.float32)              # (H, W, 3)
            depth = f["depth_metric"][i].astype(np.float32)

            mask = planarity > SEG_THRESHOLD_PLANARITY
            labels, _ = compute_vectorized_planar_segments_v5_relative(
                planarity_mask=mask,
                normal=normal,
                depth=depth,
                normal_threshold_rad=float(np.deg2rad(SEG_NORMAL_THRESHOLD_DEG)),
                depth_threshold=SEG_DEPTH_THRESHOLD_REL,
                neighbor_match_count_thresh=SEG_NEIGHBOR_MATCH_COUNT,
                device="cpu",
            )
            if hasattr(labels, "cpu"):
                labels = labels.cpu().numpy()
            out[_norm_fid(raw_fid)] = {
                "labels": labels.astype(np.int32),
                "depth": depth,
                "normals": normal,
            }
    return out


def _load_zeroplane_scene(scene_dir: str) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Read ZeroPlane planes.h5; remap to standard convention (0=non-planar);
    transpose normals to HWC.

    Path resolution: ZeroPlane stores per-scene predictions for ScanNet++ and
    7-Scenes at ``<scene_dir>/planes.h5`` (nested) but NYU-v2 as a single flat
    ``<dataset_dir>/planes.h5`` (no per-scene subdir). We try the nested path
    first, then fall back to the parent dir.

    ZeroPlane uses label ``20`` for non-planar AND labels ``0..19`` for actual
    planes (verified — plane id 0 occurs in real frames). A naive
    ``labels[labels==20] = 0`` would silently collide plane 0 with non-planar
    (and downstream metrics that ignore label 0 would drop that plane). To
    preserve plane id 0 we shift first: ``0..19 → 1..20`` then map the
    (post-shift) ``21`` (was non-planar 20) → ``0``.

    For gt and moge predictions this remap is unnecessary — they already use
    label 0 = non-planar.
    """
    h5_path = os.path.join(scene_dir, "planes.h5")
    if not os.path.isfile(h5_path):
        # Fallback: flat layout (single H5 for the whole dataset, e.g. NYU-v2).
        flat = os.path.join(os.path.dirname(scene_dir), "planes.h5")
        if os.path.isfile(flat):
            h5_path = flat
        else:
            return None

    out: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:
        frame_ids = _decode_frame_ids(f["frame_ids"][:])
        has_pp_group = "plane_params" in f
        for i, raw_fid in enumerate(frame_ids):
            labels = f["planes"][i].astype(np.int32)
            # Shift all labels up by 1, then map original non-planar (now 21) to 0.
            labels = labels + 1
            labels[labels == ZP_NONPLANAR_LABEL + 1] = 0
            depth = f["planes_depth"][i].astype(np.float32)
            normals = f["pixel_normals"][i].astype(np.float32)      # (3, H, W)
            normals = normals.transpose(1, 2, 0).copy()             # (H, W, 3)
            n = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = np.divide(normals, np.clip(n, 1e-6, None), where=n > 1e-6)

            # Per-frame plane parameters in n/d form (a*x + b*y + c*z = 1).
            # Row index i ↔ original label i (pre-shift). May have N < 20 rows
            # for frames where some queries didn't produce a valid plane.
            plane_params_nd: Optional[np.ndarray] = None
            if has_pp_group and raw_fid in f["plane_params"]:
                plane_params_nd = f["plane_params"][raw_fid][:].astype(np.float32)

            out[_norm_fid(raw_fid)] = {
                "labels": labels,
                "depth": depth,
                "normals": normals,
                "plane_params_nd": plane_params_nd,   # may be None if missing
            }
    return out


# ---------------------------------------------------------------------------
# Shape matching (resize pred to GT HxW if needed)
# ---------------------------------------------------------------------------

_SPATIAL_KEYS = {"labels", "depth", "normals"}


def _match_to_gt_shape(pred: Dict[str, np.ndarray], gt_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Resize per-pixel pred maps to GT (H, W).

    NEAREST for ``labels``, LINEAR for ``depth`` / ``normals``. Normals are
    renormalized after a LINEAR resize so the cosine-based angle metrics get
    valid unit vectors.

    Only keys in ``_SPATIAL_KEYS`` are touched. Non-spatial entries like
    ``plane_params_nd`` (shape ``(N, 3)``) are passed through unchanged —
    treating them as images would corrupt the plane parameters.
    """
    H, W = gt_hw
    out = {}
    for k, arr in pred.items():
        if k not in _SPATIAL_KEYS or arr is None:
            out[k] = arr
            continue
        if arr.shape[:2] == (H, W):
            out[k] = arr
            continue
        if k == "labels":
            out[k] = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST).astype(arr.dtype)
        elif arr.ndim == 3:
            r = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
            if k == "normals":
                n = np.linalg.norm(r, axis=-1, keepdims=True)
                r = np.divide(r, np.clip(n, 1e-6, None), where=n > 1e-6)
            out[k] = r
        else:
            out[k] = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR).astype(arr.dtype)
    # Defensive renorm in case input normals are already non-unit (float16 drift).
    if "normals" in out and out["normals"] is not None:
        n = np.linalg.norm(out["normals"], axis=-1, keepdims=True)
        out["normals"] = np.divide(out["normals"], np.clip(n, 1e-6, None), where=n > 1e-6)
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _summarize_rows(df: pd.DataFrame) -> Dict[str, float]:
    """Return one-row dict: <metric>_mean / <metric>_std for every numeric column."""
    out: Dict[str, float] = {}
    for c in df.select_dtypes(include="number").columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std())
    return out


def _save_scene_csvs(scene_id: str, frame_rows: List[Dict], scene_dir: str) -> Dict[str, float]:
    """Write <scene>/results.csv (per-frame) + <scene>/summary.csv (1-row mean/std).

    Returns the scene-summary dict (used by the dataset-level aggregator).
    """
    os.makedirs(scene_dir, exist_ok=True)
    df = pd.DataFrame.from_records(frame_rows)
    df.to_csv(os.path.join(scene_dir, "results.csv"), index=False)

    row: Dict[str, float] = {"scene_id": scene_id, "num_frames": int(len(df))}
    row.update(_summarize_rows(df))
    pd.DataFrame([row]).to_csv(os.path.join(scene_dir, "summary.csv"), index=False)
    return row


def _save_dataset_aggregates(
    dataset_dir: str,
    all_frame_rows: List[Dict],
    scene_summaries: List[Dict],
    wall_time_s: float,
) -> Dict[str, float]:
    """Write the parent-level aggregate files. Returns the dataset-level summary."""
    os.makedirs(dataset_dir, exist_ok=True)

    # 1. Concat of every per-frame row.
    if all_frame_rows:
        pd.DataFrame.from_records(all_frame_rows).to_csv(
            os.path.join(dataset_dir, "aggregate_results.csv"), index=False,
        )

    # 2. Concat of scene summaries (one row per scene).
    df_scenes = pd.DataFrame(scene_summaries) if scene_summaries else pd.DataFrame()
    if not df_scenes.empty:
        df_scenes.to_csv(os.path.join(dataset_dir, "aggregate_per_scene.csv"), index=False)

    # 3. Dataset-level: mean/std across scene-mean columns.
    dataset_row: Dict[str, float] = {
        "num_scenes": int(len(df_scenes)),
        "num_frames_total": int(df_scenes["num_frames"].sum()) if not df_scenes.empty else 0,
    }
    if not df_scenes.empty:
        # Aggregate the scene-level "_mean" columns across scenes (mean of means / std of means).
        mean_cols = [c for c in df_scenes.columns if c.endswith("_mean")]
        for mc in mean_cols:
            base = mc[: -len("_mean")]
            vals = df_scenes[mc].dropna()
            dataset_row[f"{base}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            dataset_row[f"{base}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
    pd.DataFrame([dataset_row]).to_csv(
        os.path.join(dataset_dir, "aggregate_dataset.csv"), index=False,
    )

    # 4. Runtime row.
    n_frames = len(all_frame_rows)
    pd.DataFrame([{
        "wall_time_seconds": wall_time_s,
        "num_frames": n_frames,
        "frames_per_second": n_frames / wall_time_s if wall_time_s > 0 else float("nan"),
    }]).to_csv(os.path.join(dataset_dir, "runtime.csv"), index=False)

    return dataset_row


def _write_top_summary(
    method_dataset_stats: Dict[Tuple[str, str], Dict[str, float]],
    summary_path: str,
) -> None:
    """One row per (method, dataset) pair with all dataset-level stats."""
    if not method_dataset_stats:
        return
    rows = []
    for (method, dataset), stats in method_dataset_stats.items():
        rows.append({"method": method, "dataset": dataset, **stats})
    pd.DataFrame(rows).to_csv(summary_path, index=False)


def _read_scene_csvs_under(dataset_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Walk <dataset_dir>/<scene>/{results,summary}.csv to reconstruct
    per-frame rows + scene summaries. Used by --aggregate_only mode.
    """
    all_frame_rows: List[Dict] = []
    scene_summaries: List[Dict] = []
    if not os.path.isdir(dataset_dir):
        return all_frame_rows, scene_summaries
    for entry in sorted(os.listdir(dataset_dir)):
        scene_dir = os.path.join(dataset_dir, entry)
        if not os.path.isdir(scene_dir):
            continue
        results_csv = os.path.join(scene_dir, "results.csv")
        summary_csv = os.path.join(scene_dir, "summary.csv")
        if os.path.isfile(results_csv):
            try:
                df = pd.read_csv(results_csv)
                all_frame_rows.extend(df.to_dict(orient="records"))
            except Exception as e:
                print(f"  [warn] failed to read {results_csv}: {e}")
        if os.path.isfile(summary_csv):
            try:
                df = pd.read_csv(summary_csv)
                scene_summaries.extend(df.to_dict(orient="records"))
            except Exception as e:
                print(f"  [warn] failed to read {summary_csv}: {e}")
    return all_frame_rows, scene_summaries


def _aggregate_from_disk(
    args, out_root: str,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Walk <out_root>/<method>/<dataset>/<scene>/{results,summary}.csv files
    and produce dataset-level aggregate_*.csv + top-level summary.csv.

    Used by --aggregate_only after parallel worker jobs finish.
    """
    method_dataset_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for method in args.methods:
        for ds_name in args.datasets:
            dataset_dir = os.path.join(out_root, method, ds_name)
            if not os.path.isdir(dataset_dir):
                print(f"  [skip aggregate] {method}/{ds_name}: no dir")
                continue
            all_frame_rows, scene_summaries = _read_scene_csvs_under(dataset_dir)
            print(f"  aggregating {method}/{ds_name}: "
                  f"{len(all_frame_rows)} frames over {len(scene_summaries)} scenes")
            stats = _save_dataset_aggregates(
                dataset_dir=dataset_dir,
                all_frame_rows=all_frame_rows,
                scene_summaries=scene_summaries,
                wall_time_s=0.0,
            )
            method_dataset_stats[(method, ds_name)] = stats
    _write_top_summary(method_dataset_stats, os.path.join(out_root, "summary.csv"))
    return method_dataset_stats


# ---------------------------------------------------------------------------
# Per-(method, dataset) evaluation
# ---------------------------------------------------------------------------

def _evaluate_method_dataset(
    method: str,
    dataset_name: str,
    ds,
    scene_index: Dict[str, List[Tuple[int, str]]],
    args,
) -> Tuple[List[Dict], List[Dict], float]:
    """Evaluate one (method, dataset) — load samples lazily, write per-scene CSVs.

    Returns (all_frame_rows, scene_summary_rows, wall_time_s).

    Per-scene memory is bounded: only that scene's predictions + GT samples are
    held at any one time. Predictions are released before moving to the next
    scene.
    """
    out_dataset_dir = os.path.join(args.eval_root, args.exp, method, dataset_name)
    os.makedirs(out_dataset_dir, exist_ok=True)

    moge_dataset_root = os.path.join(args.moge_signals_root, dataset_name)
    zp_dataset_root = os.path.join(args.zeroplane_h5_root, dataset_name)

    scene_ids = sorted(scene_index.keys())
    scene_filter = getattr(args, "_scene_filter", None)
    if scene_filter is not None:
        before = len(scene_ids)
        scene_ids = [s for s in scene_ids if s in scene_filter]
        skipped = before - len(scene_ids)
        if skipped:
            tqdm.write(f"  [filter] {dataset_name}: {len(scene_ids)}/{before} scenes (skipped {skipped})")

    all_frame_rows: List[Dict] = []
    scene_summaries: List[Dict] = []
    t0 = time.perf_counter()

    max_fps = getattr(args, "max_frames_per_scene", None)

    for sid in tqdm(scene_ids, desc=f"  {method:<10s}/{dataset_name:<11s}", unit="scene"):
        frame_list = scene_index[sid]   # [(ds_idx, frame_id_str), ...]
        if max_fps is not None and len(frame_list) > max_fps:
            # Evenly-spaced subsample (matches compare_plane_param_methods.py:307-308).
            idx = np.linspace(0, len(frame_list) - 1, max_fps).astype(int)
            frame_list = [frame_list[i] for i in idx]

        # Load this scene's predictions (None for gt — pred will be GT itself).
        if method == "gt":
            pred_scene = None
        elif method == "moge":
            pred_scene = _load_moge_scene(os.path.join(moge_dataset_root, sid))
        elif method == "zeroplane":
            pred_scene = _load_zeroplane_scene(os.path.join(zp_dataset_root, sid))
        else:
            raise ValueError(f"unknown method: {method}")

        if method != "gt" and pred_scene is None:
            tqdm.write(f"  [skip] {method}/{dataset_name}/{sid}: predictions missing")
            continue

        # Build per-frame tasks. Each ds[ds_idx] lazily loads one sample.
        tasks = []
        for ds_idx, raw_fid in frame_list:
            nfid = _norm_fid(raw_fid)
            if method != "gt" and nfid not in pred_scene:
                continue

            gt = _load_gt_sample(ds, ds_idx)
            gt_hw = gt["labels"].shape[:2]

            if method == "gt":
                # GT method: keep raw GT depth + SVD-broadcast GT normals
                # (worker computes the latter via compute_gt_normals_from_depth_labels).
                labels_pred = gt["labels"]
                depth_pred = gt["depth"]
                normals_pred = None
            else:
                pred = _match_to_gt_shape(pred_scene[nfid], gt_hw)
                labels_pred = pred["labels"]

                # Build a K matched to the maps' resolution. ScanNet++'s GT K
                # is at iPhone native res (~1920×1440); rendering with that K
                # against 480×640 maps would produce wrong-scale depth values.
                K_render = _scale_K_to_hw(gt["K"], gt_hw)

                if method == "moge":
                    plane_params = compute_plane_params(
                        depth=pred["depth"],
                        normal=pred["normals"],
                        plane_label=labels_pred,
                        method=MOGE_FIT_METHOD,
                        K=K_render,
                        ignore_labels=(0,),
                    )
                elif method == "zeroplane":
                    nd = pred.get("plane_params_nd")
                    if nd is None:
                        # Frame is missing pre-computed plane_params (rare).
                        # Fall back to fitting from per-pixel data, same as moge.
                        plane_params = compute_plane_params(
                            depth=pred["depth"],
                            normal=pred["normals"],
                            plane_label=labels_pred,
                            method=MOGE_FIT_METHOD,
                            K=K_render,
                            ignore_labels=(0,),
                        )
                    else:
                        plane_params = _zp_plane_params_to_dict(nd, label_offset=1)
                else:
                    raise ValueError(f"unknown method: {method}")

                depth_pred, normals_pred = _render_plane_params_to_maps(
                    plane_params, labels_pred, K_render, gt_hw[0], gt_hw[1],
                )

            tasks.append({
                "scene_id": sid,
                "frame_id": gt["frame_idx"],
                "depth_gt": gt["depth"],
                "labels_gt": gt["labels"],
                "K": gt["K"],
                "c2w": gt["c2w"],
                "labels_pred": labels_pred,
                "depth_pred": depth_pred,
                "normals_pred": normals_pred,
            })

        # Free this scene's pred dict before the heavy parallel pass — joblib
        # will memmap the per-task arrays from the `tasks` list anyway.
        pred_scene = None

        if not tasks:
            continue

        outputs = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_eval_one_frame)(
                t["scene_id"], t["frame_id"],
                t["depth_gt"], t["labels_gt"], t["K"], t["c2w"],
                t["labels_pred"], t["depth_pred"], t["normals_pred"],
                args.ransac_thresholds,
                args.depth_recall_thresholds,
                args.normal_recall_thresholds_deg,
                args.ransac_iterations,
                args.inlier_ratio_gate,
            )
            for t in tasks
        )

        scene_dir = os.path.join(out_dataset_dir, sid)
        scene_summary = _save_scene_csvs(sid, list(outputs), scene_dir)
        scene_summaries.append(scene_summary)
        all_frame_rows.extend(outputs)

        # Release per-scene buffers explicitly.
        del tasks
        del outputs

    wall = time.perf_counter() - t0
    return all_frame_rows, scene_summaries, wall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True,
                    help="Experiment name. Output goes to <eval_root>/<exp>/.")
    ap.add_argument("--methods", nargs="+",
                    default=["gt", "moge", "zeroplane"],
                    choices=["gt", "moge", "zeroplane"],
                    help="Subset of methods to run (default: all three).")
    ap.add_argument("--datasets", nargs="+",
                    default=["scannetpp", "nyuv2", "sevenscenes"],
                    choices=["scannetpp", "nyuv2", "sevenscenes"],
                    help="Subset of datasets to evaluate (default: all three).")
    ap.add_argument("--moge_signals_root", default=DEFAULT_MOGE_ROOT,
                    help="Base dir; expects <root>/<dataset>/<scene>/moge_signals.h5.")
    ap.add_argument("--zeroplane_h5_root", default=DEFAULT_ZP_ROOT,
                    help="Base dir; expects <root>/<dataset>/<scene>/planes.h5.")
    ap.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT,
                    help="Output root. Per-(method, dataset, scene) folders go under <eval_root>/<exp>/.")
    ap.add_argument("--split", default="test",
                    help="ScanNet++ split (default: test). Ignored for nyuv2/sevenscenes "
                         "(they use their own canonical splits).")
    ap.add_argument("--max_scenes", "--scenes", dest="max_scenes",
                    type=int, default=None,
                    help="Cap number of scenes. NYU-v2 has only one virtual scene "
                         "and ignores this flag — use --frames to cap its samples. "
                         "(--scenes is the v1-style alias.)")
    ap.add_argument("--max_frames_per_scene", "--frames", dest="max_frames_per_scene",
                    type=int, default=None,
                    help="Cap frames PER scene via evenly-spaced subsampling (useful "
                         "for smoke tests). For NYU-v2 / 7-Scenes (flat) this slices "
                         "the single-scene frame list the same way. (--frames is the "
                         "v1-style alias.)")
    ap.add_argument("--n_jobs", type=int,
                    default=min(16, os.cpu_count() or 16),
                    help="joblib parallel workers for per-frame evaluation.")

    ap.add_argument("--ransac_thresholds", nargs="+", type=float,
                    default=list(RANSAC_THRESHOLDS),
                    help="Distance thresholds (m) for prec/rec@<τ>cm (default: 0.001 0.005 0.01).")
    ap.add_argument("--depth_recall_thresholds", nargs="+", type=float,
                    default=list(DEPTH_RECALL_THRESHOLDS),
                    help="Thresholds (m) for plane_recall_at_depth (default: 0.05 0.1 0.6).")
    ap.add_argument("--normal_recall_thresholds_deg", nargs="+", type=float,
                    default=list(NORMAL_RECALL_THRESHOLDS_DEG),
                    help="Thresholds (deg) for plane_recall_at_normal (default: 5 10 30).")
    ap.add_argument("--ransac_iterations", type=int, default=RANSAC_ITERATIONS)
    ap.add_argument("--inlier_ratio_gate", type=float, default=INLIER_RATIO_GATE)

    ap.add_argument("--scene_ids", default=None,
                    help="Process only these scene_ids. Either a comma-separated list "
                         "('chess,fire') or a path to a .txt with one scene per line.")
    ap.add_argument("--skip_dataset_aggregates", action="store_true",
                    help="Worker mode: write per-scene CSVs only. Skip "
                         "aggregate_*.csv and top-level summary.csv (avoids races "
                         "between parallel workers writing partial aggregates).")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip evaluation. Walk existing per-scene CSVs under "
                         "<eval_root>/<exp>/<method>/<dataset>/<scene>/ and (re)write "
                         "aggregate_*.csv + top-level summary.csv from them.")

    args = ap.parse_args()
    args.ransac_thresholds = tuple(args.ransac_thresholds)
    args.depth_recall_thresholds = tuple(args.depth_recall_thresholds)
    args.normal_recall_thresholds_deg = tuple(args.normal_recall_thresholds_deg)

    # Parse --scene_ids into a set (or None = all). File or comma list.
    scene_filter: Optional[set] = None
    if args.scene_ids:
        if os.path.isfile(args.scene_ids):
            with open(args.scene_ids) as f:
                scene_filter = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        else:
            scene_filter = {s.strip() for s in args.scene_ids.split(",") if s.strip()}
    args._scene_filter = scene_filter

    # Pre-flight (skip when --aggregate_only — no source data needed).
    if not args.aggregate_only:
        for ds in args.datasets:
            for m in args.methods:
                if m == "moge":
                    d = os.path.join(args.moge_signals_root, ds)
                    if not os.path.isdir(d):
                        ap.error(f"--moge_signals_root/{ds} not found: {d}")
                if m == "zeroplane":
                    d = os.path.join(args.zeroplane_h5_root, ds)
                    if not os.path.isdir(d):
                        ap.error(f"--zeroplane_h5_root/{ds} not found: {d}")

    # Single-threaded numeric libs inside joblib workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    out_root = os.path.join(args.eval_root, args.exp)
    os.makedirs(out_root, exist_ok=True)

    print("================================================================")
    print(f" evaluate_gt_moge_zeroplane — exp={args.exp}")
    print("================================================================")
    print(f"  methods:        {args.methods}")
    print(f"  datasets:       {args.datasets}")
    print(f"  moge root:      {args.moge_signals_root}")
    print(f"  zeroplane root: {args.zeroplane_h5_root}")
    print(f"  output root:    {out_root}")
    print(f"  ransac thr:     {args.ransac_thresholds} m   iters={args.ransac_iterations}")
    print(f"  depth recall:   {args.depth_recall_thresholds} m")
    print(f"  normal recall:  {args.normal_recall_thresholds_deg} deg")
    print(f"  n_jobs:         {args.n_jobs}    max_scenes: {args.max_scenes}")
    if scene_filter is not None:
        print(f"  scene_ids:      {sorted(scene_filter)} ({len(scene_filter)} scenes)")
    if args.skip_dataset_aggregates:
        print(f"  mode:           WORKER (skip dataset aggregates / top summary)")
    if args.aggregate_only:
        print(f"  mode:           AGGREGATE-ONLY (skip evaluation)")
    print("  seg params:     "
          f"plan>{SEG_THRESHOLD_PLANARITY}, "
          f"normal<{SEG_NORMAL_THRESHOLD_DEG}°, "
          f"depth_rel<{SEG_DEPTH_THRESHOLD_REL}, "
          f"match≥{SEG_NEIGHBOR_MATCH_COUNT}")
    print("================================================================")

    # Aggregate-only mode: walk existing per-scene CSVs and (re)write aggregates.
    if args.aggregate_only:
        print("\n[AGG] reading per-scene CSVs from disk ...")
        _aggregate_from_disk(args, out_root)
        print(f"\n[DONE] aggregates rewritten under {out_root}")
        return

    # Build per-dataset {scene_id: [(ds_idx, raw_frame_id_str), ...]} maps.
    # These are CHEAP — just walks `valid_pairs` without touching any sample I/O.
    # Then the dataset object itself is held and `ds[i]` is called lazily inside
    # the per-scene loop, so we never hold all of GT in memory at once.
    datasets: Dict[str, object] = {}
    scene_indices: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}
    for ds_name in args.datasets:
        print(f"\n[GT] building scene index for {ds_name} ...")
        ds_obj = _build_gt_dataset(ds_name, args)
        idx = _build_scene_index(ds_name, ds_obj)
        # Apply --scenes / --max_scenes cap (per-scene frame cap is handled
        # in _evaluate_method_dataset via --max_frames_per_scene).
        idx = _apply_scene_cap_to_index(idx, ds_name, args.max_scenes)
        n_frames = sum(len(v) for v in idx.values())
        print(f"  {ds_name}: {n_frames} frames over {len(idx)} scenes (no GT loaded yet)"
              + (f"  [capped to {args.max_scenes} scenes]"
                 if args.max_scenes is not None and ds_name != "nyuv2" else ""))
        datasets[ds_name] = ds_obj
        scene_indices[ds_name] = idx

    method_dataset_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    for method in args.methods:
        for ds_name in args.datasets:
            print(f"\n==> {method} / {ds_name}")
            frame_rows, scene_summaries, wall = _evaluate_method_dataset(
                method=method, dataset_name=ds_name,
                ds=datasets[ds_name],
                scene_index=scene_indices[ds_name],
                args=args,
            )
            print(f"  done: {len(frame_rows)} frames, {len(scene_summaries)} scenes "
                  f"in {wall:.1f}s ({len(frame_rows)/max(wall,1e-6):.2f} fps)")
            if args.skip_dataset_aggregates:
                # Worker mode: per-scene CSVs only. Skip dataset-level aggregates
                # (a separate aggregator job will combine all parts after this finishes).
                continue
            stats = _save_dataset_aggregates(
                dataset_dir=os.path.join(out_root, method, ds_name),
                all_frame_rows=frame_rows,
                scene_summaries=scene_summaries,
                wall_time_s=wall,
            )
            method_dataset_stats[(method, ds_name)] = stats

    if args.skip_dataset_aggregates:
        print(f"\n[DONE worker] per-scene CSVs written under {out_root}")
        return

    summary_path = os.path.join(out_root, "summary.csv")
    _write_top_summary(method_dataset_stats, summary_path)
    print(f"\n[DONE] cross-(method, dataset) summary: {summary_path}")


if __name__ == "__main__":
    main()
