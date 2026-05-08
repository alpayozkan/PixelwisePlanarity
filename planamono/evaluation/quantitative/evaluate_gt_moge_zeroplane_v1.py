"""Multi-dataset, 3-method plane-segmentation evaluation (v1: streaming/parallel).

v1 changes vs v0 (evaluate_gt_moge_zeroplane.py)
------------------------------------------------
1. GT loading uses torch ``DataLoader(num_workers=N)`` instead of single-threaded
   sample-by-sample loading on the main process. Workers prefetch GT in
   parallel and overlap with the joblib eval pass.
2. MoGe segmentation + per-plane fitting + map re-rendering all run INSIDE the
   joblib worker (was: serial main-process sweep before joblib started).
3. Loop inversion — every method is evaluated in a SINGLE streaming pass over
   each dataset; GT is loaded once and reused across methods. This saves
   ``N_methods × dataset_io`` when running multiple methods together.
4. Each joblib worker calls ``torch.set_num_threads(1)`` to avoid oversubscription
   (forked workers don't inherit the main process's torch thread setting).

The metric pipeline, output layout, CSV columns and per-method semantics are
unchanged from v0. Only the orchestration is different. ``wall_time_seconds`` in
``runtime.csv`` is now the SHARED dataset-streaming time (all methods amortized
into one DataLoader pass); per-method joblib wall time is reported as
``joblib_time_seconds``.

Example
-------
    python evaluate_gt_moge_zeroplane_v1.py \\
        --exp gt_moge_zp_v1 \\
        --moge_signals_root  /cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1 \\
        --zeroplane_h5_root  /cluster/scratch/aoezkan/planeseg/inference/zeroplane_default_dust3r_released_h5 \\
        --eval_root          /cluster/scratch/aoezkan/planeseg/eval \\
        --num_workers 4 --n_jobs 16
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
from torch.utils.data import DataLoader
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
# Config defaults (identical to v0 — comparability across versions)
# ---------------------------------------------------------------------------

SEG_THRESHOLD_PLANARITY = 0.3
SEG_NORMAL_THRESHOLD_DEG = 5.0
SEG_DEPTH_THRESHOLD_REL = 0.025
SEG_NEIGHBOR_MATCH_COUNT = 8

RANSAC_THRESHOLDS = (0.001, 0.005, 0.01)
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9

DEPTH_RECALL_THRESHOLDS = (0.05, 0.1, 0.6)
NORMAL_RECALL_THRESHOLDS_DEG = (5.0, 10.0, 30.0)

ZP_NONPLANAR_LABEL = 20
MOGE_FIT_METHOD = "svd"

DEFAULT_MOGE_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1"
DEFAULT_ZP_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/zeroplane_default_dust3r_released_h5"
DEFAULT_EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/eval"

BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Frame-id normalization
# ---------------------------------------------------------------------------

def _norm_fid(fid) -> str:
    """Normalize a frame id for cross-convention lookup.

    Handles 0-d torch.Tensor, bytes, str, int. ScanNet++ ``"frame_000123"`` →
    ``"123"``; NYUv2 / 7-Scenes ints/strings ``"5"`` → ``"5"``.
    """
    if hasattr(fid, "item"):
        try:
            fid = fid.item()
        except Exception:
            pass
    if isinstance(fid, (bytes, bytearray)):
        fid = fid.decode("utf-8")
    s = str(fid)
    if s.startswith("frame_"):
        s = s[len("frame_"):]
    try:
        return str(int(s))
    except ValueError:
        return s


def _decode_frame_ids(arr) -> List[str]:
    return [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in np.asarray(arr).tolist()
    ]


def _to_str_fid(raw_fid) -> str:
    """Stringify a raw frame id (preserves "frame_000123" form for ScanNet++)."""
    if hasattr(raw_fid, "item"):
        try:
            raw_fid = raw_fid.item()
        except Exception:
            pass
    if isinstance(raw_fid, (bytes, bytearray)):
        return raw_fid.decode("utf-8")
    return str(raw_fid)


# ---------------------------------------------------------------------------
# Plane-parameter computation + rendering (matches v0 behaviour)
# ---------------------------------------------------------------------------

def _scale_K_to_hw(K: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Scale K so its principal point matches the centre of an (H, W) image.

    ScanNet++ stores intrinsics at iPhone native resolution (~1920×1440) but
    the H5 depth/plane maps are 480×640. The reference eval scripts pass
    unscaled K to ``backproject_v1`` (a known pre-existing scale bug). For
    plane fitting + re-rendering we DO scale K — rendered depth must match raw
    GT depth in absolute meters for ``plane_recall_at_depth`` to be meaningful.

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
    """Re-render per-pixel depth + normal maps from {label_id: (a,b,c,d)}."""
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
    """Convert ZeroPlane ``plane_params`` (n/d form) to standard (a,b,c,d).

    ZeroPlane stores rows as (a, b, c) such that  a·x + b·y + c·z = 1. Standard
    form is  A·X + B·Y + C·Z + D = 0  with  ‖(A, B, C)‖ = 1. Conversion:
        nrm = ‖(a, b, c)‖
        (A, B, C) = (a, b, c) / nrm
        D = −1 / nrm

    Row ``i`` ↔ ZeroPlane label ``i`` BEFORE the +1 shift the loader applies.
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
# Spatial resize (NEAREST for labels, LINEAR for depth/normals; renorm)
# ---------------------------------------------------------------------------

def _resize_pred_spatial(
    labels_pred: np.ndarray,
    depth_pred: Optional[np.ndarray],
    normals_pred: Optional[np.ndarray],
    gt_hw: Tuple[int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Resize labels (NEAREST), depth/normals (LINEAR) to GT (H, W).
    Renormalize normals to unit length after the LINEAR resize.
    """
    H, W = gt_hw
    if labels_pred.shape[:2] != (H, W):
        labels_pred = cv2.resize(labels_pred, (W, H),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32)
    if depth_pred is not None and depth_pred.shape[:2] != (H, W):
        depth_pred = cv2.resize(depth_pred, (W, H), interpolation=cv2.INTER_LINEAR)
    if normals_pred is not None and normals_pred.shape[:2] != (H, W):
        normals_pred = cv2.resize(normals_pred, (W, H), interpolation=cv2.INTER_LINEAR)
    if normals_pred is not None:
        n = np.linalg.norm(normals_pred, axis=-1, keepdims=True)
        normals_pred = np.divide(normals_pred, np.clip(n, 1e-6, None), where=n > 1e-6)
    return labels_pred, depth_pred, normals_pred


# ---------------------------------------------------------------------------
# Lazy per-scene prediction loaders (one scene cached at a time)
# ---------------------------------------------------------------------------

class _LazyMogeRawScene:
    """Caches one MoGe scene's RAW signals (planarity, normal, depth_metric).
    Segmentation is deferred to the joblib worker (was sequential on main
    process in v0).
    """
    def __init__(self, dataset_root: str):
        self.root = dataset_root
        self._scene: Optional[str] = None
        self._planarity = self._normal = self._depth = None
        self._fid_to_idx: Dict[str, int] = {}

    def _load(self, scene_id: str) -> bool:
        if scene_id == self._scene:
            return self._planarity is not None
        # Evict
        self._scene = scene_id
        self._planarity = self._normal = self._depth = None
        self._fid_to_idx = {}
        h5_path = os.path.join(self.root, scene_id, "moge_signals.h5")
        if not os.path.isfile(h5_path):
            return False
        with h5py.File(h5_path, "r") as f:
            self._planarity = f["planarity"][:].astype(np.float32)
            self._normal = f["normal"][:].astype(np.float32)            # (B, H, W, 3)
            self._depth = f["depth_metric"][:].astype(np.float32)
            fids = _decode_frame_ids(f["frame_ids"][:])
            self._fid_to_idx = {_norm_fid(x): i for i, x in enumerate(fids)}
        return True

    def get(self, scene_id: str, frame_id) -> Optional[Dict[str, np.ndarray]]:
        if not self._load(scene_id):
            return None
        nfid = _norm_fid(frame_id)
        idx = self._fid_to_idx.get(nfid)
        if idx is None:
            return None
        return {
            "planarity": self._planarity[idx],
            "normal": self._normal[idx],
            "depth": self._depth[idx],
        }


class _LazyZeroPlaneScene:
    """Caches one ZeroPlane scene's processed predictions.

    Labels are remapped from ZeroPlane convention (20=non-planar, 0..19=planes)
    to standard (0=non-planar, 1..20=planes) via the +1 shift trick.
    Normals are transposed (3,H,W)→(H,W,3) and unit-normalized.
    Per-frame ``plane_params`` (n/d form) are kept raw — converted in the
    worker.
    """
    def __init__(self, dataset_root: str):
        self.root = dataset_root
        self._scene: Optional[str] = None
        self._labels = self._depth = self._normals = None
        self._fid_to_idx: Dict[str, int] = {}
        self._plane_params_nd: Dict[str, np.ndarray] = {}

    def _load(self, scene_id: str) -> bool:
        if scene_id == self._scene:
            return self._labels is not None
        self._scene = scene_id
        self._labels = self._depth = self._normals = None
        self._fid_to_idx = {}
        self._plane_params_nd = {}
        h5_path = os.path.join(self.root, scene_id, "planes.h5")
        if not os.path.isfile(h5_path):
            return False
        with h5py.File(h5_path, "r") as f:
            labels = f["planes"][:].astype(np.int32)
            # Shift 0..19 → 1..20, then map original 20 (now 21) → 0.
            labels = labels + 1
            labels[labels == ZP_NONPLANAR_LABEL + 1] = 0
            self._labels = labels
            self._depth = f["planes_depth"][:].astype(np.float32)
            normals = f["pixel_normals"][:].astype(np.float32)          # (B, 3, H, W)
            normals = normals.transpose(0, 2, 3, 1).copy()              # (B, H, W, 3)
            n = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = np.divide(normals, np.clip(n, 1e-6, None), where=n > 1e-6)
            self._normals = normals

            raw_fids = _decode_frame_ids(f["frame_ids"][:])
            self._fid_to_idx = {_norm_fid(x): i for i, x in enumerate(raw_fids)}

            if "plane_params" in f:
                pp_group = f["plane_params"]
                for raw_fid in raw_fids:
                    if raw_fid in pp_group:
                        self._plane_params_nd[_norm_fid(raw_fid)] = (
                            pp_group[raw_fid][:].astype(np.float32)
                        )
        return True

    def get(self, scene_id: str, frame_id) -> Optional[Dict[str, np.ndarray]]:
        if not self._load(scene_id):
            return None
        nfid = _norm_fid(frame_id)
        idx = self._fid_to_idx.get(nfid)
        if idx is None:
            return None
        return {
            "labels": self._labels[idx],
            "depth": self._depth[idx],
            "normals": self._normals[idx],
            "plane_params_nd": self._plane_params_nd.get(nfid),
        }


# ---------------------------------------------------------------------------
# Per-frame evaluation (joblib worker)
# ---------------------------------------------------------------------------

def _eval_one_frame_dispatch(
    method: str,
    scene_id: str,
    frame_id: str,
    depth_gt: np.ndarray,
    labels_gt: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    pred_payload: Optional[Dict[str, np.ndarray]],
    ransac_thresholds: Tuple[float, ...],
    depth_recall_thresholds: Tuple[float, ...],
    normal_recall_thresholds_deg: Tuple[float, ...],
    ransac_iterations: int,
    inlier_ratio_gate: float,
    seg_threshold_planarity: float,
    seg_normal_threshold_deg: float,
    seg_depth_threshold_rel: float,
    seg_neighbor_match_count: int,
    moge_fit_method: str,
) -> Dict[str, float]:
    """Build pred maps (segmenting + plane-fitting + re-rendering as needed),
    then compute the full metric block.

    For ``method=="moge"``: segmentation runs HERE in the worker (was a serial
    main-process step in v0). Plane params are then SVD-fitted from the
    segmented labels and per-pixel (depth, normal); maps are re-rendered.

    For ``method=="zeroplane"``: labels are pre-segmented in the loader. Plane
    params come from the H5 (n/d → (a,b,c,d)) when present, otherwise we fit
    from per-pixel data the same way as moge.

    For ``method=="gt"``: pred == GT (upper bound). No fitting / re-rendering.
    """
    # Defensive: forked workers don't inherit the main process's torch threads
    # cap, so this avoids 16x16 oversubscription on multi-CPU nodes.
    torch.set_num_threads(1)

    H, W = labels_gt.shape[:2]

    if method == "gt":
        labels_pred = labels_gt
        depth_pred = depth_gt
        normals_pred = None  # downstream sets normals_pred = normals_gt

    elif method == "moge":
        planarity = pred_payload["planarity"]
        normal = pred_payload["normal"]
        depth_metric = pred_payload["depth"]

        # 1. Segment from raw signals (this is the step that was serial on the
        #    main process in v0 — now parallelized across joblib workers).
        mask = planarity > seg_threshold_planarity
        labels_raw, _ = compute_vectorized_planar_segments_v5_relative(
            planarity_mask=mask,
            normal=normal,
            depth=depth_metric,
            normal_threshold_rad=float(np.deg2rad(seg_normal_threshold_deg)),
            depth_threshold=seg_depth_threshold_rel,
            neighbor_match_count_thresh=seg_neighbor_match_count,
            device="cpu",
        )
        if hasattr(labels_raw, "cpu"):
            labels_raw = labels_raw.cpu().numpy()
        labels_raw = labels_raw.astype(np.int32)

        # 2. Resize to GT (no-op if pred is already at GT shape).
        labels_pred, depth_resized, normal_resized = _resize_pred_spatial(
            labels_raw, depth_metric, normal, (H, W),
        )

        # 3. Fit per-plane params at GT resolution; K must be scaled to match.
        K_render = _scale_K_to_hw(K, (H, W))
        plane_params = compute_plane_params(
            depth=depth_resized,
            normal=normal_resized,
            plane_label=labels_pred,
            method=moge_fit_method,
            K=K_render,
            ignore_labels=(0,),
        )

        # 4. Re-render depth + normal maps from plane params.
        depth_pred, normals_pred = _render_plane_params_to_maps(
            plane_params, labels_pred, K_render, H, W,
        )

    elif method == "zeroplane":
        labels_raw = pred_payload["labels"].astype(np.int32)
        depth_raw = pred_payload["depth"].astype(np.float32)
        normals_raw = pred_payload["normals"].astype(np.float32)
        plane_params_nd = pred_payload.get("plane_params_nd")

        labels_pred, depth_resized, normals_resized = _resize_pred_spatial(
            labels_raw, depth_raw, normals_raw, (H, W),
        )

        K_render = _scale_K_to_hw(K, (H, W))
        if plane_params_nd is None:
            # H5 missing pre-computed plane_params for this frame — fall back
            # to fitting from per-pixel data (same recipe as moge).
            plane_params = compute_plane_params(
                depth=depth_resized,
                normal=normals_resized,
                plane_label=labels_pred,
                method=moge_fit_method,
                K=K_render,
                ignore_labels=(0,),
            )
        else:
            plane_params = _zp_plane_params_to_dict(plane_params_nd, label_offset=1)

        depth_pred, normals_pred = _render_plane_params_to_maps(
            plane_params, labels_pred, K_render, H, W,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    # ---- evaluate ----------------------------------------------------------
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
# Dataset adapter (same as v0)
# ---------------------------------------------------------------------------

def _build_gt_dataset(dataset_name: str, args):
    """Build the GT dataset without any sample/scene caps. ``--scenes`` and
    ``--frames`` are applied uniformly post-hoc via ``_apply_data_caps`` so
    each dataset gets identical, predictable semantics.
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


# ---------------------------------------------------------------------------
# Scene sharding (for parallel SLURM submission)
# ---------------------------------------------------------------------------

def _shard_scenes(scene_ids: List[str], shard_id: int, num_shards: int) -> List[str]:
    """Sequential split of ``scene_ids`` into ``num_shards`` equal-ish chunks.

    Distributes the remainder to the first ``len(scene_ids) % num_shards``
    shards so the sizes differ by at most 1.
    """
    n = len(scene_ids)
    base = n // num_shards
    rem = n % num_shards
    start = shard_id * base + min(shard_id, rem)
    end = start + base + (1 if shard_id < rem else 0)
    return scene_ids[start:end]


def _apply_data_caps(
    ds,
    dataset_name: str,
    scenes: Optional[int],
    frames: Optional[int],
) -> None:
    """In-place: cap ds.valid_pairs to first ``scenes`` scenes × ``frames``
    frames-per-scene (preserving original ordering).

    For NYU-v2 (single virtual scene), ``--scenes`` is ignored and
    ``--frames`` acts as a total-sample cap. For ScanNet++ and 7-Scenes,
    ``--scenes`` keeps the first N scene_ids encountered (sorted by their
    appearance in valid_pairs) and ``--frames`` truncates each scene's frame
    list.

    No-op when both args are None.
    """
    if scenes is None and frames is None:
        return

    def _subsample_evenly(items: List, n: int) -> List:
        """Evenly-spaced subsample of ``items`` to ``n`` elements (preserves
        endpoints). Matches v0's --max_frames_per_scene behavior."""
        if len(items) <= n:
            return items
        idxs = np.linspace(0, len(items) - 1, n).astype(int)
        return [items[i] for i in idxs]

    if dataset_name == "nyuv2":
        if frames is not None:
            ds.valid_pairs = _subsample_evenly(ds.valid_pairs, frames)
        return

    from collections import OrderedDict
    groups: "OrderedDict[str, List]" = OrderedDict()
    for vp in ds.valid_pairs:
        if dataset_name == "scannetpp":
            sid = vp[0].split("/")[-4]
        elif dataset_name == "sevenscenes":
            sid = vp[2]
        else:
            raise ValueError(f"unknown dataset: {dataset_name}")
        groups.setdefault(sid, []).append(vp)

    if scenes is not None:
        groups = OrderedDict(list(groups.items())[:scenes])
    if frames is not None:
        for sid in groups:
            groups[sid] = _subsample_evenly(groups[sid], frames)

    ds.valid_pairs = [vp for vps in groups.values() for vp in vps]
    if hasattr(ds, "scene_ids"):
        ds.scene_ids = list(groups.keys())


def _apply_shard_filter(ds, dataset_name: str, shard_id: int, num_shards: int) -> List[str]:
    """In-place filter ``ds.valid_pairs`` (and ``ds.scene_ids`` for ScanNet++)
    to the shard's scene assignment. Returns the assigned scene_id list.

    NYU-v2 is a single virtual scene → only shard 0 gets work; other shards
    return an empty list and the caller skips evaluation.
    """
    if dataset_name == "scannetpp":
        all_scenes = sorted(getattr(ds, "scene_ids", None) or [
            p[0].split("/")[-4] for p in ds.valid_pairs
        ])
        sub = _shard_scenes(all_scenes, shard_id, num_shards)
        sub_set = set(sub)
        ds.valid_pairs = [
            p for p in ds.valid_pairs if p[0].split("/")[-4] in sub_set
        ]
        ds.scene_ids = [s for s in all_scenes if s in sub_set]
        return sub
    if dataset_name == "nyuv2":
        return ["nyuv2"] if shard_id == 0 else []
    if dataset_name == "sevenscenes":
        all_scenes = sorted({vp[2] for vp in ds.valid_pairs})
        sub = _shard_scenes(all_scenes, shard_id, num_shards)
        sub_set = set(sub)
        ds.valid_pairs = [vp for vp in ds.valid_pairs if vp[2] in sub_set]
        return sub
    raise ValueError(f"unknown dataset: {dataset_name}")


# ---------------------------------------------------------------------------
# Output writers (same as v0; runtime.csv gains joblib_time_seconds)
# ---------------------------------------------------------------------------

def _summarize_rows(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in df.select_dtypes(include="number").columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std())
    return out


def _save_scene_csvs(scene_id: str, frame_rows: List[Dict], scene_dir: str) -> Dict[str, float]:
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
    extra_runtime_cols: Optional[Dict[str, float]] = None,
    *,
    write_dataset_aggregates: bool = True,
    runtime_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Write aggregate CSVs and the runtime row.

    When ``write_dataset_aggregates=False`` (sharded mode), only the runtime
    row is written and to ``runtime_dir`` instead of ``dataset_dir``. The
    returned dataset_row dict is empty in that case.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_row: Dict[str, float] = {}

    if write_dataset_aggregates:
        if all_frame_rows:
            pd.DataFrame.from_records(all_frame_rows).to_csv(
                os.path.join(dataset_dir, "aggregate_results.csv"), index=False,
            )
        df_scenes = pd.DataFrame(scene_summaries) if scene_summaries else pd.DataFrame()
        if not df_scenes.empty:
            df_scenes.to_csv(os.path.join(dataset_dir, "aggregate_per_scene.csv"), index=False)

        dataset_row = {
            "num_scenes": int(len(df_scenes)),
            "num_frames_total": int(df_scenes["num_frames"].sum()) if not df_scenes.empty else 0,
        }
        if not df_scenes.empty:
            mean_cols = [c for c in df_scenes.columns if c.endswith("_mean")]
            for mc in mean_cols:
                base = mc[: -len("_mean")]
                vals = df_scenes[mc].dropna()
                dataset_row[f"{base}_mean"] = float(vals.mean()) if len(vals) else float("nan")
                dataset_row[f"{base}_std"] = float(vals.std()) if len(vals) > 1 else float("nan")
        pd.DataFrame([dataset_row]).to_csv(
            os.path.join(dataset_dir, "aggregate_dataset.csv"), index=False,
        )

    n_frames = len(all_frame_rows)
    runtime_row: Dict[str, float] = {
        "wall_time_seconds": wall_time_s,
        "num_frames": n_frames,
        "frames_per_second": n_frames / wall_time_s if wall_time_s > 0 else float("nan"),
    }
    if extra_runtime_cols:
        runtime_row.update(extra_runtime_cols)

    rt_dir = runtime_dir if runtime_dir is not None else dataset_dir
    os.makedirs(rt_dir, exist_ok=True)
    pd.DataFrame([runtime_row]).to_csv(os.path.join(rt_dir, "runtime.csv"), index=False)
    return dataset_row


def _write_top_summary(
    method_dataset_stats: Dict[Tuple[str, str], Dict[str, float]],
    summary_path: str,
) -> None:
    if not method_dataset_stats:
        return
    rows = []
    for (method, dataset), stats in method_dataset_stats.items():
        rows.append({"method": method, "dataset": dataset, **stats})
    pd.DataFrame(rows).to_csv(summary_path, index=False)


# ---------------------------------------------------------------------------
# Per-dataset streaming evaluation (loop-inverted: ALL methods per batch)
# ---------------------------------------------------------------------------

def _evaluate_methods_streaming(
    methods: List[str],
    dataset_name: str,
    args,
) -> Tuple[Dict[str, List[Dict]], Dict[str, float], float, Dict[str, int]]:
    """Stream the dataset once via DataLoader; evaluate ALL methods per batch.

    Returns:
        per_method_rows[m]   list of per-frame metric dicts (with scene_id, frame_id)
        per_method_joblib[m] cumulative joblib wall time for that method
        wall_streaming        total wall time of the streaming pass (shared)
        per_method_skipped[m] count of frames where pred was missing
    """
    val_dataset = _build_gt_dataset(dataset_name, args)

    # Optional: cap scenes / frames-per-scene before sharding so all shards
    # see a consistent capped set.
    _apply_data_caps(val_dataset, dataset_name, args.scenes, args.frames)
    if args.scenes is not None or args.frames is not None:
        print(f"   [{dataset_name}] caps: scenes={args.scenes}, frames/scene={args.frames} "
              f"→ {len(val_dataset)} frames")

    # Optional: restrict to a single shard's scene assignment.
    if args.shard_id is not None:
        sub_scenes = _apply_shard_filter(
            val_dataset, dataset_name, args.shard_id, args.num_shards,
        )
        n_after = len(val_dataset)
        print(f"   [{dataset_name}] shard {args.shard_id}/{args.num_shards}: "
              f"{len(sub_scenes)} scenes, {n_after} frames")
        if n_after == 0:
            return (
                {m: [] for m in methods},
                {m: 0.0 for m in methods},
                0.0,
                {m: 0 for m in methods},
            )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    moge_loader = (
        _LazyMogeRawScene(os.path.join(args.moge_signals_root, dataset_name))
        if "moge" in methods else None
    )
    zp_loader = (
        _LazyZeroPlaneScene(os.path.join(args.zeroplane_h5_root, dataset_name))
        if "zeroplane" in methods else None
    )

    per_method_rows: Dict[str, List[Dict]] = {m: [] for m in methods}
    per_method_joblib: Dict[str, float] = {m: 0.0 for m in methods}
    per_method_skipped: Dict[str, int] = {m: 0 for m in methods}

    t0 = time.perf_counter()
    pbar = tqdm(val_loader, desc=f"  stream/{dataset_name:<11s}", unit="batch")

    for batch in pbar:
        scene_ids = batch["scene_id"]
        frame_ids_raw = batch["frame_idx"]
        gt_planes = batch["plane"]
        depths = batch["depth"]
        Ks = batch["K"]
        c2ws = batch["c2w"]

        B = len(scene_ids)

        # Per-method tasks for this batch.
        method_tasks: Dict[str, List[Dict]] = {m: [] for m in methods}

        for i in range(B):
            sid = scene_ids[i]
            raw_fid = (frame_ids_raw[i] if not isinstance(frame_ids_raw, torch.Tensor)
                       else frame_ids_raw[i])
            fid_str = _to_str_fid(raw_fid)

            depth_t = depths[i]
            depth_gt = (depth_t[0].numpy() if depth_t.ndim == 3
                        else depth_t.numpy()).astype(np.float32)

            plane_t = gt_planes[i]
            labels_gt = (plane_t[0].numpy() if plane_t.ndim == 3
                         else plane_t.numpy()).astype(np.int32)

            K = Ks[i].numpy().astype(np.float32)
            c2w = c2ws[i].numpy().astype(np.float32)

            for method in methods:
                if method == "gt":
                    pred_payload = None
                elif method == "moge":
                    pred_payload = moge_loader.get(sid, raw_fid)
                    if pred_payload is None:
                        per_method_skipped[method] += 1
                        continue
                elif method == "zeroplane":
                    pred_payload = zp_loader.get(sid, raw_fid)
                    if pred_payload is None:
                        per_method_skipped[method] += 1
                        continue
                else:
                    raise ValueError(f"unknown method: {method}")

                method_tasks[method].append({
                    "scene_id": sid,
                    "frame_id": fid_str,
                    "depth_gt": depth_gt,
                    "labels_gt": labels_gt,
                    "K": K,
                    "c2w": c2w,
                    "pred_payload": pred_payload,
                })

        # Joblib eval per method (separate calls give clean per-method timing;
        # loky's worker pool persists across calls, so per-call startup is
        # negligible).
        for method in methods:
            tasks = method_tasks[method]
            if not tasks:
                continue
            t_m = time.perf_counter()
            outputs = Parallel(n_jobs=args.n_jobs, backend="loky")(
                delayed(_eval_one_frame_dispatch)(
                    method,
                    t["scene_id"], t["frame_id"],
                    t["depth_gt"], t["labels_gt"], t["K"], t["c2w"],
                    t["pred_payload"],
                    args.ransac_thresholds,
                    args.depth_recall_thresholds,
                    args.normal_recall_thresholds_deg,
                    args.ransac_iterations,
                    args.inlier_ratio_gate,
                    SEG_THRESHOLD_PLANARITY,
                    SEG_NORMAL_THRESHOLD_DEG,
                    SEG_DEPTH_THRESHOLD_REL,
                    SEG_NEIGHBOR_MATCH_COUNT,
                    MOGE_FIT_METHOD,
                )
                for t in tasks
            )
            per_method_joblib[method] += time.perf_counter() - t_m

            for output, t in zip(outputs, tasks):
                # Attach scene/frame ids for later per-scene grouping. Use
                # setdefault so we don't clobber any keys evaluate_single_frame
                # might already have set.
                output.setdefault("scene_id", t["scene_id"])
                output.setdefault("frame_id", t["frame_id"])
                per_method_rows[method].append(output)

    wall = time.perf_counter() - t0
    return per_method_rows, per_method_joblib, wall, per_method_skipped


def _group_rows_by_scene(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for row in rows:
        sid = row.get("scene_id", "_")
        out.setdefault(sid, []).append(row)
    return out


# ---------------------------------------------------------------------------
# Aggregate-only mode (post-shard merge)
# ---------------------------------------------------------------------------

def _aggregate_only(args) -> None:
    """Walk per-scene CSVs and produce dataset-level aggregates.

    Reads ``<exp>/<method>/<dataset>/<scene>/results.csv`` + ``summary.csv``
    and writes ``aggregate_results.csv`` / ``aggregate_per_scene.csv`` /
    ``aggregate_dataset.csv`` / ``runtime.csv`` at the dataset level. Sums
    per-shard runtime info from ``<dataset>/_shards/shard_*/runtime.csv`` if
    present.
    """
    out_root = os.path.join(args.eval_root, args.exp)
    method_dataset_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    for method in args.methods:
        for ds_name in args.datasets:
            dataset_dir = os.path.join(out_root, method, ds_name)
            if not os.path.isdir(dataset_dir):
                print(f"  [skip] {method}/{ds_name}: dir not found")
                continue

            # Collect scene dirs (skip dot-files and the _shards bookkeeping dir).
            scene_ids: List[str] = []
            for entry in sorted(os.listdir(dataset_dir)):
                if entry.startswith(("_", ".")):
                    continue
                full = os.path.join(dataset_dir, entry)
                if not os.path.isdir(full):
                    continue
                if not os.path.isfile(os.path.join(full, "results.csv")):
                    continue
                scene_ids.append(entry)

            if not scene_ids:
                print(f"  [skip] {method}/{ds_name}: no scene dirs with results.csv")
                continue

            all_frame_rows: List[Dict] = []
            scene_summaries: List[Dict] = []
            for sid in scene_ids:
                sd = os.path.join(dataset_dir, sid)
                df_frames = pd.read_csv(os.path.join(sd, "results.csv"))
                all_frame_rows.extend(df_frames.to_dict("records"))
                summary_csv = os.path.join(sd, "summary.csv")
                if os.path.isfile(summary_csv):
                    df_sum = pd.read_csv(summary_csv)
                    scene_summaries.extend(df_sum.to_dict("records"))

            # Sum / max per-shard runtime info.
            shard_walls: List[float] = []
            shard_joblibs: List[float] = []
            shards_dir = os.path.join(dataset_dir, "_shards")
            if os.path.isdir(shards_dir):
                for sd_name in sorted(os.listdir(shards_dir)):
                    rt = os.path.join(shards_dir, sd_name, "runtime.csv")
                    if os.path.isfile(rt):
                        df = pd.read_csv(rt)
                        if not df.empty:
                            shard_walls.append(float(df["wall_time_seconds"].iloc[0]))
                            if "joblib_time_seconds" in df.columns:
                                shard_joblibs.append(float(df["joblib_time_seconds"].iloc[0]))

            extra: Dict[str, float] = {"aggregate_only": 1}
            if shard_walls:
                extra["max_shard_wall_time_seconds"] = max(shard_walls)
                extra["sum_shard_wall_time_seconds"] = sum(shard_walls)
                extra["num_shards"] = len(shard_walls)
            if shard_joblibs:
                extra["sum_shard_joblib_time_seconds"] = sum(shard_joblibs)

            wall = max(shard_walls) if shard_walls else 0.0

            print(f"  [aggregate] {method}/{ds_name}: "
                  f"{len(scene_ids)} scenes, {len(all_frame_rows)} frames"
                  + (f", {len(shard_walls)} shards" if shard_walls else ""))

            stats = _save_dataset_aggregates(
                dataset_dir=dataset_dir,
                all_frame_rows=all_frame_rows,
                scene_summaries=scene_summaries,
                wall_time_s=wall,
                extra_runtime_cols=extra,
            )
            method_dataset_stats[(method, ds_name)] = stats

    summary_path = os.path.join(out_root, "summary.csv")
    _write_top_summary(method_dataset_stats, summary_path)
    print(f"\n[DONE] aggregate-only summary: {summary_path}")


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
    ap.add_argument("--moge_signals_root", default=DEFAULT_MOGE_ROOT)
    ap.add_argument("--zeroplane_h5_root", default=DEFAULT_ZP_ROOT)
    ap.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--split", default="test",
                    help="ScanNet++ split (default: test). Ignored for nyuv2/sevenscenes.")
    ap.add_argument("--scenes", "--max_scenes", dest="scenes",
                    type=int, default=None,
                    help="Cap number of scenes. NYU-v2 has only one virtual scene "
                         "and ignores this flag — use --frames to cap its samples. "
                         "(--max_scenes is the v0-style alias.)")
    ap.add_argument("--frames", "--max_frames_per_scene", dest="frames",
                    type=int, default=None,
                    help="Cap frames PER scene via evenly-spaced subsampling. For "
                         "NYU-v2 (single virtual scene) this caps total samples. "
                         "(--max_frames_per_scene is the v0-style alias.)")
    ap.add_argument("--n_jobs", type=int,
                    default=min(16, os.cpu_count() or 16),
                    help="joblib parallel workers for per-frame evaluation.")
    ap.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader workers for GT loading (parallel JPEG/H5 decode).")

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

    # Scene sharding (for parallel SLURM submission). Both must be set together.
    # When set, the dataset is filtered to a single shard's scene assignment;
    # per-scene CSVs go to the standard <method>/<dataset>/<scene>/ location;
    # runtime.csv goes to <method>/<dataset>/_shards/shard_<NNN>/runtime.csv;
    # dataset-level aggregates are SKIPPED (a final --aggregate_only run merges
    # them after all shards finish).
    ap.add_argument("--shard_id", type=int, default=None,
                    help="Shard index in [0, --num_shards).")
    ap.add_argument("--num_shards", type=int, default=None,
                    help="Total number of shards for this (method, dataset) pass.")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip evaluation; walk per-scene CSVs and write "
                         "dataset aggregates + cross-(method, dataset) summary.")

    args = ap.parse_args()
    args.ransac_thresholds = tuple(args.ransac_thresholds)
    args.depth_recall_thresholds = tuple(args.depth_recall_thresholds)
    args.normal_recall_thresholds_deg = tuple(args.normal_recall_thresholds_deg)


    # Validate shard args
    if (args.shard_id is None) != (args.num_shards is None):
        ap.error("--shard_id and --num_shards must be set together")
    if args.shard_id is not None:
        if args.num_shards <= 0:
            ap.error(f"--num_shards must be > 0, got {args.num_shards}")
        if not (0 <= args.shard_id < args.num_shards):
            ap.error(f"--shard_id ({args.shard_id}) must be in "
                     f"[0, --num_shards={args.num_shards})")

    # Aggregate-only mode bypasses all evaluation work.
    if args.aggregate_only:
        os.makedirs(os.path.join(args.eval_root, args.exp), exist_ok=True)
        print(f"[aggregate-only] exp={args.exp}")
        _aggregate_only(args)
        return

    # Pre-flight
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

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    out_root = os.path.join(args.eval_root, args.exp)
    os.makedirs(out_root, exist_ok=True)

    print("================================================================")
    print(f" evaluate_gt_moge_zeroplane_v1 — exp={args.exp}")
    print("================================================================")
    print(f"  methods:        {args.methods}")
    print(f"  datasets:       {args.datasets}")
    print(f"  moge root:      {args.moge_signals_root}")
    print(f"  zeroplane root: {args.zeroplane_h5_root}")
    print(f"  output root:    {out_root}")
    print(f"  ransac thr:     {args.ransac_thresholds} m   iters={args.ransac_iterations}")
    print(f"  depth recall:   {args.depth_recall_thresholds} m")
    print(f"  normal recall:  {args.normal_recall_thresholds_deg} deg")
    print(f"  n_jobs:         {args.n_jobs}    num_workers: {args.num_workers}")
    print(f"  caps:           scenes={args.scenes}, frames/scene={args.frames}")
    print("  seg params:     "
          f"plan>{SEG_THRESHOLD_PLANARITY}, "
          f"normal<{SEG_NORMAL_THRESHOLD_DEG}°, "
          f"depth_rel<{SEG_DEPTH_THRESHOLD_REL}, "
          f"match≥{SEG_NEIGHBOR_MATCH_COUNT}")
    print("================================================================")

    is_sharded = args.shard_id is not None
    method_dataset_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    for ds_name in args.datasets:
        print(f"\n==> streaming dataset: {ds_name}  (methods: {args.methods})"
              + (f"  [shard {args.shard_id}/{args.num_shards}]" if is_sharded else ""))
        per_rows, per_joblib, wall, per_skipped = _evaluate_methods_streaming(
            args.methods, ds_name, args,
        )
        for method in args.methods:
            n_frames = len(per_rows[method])
            print(f"   [{method:<10s}] {n_frames} frames evaluated, "
                  f"{per_skipped[method]} skipped, "
                  f"joblib={per_joblib[method]:.1f}s, total_wall={wall:.1f}s")

            by_scene = _group_rows_by_scene(per_rows[method])
            out_dataset_dir = os.path.join(out_root, method, ds_name)
            scene_summaries: List[Dict] = []
            for sid in sorted(by_scene.keys()):
                scene_dir = os.path.join(out_dataset_dir, sid)
                summary = _save_scene_csvs(sid, by_scene[sid], scene_dir)
                scene_summaries.append(summary)

            extra_runtime_cols: Dict[str, float] = {
                "joblib_time_seconds": per_joblib[method],
                "frames_skipped": per_skipped[method],
                "shared_streaming_pass": 1,
            }
            if is_sharded:
                extra_runtime_cols["shard_id"] = args.shard_id
                extra_runtime_cols["num_shards"] = args.num_shards
                runtime_dir = os.path.join(
                    out_dataset_dir, "_shards", f"shard_{args.shard_id:03d}",
                )
            else:
                runtime_dir = None

            stats = _save_dataset_aggregates(
                dataset_dir=out_dataset_dir,
                all_frame_rows=per_rows[method],
                scene_summaries=scene_summaries,
                wall_time_s=wall,
                extra_runtime_cols=extra_runtime_cols,
                write_dataset_aggregates=not is_sharded,
                runtime_dir=runtime_dir,
            )
            if not is_sharded:
                method_dataset_stats[(method, ds_name)] = stats

    if is_sharded:
        print(f"\n[DONE] shard {args.shard_id}/{args.num_shards} written. "
              f"Run --aggregate_only after all shards complete.")
    else:
        summary_path = os.path.join(out_root, "summary.csv")
        _write_top_summary(method_dataset_stats, summary_path)
        print(f"\n[DONE] cross-(method, dataset) summary: {summary_path}")


if __name__ == "__main__":
    main()
