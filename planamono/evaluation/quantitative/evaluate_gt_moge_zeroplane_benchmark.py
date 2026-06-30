"""
Benchmark evaluation: gt / moge / moge_ep2 / zeroplane / metric3d on
scannetpp / nyuv2 / sevenscenes using the merged ZeroPlane + PlaneRCNN +
PlaneRecTR metric kernels. metric3d is currently only available for ScanNet++
(``<root>/<dataset>/<split>/<scene>/inference.h5``); its depth is scaled by
``--metric3d_depth_scale`` (default 1.2833 ≈ min(616/480, 1064/640) for the
ScanNet++ canonical-resize) before plane fitting.

Same CLI / output layout as ``evaluate_gt_moge_zeroplane.py``. Only the
per-frame metric block is replaced — instead of planamono-native metrics
(prec@<τ>cm, bp_*, plane_recall_d/n_*, normal_err_deg_*, offset_err_m_*)
this script runs ``compute_benchmark_metrics()`` from
``metrics_benchmark.py``.

Per-frame metric columns (≈40):
    RI, VI, SC                                                  ZeroPlane evaluateMasks
    DE_{rel, rel_sqr, log10, rmse, rmse_log,
        accuracy_1, accuracy_2, accuracy_3}                     ZeroPlane evaluateDepths (plane_depth)
    per_pixel_depth_{01, 06}                                    ZP / PlaneRecTR (indoor)
    per_plane_depth_{005, 01, 06}                               ZP / PlaneRecTR (indoor)
    per_pixel_normal_{5, 30}                                    ZP / PlaneRecTR
    per_plane_normal_{5, 10, 30}                                ZP / PlaneRecTR
    per_pixel_offset_{50mm, 150mm, 300mm}                       PlaneRecTR (ZP dropped)
    per_plane_offset_{50mm, 150mm, 300mm}                       PlaneRecTR (ZP dropped)
    mean_normal_error_deg, median_normal_error_deg              ZeroPlane Hungarian
    mean_offset_error_m, median_offset_error_m                  ZeroPlane Hungarian (offset half re-enabled)
    AP@{20, 30, 60, 90}cm                                       PlaneRCNN (area as proxy score)
    plane_param_L2_{mean, area_weighted}                        PlaneRCNN evaluatePlaneDepth

Output layout (identical to evaluate_gt_moge_zeroplane.py):
    <eval_root>/<exp>/
        <method>/<dataset>/<scene>/{results,summary}.csv
        <method>/<dataset>/{aggregate_results,aggregate_per_scene,aggregate_dataset,runtime}.csv
        summary.csv
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


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.paths import (                                                                # noqa: E402
    repo_path, scannetpp_path, scannetpp_rend_plane_path,
    nyuv2_path, sevenscenes_path,
)
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset                       # noqa: E402
from planamono.shared.segmentation.plan2seg import compute_vectorized_planar_segments_v5_relative  # noqa: E402
from planamono.shared.segmentation.compute_plane_params import compute_plane_params           # noqa: E402
from planamono.shared.plane_fitting.metrics_planes import compute_gt_normals_from_depth_labels  # noqa: E402
from planamono.shared.plane_fitting import backproject_v1                                    # noqa: E402
from planamono.evaluation.quantitative.eval_utils import compute_plane_metrics                # noqa: E402
from planamono.shared.plane_fitting import set_ransac_seed                                    # noqa: E402
from planamono.evaluation.quantitative.metrics_benchmark import compute_benchmark_metrics    # noqa: E402


# ---------------------------------------------------------------------------
# Config defaults — matched to evaluate_gt_moge_zeroplane.py
# ---------------------------------------------------------------------------

SEG_THRESHOLD_PLANARITY = 0.3
SEG_NORMAL_THRESHOLD_DEG = 5.0
SEG_DEPTH_THRESHOLD_REL = 0.025
SEG_NEIGHBOR_MATCH_COUNT = 8

ZP_NONPLANAR_LABEL = 20
MOGE_FIT_METHOD = "svd"

# RANSAC prec/rec@<τ>cm — borrowed from evaluate_gt_moge_zeroplane.py.
# These are NOT part of the ZP/PlaneRCNN/PlaneRecTR benchmark suite; they're
# planamono's own 3D-plane prec/rec at multiple distance thresholds. Run via
# eval_utils.compute_plane_metrics on (backprojected GT-depth points,
# pred labels). Adds 6 columns: prec@{0.1,0.5,1.0}cm + rec@{0.1,0.5,1.0}cm.
RANSAC_THRESHOLDS = (0.001, 0.005, 0.01)            # meters
RANSAC_ITERATIONS = 200
INLIER_RATIO_GATE = 0.9

# Indoor max-depth defaults (ZP convention: scannet=10, nyu=10, sevenscenes=10).
DATASET_MAX_DEPTH = {
    "scannetpp": 10.0,
    "nyuv2": 10.0,
    "sevenscenes": 10.0,
}
DATASET_INDOOR = {
    "scannetpp": True,
    "nyuv2": True,
    "sevenscenes": True,
}

DEFAULT_MOGE_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1"
DEFAULT_MOGE_EP2_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep2"
DEFAULT_ZP_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/zeroplane_default_dust3r_released_h5"
DEFAULT_METRIC3D_ROOT = "/cluster/scratch/ayavuz/inference/metric3d_v2_epoch1"
DEFAULT_EVAL_ROOT = "/cluster/scratch/aoezkan/planeseg/eval"

# Metric3D was inferred on a downscaled ScanNet++ image; multiply by this scale
# to recover true metric depth at the H5 resolution.
#   SCALE = min(616/480, 1064/640) = 1.28333...  (ScanNet++ specific)
# Equivalent to the de-canonical step `pred_depth *= focal/1000` in
# planamono/moge/baselines/metric3d_v2.py that the upstream H5 dump skipped.
DEFAULT_METRIC3D_DEPTH_SCALE = min(616 / 480, 1064 / 640)

# Datasets for which Metric3D inference H5s are available. The upstream
# inference job only ran on ScanNet++; passing --methods metric3d together
# with another dataset will be flagged as a config error.
METRIC3D_DATASETS = {"scannetpp"}

# Methods that share the MoGe signals.h5 loading + segmentation code path.
# Each entry maps to the CLI arg name holding its signals root.
_MOGE_METHODS: Dict[str, str] = {
    "moge": "moge_signals_root",
    "moge_ep2": "moge_ep2_signals_root",
}


def _moge_root_for_method(method: str, args) -> str:
    return getattr(args, _MOGE_METHODS[method])


# ---------------------------------------------------------------------------
# Utilities (cribbed verbatim from evaluate_gt_moge_zeroplane.py)
# ---------------------------------------------------------------------------

def _norm_fid(fid) -> str:
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


def _scale_K_to_hw(K: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
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
    out: Dict[int, np.ndarray] = {}
    for i in range(nd_arr.shape[0]):
        nd = np.asarray(nd_arr[i], dtype=np.float64)
        nrm = float(np.linalg.norm(nd))
        if nrm < 1e-12:
            continue
        unit_n = nd / nrm
        D = -1.0 / nrm
        out[i + label_offset] = np.array(
            [unit_n[0], unit_n[1], unit_n[2], D], dtype=np.float64
        )
    return out


# ---------------------------------------------------------------------------
# Shape matching
# ---------------------------------------------------------------------------

_SPATIAL_KEYS = {"labels", "depth", "normals"}


def _match_to_gt_shape(pred: Dict[str, np.ndarray], gt_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
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
    if "normals" in out and out["normals"] is not None:
        n = np.linalg.norm(out["normals"], axis=-1, keepdims=True)
        out["normals"] = np.divide(out["normals"], np.clip(n, 1e-6, None), where=n > 1e-6)
    return out


# ---------------------------------------------------------------------------
# Per-frame benchmark worker
# ---------------------------------------------------------------------------

def _eval_one_frame_benchmark(
    scene_id: str,
    frame_id: str,
    depth_gt: np.ndarray,
    labels_gt: np.ndarray,
    K: np.ndarray,
    K_ransac: np.ndarray,
    c2w: np.ndarray,
    labels_pred: np.ndarray,
    depth_pred: np.ndarray,
    pred_plane_params: Dict[int, np.ndarray],
    gt_plane_params: Dict[int, np.ndarray],
    eval_indoor: bool,
    max_depth: float,
    ransac_thresholds: Tuple[float, ...],
    ransac_iterations: int,
    inlier_ratio_gate: float,
    rivoisc_ver: str,
    ransac_seed: Optional[int] = 0,
) -> Dict[str, float]:
    # Reproducible RANSAC for the prec/rec block below (see
    # docs/ransac_seeding_reproducibility.md). Runs inside the loky worker.
    set_ransac_seed(ransac_seed)
    metrics = compute_benchmark_metrics(
        pred_seg=labels_pred,
        gt_seg=labels_gt,
        pred_depth=depth_pred,
        gt_depth=depth_gt,
        pred_plane_params=pred_plane_params,
        gt_plane_params=gt_plane_params,
        K=K,
        eval_indoor=eval_indoor,
        max_depth=max_depth,
        rivoisc_ver=rivoisc_ver,
    )

    # planamono RANSAC prec/rec at multiple thresholds, computed on
    # (backprojected GT depth, pred labels). Same recipe as
    # evaluate_gt_moge_zeroplane.py's evaluate_single_frame.
    #
    # K_ransac controls whether scaled (correct, default) or unscaled (v0
    # convention, has the K-scale bug) intrinsics are used for backprojection.
    # See --kscaled flag.
    pts_world, pt_labels, _ = backproject_v1(depth_gt, K_ransac, c2w, labels_pred)
    if pts_world.shape[0] == 0:
        ransac_block = {
            **{f"prec@{t*100:.1f}cm": 0.0 for t in ransac_thresholds},
            **{f"rec@{t*100:.1f}cm": 0.0 for t in ransac_thresholds},
        }
    else:
        ransac_block = compute_plane_metrics(
            pts_world, pt_labels, ransac_thresholds,
            num_iterations=ransac_iterations,
            inlier_ratio_gate=inlier_ratio_gate,
            ransac_seed=ransac_seed,
        )

    return {"scene_id": scene_id, "frame_idx": frame_id, **metrics, **ransac_block}


# ---------------------------------------------------------------------------
# Dataset GT adapters / scene index (cribbed from base script)
# ---------------------------------------------------------------------------

def _build_gt_dataset(dataset_name: str, args):
    if dataset_name == "scannetpp":
        return ScanNetPPPlaneDataset(
            rgb_root=os.path.join(scannetpp_path, "data"),
            plane_label_root=scannetpp_rend_plane_path,
            sem_label_root=scannetpp_rend_plane_path,
            depth_label_root=scannetpp_rend_plane_path,
            split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
            split=args.split,
            max_scenes=args.max_scenes,
        )
    if dataset_name == "nyuv2":
        from planamono.shared.datasets.nyuv2_plane_dataset import NYUv2PlaneDataset
        return NYUv2PlaneDataset(
            data_root=nyuv2_path, split="test",
            max_samples=args.max_scenes,
        )
    if dataset_name == "sevenscenes":
        from planamono.shared.datasets.sevenscenes_plane_dataset import SevenScenesPlaneDataset
        return SevenScenesPlaneDataset(
            data_root=sevenscenes_path, split="val",
            max_samples=args.max_scenes,
        )
    raise ValueError(f"unknown dataset: {dataset_name}")


def _build_scene_index(dataset_name: str, ds) -> Dict[str, List[Tuple[int, str]]]:
    out: Dict[str, List[Tuple[int, str]]] = {}
    if dataset_name == "scannetpp":
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
            sample_idx = int(vp[1])
            out.setdefault("nyuv2", []).append((i, str(sample_idx)))
        return out
    if dataset_name == "sevenscenes":
        for i, vp in enumerate(ds.valid_pairs):
            sample_idx = int(vp[1])
            scene_id = vp[2]
            out.setdefault(scene_id, []).append((i, str(sample_idx)))
        return out
    raise ValueError(f"unknown dataset: {dataset_name}")


def _load_gt_sample(ds, ds_idx: int) -> Dict[str, np.ndarray]:
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
# Per-method scene loaders
# ---------------------------------------------------------------------------

def _load_moge_scene(scene_dir: str) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    h5_path = os.path.join(scene_dir, "moge_signals.h5")
    if not os.path.isfile(h5_path):
        return None
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:
        frame_ids = _decode_frame_ids(f["frame_ids"][:])
        for i, raw_fid in enumerate(frame_ids):
            planarity = f["planarity"][i].astype(np.float32)
            normal = f["normal"][i].astype(np.float32)
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
    # Path resolution: ZeroPlane stores per-scene predictions for ScanNet++ and
    # 7-Scenes at <scene_dir>/planes.h5 (nested), but NYU-v2 as a single flat
    # <dataset_dir>/planes.h5 (no per-scene subdir, since NYU is one virtual
    # scene with 654 samples). Try nested first, then fall back to the parent.
    h5_path = os.path.join(scene_dir, "planes.h5")
    if not os.path.isfile(h5_path):
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
            labels = labels + 1
            labels[labels == ZP_NONPLANAR_LABEL + 1] = 0
            depth = f["planes_depth"][i].astype(np.float32)
            normals = f["pixel_normals"][i].astype(np.float32)
            normals = normals.transpose(1, 2, 0).copy()
            n = np.linalg.norm(normals, axis=-1, keepdims=True)
            normals = np.divide(normals, np.clip(n, 1e-6, None), where=n > 1e-6)

            plane_params_nd: Optional[np.ndarray] = None
            if has_pp_group and raw_fid in f["plane_params"]:
                plane_params_nd = f["plane_params"][raw_fid][:].astype(np.float32)

            out[_norm_fid(raw_fid)] = {
                "labels": labels,
                "depth": depth,
                "normals": normals,
                "plane_params_nd": plane_params_nd,
            }
    return out


def _load_metric3d_scene(
    scene_dir: str,
    depth_scale: float = 1.0,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Read Metric3D inference.h5 and run our segmentation per frame.

    Mirrors ``_load_moge_scene`` but reads from ``inference.h5`` (Metric3D
    job format) and applies ``depth_scale`` to recover true metric units.
    Field names: depth/normals/planarity (note plural ``normals``).
    """
    h5_path = os.path.join(scene_dir, "inference.h5")
    if not os.path.isfile(h5_path):
        return None
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(h5_path, "r") as f:
        frame_ids = _decode_frame_ids(f["frame_ids"][:])
        for i, raw_fid in enumerate(frame_ids):
            planarity = f["planarity"][i].astype(np.float32)
            normal = f["normals"][i].astype(np.float32)
            depth = f["depth"][i].astype(np.float32) * depth_scale
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


# ---------------------------------------------------------------------------
# Output writers (cribbed from base script)
# ---------------------------------------------------------------------------

def _summarize_rows(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c in df.select_dtypes(include="number").columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std())
    return out


def _save_moge_pred_h5(
    pred_h5_data: List[Dict],
    scene_dir: str,
) -> None:
    """Write per-scene plane_labels.h5 and plane_params.h5 for MoGe predictions.

    Layout:
        plane_labels.h5
            frame_ids:   (N_frames,) variable-length utf-8
            plane_labels: (N_frames, H, W) int32  (planamono convention,
                          0 = non-planar, positive ints = planes)

        plane_params.h5
            frame_ids:    (N_frames,) variable-length utf-8
            label_ids/<frame_id>:    (N_planes,) int32   — the planamono
                                     label each row of `params` belongs to
            plane_params/<frame_id>: (N_planes, 4) float64 — Hessian normal
                                     form (a, b, c, d) where ‖(a,b,c)‖ = 1
                                     and the plane is a·X + b·Y + c·Z + d = 0
    """
    if not pred_h5_data:
        return
    os.makedirs(scene_dir, exist_ok=True)

    frame_ids = [str(d["frame_id"]) for d in pred_h5_data]
    str_dtype = h5py.string_dtype(encoding="utf-8")

    # plane_labels.h5 — stacked (N, H, W). All frames in a scene share
    # gt_hw (because labels_pred was already resized to gt_hw upstream).
    labels_arr = np.stack(
        [d["labels_pred"].astype(np.int32) for d in pred_h5_data], axis=0
    )
    with h5py.File(os.path.join(scene_dir, "plane_labels.h5"), "w") as f:
        f.create_dataset("frame_ids", data=np.array(frame_ids, dtype=object),
                         dtype=str_dtype)
        f.create_dataset("plane_labels", data=labels_arr,
                         compression="gzip", compression_opts=4)

    # plane_params.h5 — variable-length per frame, group-keyed by frame id.
    with h5py.File(os.path.join(scene_dir, "plane_params.h5"), "w") as f:
        f.create_dataset("frame_ids", data=np.array(frame_ids, dtype=object),
                         dtype=str_dtype)
        params_grp = f.create_group("plane_params")
        ids_grp = f.create_group("label_ids")
        for d in pred_h5_data:
            fid = str(d["frame_id"])
            params_dict = d["pred_plane_params"] or {}
            if params_dict:
                sorted_lids = sorted(int(k) for k in params_dict.keys())
                params_arr = np.stack(
                    [np.asarray(params_dict[lid], dtype=np.float64)
                     for lid in sorted_lids],
                    axis=0,
                )
                params_grp.create_dataset(fid, data=params_arr)
                ids_grp.create_dataset(
                    fid, data=np.asarray(sorted_lids, dtype=np.int32)
                )
            else:
                params_grp.create_dataset(fid, data=np.zeros((0, 4), dtype=np.float64))
                ids_grp.create_dataset(fid, data=np.zeros(0, dtype=np.int32))


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
) -> Dict[str, float]:
    os.makedirs(dataset_dir, exist_ok=True)
    if all_frame_rows:
        pd.DataFrame.from_records(all_frame_rows).to_csv(
            os.path.join(dataset_dir, "aggregate_results.csv"), index=False,
        )
    df_scenes = pd.DataFrame(scene_summaries) if scene_summaries else pd.DataFrame()
    if not df_scenes.empty:
        df_scenes.to_csv(os.path.join(dataset_dir, "aggregate_per_scene.csv"), index=False)

    dataset_row: Dict[str, float] = {
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
    pd.DataFrame([{
        "wall_time_seconds": wall_time_s,
        "num_frames": n_frames,
        "frames_per_second": n_frames / wall_time_s if wall_time_s > 0 else float("nan"),
    }]).to_csv(os.path.join(dataset_dir, "runtime.csv"), index=False)

    return dataset_row


def _write_top_summary(stats: Dict[Tuple[str, str], Dict[str, float]], path: str) -> None:
    if not stats:
        return
    rows = []
    for (method, dataset), s in stats.items():
        rows.append({"method": method, "dataset": dataset, **s})
    pd.DataFrame(rows).to_csv(path, index=False)


def _read_scene_csvs_under(dataset_dir: str) -> Tuple[List[Dict], List[Dict]]:
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


def _aggregate_from_disk(args, out_root: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    out_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
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
            out_stats[(method, ds_name)] = stats
    _write_top_summary(out_stats, os.path.join(out_root, "summary.csv"))
    return out_stats


# ---------------------------------------------------------------------------
# Per-(method, dataset) evaluation
# ---------------------------------------------------------------------------

def _build_pred_arrays(
    method: str,
    pred: Dict[str, np.ndarray],
    gt: Dict[str, np.ndarray],
    gt_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """For non-GT methods: produce (labels_pred, depth_pred, plane_params_dict)
    rendered to GT shape, with K scaled to GT resolution.
    """
    K_render = _scale_K_to_hw(gt["K"], gt_hw)
    labels_pred = pred["labels"]

    if method in _MOGE_METHODS or method == "metric3d":
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

    depth_pred, _ = _render_plane_params_to_maps(
        plane_params, labels_pred, K_render, gt_hw[0], gt_hw[1],
    )
    return labels_pred, depth_pred, plane_params


def _build_gt_plane_params(
    depth_gt: np.ndarray,
    labels_gt: np.ndarray,
    K_render: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Fit GT plane params via SVD on backprojected GT XYZ, oriented by
    GT depth-derived normals.
    """
    normals_gt = compute_gt_normals_from_depth_labels(
        depth_gt, labels_gt, K_render, ignore_labels=(0,),
    ).astype(np.float64)
    return compute_plane_params(
        depth=depth_gt,
        normal=normals_gt,
        plane_label=labels_gt,
        method=MOGE_FIT_METHOD,
        K=K_render,
        ignore_labels=(0,),
    )


def _evaluate_method_dataset(
    method: str,
    dataset_name: str,
    ds,
    scene_index: Dict[str, List[Tuple[int, str]]],
    args,
) -> Tuple[List[Dict], List[Dict], float]:
    out_dataset_dir = os.path.join(args.eval_root, args.exp, method, dataset_name)
    os.makedirs(out_dataset_dir, exist_ok=True)

    if method in _MOGE_METHODS:
        moge_dataset_root = os.path.join(_moge_root_for_method(method, args), dataset_name)
    else:
        moge_dataset_root = None
    zp_dataset_root = os.path.join(args.zeroplane_h5_root, dataset_name)
    # Metric3D dump layout adds the split as an extra subdir
    # (<root>/<dataset>/<split>/<scene>/inference.h5).
    if method == "metric3d":
        metric3d_dataset_root = os.path.join(
            args.metric3d_h5_root, dataset_name, args.split,
        )
    else:
        metric3d_dataset_root = None

    eval_indoor = DATASET_INDOOR.get(dataset_name, True)
    max_depth = DATASET_MAX_DEPTH.get(dataset_name, 10.0)

    scene_ids = sorted(scene_index.keys())
    scene_filter = getattr(args, "_scene_filter", None)
    if scene_filter is not None:
        before = len(scene_ids)
        scene_ids = [s for s in scene_ids if s in scene_filter]
        skipped = before - len(scene_ids)
        if skipped:
            tqdm.write(f"  [filter] {dataset_name}: {len(scene_ids)}/{before} scenes (skipped {skipped})")

    # Sharding: split scene_ids into num_shards equal-ish chunks; remainder
    # distributed to the first (n % num_shards) shards.
    if args.shard_id is not None and args.num_shards is not None and args.num_shards > 1:
        n = len(scene_ids)
        if n == 0:
            tqdm.write(f"  [shard] {dataset_name}: empty after filter, nothing to do")
            return [], [], 0.0
        base = n // args.num_shards
        rem = n % args.num_shards
        start = args.shard_id * base + min(args.shard_id, rem)
        end = start + base + (1 if args.shard_id < rem else 0)
        before = n
        scene_ids = scene_ids[start:end]
        tqdm.write(
            f"  [shard] {dataset_name}: {len(scene_ids)}/{before} scenes "
            f"(shard {args.shard_id}/{args.num_shards}, slice [{start}:{end}])"
        )

    all_frame_rows: List[Dict] = []
    scene_summaries: List[Dict] = []
    t0 = time.perf_counter()

    max_fps = getattr(args, "max_frames_per_scene", None)

    for sid in tqdm(scene_ids, desc=f"  {method:<10s}/{dataset_name:<11s}", unit="scene"):
        frame_list = scene_index[sid]
        if max_fps is not None and len(frame_list) > max_fps:
            idx = np.linspace(0, len(frame_list) - 1, max_fps).astype(int)
            frame_list = [frame_list[i] for i in idx]

        if method == "gt":
            pred_scene = None
        elif method in _MOGE_METHODS:
            pred_scene = _load_moge_scene(os.path.join(moge_dataset_root, sid))
        elif method == "zeroplane":
            pred_scene = _load_zeroplane_scene(os.path.join(zp_dataset_root, sid))
        elif method == "metric3d":
            pred_scene = _load_metric3d_scene(
                os.path.join(metric3d_dataset_root, sid),
                depth_scale=args.metric3d_depth_scale,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        if method != "gt" and pred_scene is None:
            tqdm.write(f"  [skip] {method}/{dataset_name}/{sid}: predictions missing")
            continue

        tasks = []
        # Collect MoGe pred data for the per-scene H5 dump (only used when
        # method=="moge" and --save_moge_h5 is on). One entry per evaluated
        # frame; appended in lockstep with `tasks`.
        moge_h5_data: List[Dict] = []
        for ds_idx, raw_fid in frame_list:
            nfid = _norm_fid(raw_fid)
            if method != "gt" and nfid not in pred_scene:
                continue

            gt = _load_gt_sample(ds, ds_idx)
            gt_hw = gt["labels"].shape[:2]
            K_render = _scale_K_to_hw(gt["K"], gt_hw)

            gt_plane_params = _build_gt_plane_params(
                gt["depth"], gt["labels"], K_render,
            )

            if method == "gt":
                labels_pred = gt["labels"]
                # For GT method we render plane depth from fitted GT params, so the
                # depth_pred used by AP / per-pixel-depth-recall reflects the same
                # "perfectly aligned" plane equations.
                depth_pred, _ = _render_plane_params_to_maps(
                    gt_plane_params, labels_pred, K_render, gt_hw[0], gt_hw[1],
                )
                pred_plane_params = gt_plane_params
            else:
                pred = _match_to_gt_shape(pred_scene[nfid], gt_hw)
                labels_pred, depth_pred, pred_plane_params = _build_pred_arrays(
                    method, pred, gt, gt_hw,
                )

            # K for the RANSAC prec/rec block. Default: K_render (scaled, correct).
            # If --no-kscaled, use the unscaled GT K to match v0's convention.
            K_ransac = K_render if args.kscaled else gt["K"]

            tasks.append({
                "scene_id": sid,
                "frame_id": gt["frame_idx"],
                "depth_gt": gt["depth"],
                "labels_gt": gt["labels"],
                "K": K_render,
                "K_ransac": K_ransac,
                "c2w": gt["c2w"],
                "labels_pred": labels_pred,
                "depth_pred": depth_pred,
                "pred_plane_params": pred_plane_params,
                "gt_plane_params": gt_plane_params,
            })

            if method in _MOGE_METHODS and args.save_moge_h5:
                moge_h5_data.append({
                    "frame_id": gt["frame_idx"],
                    "labels_pred": labels_pred,
                    "pred_plane_params": pred_plane_params,
                })

        pred_scene = None

        if not tasks:
            continue

        outputs = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_eval_one_frame_benchmark)(
                t["scene_id"], t["frame_id"],
                t["depth_gt"], t["labels_gt"], t["K"], t["K_ransac"], t["c2w"],
                t["labels_pred"], t["depth_pred"],
                t["pred_plane_params"], t["gt_plane_params"],
                eval_indoor, max_depth,
                args.ransac_thresholds,
                args.ransac_iterations,
                args.inlier_ratio_gate,
                args.rivoisc_ver,
                args.ransac_seed,
            )
            for t in tasks
        )

        scene_dir = os.path.join(out_dataset_dir, sid)
        scene_summary = _save_scene_csvs(sid, list(outputs), scene_dir)
        scene_summaries.append(scene_summary)
        all_frame_rows.extend(outputs)

        # MoGe-only: dump per-scene plane_labels.h5 + plane_params.h5.
        if method in _MOGE_METHODS and args.save_moge_h5 and moge_h5_data:
            _save_moge_pred_h5(moge_h5_data, scene_dir)

        del tasks
        del outputs
        del moge_h5_data

    wall = time.perf_counter() - t0
    return all_frame_rows, scene_summaries, wall


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["gt", "moge", "zeroplane"],
                    choices=["gt", "moge", "moge_ep2", "zeroplane", "metric3d"])
    ap.add_argument("--datasets", nargs="+",
                    default=["scannetpp", "nyuv2", "sevenscenes"],
                    choices=["scannetpp", "nyuv2", "sevenscenes"])
    ap.add_argument("--moge_signals_root", default=DEFAULT_MOGE_ROOT)
    ap.add_argument("--moge_ep2_signals_root", default=DEFAULT_MOGE_EP2_ROOT,
                    help="Signals root for the moge_ep2 method "
                         "(epoch-2 4-dataset MoGe checkpoint).")
    ap.add_argument("--zeroplane_h5_root", default=DEFAULT_ZP_ROOT)
    ap.add_argument("--metric3d_h5_root", default=DEFAULT_METRIC3D_ROOT,
                    help=("Root for Metric3D inference dumps. Layout: "
                          "<root>/<dataset>/<split>/<scene>/inference.h5. "
                          "Currently only ScanNet++ has dumps."))
    ap.add_argument("--metric3d_depth_scale", type=float,
                    default=DEFAULT_METRIC3D_DEPTH_SCALE,
                    help=("Multiplicative scale applied to Metric3D depth to "
                          "recover true metric units (default: "
                          f"{DEFAULT_METRIC3D_DEPTH_SCALE:.6f} for ScanNet++; "
                          "pass 1.0 to disable)."))
    ap.add_argument("--eval_root", default=DEFAULT_EVAL_ROOT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--max_frames_per_scene", type=int, default=None)
    ap.add_argument("--n_jobs", type=int, default=min(16, os.cpu_count() or 16))

    ap.add_argument("--scene_ids", default=None,
                    help="Comma-separated list or path to a .txt with one scene per line.")
    ap.add_argument("--shard_id", type=int, default=None,
                    help="Shard index in [0, --num_shards). Used by parallel SLURM submission.")
    ap.add_argument("--num_shards", type=int, default=None,
                    help="Total number of shards. Each shard processes 1/num_shards of the scenes.")
    ap.add_argument("--skip_dataset_aggregates", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")

    # planamono RANSAC prec/rec block (added on top of the benchmark suite).
    ap.add_argument("--ransac_thresholds", nargs="+", type=float,
                    default=list(RANSAC_THRESHOLDS),
                    help="Distance thresholds (m) for prec/rec@<τ>cm "
                         "(default: 0.001 0.005 0.01).")
    ap.add_argument("--ransac_iterations", type=int, default=RANSAC_ITERATIONS)
    ap.add_argument("--inlier_ratio_gate", type=float, default=INLIER_RATIO_GATE)
    ap.add_argument("--ransac_seed", type=int, default=0,
                    help="Seed for the RANSAC prec/rec block (default 0 = "
                         "reproducible). Pass -1 to disable seeding.")

    # Whether to use K scaled to depth/label resolution for the RANSAC
    # prec/rec block. Default True (correct: 3D points in true meters).
    # --no-kscaled reproduces v0's convention (unscaled iPhone-native K), which
    # has the K-scale bug — useful only for direct comparison with v0 numbers.
    ap.add_argument("--kscaled", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Scale K to depth/label resolution for RANSAC "
                         "backprojection (default: True). --no-kscaled uses "
                         "unscaled GT K (matches v0's prec/rec, has the "
                         "pre-existing K-scale bug).")

    # Which RI/VI/SC implementation to use.
    # 'new' (default) = ZeroPlane's evaluateMasks (port of PlaneRCNN /
    #   PlaneRecTR). Reduction masked to GT-planar pixels.
    # 'old'           = planamono's compute_clustering_metrics (used by
    #   evaluate_all_baselines.py). sklearn/skimage-based, treats label 0 as
    #   just another cluster (non-planar pixels DO contribute).
    ap.add_argument("--rivoisc_ver", choices=["old", "new"], default="new",
                    help="RI/VI/SC implementation: 'new' (default, ZP's "
                         "evaluateMasks) or 'old' (planamono's "
                         "compute_clustering_metrics, used by "
                         "evaluate_all_baselines.py).")

    # Whether to dump MoGe's per-scene predicted labels + plane params as
    # H5 alongside the CSVs. Useful for downstream visualisation /
    # re-analysis without re-running inference. No-op for gt / zeroplane.
    ap.add_argument("--save_moge_h5", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Save plane_labels.h5 and plane_params.h5 per scene "
                         "for MoGe predictions (default: True). Use "
                         "--no-save_moge_h5 to disable.")

    args = ap.parse_args()
    args.ransac_thresholds = tuple(args.ransac_thresholds)
    # -1 => None => legacy non-deterministic RANSAC.
    if args.ransac_seed is not None and args.ransac_seed < 0:
        args.ransac_seed = None

    scene_filter: Optional[set] = None
    if args.scene_ids:
        if os.path.isfile(args.scene_ids):
            with open(args.scene_ids) as f:
                scene_filter = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        else:
            scene_filter = {s.strip() for s in args.scene_ids.split(",") if s.strip()}
    args._scene_filter = scene_filter

    if not args.aggregate_only:
        for ds in args.datasets:
            for m in args.methods:
                if m in _MOGE_METHODS:
                    root = _moge_root_for_method(m, args)
                    d = os.path.join(root, ds)
                    if not os.path.isdir(d):
                        ap.error(f"--{_MOGE_METHODS[m]}/{ds} not found: {d}")
                if m == "zeroplane":
                    d = os.path.join(args.zeroplane_h5_root, ds)
                    if not os.path.isdir(d):
                        ap.error(f"--zeroplane_h5_root/{ds} not found: {d}")
                if m == "metric3d":
                    if ds not in METRIC3D_DATASETS:
                        ap.error(
                            f"metric3d inference dumps only exist for "
                            f"{sorted(METRIC3D_DATASETS)}; got dataset={ds}. "
                            f"Restrict --datasets accordingly."
                        )
                    d = os.path.join(args.metric3d_h5_root, ds, args.split)
                    if not os.path.isdir(d):
                        ap.error(f"--metric3d_h5_root/{ds}/{args.split} not found: {d}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    torch.set_num_threads(1)

    out_root = os.path.join(args.eval_root, args.exp)
    os.makedirs(out_root, exist_ok=True)

    print("================================================================")
    print(f" evaluate_gt_moge_zeroplane_BENCHMARK — exp={args.exp}")
    print("================================================================")
    print(f"  methods:        {args.methods}")
    print(f"  datasets:       {args.datasets}")
    print(f"  moge root:      {args.moge_signals_root}")
    print(f"  moge_ep2 root:  {args.moge_ep2_signals_root}")
    print(f"  zeroplane root: {args.zeroplane_h5_root}")
    if "metric3d" in args.methods:
        print(f"  metric3d root:  {args.metric3d_h5_root}")
        print(f"  m3d depth scl:  {args.metric3d_depth_scale:.6f}")
    print(f"  output root:    {out_root}")
    print(f"  metric source:  metrics_benchmark.py "
          "(ZeroPlane + PlaneRCNN + PlaneRecTR)")
    print(f"  RANSAC K conv.: {'scaled (correct)' if args.kscaled else 'unscaled (v0 convention)'}")
    print(f"  RI/VI/SC ver.:  {args.rivoisc_ver} "
          f"({'ZP evaluateMasks' if args.rivoisc_ver == 'new' else 'planamono compute_clustering_metrics'})")
    print(f"  n_jobs:         {args.n_jobs}    max_scenes: {args.max_scenes}")
    if scene_filter is not None:
        print(f"  scene_ids:      {sorted(scene_filter)} ({len(scene_filter)} scenes)")
    if args.skip_dataset_aggregates:
        print(f"  mode:           WORKER (skip dataset aggregates)")
    if args.aggregate_only:
        print(f"  mode:           AGGREGATE-ONLY (skip evaluation)")
    print("  seg params:     "
          f"plan>{SEG_THRESHOLD_PLANARITY}, "
          f"normal<{SEG_NORMAL_THRESHOLD_DEG}°, "
          f"depth_rel<{SEG_DEPTH_THRESHOLD_REL}, "
          f"match≥{SEG_NEIGHBOR_MATCH_COUNT}")
    print("================================================================")

    if args.aggregate_only:
        print("\n[AGG] reading per-scene CSVs from disk ...")
        _aggregate_from_disk(args, out_root)
        print(f"\n[DONE] aggregates rewritten under {out_root}")
        return

    datasets: Dict[str, object] = {}
    scene_indices: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}
    for ds_name in args.datasets:
        print(f"\n[GT] building scene index for {ds_name} ...")
        ds_obj = _build_gt_dataset(ds_name, args)
        idx = _build_scene_index(ds_name, ds_obj)
        n_frames = sum(len(v) for v in idx.values())
        print(f"  {ds_name}: {n_frames} frames over {len(idx)} scenes")
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
