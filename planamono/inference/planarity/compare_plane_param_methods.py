"""
Compare all 6 plane-parameter estimation methods on a moge_signals.h5 dump.

For a single scene this script:
  1. Reads predicted (depth_metric, normal, planarity, mask, K) per frame from
     moge_signals.h5.
  2. Builds plane labels via `compute_vectorized_planar_segments_v5_relative`
     with the user-spec config:
        threshold_planarity = 0.3
        normal_threshold     = 5°
        depth_threshold_rel  = 0.025
        neighbor_match       = 8
  3. Fits plane params with all 6 methods from compute_plane_params.py.
  4. Renders each method's plane params back to per-pixel (depth, normal)
     maps, then evaluates against ScanNet++ GT via evaluate_plane_predictions.
  5. Logs per-step runtime (min / sec / ms / total_ms).
  6. Writes:
       results.csv         per-(frame, method) metrics + runtime
       plane_labels.h5     stacked uint16 plane labels for the evaluated frames
       plane_params.h5     per (method, frame_id): plane_ids + (a,b,c,d) arrays
       debug/<fid>.png     7-panel comparison (only with --debug)

Example
-------
python compare_plane_param_methods.py \\
    --scene_id c50d2d1d42 \\
    --frames 5 \\
    --debug
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import h5py
import numpy as np
import pandas as pd
import torch

# Make `planamono` importable when running as a script
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.shared.segmentation.plan2seg import (
    compute_vectorized_planar_segments_v5_relative,
)
from planamono.shared.segmentation.compute_plane_params import compute_plane_params
from planamono.shared.plane_fitting.metrics_planes import (
    compute_segmentation_metrics,
    compute_gt_normals_from_depth_labels,
    match_planes_by_overlap,
    plane_recall_at_depth,
    plane_recall_at_normal,
    per_plane_error_stats,
)
from planamono.paths import scannetpp_path, scannetpp_rend_plane_path


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

# Experiment-specific defaults (this checkpoint / output run); override via CLI.
DEFAULT_SIGNALS_ROOT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1"
DEFAULT_OUTPUT_ROOT  = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1_methods"
# Dataset paths come from paths.py so they stay consistent with the rest of the repo.
DEFAULT_GT_ROOT      = scannetpp_rend_plane_path
DEFAULT_RGB_ROOT     = os.path.join(scannetpp_path, "data")

THRESHOLD_PLANARITY      = 0.3
NORMAL_THRESHOLD_DEG     = 5.0
DEPTH_THRESHOLD_REL      = 0.025
NEIGHBOR_MATCH_COUNT     = 8

DEPTH_THRESHOLDS_M       = (0.05, 0.1, 0.6)
NORMAL_THRESHOLDS_DEG    = (5.0, 10.0, 30.0)

METHOD_NAMES = [
    "normal_average",
    "least_squares",
    "svd",
    "ransac",
    "ransac_normal",
    "ransac_mestimator",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _runtime_breakdown(seconds: float) -> Dict[str, float]:
    total_ms = seconds * 1000.0
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    sec_int = int(rem)
    ms = (rem - sec_int) * 1000.0
    return {"min": int(minutes), "sec": int(sec_int),
            "ms": float(ms), "total_ms": float(total_ms)}


def _method_kwargs(method: str, ransac_trials: int) -> Dict:
    if method in ("ransac", "ransac_normal", "ransac_mestimator"):
        if method == "ransac":
            return {"residual_threshold": 0.05, "max_trials": ransac_trials}
        if method == "ransac_normal":
            return {"residual_threshold": 0.05, "normal_threshold": 0.1,
                    "max_trials": ransac_trials}
        return {"soft_threshold": 0.05, "max_trials": ransac_trials}
    return {}


def _render_plane_params_to_maps(
    plane_params: Dict[int, np.ndarray],
    labels: np.ndarray,
    K: np.ndarray,
    H: int, W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each pid in plane_params, write (a,b,c) into normals[mask] and
    z = -d / (a*(u-cx)/fx + b*(v-cy)/fy + c) into depth[mask]."""
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
        # only keep positive depths inside the mask
        ok = mask & (z > 0) & np.isfinite(z)
        depth_out[ok] = z[ok].astype(np.float32)
        normal_out[mask] = (a, b, c)

    return depth_out, normal_out


def _load_predictions(signals_h5: str, idx: int):
    with h5py.File(signals_h5, "r") as f:
        depth = f["depth_metric"][idx].astype(np.float32)
        normal = f["normal"][idx].astype(np.float32)
        planarity = f["planarity"][idx].astype(np.float32)
        mask = f["mask"][idx].astype(np.uint8)
        K = f["intrinsics"][idx].astype(np.float64)
        fid = f["frame_ids"][idx]
        fid = fid.decode() if isinstance(fid, (bytes, bytearray)) else str(fid)

    nrm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, np.clip(nrm, 1e-6, None), where=nrm > 1e-6)
    return depth, normal, planarity, mask, K, fid


def _gt_index(rendered_h5: str) -> Dict[str, int]:
    with h5py.File(rendered_h5, "r") as f:
        ids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
               for x in f["frame_ids"][:]]
    return {fid: i for i, fid in enumerate(ids)}


def _load_gt_frame(rendered_h5: str, rendered_depth_h5: str, gt_idx: int):
    with h5py.File(rendered_h5, "r") as f:
        labels_gt = f["planes"][gt_idx].astype(np.int32)
    with h5py.File(rendered_depth_h5, "r") as f:
        depth_gt = f["depth"][gt_idx].astype(np.float32) / 1000.0
    return depth_gt, labels_gt


def _segment_one_frame(planarity, normal, depth, device):
    mask = planarity > THRESHOLD_PLANARITY
    t0 = time.perf_counter()
    labels, num_components = compute_vectorized_planar_segments_v5_relative(
        planarity_mask=mask,
        normal=normal,
        depth=depth,
        normal_threshold_rad=float(np.deg2rad(NORMAL_THRESHOLD_DEG)),
        depth_threshold=DEPTH_THRESHOLD_REL,
        neighbor_match_count_thresh=NEIGHBOR_MATCH_COUNT,
        device=device,
    )
    elapsed = time.perf_counter() - t0
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    return labels.astype(np.int32), int(num_components), elapsed


def _save_debug_png(
    out_path: str, fid: str, rgb, depth_pred, normal_pred, planarity,
    labels_pred, depth_gt, labels_gt,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(0)
    palette = rng.uniform(0.2, 1.0, size=(2048, 3))
    palette[0] = [0.05, 0.05, 0.05]
    cmap = ListedColormap(palette)

    H, W = depth_pred.shape

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    axes = axes.flatten()

    def _show(ax, img, title, **kw):
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    _show(axes[0], rgb, "RGB")
    _show(axes[1], depth_pred, "depth_pred (m)", cmap="viridis",
          vmin=np.nanmin(depth_pred), vmax=np.nanpercentile(depth_pred, 99))
    n_vis = (np.clip(normal_pred, -1, 1) + 1) / 2
    _show(axes[2], n_vis, "normal_pred")
    _show(axes[3], planarity, "planarity", cmap="magma", vmin=0, vmax=1)

    n_classes = max(int(labels_pred.max()), int(labels_gt.max()), 1) + 1
    _show(axes[4], labels_pred % len(palette),
          f"labels_pred ({n_classes - 1} ids)", cmap=cmap, vmin=0, vmax=len(palette) - 1)

    valid_gt = depth_gt > 0
    if valid_gt.any():
        vmax = float(np.nanpercentile(depth_gt[valid_gt], 99))
        _show(axes[5], depth_gt, "depth_gt (m)", cmap="viridis", vmin=0, vmax=vmax)
    else:
        _show(axes[5], depth_gt, "depth_gt (empty)", cmap="viridis")
    _show(axes[6], labels_gt % len(palette),
          f"labels_gt ({int(labels_gt.max())} ids)", cmap=cmap, vmin=0, vmax=len(palette) - 1)

    axes[7].axis("off")
    axes[7].text(0.02, 0.95, f"frame_id: {fid}",
                 transform=axes[7].transAxes, fontsize=10, va="top")

    fig.suptitle(f"compare_plane_param_methods — {fid}", fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene_id", required=True)
    p.add_argument("--signals_root", default=DEFAULT_SIGNALS_ROOT)
    p.add_argument("--gt_root", default=DEFAULT_GT_ROOT)
    p.add_argument("--rgb_root", default=DEFAULT_RGB_ROOT)
    p.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--frames", type=int, default=None,
                   help="Number of evenly-spaced frames to sample (default: all).")
    p.add_argument("--ransac_trials", type=int, default=200,
                   help="RANSAC iterations per plane for methods 4-6. Default 200.")
    p.add_argument("--methods", nargs="+", default=METHOD_NAMES,
                   help=f"Which methods to run. Default: all 6 of {METHOD_NAMES}")
    p.add_argument("--depth_thresholds", nargs="+", type=float,
                   default=list(DEPTH_THRESHOLDS_M),
                   metavar="M",
                   help=f"Depth thresholds (meters) for plane_recall_at_depth. "
                        f"Default: {list(DEPTH_THRESHOLDS_M)}")
    p.add_argument("--normal_thresholds", nargs="+", type=float,
                   default=list(NORMAL_THRESHOLDS_DEG),
                   metavar="DEG",
                   help=f"Normal thresholds (degrees) for plane_recall_at_normal. "
                        f"Default: {list(NORMAL_THRESHOLDS_DEG)}")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug", action="store_true",
                   help="Save per-frame 7-panel comparison PNGs.")
    args = p.parse_args()

    signals_h5 = os.path.join(args.signals_root, args.scene_id, "moge_signals.h5")
    rendered_h5 = os.path.join(args.gt_root, args.scene_id, "rendered.h5")
    rendered_depth_h5 = os.path.join(args.gt_root, args.scene_id, "rendered_depth.h5")
    rgb_dir = os.path.join(args.rgb_root, args.scene_id, "iphone", "rgb")

    for path in (signals_h5, rendered_h5, rendered_depth_h5, rgb_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    out_dir = os.path.join(args.output_root, args.scene_id)
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "debug")
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)

    # Frame indexing: align prediction frames to GT by frame_id
    with h5py.File(signals_h5, "r") as f:
        all_pred_ids = [x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                        for x in f["frame_ids"][:]]
        H, W = f["depth_metric"].shape[1], f["depth_metric"].shape[2]
    gt_idx_lookup = _gt_index(rendered_h5)

    pred_indices = list(range(len(all_pred_ids)))
    if args.frames is not None and args.frames < len(pred_indices):
        pred_indices = list(np.linspace(0, len(all_pred_ids) - 1, args.frames).astype(int))

    print(f"[INFO] scene={args.scene_id}  device={args.device}")
    print(f"[INFO] {len(pred_indices)}/{len(all_pred_ids)} frames; methods={args.methods}")
    print(f"[INFO] seg cfg: planarity>{THRESHOLD_PLANARITY}, "
          f"normal<{NORMAL_THRESHOLD_DEG}deg, depth_rel<{DEPTH_THRESHOLD_REL}, "
          f"match>={NEIGHBOR_MATCH_COUNT}")

    rows: List[Dict] = []
    labels_stack: List[np.ndarray] = []
    fids_stack: List[str] = []

    pp_h5_path = os.path.join(out_dir, "plane_params.h5")
    pp_h5 = h5py.File(pp_h5_path, "w")
    for method in args.methods:
        pp_h5.create_group(method)

    for idx in pred_indices:
        depth, normal, planarity, mask, K, fid = _load_predictions(signals_h5, idx)

        if fid not in gt_idx_lookup:
            print(f"[SKIP] {fid}: no matching GT frame_id")
            continue
        gt_idx = gt_idx_lookup[fid]
        depth_gt, labels_gt = _load_gt_frame(rendered_h5, rendered_depth_h5, gt_idx)

        if labels_gt.shape != (H, W):
            labels_gt = cv2.resize(labels_gt, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)

        labels_pred, n_seg, seg_secs = _segment_one_frame(planarity, normal, depth, args.device)
        seg_rt = _runtime_breakdown(seg_secs)

        labels_stack.append(labels_pred.astype(np.uint16))
        fids_stack.append(fid)

        # GT normals computed once per frame (cheap; reused across all methods)
        normals_gt_dense = compute_gt_normals_from_depth_labels(
            depth_gt, labels_gt, K, ignore_labels=(0,))
        matches = match_planes_by_overlap(labels_gt, labels_pred,
                                          ignore_labels_gt=(0,), ignore_labels_pred=(0,))

        seg_metrics = compute_segmentation_metrics(labels_gt, labels_pred)

        for method in args.methods:
            mk = _method_kwargs(method, args.ransac_trials)
            t0 = time.perf_counter()
            params = compute_plane_params(
                depth, normal, labels_pred, method=method, K=K, **mk)
            method_secs = time.perf_counter() - t0
            method_rt = _runtime_breakdown(method_secs)

            depth_method, normal_method = _render_plane_params_to_maps(
                params, labels_pred, K, H, W)

            rec_d = {f"plane_recall_d_{int(round(t * 1000))}mm":
                     plane_recall_at_depth(depth_method, depth_gt, labels_pred, labels_gt,
                                           t, matches=matches)["recall"]
                     for t in args.depth_thresholds}
            rec_n = {f"plane_recall_n_{int(round(t))}deg":
                     plane_recall_at_normal(normal_method, normals_gt_dense,
                                            labels_pred, labels_gt, t, matches=matches)["recall"]
                     for t in args.normal_thresholds}
            err_stats = per_plane_error_stats(
                depth_method, depth_gt, normal_method, normals_gt_dense,
                labels_pred, labels_gt, matches=matches)

            row = {
                "scene_id": args.scene_id, "frame_id": fid, "method": method,
                "n_segments": n_seg,
                "seg_min": seg_rt["min"], "seg_sec": seg_rt["sec"],
                "seg_ms": seg_rt["ms"], "seg_total_ms": seg_rt["total_ms"],
                "method_min": method_rt["min"], "method_sec": method_rt["sec"],
                "method_ms": method_rt["ms"], "method_total_ms": method_rt["total_ms"],
                **seg_metrics,
                **rec_d, **rec_n, **err_stats,
                "n_gt_planes": int(len(set(np.unique(labels_gt).tolist()) - {0})),
                "n_pred_planes": int(len(set(np.unique(labels_pred).tolist()) - {0})),
            }
            rows.append(row)

            grp = pp_h5[method].create_group(fid)
            pids = np.array(sorted(params.keys()), dtype=np.int32)
            par = np.stack([params[p] for p in pids], axis=0) if len(pids) else np.zeros((0, 4))
            grp.create_dataset("plane_ids", data=pids)
            grp.create_dataset("params", data=par.astype(np.float32))

            d_key = next(iter(rec_d))
            n_key = next(iter(rec_n))
            print(f"[{fid}] seg={seg_rt['total_ms']:.1f}ms  "
                  f"{method:<18s} {method_rt['total_ms']:.1f}ms  "
                  f"R@{d_key.split('_')[-1]}={rec_d[d_key]:.2f} "
                  f"R@{n_key.split('_')[-1]}={rec_n[n_key]:.2f}")

        if args.debug:
            rgb_path = os.path.join(rgb_dir, f"{fid}.jpg")
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
            _save_debug_png(
                os.path.join(debug_dir, f"{fid}.png"),
                fid, rgb, depth, normal, planarity, labels_pred, depth_gt, labels_gt,
            )

    pp_h5.close()

    # Save plane_labels.h5
    if labels_stack:
        with h5py.File(os.path.join(out_dir, "plane_labels.h5"), "w") as f:
            f.create_dataset("plane_labels",
                             data=np.stack(labels_stack, axis=0),
                             compression="gzip", compression_opts=4)
            f.create_dataset("frame_ids",
                             data=np.array(fids_stack, dtype="S"))

    # Save per-frame CSV + per-method aggregate (mean / std across frames).
    summary_path = None
    n_summary_rows = 0
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

        # Aggregate every numeric column into mean/std per method.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        agg = df.groupby("method")[numeric_cols].agg(["mean", "std"])
        agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
        agg["n_frames"] = df.groupby("method").size()
        # Preserve --methods order (groupby returns alphabetical otherwise).
        method_order = [m for m in args.methods if m in agg.index]
        agg = agg.reindex(method_order).reset_index()
        summary_path = os.path.join(out_dir, "summary.csv")
        agg.to_csv(summary_path, index=False)
        n_summary_rows = len(agg)

        # Compact console summary: all depth + normal recalls and method runtime.
        d_keys = [c for c in df.columns if c.startswith("plane_recall_d_")]
        n_keys = [c for c in df.columns if c.startswith("plane_recall_n_")]
        n_unique_frames = int(df["frame_id"].nunique())
        print(f"\n========== SUMMARY across {n_unique_frames} frames ==========")
        for _, r in agg.iterrows():
            parts = [f"{r['method']:<18s}"]
            for k in d_keys:
                parts.append(
                    f"R@{k.split('_')[-1]}="
                    f"{r[f'{k}_mean']:.2f}±{r[f'{k}_std']:.2f}"
                )
            for k in n_keys:
                parts.append(
                    f"R@{k.split('_')[-1]}="
                    f"{r[f'{k}_mean']:.2f}±{r[f'{k}_std']:.2f}"
                )
            parts.append(
                f"rt={r['method_total_ms_mean']:.0f}±"
                f"{r['method_total_ms_std']:.0f}ms"
            )
            print("  " + "  ".join(parts))

    print(f"\n[DONE] outputs under {out_dir}")
    print(f"  results.csv         {len(rows)} rows ({len(args.methods)} methods × "
          f"{len(rows) // max(len(args.methods), 1)} frames)")
    if summary_path is not None:
        print(f"  summary.csv         {n_summary_rows} methods "
              f"(mean ± std across frames)")
    print(f"  plane_labels.h5     {len(labels_stack)} frames")
    print(f"  plane_params.h5     {len(args.methods)} methods × {len(labels_stack)} frames")
    if args.debug:
        print(f"  debug/*.png         {len(labels_stack)} files")


if __name__ == "__main__":
    main()
