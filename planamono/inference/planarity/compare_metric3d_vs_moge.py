"""
Compare Metric3D vs MoGe metric depth against ScanNet++ GT depth (per-frame PNGs).

Each PNG is a 2-row figure matching ``compare_metric_depth.py``'s layout:

    Row 0:  RGB | Metric3D | MoGe | GT depth        (turbo, shared scale)
    Row 1:        Metric3D − GT | MoGe − GT         (RdBu_r, shared symmetric scale)
                                                    horizontal colorbars below each row

- Metric3D depth + GT depth come from the per-scene ``inference.h5`` produced
  by the Metric3D v2 inference job (``depth``, ``gt_depth``).
- MoGe depth (``depth_metric``) comes from the per-scene ``moge_signals.h5``
  produced by the 4-head MoGe pipeline.
- RGB is loaded from the original ScanNet++ iPhone tree
  ``<scannetpp>/data/<scene>/iphone/rgb/<frame>.jpg`` and resized to the H5
  resolution.

All depth panels (Metric3D, MoGe, GT) share one turbo colour scale (2nd–98th
percentile of pooled valid pixels). Both signed-diff panels share one
symmetric RdBu_r scale (±p98 across both methods) so a wider colour band on
one side means *that* method drifts more.

Title shows per-frame median ratio (pred/gt) and mean relative error for each
method — handy for spotting systematic global-scale offsets.

Output layout:

    <save_root>/
      ├── <scene_id>/<frame_id>.png
      └── manifest.json

Example
-------
python compare_metric3d_vs_moge.py \\
    --save_root /cluster/scratch/aoezkan/planeseg/audit/metric_depth_viz_with_ours \\
    --frames 5 --scenes 8 --seed 0
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.paths import scannetpp_path  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

METRIC3D_ROOT_DEFAULT = "/cluster/scratch/ayavuz/inference/metric3d_v2_epoch1/scannetpp/test"
MOGE_ROOT_DEFAULT = "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1/scannetpp"
RGB_ROOT_DEFAULT = os.path.join(scannetpp_path, "data")
SAVE_ROOT_DEFAULT = "/cluster/scratch/aoezkan/planeseg/audit/metric_depth_viz_with_ours"

# Metric3D was inferred on a downscaled ScanNet++ image; multiply by this scale
# to recover true metric depth at the H5 resolution.
#   SCALE = min(616/480, 1064/640) = 1.28333...  (ScanNet++ specific)
METRIC3D_DEPTH_SCALE_DEFAULT = min(616 / 480, 1064 / 640)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_fid(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _list_scenes_intersect(metric3d_root: str, moge_root: str) -> List[str]:
    m3d = {d for d in os.listdir(metric3d_root)
           if os.path.isfile(os.path.join(metric3d_root, d, "inference.h5"))}
    moge = {d for d in os.listdir(moge_root)
            if os.path.isfile(os.path.join(moge_root, d, "moge_signals.h5"))}
    return sorted(m3d & moge)


def _frame_index(h5_path: str) -> Dict[str, int]:
    with h5py.File(h5_path, "r") as f:
        return {_decode_fid(x): i for i, x in enumerate(f["frame_ids"][:])}


def load_rgb(rgb_root: str, scene_id: str, frame_id: str,
             target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    base = os.path.join(rgb_root, scene_id, "iphone", "rgb")
    for ext in (".jpg", ".JPG", ".png"):
        p = os.path.join(base, f"{frame_id}{ext}")
        if os.path.isfile(p):
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = target_hw
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            return img
    return None


def _resize_to(arr: np.ndarray, hw: Tuple[int, int],
               interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    H, W = hw
    if arr.shape[:2] == (H, W):
        return arr
    return cv2.resize(arr, (W, H), interpolation=interp)


def _depth_stats(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Return (median(pred/gt), mean(|pred-gt|/gt)) over jointly-valid pixels."""
    v = (gt > 0.05) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if not v.any():
        return float("nan"), float("nan")
    p, g = pred[v], gt[v]
    med_ratio = float(np.median(p / g))
    rel_err = float(np.mean(np.abs(p - g) / g))
    return med_ratio, rel_err


# ---------------------------------------------------------------------------
# Per-frame plot
# ---------------------------------------------------------------------------

def _save_panel(
    save_path: str,
    title: str,
    rgb: Optional[np.ndarray],
    gt_depth: np.ndarray,
    methods: List[Tuple[str, np.ndarray]],
):
    """Save a 2-row comparison figure (matches ``compare_metric_depth.py``).

    Layout: ``RGB | method_1 | method_2 | ... | GT`` on row 0, with
    ``(method_k − GT)`` panels on row 1. All depth panels share one turbo
    scale; all diff panels share one symmetric RdBu_r scale. Two horizontal
    colorbars (depth, Δdepth) span the depth/diff columns under each row.
    """
    n_methods = len(methods)
    n_cols = 2 + n_methods  # RGB | methods... | GT

    # ── Joint validity mask ───────────────────────────────────────────────
    valid = (gt_depth > 0.05) & np.isfinite(gt_depth)
    for _, arr in methods:
        valid &= np.isfinite(arr) & (arr > 0)

    # ── Shared depth scale across methods + GT ────────────────────────────
    depths_for_range = [arr for _, arr in methods] + [gt_depth]
    if valid.sum() < 100:
        vmin_d, vmax_d = 0.0, 5.0
    else:
        all_d = np.concatenate([d[valid] for d in depths_for_range])
        vmin_d, vmax_d = np.percentile(all_d, [2, 98])

    # ── Shared symmetric diff scale across all methods ────────────────────
    diffs = [(label, arr - gt_depth) for label, arr in methods]
    if valid.sum() < 100:
        vlim_diff = 1.0
    else:
        vlim_diff = float(np.percentile(
            np.abs(np.concatenate([d[valid] for _, d in diffs])), 98))
        vlim_diff = max(vlim_diff, 1e-3)

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(4.4 * n_cols, 9))
    gs = fig.add_gridspec(
        nrows=2, ncols=n_cols,
        height_ratios=[1, 1],
        hspace=0.35, wspace=0.05,
        left=0.03, right=0.99, top=0.93, bottom=0.10,
    )
    cmap_d = plt.colormaps["turbo"]
    cmap_diff = plt.colormaps["RdBu_r"]

    def _show(ax, arr, vmin, vmax, cmap, panel_title):
        im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        return im

    # Row 0: RGB | method_1 | method_2 | ... | GT
    ax_rgb = fig.add_subplot(gs[0, 0])
    if rgb is None:
        H, W = gt_depth.shape
        ax_rgb.imshow(np.zeros((H, W, 3), dtype=np.uint8))
        ax_rgb.set_title("RGB (missing)", fontsize=11)
    else:
        ax_rgb.imshow(rgb)
        ax_rgb.set_title("RGB", fontsize=11)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    im_d = None
    for k, (label, arr) in enumerate(methods):
        ax = fig.add_subplot(gs[0, 1 + k])
        im_d = _show(ax, np.where(valid, arr, np.nan),
                     vmin_d, vmax_d, cmap_d, label)

    gt_col = 1 + n_methods
    ax_gt = fig.add_subplot(gs[0, gt_col])
    gt_v = np.where(valid, gt_depth, np.nan)
    if im_d is None:
        im_d = _show(ax_gt, gt_v, vmin_d, vmax_d, cmap_d, "GT depth")
    else:
        _show(ax_gt, gt_v, vmin_d, vmax_d, cmap_d, "GT depth")

    # Row 1: blank | method_1 − GT | method_2 − GT | ... | blank
    im_diff = None
    for k, (label, diff) in enumerate(diffs):
        ax = fig.add_subplot(gs[1, 1 + k])
        im_diff = _show(ax, np.where(valid, diff, np.nan),
                        -vlim_diff, vlim_diff, cmap_diff, f"{label} − GT")

    # ── Colorbars (span method+GT cols, normalized to figure coords) ──────
    left_frac = 1.0 / n_cols + 0.02
    right_frac = 0.99
    width = right_frac - left_frac
    cax_d = fig.add_axes([left_frac, 0.535, width, 0.018])
    cax_diff = fig.add_axes([left_frac, 0.06, width, 0.018])

    if im_d is not None:
        cbar_d = fig.colorbar(im_d, cax=cax_d, orientation="horizontal")
        cbar_d.set_label("depth [m]", fontsize=10)
    if im_diff is not None:
        cbar_diff = fig.colorbar(im_diff, cax=cax_diff, orientation="horizontal")
        cbar_diff.set_label("Δdepth [m]", fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-frame pipeline
# ---------------------------------------------------------------------------

def process_one(
    scene_id: str,
    frame_id: str,
    m3d_h5: str,
    m3d_idx: int,
    moge_h5: str,
    moge_idx: int,
    rgb_root: str,
    save_root: str,
    m3d_scale: float = 1.0,
) -> Optional[Tuple[str, dict]]:
    with h5py.File(m3d_h5, "r") as f:
        m3d_depth = f["depth"][m3d_idx].astype(np.float32) * m3d_scale
        gt_depth = f["gt_depth"][m3d_idx].astype(np.float32)
    with h5py.File(moge_h5, "r") as f:
        moge_depth = f["depth_metric"][moge_idx].astype(np.float32)

    # Match resolution if MoGe and Metric3D differ.
    target_hw = m3d_depth.shape
    if moge_depth.shape != target_hw:
        moge_depth = _resize_to(moge_depth, target_hw, interp=cv2.INTER_LINEAR)

    H, W = target_hw
    rgb = load_rgb(rgb_root, scene_id, frame_id, (H, W))

    m3d_med, m3d_rel = _depth_stats(m3d_depth, gt_depth)
    moge_med, moge_rel = _depth_stats(moge_depth, gt_depth)

    out_dir = os.path.join(save_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{frame_id}.png")

    title = (
        f"scannetpp / {scene_id} / {frame_id}      "
        f"Metric3D: med(pred/gt)={m3d_med:.3f}  rel={m3d_rel*100:.1f}%      "
        f"MoGe: med(pred/gt)={moge_med:.3f}  rel={moge_rel*100:.1f}%"
    )
    _save_panel(
        out_path, title,
        rgb=rgb, gt_depth=gt_depth,
        methods=[("Metric3D", m3d_depth), ("MoGe", moge_depth)],
    )
    stats = {
        "metric3d_median_ratio": m3d_med, "metric3d_rel_err": m3d_rel,
        "moge_median_ratio":     moge_med, "moge_rel_err":     moge_rel,
    }
    return out_path, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--metric3d_root", type=str, default=METRIC3D_ROOT_DEFAULT,
                    help="Root with <scene>/inference.h5 (Metric3D outputs).")
    ap.add_argument("--moge_root", type=str, default=MOGE_ROOT_DEFAULT,
                    help="Root with <scene>/moge_signals.h5 (MoGe 4-head signals).")
    ap.add_argument("--rgb_root", type=str, default=RGB_ROOT_DEFAULT,
                    help="ScanNet++ data root (must contain <scene>/iphone/rgb/).")
    ap.add_argument("--save_root", type=str, default=SAVE_ROOT_DEFAULT,
                    help="Root directory for output PNGs.")
    ap.add_argument("--m3d_scale", type=float, default=METRIC3D_DEPTH_SCALE_DEFAULT,
                    help=("Multiplicative scale applied to Metric3D depth to "
                          "recover true metric units (default: "
                          f"{METRIC3D_DEPTH_SCALE_DEFAULT:.6f} for ScanNet++; "
                          "pass 1.0 to disable)."))
    ap.add_argument("--frames", type=int, default=5,
                    help="Frames per scene (capped at the scene's frame count).")
    ap.add_argument("--scenes", type=int, default=None,
                    help="Number of scenes to sample. Default: all available.")
    ap.add_argument("--scene", type=str, action="append", default=None,
                    help="Restrict to specific scene(s). Repeatable.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scenes_pool = _list_scenes_intersect(args.metric3d_root, args.moge_root)
    if not scenes_pool:
        raise SystemExit(
            f"No scenes have BOTH inference.h5 ({args.metric3d_root}) and "
            f"moge_signals.h5 ({args.moge_root})."
        )

    if args.scene:
        missing = sorted(set(args.scene) - set(scenes_pool))
        if missing:
            raise SystemExit(f"Scene(s) missing or without both H5 files: {missing}")
        scenes_pool = sorted(args.scene)

    n_scenes_pick = (len(scenes_pool) if args.scenes is None
                     else min(args.scenes, len(scenes_pool)))
    scene_picks_idx = rng.choice(len(scenes_pool), size=n_scenes_pick, replace=False)
    picked_scenes = sorted(scenes_pool[i] for i in scene_picks_idx)

    manifest = {
        "seed": args.seed,
        "frames_per_scene": args.frames,
        "scenes_per_dataset": args.scenes,
        "metric3d_root": args.metric3d_root,
        "moge_root": args.moge_root,
        "rgb_root": args.rgb_root,
        "m3d_scale": args.m3d_scale,
        "saved": [],
    }

    print(f"Picking {args.frames} frame(s) from each of {len(picked_scenes)} "
          f"scene(s) (out of {len(scenes_pool)} with both H5 files).")

    saved_count = 0
    for sid in tqdm(picked_scenes, desc="scenes"):
        m3d_h5 = os.path.join(args.metric3d_root, sid, "inference.h5")
        moge_h5 = os.path.join(args.moge_root, sid, "moge_signals.h5")
        try:
            m3d_map = _frame_index(m3d_h5)
            moge_map = _frame_index(moge_h5)
        except Exception as e:
            print(f"  [skip] {sid}: cannot read frame_ids ({e})")
            continue

        common = sorted(set(m3d_map) & set(moge_map))
        if not common:
            print(f"  [skip] {sid}: no overlapping frames")
            continue

        k = min(args.frames, len(common))
        sel = rng.choice(len(common), size=k, replace=False).tolist()
        picks = sorted(common[i] for i in sel)

        for fid in tqdm(picks, desc=f"  {sid}", leave=False):
            try:
                out = process_one(
                    scene_id=sid, frame_id=fid,
                    m3d_h5=m3d_h5, m3d_idx=m3d_map[fid],
                    moge_h5=moge_h5, moge_idx=moge_map[fid],
                    rgb_root=args.rgb_root, save_root=args.save_root,
                    m3d_scale=args.m3d_scale,
                )
            except Exception as e:
                print(f"  [fail] {sid}/{fid}: {type(e).__name__}: {e}")
                continue
            if out:
                path, stats = out
                manifest["saved"].append({
                    "scene_id": sid, "frame_id": fid,
                    "png": os.path.relpath(path, args.save_root),
                    **stats,
                })
                saved_count += 1

    manifest_path = os.path.join(args.save_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {saved_count} PNGs under {args.save_root}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
