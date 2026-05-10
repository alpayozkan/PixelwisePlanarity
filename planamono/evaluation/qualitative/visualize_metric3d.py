"""
Visualize Metric3D v2 inference output on ScanNet++ test split as per-frame PNGs.

Each PNG shows one frame with 7 panels (single row):

    RGB | Metric3D depth | Metric3D normal | GT depth |
        Metric3D - GT depth | Metric3D planarity | Plane labels (ours seg)

- Metric3D signals are read from the per-scene ``inference.h5`` produced by the
  Metric3D v2 inference job (planarity / normals / depth / gt_depth / gt_planes).
- "Plane labels" is computed on the fly from the dump's planarity + normal +
  depth via ``compute_vectorized_planar_segments_v5_relative`` with the
  ECCV ``config3_default`` parameters (plan=0.3, norm=5°, depth_rel=0.025,
  match=8) — same defaults as ``compare_signals_vs_zeroplane.py``.
- "RGB" is loaded from the original ScanNet++ iPhone tree
  ``<scannetpp>/data/<scene>/iphone/rgb/<frame>.jpg`` and resized to the H5
  resolution.

Output layout (one PNG per frame, plus a JSON manifest):

    <save_root>/
      ├── <scene_id>/<frame_id>.png
      └── manifest.json

Example
-------
python visualize_metric3d.py \\
    --save_root /cluster/scratch/aoezkan/planeseg/audit/metric3d_v2_epoch1 \\
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
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.paths import scannetpp_path  # noqa: E402
from planamono.shared.segmentation import (  # noqa: E402
    compute_vectorized_planar_segments_v5_relative,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

INFERENCE_ROOT_DEFAULT = "/cluster/scratch/ayavuz/inference/metric3d_v2_epoch1/scannetpp/test"
RGB_ROOT_DEFAULT = os.path.join(scannetpp_path, "data")
SAVE_ROOT_DEFAULT = "/cluster/scratch/aoezkan/planeseg/audit/metric3d_v2_epoch1"

# Metric3D was inferred on a downscaled ScanNet++ image; multiply by this scale
# to recover true metric depth at the H5 resolution.
#   SCALE = min(616/480, 1064/640) = 1.28333...  (ScanNet++ specific)
# Equivalent to the de-canonical step `pred_depth *= focal/1000` in
# planamono/moge/baselines/metric3d_v2.py that the upstream H5 dump skipped.
METRIC3D_DEPTH_SCALE_DEFAULT = min(616 / 480, 1064 / 640)

# v5_relative config3_default
THRESHOLD_PLANARITY = 0.3
NORMAL_THRESHOLD_DEG = 5.0
DEPTH_THRESHOLD_REL = 0.025
NEIGHBOR_MATCH_COUNT = 8


# ---------------------------------------------------------------------------
# H5 helpers
# ---------------------------------------------------------------------------

def _decode_fid(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _list_scenes(inference_root: str) -> List[str]:
    return sorted(
        d for d in os.listdir(inference_root)
        if os.path.isdir(os.path.join(inference_root, d))
        and os.path.isfile(os.path.join(inference_root, d, "inference.h5"))
    )


def _list_frame_ids(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as f:
        return [_decode_fid(x) for x in f["frame_ids"][:]]


# ---------------------------------------------------------------------------
# Per-frame loading
# ---------------------------------------------------------------------------

def load_frame(h5_path: str, idx: int, depth_scale: float = 1.0) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        return {
            "depth":     f["depth"][idx].astype(np.float32) * depth_scale,
            "normal":    f["normals"][idx].astype(np.float32),
            "planarity": f["planarity"][idx].astype(np.float32),
            "gt_depth":  f["gt_depth"][idx].astype(np.float32),
            "gt_planes": f["gt_planes"][idx].astype(np.int32),
        }


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


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _label_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    pal = rng.integers(0, 255, size=(max(n, 2), 3), dtype=np.uint8)
    return pal


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    n = int(labels.max()) + 1 if labels.size else 1
    pal = _label_palette(max(n, 2))
    pal[0] = 0  # 0 = non-planar → black
    return pal[np.clip(labels.astype(np.int64), 0, n - 1)]


def _normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    return ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)


def _compute_ours_seg(planarity: np.ndarray, normal: np.ndarray,
                      depth: np.ndarray, device: str = "cuda") -> np.ndarray:
    mask = (planarity > THRESHOLD_PLANARITY).astype(np.int32)
    labels, _ = compute_vectorized_planar_segments_v5_relative(
        mask, normal, depth,
        np.deg2rad(NORMAL_THRESHOLD_DEG),
        DEPTH_THRESHOLD_REL,
        NEIGHBOR_MATCH_COUNT,
        device=device,
    )
    return labels.astype(np.int32)


def _save_panel(
    save_path: str,
    title: str,
    rgb: Optional[np.ndarray],
    moge_depth: np.ndarray,
    moge_normal: np.ndarray,
    gt_depth: np.ndarray,
    depth_diff: np.ndarray,
    moge_planarity: np.ndarray,
    ours_seg: np.ndarray,
):
    valid = (gt_depth > 0.05) & np.isfinite(gt_depth)

    # Shared depth color limits across GT / Metric3D depth
    finite_gt = gt_depth[valid]
    finite_pred = moge_depth[valid & np.isfinite(moge_depth)]
    if finite_gt.size > 100 and finite_pred.size > 100:
        all_d = np.concatenate([finite_gt, finite_pred])
        vmin_d, vmax_d = np.percentile(all_d, [2, 98])
    else:
        vmin_d, vmax_d = 0.0, 5.0

    # Symmetric color limit for the diff panel (mask to valid GT)
    valid_diff = valid & np.isfinite(moge_depth)
    if valid_diff.any():
        vabs = float(np.percentile(np.abs(depth_diff[valid_diff]), 95))
        vabs = max(vabs, 1e-3)
    else:
        vabs = 0.5

    cols = []
    if rgb is None:
        H, W = moge_depth.shape
        cols.append(("RGB (missing)", np.zeros((H, W, 3), dtype=np.uint8), None, None))
    else:
        cols.append(("RGB", rgb, None, None))
    cols.append(("Metric3D depth",  moge_depth, "turbo", "depth"))
    cols.append(("Metric3D normal", _normal_to_rgb(moge_normal), None, None))
    cols.append(("GT depth",        gt_depth,   "turbo", "depth"))
    cols.append((f"Metric3D − GT (±{vabs:.3f} m)", depth_diff, "RdBu_r", "diff"))
    cols.append(("Metric3D planarity", moge_planarity, "magma", "prob"))
    cols.append(("Ours seg (v5_relative)", _colorize_labels(ours_seg), None, None))

    n = len(cols)
    fig = plt.figure(figsize=(3.6 * n, 4.0))
    gs = fig.add_gridspec(nrows=1, ncols=n, wspace=0.04, left=0.01, right=0.99,
                          top=0.86, bottom=0.04)
    for i, (name, img, cmap, kind) in enumerate(cols):
        ax = fig.add_subplot(gs[0, i])
        if cmap is None:
            ax.imshow(img)
        elif kind == "depth":
            arr = np.where(valid, img, np.nan)
            ax.imshow(arr, cmap=cmap, vmin=vmin_d, vmax=vmax_d)
        elif kind == "diff":
            arr = np.where(valid_diff, img, np.nan)
            ax.imshow(arr, cmap=cmap, vmin=-vabs, vmax=vabs)
        elif kind == "prob":
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-frame pipeline
# ---------------------------------------------------------------------------

def process_one(
    scene_id: str,
    frame_id: str,
    h5_path: str,
    h5_idx: int,
    rgb_root: str,
    save_root: str,
    device: str,
    depth_scale: float = 1.0,
) -> Optional[str]:
    f = load_frame(h5_path, h5_idx, depth_scale=depth_scale)
    H, W = f["depth"].shape
    rgb = load_rgb(rgb_root, scene_id, frame_id, (H, W))

    valid = (f["gt_depth"] > 0.05) & np.isfinite(f["gt_depth"]) & np.isfinite(f["depth"])
    diff = np.where(valid, f["depth"] - f["gt_depth"], 0.0).astype(np.float32)

    ours_seg = _compute_ours_seg(f["planarity"], f["normal"], f["depth"], device=device)

    out_dir = os.path.join(save_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{frame_id}.png")

    title = (
        f"scannetpp / {scene_id} / {frame_id}      "
        f"plan={THRESHOLD_PLANARITY}  norm={NORMAL_THRESHOLD_DEG}°  "
        f"depth_rel={DEPTH_THRESHOLD_REL}  match={NEIGHBOR_MATCH_COUNT}"
    )
    _save_panel(
        out_path, title,
        rgb=rgb,
        moge_depth=f["depth"],
        moge_normal=f["normal"],
        gt_depth=f["gt_depth"],
        depth_diff=diff,
        moge_planarity=f["planarity"],
        ours_seg=ours_seg,
    )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--inference_root", type=str, default=INFERENCE_ROOT_DEFAULT,
                    help="Root with <scene>/inference.h5 (Metric3D outputs).")
    ap.add_argument("--rgb_root", type=str, default=RGB_ROOT_DEFAULT,
                    help="ScanNet++ data root (must contain <scene>/iphone/rgb/).")
    ap.add_argument("--save_root", type=str, default=SAVE_ROOT_DEFAULT,
                    help="Root directory for output PNGs.")
    ap.add_argument("--frames", type=int, default=5,
                    help="Frames per scene (capped at the scene's frame count).")
    ap.add_argument("--scenes", type=int, default=None,
                    help="Number of scenes to sample. Default: all available.")
    ap.add_argument("--scene", type=str, action="append", default=None,
                    help="Restrict to specific scene(s). Repeatable.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda",
                    help='Torch device for segmentation ("cuda" or "cpu").')
    ap.add_argument("--m3d_scale", type=float, default=METRIC3D_DEPTH_SCALE_DEFAULT,
                    help=("Multiplicative scale applied to Metric3D depth to "
                          "recover true metric units (default: "
                          f"{METRIC3D_DEPTH_SCALE_DEFAULT:.6f} for ScanNet++; "
                          "pass 1.0 to disable)."))
    args = ap.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if not torch.cuda.is_available() and args.device == "cuda":
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    available = _list_scenes(args.inference_root)
    if not available:
        raise SystemExit(f"No <scene>/inference.h5 under {args.inference_root}")

    if args.scene:
        missing = sorted(set(args.scene) - set(available))
        if missing:
            raise SystemExit(f"Scene(s) not found: {missing}")
        scenes_pool = sorted(args.scene)
    else:
        scenes_pool = available

    n_scenes_pick = (len(scenes_pool) if args.scenes is None
                     else min(args.scenes, len(scenes_pool)))
    scene_picks_idx = rng.choice(len(scenes_pool), size=n_scenes_pick, replace=False)
    picked_scenes = sorted(scenes_pool[i] for i in scene_picks_idx)

    manifest = {
        "seed": args.seed,
        "frames_per_scene": args.frames,
        "scenes_per_dataset": args.scenes,
        "inference_root": args.inference_root,
        "rgb_root": args.rgb_root,
        "m3d_scale": args.m3d_scale,
        "seg_params": {
            "threshold_planarity": THRESHOLD_PLANARITY,
            "normal_threshold_deg": NORMAL_THRESHOLD_DEG,
            "depth_threshold_rel": DEPTH_THRESHOLD_REL,
            "neighbor_match_count": NEIGHBOR_MATCH_COUNT,
            "seg_function": "compute_vectorized_planar_segments_v5_relative",
        },
        "saved": [],
    }

    print(f"Picking {args.frames} frame(s) from each of {len(picked_scenes)} "
          f"scene(s) (out of {len(scenes_pool)} available).")

    saved_count = 0
    for sid in tqdm(picked_scenes, desc="scenes"):
        h5_path = os.path.join(args.inference_root, sid, "inference.h5")
        try:
            fids = _list_frame_ids(h5_path)
        except Exception as e:
            print(f"  [skip] {sid}: cannot read frame_ids ({e})")
            continue
        if not fids:
            continue

        k = min(args.frames, len(fids))
        idxs = rng.choice(len(fids), size=k, replace=False).tolist()
        idxs.sort()

        for h5_idx in tqdm(idxs, desc=f"  {sid}", leave=False):
            fid = fids[h5_idx]
            try:
                out = process_one(
                    scene_id=sid, frame_id=fid,
                    h5_path=h5_path, h5_idx=h5_idx,
                    rgb_root=args.rgb_root, save_root=args.save_root,
                    device=args.device,
                    depth_scale=args.m3d_scale,
                )
            except Exception as e:
                print(f"  [fail] {sid}/{fid}: {type(e).__name__}: {e}")
                continue
            if out:
                manifest["saved"].append({
                    "scene_id": sid,
                    "frame_id": fid,
                    "h5_idx": int(h5_idx),
                    "png": os.path.relpath(out, args.save_root),
                })
                saved_count += 1

    manifest_path = os.path.join(args.save_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {saved_count} PNGs under {args.save_root}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
