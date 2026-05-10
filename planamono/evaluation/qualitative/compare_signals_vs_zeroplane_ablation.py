"""
Ablation of compare_signals_vs_zeroplane.py: replace MoGe's predicted normal
with a normal computed **geometrically from the (MoGe) depth map** when
running ``compute_vectorized_planar_segments_v5_relative``. The same v5_relative
seg parameters (plan>0.3, norm<5°, depth_rel<0.025, match≥8) and the same
inputs (planarity, depth) are used; only the *normal* signal differs.

Layout — one row per frame, **9 panels** (was 7 in the non-ablation version):

    RGB | GT depth | GT planes | MoGe depth |
    MoGe normal | Geo normal (from depth) |
    MoGe planarity |
    Ours seg (MoGe normal) | Ours seg (geo normal) | ZeroPlane seg

Geometric normal source
-----------------------
``planamono.shared.utils.depth_normal.depth_to_normal_remi`` — gradient-based
cross-product of the backprojected XYZ map. Requires (fx, fy, cx, cy), which
we read from the GT dataset's ``K`` field and scale to the panel resolution.

Output layout
-------------
Default save_root is ``signals_vs_zeroplane_ablation`` (sibling of the
non-ablation directory). Same per-(dataset, scene) tree as the original.

Example
-------
python compare_signals_vs_zeroplane_ablation.py \\
    --save_root /cluster/scratch/aoezkan/planeseg/audit/signals_vs_zeroplane_ablation \\
    --frames 8 --seed 0
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

from planamono.paths import (  # noqa: E402
    scannetpp_path,
    scannetpp_rend_plane_path,
    repo_path,
)
from planamono.shared.datasets.scannetpp import ScanNetPPPlaneDataset  # noqa: E402
from planamono.shared.datasets.nyuv2_plane_dataset import NYUv2PlaneDataset  # noqa: E402
from planamono.shared.datasets.sevenscenes_plane_dataset import SevenScenesPlaneDataset  # noqa: E402
from planamono.shared.segmentation import compute_vectorized_planar_segments_v5_relative  # noqa: E402
from planamono.shared.utils.depth_normal import depth_to_normal_remi  # noqa: E402


# ---------------------------------------------------------------------------
# Segmentation parameters (ECCV v5_relative config3_default)
# ---------------------------------------------------------------------------

THRESHOLD_PLANARITY = 0.3
NORMAL_THRESHOLD_DEG = 5.0
DEPTH_THRESHOLD_REL = 0.025
NEIGHBOR_MATCH_COUNT = 8

ZEROPLANE_NONPLANAR_LABEL = 20


# ---------------------------------------------------------------------------
# Dataset configurations (must match the non-ablation script).
# ---------------------------------------------------------------------------

DATASETS = {
    "scannetpp": {
        "moge_root": "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1/scannetpp",
        "zeroplane_root": "/cluster/scratch/aoezkan/planeseg/scannetpp/inference/zeroplane_default_dust3r_released_h5",
        "split": "test",
    },
    "nyuv2": {
        "moge_root": "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1/nyuv2",
        "zeroplane_root": "/cluster/scratch/aoezkan/planeseg/nyuv2/inference/zeroplane_default_dust3r_released_h5",
        "data_root": "/cluster/scratch/aoezkan/planeseg/dataset/nyuv2_plane",
        "split": "test",
    },
    "sevenscenes": {
        "moge_root": "/cluster/scratch/aoezkan/planeseg/inference/moge_signals_4ds_ep1/sevenscenes",
        "zeroplane_root": "/cluster/scratch/aoezkan/planeseg/sevenscenes/inference/zeroplane_default_dust3r_released_h5",
        "data_root": "/cluster/scratch/aoezkan/planeseg/dataset/sevenscenes_plane",
        "split": "val",
    },
}


# ---------------------------------------------------------------------------
# Per-scene H5 frame_id index (cached lookup)
# ---------------------------------------------------------------------------

class FidIndex:
    def __init__(self, root: str, h5_filename: str = "planes.h5"):
        self.root = root
        self.h5_filename = h5_filename
        self._cache: Dict[str, Optional[Tuple[str, Dict[str, int]]]] = {}

    def info(self, scene_id: str):
        if scene_id in self._cache:
            return self._cache[scene_id]
        h5p = os.path.join(self.root, scene_id, self.h5_filename)
        if not os.path.isfile(h5p):
            self._cache[scene_id] = None
            return None
        with h5py.File(h5p, "r") as f:
            fids_raw = f["frame_ids"][:]
            fid_to_idx = {}
            for i, x in enumerate(fids_raw):
                fid = x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
                fid_to_idx[fid] = i
        self._cache[scene_id] = (h5p, fid_to_idx)
        return self._cache[scene_id]


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def _decode_fid(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def _moge_fid_to_canonical(dataset_name: str, moge_fid: str) -> str:
    if dataset_name == "scannetpp":
        return moge_fid
    if moge_fid.startswith("frame_"):
        tail = moge_fid.split("_", 1)[1]
        try:
            return str(int(tail))
        except ValueError:
            return moge_fid
    return moge_fid


def load_moge_frame(moge_h5: str, idx: int):
    with h5py.File(moge_h5, "r") as f:
        plan = f["planarity"][idx].astype(np.float32)
        nrm = f["normal"][idx].astype(np.float32)
        dep = f["depth_metric"][idx].astype(np.float32)
        msk = f["mask"][idx].astype(bool) if "mask" in f else np.ones_like(dep, dtype=bool)
    return plan, nrm, dep, msk


def load_zeroplane_seg(zp_h5: str, idx: int) -> np.ndarray:
    with h5py.File(zp_h5, "r") as f:
        labels = f["planes"][idx].astype(np.int32)
    labels = labels + 1
    labels[labels == ZEROPLANE_NONPLANAR_LABEL + 1] = 0
    return labels


# ---------------------------------------------------------------------------
# Frame-pair construction
# ---------------------------------------------------------------------------

def _list_moge_scenes(moge_root: str) -> List[str]:
    return sorted([
        d for d in os.listdir(moge_root)
        if os.path.isdir(os.path.join(moge_root, d))
        and os.path.isfile(os.path.join(moge_root, d, "moge_signals.h5"))
    ])


def _build_pool(dataset_name: str) -> List[Dict]:
    cfg = DATASETS[dataset_name]
    moge_root = cfg["moge_root"]
    zp_root = cfg["zeroplane_root"]

    scenes = _list_moge_scenes(moge_root)
    pool = []
    for sid in scenes:
        moge_h5 = os.path.join(moge_root, sid, "moge_signals.h5")
        with h5py.File(moge_h5, "r") as f:
            moge_fids = [_decode_fid(x) for x in f["frame_ids"][:]]
        zp_h5 = os.path.join(zp_root, sid, "planes.h5")
        if not os.path.isfile(zp_h5):
            continue
        with h5py.File(zp_h5, "r") as f:
            zp_fids = set(_decode_fid(x) for x in f["frame_ids"][:])
        for moge_fid in moge_fids:
            canonical = _moge_fid_to_canonical(dataset_name, moge_fid)
            if canonical in zp_fids:
                pool.append({
                    "scene_id": sid,
                    "moge_fid": moge_fid,
                    "fid": canonical,
                })
    return pool


# ---------------------------------------------------------------------------
# GT lookup
# ---------------------------------------------------------------------------

class ScanNetPPGT:
    def __init__(self, target_hw: Tuple[int, int]):
        H, W = target_hw
        rgb_root = os.path.join(scannetpp_path, "data")
        if not os.path.isdir(rgb_root):
            rgb_root = scannetpp_path
        self.ds = ScanNetPPPlaneDataset(
            rgb_root=rgb_root,
            plane_label_root=scannetpp_rend_plane_path,
            sem_label_root=scannetpp_rend_plane_path,
            depth_label_root=scannetpp_rend_plane_path,
            split_txt_dir=os.path.join(repo_path, "splits", "scannetpp"),
            split=DATASETS["scannetpp"]["split"],
            image_height=H,
            image_width=W,
        )
        self._lookup = {
            (s, f): i for i, (s, f) in enumerate(
                ((rp.split("/")[-4], os.path.splitext(os.path.basename(rp))[0])
                 for (rp, *_rest) in self.ds.valid_pairs)
            )
        }

    def get(self, scene_id: str, frame_id: str):
        idx = self._lookup.get((scene_id, frame_id))
        if idx is None:
            return None
        return self.ds[idx]


class FlatNPZGT:
    def __init__(self, name: str, target_hw: Tuple[int, int]):
        H, W = target_hw
        cfg = DATASETS[name]
        if name == "nyuv2":
            self.ds = NYUv2PlaneDataset(
                data_root=cfg["data_root"], split=cfg["split"],
                image_height=H, image_width=W,
            )
        elif name == "sevenscenes":
            self.ds = SevenScenesPlaneDataset(
                data_root=cfg["data_root"], split=cfg["split"],
                image_height=H, image_width=W,
            )
        else:
            raise ValueError(name)

        self._lookup = {}
        for i in range(len(self.ds)):
            if hasattr(self.ds, "valid_pairs"):
                pair = self.ds.valid_pairs[i]
                if name == "nyuv2":
                    sid = "nyuv2"
                    fid = str(pair[1])
                else:
                    sid = pair[2]
                    fid = str(pair[1])
                self._lookup[(sid, fid)] = i

    def get(self, scene_id: str, frame_id: str):
        idx = self._lookup.get((scene_id, frame_id))
        if idx is None:
            return None
        return self.ds[idx]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _label_palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    pal = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    return pal


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    H, W = labels.shape
    n = int(labels.max()) + 1 if labels.size else 1
    n = max(n, 2)
    pal = _label_palette(n)
    pal[0] = 0
    out = pal[np.clip(labels, 0, n - 1)]
    return out


def _normal_to_rgb(normal: np.ndarray) -> np.ndarray:
    rgb = ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def _save_panel_ablation(
    save_path: str,
    title: str,
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    moge_depth: np.ndarray,
    moge_normal: np.ndarray,
    geo_normal: np.ndarray,
    moge_planarity: np.ndarray,
    ours_seg_moge: np.ndarray,
    ours_seg_geo: np.ndarray,
    zp_seg: np.ndarray,
    gt_plane: Optional[np.ndarray] = None,
):
    """9-panel ablation row: like the non-ablation but with extra 'Geo normal'
    and 'Ours seg (geo normal)' columns. Bracketed by RGB / ZeroPlane on the
    ends so the eye can pivot between predictions and inputs."""
    valid = (gt_depth > 0.05) & np.isfinite(gt_depth)

    cols = []
    cols.append(("RGB",        rgb,                            None,    None))
    cols.append(("GT depth",   gt_depth,                       "turbo", "depth"))
    if gt_plane is not None:
        cols.append(("GT planes", _colorize_labels(gt_plane), None,    None))
    cols.append(("MoGe depth",      moge_depth,                "turbo", "depth"))
    cols.append(("MoGe normal",     _normal_to_rgb(moge_normal), None, None))
    cols.append(("Geo normal\n(from depth)", _normal_to_rgb(geo_normal), None, None))
    cols.append(("MoGe planarity",  moge_planarity,            "magma", "prob"))
    cols.append(("Ours seg\n(MoGe nrm)",   _colorize_labels(ours_seg_moge), None, None))
    cols.append(("Ours seg\n(geo nrm)",    _colorize_labels(ours_seg_geo),  None, None))
    cols.append(("ZeroPlane seg",   _colorize_labels(zp_seg),  None,    None))

    finite_gt = gt_depth[valid]
    finite_moge = moge_depth[valid & np.isfinite(moge_depth)]
    if finite_gt.size > 100 and finite_moge.size > 100:
        all_d = np.concatenate([finite_gt, finite_moge])
        vmin_d, vmax_d = np.percentile(all_d, [2, 98])
    else:
        vmin_d, vmax_d = 0.0, 5.0

    n = len(cols)
    fig = plt.figure(figsize=(3.4 * n, 4.0))
    gs = fig.add_gridspec(nrows=1, ncols=n, wspace=0.04, left=0.01, right=0.99,
                          top=0.84, bottom=0.04)
    for i, (name, img, cmap, kind) in enumerate(cols):
        ax = fig.add_subplot(gs[0, i])
        if cmap is None:
            ax.imshow(img)
        elif kind == "depth":
            arr = np.where(valid, img, np.nan)
            ax.imshow(arr, cmap=cmap, vmin=vmin_d, vmax=vmax_d)
        elif kind == "prob":
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _scale_K_to_hw(K: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    """Scale K's principal point + focal length from its source resolution
    (inferred as 2·cx, 2·cy assuming centred principal point) to ``hw``."""
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


def _normal_from_depth_and_K(depth: np.ndarray, K_at_panel: np.ndarray) -> np.ndarray:
    """Gradient-based depth → unit normal. K must be at the same resolution
    as ``depth`` (use ``_scale_K_to_hw``)."""
    fx = float(K_at_panel[0, 0])
    fy = float(K_at_panel[1, 1])
    cx = float(K_at_panel[0, 2])
    cy = float(K_at_panel[1, 2])
    n = depth_to_normal_remi(depth.astype(np.float64), fx, fy, cx, cy)
    return n.astype(np.float32)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _compute_ours_seg(planarity: np.ndarray, normal: np.ndarray, depth: np.ndarray,
                     device: str = "cuda") -> np.ndarray:
    mask = (planarity > THRESHOLD_PLANARITY).astype(np.int32)
    labels, _ = compute_vectorized_planar_segments_v5_relative(
        mask, normal, depth,
        np.deg2rad(NORMAL_THRESHOLD_DEG),
        DEPTH_THRESHOLD_REL,
        NEIGHBOR_MATCH_COUNT,
        device=device,
    )
    return labels.astype(np.int32)


def _resize_to(arr: np.ndarray, hw: Tuple[int, int],
               interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    H, W = hw
    if arr.shape[:2] == (H, W):
        return arr
    return cv2.resize(arr, (W, H), interpolation=interp)


def process_one(
    dataset_name: str,
    sample: Dict,
    moge_idx: FidIndex,
    zp_idx: FidIndex,
    gt_provider,
    save_root: str,
    target_hw: Tuple[int, int],
    device: str,
) -> Optional[str]:
    sid = sample["scene_id"]
    moge_fid = sample["moge_fid"]
    fid = sample["fid"]

    moge_info = moge_idx.info(sid)
    zp_info = zp_idx.info(sid)
    if moge_info is None or zp_info is None:
        return None
    moge_h5, moge_map = moge_info
    zp_h5, zp_map = zp_info
    if moge_fid not in moge_map or fid not in zp_map:
        return None

    plan, nrm, dep, msk = load_moge_frame(moge_h5, moge_map[moge_fid])
    zp_seg = load_zeroplane_seg(zp_h5, zp_map[fid])

    H, W = target_hw
    plan = _resize_to(plan, target_hw)
    dep = _resize_to(dep, target_hw)
    nrm = _resize_to(nrm, target_hw)
    zp_seg = _resize_to(zp_seg, target_hw, cv2.INTER_NEAREST)

    # Renormalise predicted normals after resize (linear interp leaves them
    # non-unit, which biases the angular thresholds inside the seg).
    nrm_norm = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.divide(nrm, np.clip(nrm_norm, 1e-6, None), where=nrm_norm > 1e-6)

    gt = gt_provider.get(sid, fid)
    if gt is None:
        return None
    rgb = (gt["image"].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    gt_depth = gt["depth"].squeeze(0).cpu().numpy().astype(np.float32)
    gt_plane = gt["plane"].squeeze(0).cpu().numpy().astype(np.int32)
    K_native = (gt["K"].cpu().numpy() if hasattr(gt["K"], "cpu") else np.asarray(gt["K"]))
    K_native = K_native.astype(np.float64)

    rgb = _resize_to(rgb, target_hw)
    gt_depth = _resize_to(gt_depth, target_hw)
    gt_plane = _resize_to(gt_plane, target_hw, cv2.INTER_NEAREST)

    # Geometric normal from MoGe depth, using GT K scaled to the panel HW.
    K_panel = _scale_K_to_hw(K_native, target_hw)
    geo_normal = _normal_from_depth_and_K(dep, K_panel)

    # Two segmentations: same planarity + depth, different normal.
    ours_seg_moge = _compute_ours_seg(plan, nrm, dep, device=device)
    ours_seg_geo = _compute_ours_seg(plan, geo_normal, dep, device=device)

    out_dir = os.path.join(save_root, dataset_name, sid)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{fid}.png")

    title = (
        f"{dataset_name}  /  {sid}  /  {fid}      "
        f"plan={THRESHOLD_PLANARITY}  norm={NORMAL_THRESHOLD_DEG}°  "
        f"depth_rel={DEPTH_THRESHOLD_REL}  match={NEIGHBOR_MATCH_COUNT}      "
        f"ablation: normal source = MoGe vs depth-gradient"
    )
    _save_panel_ablation(
        out_path, title,
        rgb=rgb, gt_depth=gt_depth, gt_plane=gt_plane,
        moge_depth=dep, moge_normal=nrm, geo_normal=geo_normal,
        moge_planarity=plan,
        ours_seg_moge=ours_seg_moge, ours_seg_geo=ours_seg_geo,
        zp_seg=zp_seg,
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
    ap.add_argument(
        "--save_root", type=str,
        default="/cluster/scratch/aoezkan/planeseg/audit/signals_vs_zeroplane_ablation",
        help="Root directory for ablation PNGs.",
    )
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--scenes", type=int, default=None)
    ap.add_argument("--datasets", nargs="+",
                    default=["scannetpp", "nyuv2", "sevenscenes"],
                    choices=list(DATASETS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    target_hw = (args.height, args.width)

    if not torch.cuda.is_available() and args.device == "cuda":
        print("[WARN] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    manifest = {
        "seed": args.seed,
        "frames_per_scene": args.frames,
        "scenes_per_dataset": args.scenes,
        "target_hw": list(target_hw),
        "ablation": {
            "normal_source_for_seg": "depth_to_normal_remi (gradient-based)",
            "panel_columns": [
                "RGB", "GT depth", "GT planes",
                "MoGe depth", "MoGe normal", "Geo normal (from depth)",
                "MoGe planarity",
                "Ours seg (MoGe normal)", "Ours seg (geo normal)",
                "ZeroPlane seg",
            ],
        },
        "seg_params": {
            "threshold_planarity": THRESHOLD_PLANARITY,
            "normal_threshold_deg": NORMAL_THRESHOLD_DEG,
            "depth_threshold_rel": DEPTH_THRESHOLD_REL,
            "neighbor_match_count": NEIGHBOR_MATCH_COUNT,
            "seg_function": "compute_vectorized_planar_segments_v5_relative",
        },
        "datasets": {},
    }

    for ds_name in args.datasets:
        print(f"\n=== {ds_name} ===")
        cfg = DATASETS[ds_name]

        pool = _build_pool(ds_name)
        if not pool:
            print(f"  [skip] empty pool — check moge_root and zeroplane_root paths.")
            manifest["datasets"][ds_name] = {"pool_size": 0, "saved": []}
            continue
        print(f"  pool size: {len(pool)}")

        by_scene: Dict[str, List[Dict]] = {}
        for entry in pool:
            by_scene.setdefault(entry["scene_id"], []).append(entry)

        all_scene_ids = sorted(by_scene.keys())
        n_scenes_pick = (len(all_scene_ids) if args.scenes is None
                         else min(args.scenes, len(all_scene_ids)))
        scene_picks_idx = rng.choice(len(all_scene_ids), size=n_scenes_pick, replace=False)
        picked_scenes = sorted(all_scene_ids[i] for i in scene_picks_idx)

        picks: List[Dict] = []
        for sid in picked_scenes:
            entries = by_scene[sid]
            k = min(args.frames, len(entries))
            sel_idx = rng.choice(len(entries), size=k, replace=False)
            picks.extend(entries[i] for i in sorted(sel_idx.tolist()))
        print(f"  picked: {len(picks)} frames across {n_scenes_pick}/{len(all_scene_ids)} scenes "
              f"({args.frames}/scene)")

        moge_idx = FidIndex(cfg["moge_root"], h5_filename="moge_signals.h5")
        zp_idx = FidIndex(cfg["zeroplane_root"], h5_filename="planes.h5")

        if ds_name == "scannetpp":
            gt_provider = ScanNetPPGT(target_hw=target_hw)
        else:
            gt_provider = FlatNPZGT(ds_name, target_hw=target_hw)

        saved_list = []
        for sample in tqdm(picks, desc=f"  {ds_name}"):
            try:
                path = process_one(
                    dataset_name=ds_name, sample=sample,
                    moge_idx=moge_idx, zp_idx=zp_idx,
                    gt_provider=gt_provider,
                    save_root=args.save_root,
                    target_hw=target_hw, device=args.device,
                )
            except Exception as e:
                print(f"  [fail] {sample}: {type(e).__name__}: {e}")
                continue
            if path:
                saved_list.append({"scene_id": sample["scene_id"],
                                   "moge_fid": sample["moge_fid"],
                                   "fid": sample["fid"],
                                   "png": os.path.relpath(path, args.save_root)})
        manifest["datasets"][ds_name] = {
            "pool_size": len(pool),
            "saved": saved_list,
        }
        print(f"  saved {len(saved_list)}/{len(picks)} PNGs under {args.save_root}/{ds_name}/")

    manifest_path = os.path.join(args.save_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
