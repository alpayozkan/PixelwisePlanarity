"""
Segment plane labels from an existing inference.h5 dump.

Input H5 schema (per scene, all frames stacked along axis 0):
    frame_ids   (N,)            object        e.g. b'frame_000000'
    depth       (N, H, W)       float32       predicted depth (m, possibly affine)
    normals     (N, H, W, 3)    float32       predicted normals in [-1, 1]
    planarity   (N, H, W)       float32       predicted planarity in [0, 1]
    mask        (N, H, W)       float32       validity (~1.0)
    intrinsics  (N, 3, 3)       float32       (used for reference only)
    gt_*        ...                           ground truth (ignored here)

Runs `compute_vectorized_planar_segments_v5_relative` per frame and writes:

    <output_dir>/<scene_id>/planes.h5
        plane_labels  (N, H, W)  uint16   0 = non-planar, >0 = plane id
        frame_ids     (N,)       object

With --debug, also writes side-by-side PNG previews under
    <output_dir>/<scene_id>/debug/<frame_id>.png

Example
-------
python segment_from_inference_h5.py \\
    --input_root /cluster/scratch/ayavuz/inference/moge_HIRES_4datasets_epoch1/scannetpp/test \\
    --output_root /cluster/scratch/aoezkan/planeseg/inference/moge_HIRES_4datasets_epoch1_v5rel \\
    --frames 5 --debug
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# Make `planamono` importable when run as a script
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from planamono.shared.segmentation.plan2seg import (
    compute_vectorized_planar_segments_v5_relative,
)
from planamono.paths import scannetpp_path

# Default ScanNet++ RGB root — only used when --debug is set
SCANNETPP_RGB_ROOT_DEFAULT = os.path.join(scannetpp_path, "data")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    norm[~valid] = 0
    return (cm.get_cmap("turbo")(norm)[..., :3] * 255).astype(np.uint8)


def _colorize_normal(normal: np.ndarray) -> np.ndarray:
    # (H, W, 3) in [-1, 1] → RGB in [0, 255]
    return ((normal + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)


def _colorize_planarity(planarity: np.ndarray) -> np.ndarray:
    p = np.clip(planarity, 0, 1)
    return (cm.get_cmap("viridis")(p)[..., :3] * 255).astype(np.uint8)


def _colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Random color per label, label 0 → black."""
    H, W = labels.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    uniq = np.unique(labels)
    uniq = uniq[uniq != 0]
    if uniq.size == 0:
        return out
    rng = np.random.default_rng(seed=0)
    palette = rng.integers(50, 256, size=(uniq.max() + 1, 3), dtype=np.int32)
    palette[0] = 0
    out = palette[labels].astype(np.uint8)
    return out


def _load_rgb(scannetpp_rgb_root: str, scene_id: str, frame_id: str,
              target_hw: tuple) -> np.ndarray:
    """Load and resize RGB to target (H, W). Returns uint8 RGB or zeros if missing."""
    H, W = target_hw
    rgb_path = os.path.join(scannetpp_rgb_root, scene_id, "iphone", "rgb", f"{frame_id}.jpg")
    img = cv2.imread(rgb_path)
    if img is None:
        return np.zeros((H, W, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return img


def _save_debug_png(out_path: str, panels: list, titles: list) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-scene processing
# ---------------------------------------------------------------------------

def process_scene(
    inference_h5: str,
    output_dir: str,
    scene_id: str,
    threshold_planarity: float,
    normal_threshold_deg: float,
    neighbor_match_count_thresh: int,
    depth_threshold_relative: float,
    n_frames: int | None,
    debug: bool,
    scannetpp_rgb_root: str,
    device: str,
    show_frame_pbar: bool = True,
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    normal_threshold_rad = np.deg2rad(normal_threshold_deg)

    with h5py.File(inference_h5, "r") as f:
        N = f["planarity"].shape[0]
        H, W = f["planarity"].shape[1:]
        frame_ids_all = f["frame_ids"][:]
        n_to_process = N if n_frames is None else min(n_frames, N)

        all_labels = np.zeros((n_to_process, H, W), dtype=np.uint16)
        frame_ids_out = []

        frame_iter = range(n_to_process)
        if show_frame_pbar:
            frame_iter = tqdm(frame_iter, total=n_to_process,
                              desc=f"  {scene_id}", leave=False, unit="frame")

        for i in frame_iter:
            frame_id_b = frame_ids_all[i]
            frame_id = frame_id_b.decode() if isinstance(frame_id_b, (bytes, bytearray)) else str(frame_id_b)
            frame_ids_out.append(frame_id)

            planarity = f["planarity"][i, :].astype(np.float32)        # (H, W)
            depth = f["depth"][i, :].astype(np.float32)                # (H, W)
            normal = f["normals"][i, :].astype(np.float32)             # (H, W, 3)
            gt_planes = f["gt_planes"][i, :].astype(np.int32) if debug and "gt_planes" in f else None

            planarity_mask = (planarity > threshold_planarity).astype(np.int32)

            labels, num_components = compute_vectorized_planar_segments_v5_relative(
                planarity_mask=planarity_mask,
                normal=normal,
                depth=depth,
                normal_threshold_rad=normal_threshold_rad,
                depth_threshold=depth_threshold_relative,
                neighbor_match_count_thresh=neighbor_match_count_thresh,
                device=device,
            )
            labels_u16 = labels.astype(np.uint16)
            all_labels[i] = labels_u16

            if not show_frame_pbar:
                print(f"  [{i+1}/{n_to_process}] {frame_id}: {num_components} segments, "
                      f"{(labels_u16 > 0).mean()*100:.1f}% planar")

            if debug:
                rgb = _load_rgb(scannetpp_rgb_root, scene_id, frame_id, (H, W))
                panels = [
                    rgb,
                    _colorize_depth(depth),
                    _colorize_normal(normal),
                    _colorize_planarity(planarity),
                    _colorize_labels(labels_u16.astype(np.int32)),
                ]
                titles = [
                    f"RGB ({frame_id})",
                    "MoGe depth (pred)",
                    "Normals (pred)",
                    f"Planarity (>{threshold_planarity})",
                    f"Plane labels ({num_components} segs)",
                ]
                if gt_planes is not None:
                    n_gt = int((np.unique(gt_planes) != 0).sum())
                    panels.append(_colorize_labels(gt_planes))
                    titles.append(f"GT plane labels ({n_gt} segs)")
                _save_debug_png(
                    os.path.join(debug_dir, f"{frame_id}.png"),
                    panels, titles,
                )

    # Write planes.h5
    out_h5 = os.path.join(output_dir, "planes.h5")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset(
            "plane_labels",
            data=all_labels,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "frame_ids",
            data=np.array(frame_ids_out, dtype="S"),
        )
        f.attrs["seg_version"] = "v5_relative"
        f.attrs["threshold_planarity"] = threshold_planarity
        f.attrs["normal_threshold_deg"] = normal_threshold_deg
        f.attrs["depth_threshold_relative"] = depth_threshold_relative
        f.attrs["neighbor_match_count_thresh"] = neighbor_match_count_thresh

    if not show_frame_pbar:
        print(f"Wrote {out_h5}  ({n_to_process} frames)")
        if debug:
            print(f"Wrote debug PNGs → {debug_dir}")
    return n_to_process


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root", type=str, required=True,
                    help="Directory containing <scene_id>/inference.h5 "
                         "(typically .../<model>/<dataset>/<split>)")
    ap.add_argument("--output_root", type=str, required=True,
                    help="Output base directory. Final layout: "
                         "<output_root>/<model>/<dataset>/<split>/<scene_id>/planes.h5 "
                         "(model/dataset/split inferred from input_root unless overridden).")
    ap.add_argument("--scene_id", type=str, default=None,
                    help="If set, process only this scene; otherwise process all scenes under input_root")
    ap.add_argument("--model_name", type=str, default=None,
                    help="Override model name in output path (default: 3rd-from-last component of input_root)")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Override dataset name in output path (default: 2nd-from-last component of input_root)")
    ap.add_argument("--split", type=str, default=None,
                    help="Override split name in output path (default: last component of input_root)")

    # Segmentation hyperparameters (defaults match the user's request)
    ap.add_argument("--threshold_planarity", type=float, default=0.3)
    ap.add_argument("--normal_threshold_deg", type=float, default=5.0)
    ap.add_argument("--neighbor_match_count_thresh", type=int, default=8)
    ap.add_argument("--depth_threshold_relative", type=float, default=0.025,
                    help="Relative depth threshold (fraction of center depth)")

    ap.add_argument("--frames", type=int, default=None,
                    help="If set, process only the first N frames per scene")
    ap.add_argument("--debug", action="store_true",
                    help="Save side-by-side PNGs (RGB, depth, normal, planarity, labels) per frame")
    ap.add_argument("--scannetpp_rgb_root", type=str, default=SCANNETPP_RGB_ROOT_DEFAULT,
                    help="Root for ScanNet++ RGB images, used when --debug is set")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.scene_id is not None:
        scene_ids = [args.scene_id]
    else:
        scene_ids = sorted([
            d for d in os.listdir(args.input_root)
            if os.path.isdir(os.path.join(args.input_root, d))
        ])

    # Infer <model>/<dataset>/<split> from input_root tail (with overrides)
    parts = Path(args.input_root.rstrip("/")).parts
    inferred_split = parts[-1] if len(parts) >= 1 else "split"
    inferred_dataset = parts[-2] if len(parts) >= 2 else "dataset"
    inferred_model = parts[-3] if len(parts) >= 3 else "model"
    model_name = args.model_name or inferred_model
    dataset = args.dataset or inferred_dataset
    split = args.split or inferred_split
    out_base = os.path.join(args.output_root, model_name, dataset, split)

    print(f"Found {len(scene_ids)} scene(s)")
    print(f"Output base: {out_base}")
    print(f"Config: planarity>{args.threshold_planarity}, "
          f"normal<{args.normal_threshold_deg}°, "
          f"match>={args.neighbor_match_count_thresh}, "
          f"depth_rel<{args.depth_threshold_relative}")

    total = 0
    scene_pbar = tqdm(scene_ids, desc="Scenes", unit="scene")
    for sid in scene_pbar:
        scene_pbar.set_postfix_str(sid)
        inf_h5 = os.path.join(args.input_root, sid, "inference.h5")
        if not os.path.isfile(inf_h5):
            tqdm.write(f"[skip] {sid}: no inference.h5")
            continue
        out_dir = os.path.join(out_base, sid)
        total += process_scene(
            inference_h5=inf_h5,
            output_dir=out_dir,
            scene_id=sid,
            threshold_planarity=args.threshold_planarity,
            normal_threshold_deg=args.normal_threshold_deg,
            neighbor_match_count_thresh=args.neighbor_match_count_thresh,
            depth_threshold_relative=args.depth_threshold_relative,
            n_frames=args.frames,
            debug=args.debug,
            scannetpp_rgb_root=args.scannetpp_rgb_root,
            device=args.device,
        )

    print(f"\nDone. Processed {total} frames across {len(scene_ids)} scene(s).")


if __name__ == "__main__":
    main()
