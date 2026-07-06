#!/usr/bin/env python3
"""
Demo: RGB -> MoGe 4-head model -> depth / normal / planarity / plane segmentation.

Runs the full single-image pipeline on every image in demo/inputs/ and writes,
for each frame, demo/outputs/<frame>/:
    depth.png       metric depth (turbo colormap)
    normal.png      surface normals ((n+1)/2 as RGB)
    planarity.png   planarity probability (viridis colormap)
    planeseg.png    plane segmentation (region growing on planarity+normal+depth)
    combined.png    RGB | depth | normal | planarity | planeseg montage (equal panel sizes)

The checkpoint defaults to paths.planarity_model_path; MoGe base weights come
from HuggingFace (cache via HF_HOME, defaulted to paths.moge_cache_dir).

Usage (from the repo root, `pxwplanar` env, GPU recommended):
    python demo/run_demo.py [--model_path <ckpt.pt>]
"""
import os
import sys
import glob
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from pxwplanar.paths import planarity_model_path, moge_cache_dir  # noqa: E402
os.environ.setdefault("HF_HOME", moge_cache_dir)

from pxwplanar.shared.segmentation import compute_vectorized_planar_segments  # noqa: E402
from pxwplanar.shared.utils.label_utils import remap_labels  # noqa: E402
from pxwplanar.shared.utils import visualize_top_components  # noqa: E402
from pxwplanar.inference.planarity.moge_inference import MoGePlanarityInference  # noqa: E402


def colorize_depth(depth, mask):
    """Metric depth -> turbo colormap RGB (invalid pixels black)."""
    valid = mask & (depth > 0)
    vis = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        colored = plt.get_cmap("turbo")(norm)[..., :3]
        vis = (colored * 255).astype(np.uint8)
        vis[~valid] = 0
    return vis


def colorize_normal(normal, mask):
    """Unit normals -> RGB in [0, 1] (invalid pixels black)."""
    vis = ((normal * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    vis[~mask] = 0
    return vis


def colorize_planarity(planarity):
    """Planarity probability -> viridis colormap RGB."""
    colored = plt.get_cmap("viridis")(np.clip(planarity, 0, 1))[..., :3]
    return (colored * 255).astype(np.uint8)


def process_image(image_path, model, output_root, args):
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    rgb = np.array(Image.open(image_path).convert("RGB"))

    res = model.predict_metric(image_path, num_tokens=args.num_tokens, return_all_heads=True)
    depth = res["depth"]
    normal = res["normal"]
    planarity = res["planarity_probability"]
    mask = res["mask"].astype(bool)

    # Plane segmentation with the canonical parameters (same as the benchmark).
    planarity_mask = (planarity > args.threshold_planarity).astype(np.int16)
    labels, _ = compute_vectorized_planar_segments(
        planarity_mask, normal, depth,
        np.deg2rad(args.normal_threshold_deg), args.depth_threshold,
        neighbor_match_count_thresh=args.neighbor_match_count_thresh,
    )
    labels, _ = remap_labels(labels)
    n_planes = len(np.unique(labels)) - (1 if (labels == 0).any() else 0)

    # Individual visualizations, all resized to the input RGB resolution so
    # every panel has identical dimensions (model heads run at a slightly
    # different processing resolution).
    H, W = rgb.shape[:2]
    depth_vis = cv2.resize(colorize_depth(depth, mask), (W, H), interpolation=cv2.INTER_AREA)
    normal_vis = cv2.resize(colorize_normal(normal, mask), (W, H), interpolation=cv2.INTER_AREA)
    planarity_vis = cv2.resize(colorize_planarity(planarity), (W, H), interpolation=cv2.INTER_AREA)
    # tab20 has 20 distinct colors — show the top-20 planes so no color repeats.
    seg_vis = visualize_top_components(labels, k=min(n_planes, 20), ignore_label=0,
                                       return_colors=True)
    seg_vis = cv2.resize(seg_vis, (W, H), interpolation=cv2.INTER_NEAREST)

    for fname, vis in [("depth.png", depth_vis), ("normal.png", normal_vis),
                       ("planarity.png", planarity_vis), ("planeseg.png", seg_vis)]:
        cv2.imwrite(os.path.join(out_dir, fname), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # Combined figure: plain pixel montage (RGB | depth | normal | planarity |
    # plane segmentation), equal panel sizes, thin white separators, no text.
    gap = np.full((H, 12, 3), 255, dtype=np.uint8)
    panels = [rgb, depth_vis, normal_vis, planarity_vis, seg_vis]
    combined = np.hstack([np.hstack([p, gap]) for p in panels[:-1]] + [panels[-1]])
    cv2.imwrite(os.path.join(out_dir, "combined.png"),
                cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"  {name}: {n_planes} planes -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str, default=planarity_model_path,
                        help="4-head MoGe checkpoint (.pt); default: paths.planarity_model_path")
    parser.add_argument("--input_dir", type=str, default=str(REPO_ROOT / "demo" / "inputs"))
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "demo" / "outputs"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_tokens", type=int, default=1600)

    # Canonical segmentation parameters (keep in sync with the benchmark).
    parser.add_argument("--threshold_planarity", type=float, default=0.3)
    parser.add_argument("--normal_threshold_deg", type=float, default=5.0)
    parser.add_argument("--depth_threshold", type=float, default=0.025,
                        help="Relative depth threshold (fraction of center depth)")
    parser.add_argument("--neighbor_match_count_thresh", type=int, default=8)
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg"))
                    + glob.glob(os.path.join(args.input_dir, "*.png")))
    if not images:
        sys.exit(f"No images found in {args.input_dir}")
    if not os.path.isfile(args.model_path):
        sys.exit(f"Checkpoint not found: {args.model_path}")

    print(f"Loading 4-head checkpoint {args.model_path} ...")
    model = MoGePlanarityInference(args.model_path, device=args.device)

    print(f"Processing {len(images)} image(s) -> {args.output_dir}")
    for image_path in images:
        process_image(image_path, model, args.output_dir, args)
    print("Done.")


if __name__ == "__main__":
    main()
