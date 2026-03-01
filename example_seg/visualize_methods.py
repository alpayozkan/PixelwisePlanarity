"""
Visualize segmentation methods on random frames from ScanNet++ val inference.

Saves one figure per frame showing GT, each variant (no merge), and each variant (with merge),
with SC scores annotated.

Usage:
    python example_seg/visualize_methods.py --device cpu
    python example_seg/visualize_methods.py --device cpu --variants v5,v9,v10
"""

import argparse
import os
import sys
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan2seg import (
    compute_vectorized_planar_segments_v1,
    compute_vectorized_planar_segments_v4,
    compute_vectorized_planar_segments_v5,
    compute_vectorized_planar_segments_v6,
    compute_vectorized_planar_segments_v7,
    compute_vectorized_planar_segments_v8,
    compute_vectorized_planar_segments_v9,
    compute_vectorized_planar_segments_v10,
    compute_vectorized_planar_segments_v11,
    compute_vectorized_planar_segments_v12,
    filter_small_segments,
    merge_similar_planes,
    merge_planes_3d,
)

VARIANT_FNS = {
    "v1": compute_vectorized_planar_segments_v1,
    "v4": compute_vectorized_planar_segments_v4,
    "v5": compute_vectorized_planar_segments_v5,
    "v6": compute_vectorized_planar_segments_v6,
    "v7": compute_vectorized_planar_segments_v7,
    "v8": compute_vectorized_planar_segments_v8,
    "v9": compute_vectorized_planar_segments_v9,
    "v10": compute_vectorized_planar_segments_v10,
    "v11": compute_vectorized_planar_segments_v11,
    "v12": compute_vectorized_planar_segments_v12,
}


# ---------------------------------------------------------------------------
# Inlined metrics
# ---------------------------------------------------------------------------
def segmentation_covering_fast(gt_seg, pred_seg):
    gt = gt_seg.ravel().astype(np.int64)
    pr = pred_seg.ravel().astype(np.int64)
    if gt.size == 0:
        return 0.0
    gt_labels, gt_inv = np.unique(gt, return_inverse=True)
    pr_labels, pr_inv = np.unique(pr, return_inverse=True)
    n_gt, n_pr = gt_labels.size, pr_labels.size
    combined = gt_inv * n_pr + pr_inv
    counts = np.bincount(combined, minlength=n_gt * n_pr).astype(np.int64)
    contingency = counts.reshape((n_gt, n_pr))
    gt_areas = contingency.sum(axis=1)
    pr_areas = contingency.sum(axis=0)
    union = gt_areas[:, None] + pr_areas[None, :] - contingency
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_matrix = np.where(union > 0, contingency / union, 0.0)
    best_iou = iou_matrix.max(axis=1)
    total_area = gt_areas.sum()
    return float((best_iou * gt_areas).sum() / total_area) if total_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def label_to_rgb(seg):
    """Convert segmentation labels to RGB. Background (-1 or 0 in eval) is black."""
    rng = np.random.RandomState(42)
    palette = rng.rand(1000, 3) * 0.7 + 0.2  # avoid too dark/light
    out = np.zeros((*seg.shape, 3))
    for label in np.unique(seg):
        if label <= 0:
            continue
        out[seg == label] = palette[label % len(palette)]
    return out


def pred_to_eval_labels(pred_seg):
    out = np.zeros_like(pred_seg, dtype=np.int64)
    valid = pred_seg >= 0
    out[valid] = pred_seg[valid] + 1
    return out


# ---------------------------------------------------------------------------
# Segmentation runner (simplified from run_eval.py)
# ---------------------------------------------------------------------------
def run_variant(variant, planarity_mask, planarity, normals, depth, args):
    normal_rad = np.deg2rad(args.normal_thresh)
    common_gpu = dict(normal_threshold_rad=normal_rad, depth_threshold=args.depth_thresh)

    if variant == "v1":
        labels, _ = compute_vectorized_planar_segments_v1(
            planarity_mask, normals, depth, **common_gpu,
            neighbor_match_count_thresh=args.neighbor_thresh_v1)
    elif variant in ("v4", "v5"):
        fn = VARIANT_FNS[variant]
        labels, _ = fn(planarity_mask, normals, depth, **common_gpu,
                       neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
                       device=args.device)
    elif variant == "v6":
        labels, _ = compute_vectorized_planar_segments_v6(
            planarity, normals, depth, **common_gpu,
            weighted_thresh=3.0, planarity_floor=0.1, device=args.device)
    elif variant == "v7":
        labels, _ = compute_vectorized_planar_segments_v7(
            planarity, normals, depth, **common_gpu,
            seed_thresh=0.7, grow_thresh=0.2,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            min_seed_frac=0.1, device=args.device)
    elif variant == "v8":
        labels, _ = compute_vectorized_planar_segments_v8(
            planarity, normals, depth, **common_gpu,
            base_thresh=args.planarity_thresh, adapt_strength=0.3,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device)
    elif variant == "v9":
        labels, _ = compute_vectorized_planar_segments_v9(
            planarity, normals, depth, **common_gpu,
            joint_thresh=0.5, planarity_weight=0.5, planarity_floor=0.1,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device)
    elif variant == "v10":
        labels, _ = compute_vectorized_planar_segments_v10(
            planarity, normals, depth, **common_gpu)
    elif variant == "v11":
        labels, _ = compute_vectorized_planar_segments_v11(
            planarity, normals, depth, **common_gpu)
    elif variant == "v12":
        labels, _ = compute_vectorized_planar_segments_v12(
            planarity, normals, depth, **common_gpu,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    labels = labels.astype(np.int32) - 1  # bg(0) → -1
    filtered = filter_small_segments(labels, min_size=args.min_size)
    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation methods")
    parser.add_argument("--data_dir", default="example_seg/scannetpp_val_inference_intrinsics")
    parser.add_argument("--out_dir", default="example_seg/vis_output")
    parser.add_argument("--n_frames", type=int, default=10,
                        help="Number of random frames to visualize")
    parser.add_argument("--variants", default="v5,v9,v10,v11,v12",
                        help="Comma-separated variants to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Segmentation params (same defaults as run_eval.py)
    parser.add_argument("--planarity_thresh", type=float, default=0.5)
    parser.add_argument("--normal_thresh", type=float, default=10.0)
    parser.add_argument("--depth_thresh", type=float, default=0.05)
    parser.add_argument("--min_size", type=int, default=50)
    parser.add_argument("--neighbor_thresh_v1", type=int, default=1)
    parser.add_argument("--neighbor_thresh_v4v5", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    # Merge params
    parser.add_argument("--merge_normal_thresh", type=float, default=10.0)
    parser.add_argument("--merge_depth_thresh", type=float, default=0.05)
    # 3D plane merge
    parser.add_argument("--merge_3d", action="store_true",
                        help="Enable 3D plane-equation merge (requires intrinsics)")
    parser.add_argument("--merge_3d_dist_thresh", type=float, default=0.05,
                        help="Max point-to-plane distance (meters) for 3D merge")
    args = parser.parse_args()

    variants = args.variants.split(",")
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Discover H5 files: supports both flat (scene/inference.h5) and
    # nested (scene/camera/inference.h5) layouts.
    h5_entries = []  # list of (label, h5_path)
    for scene in sorted(os.listdir(args.data_dir)):
        scene_path = os.path.join(args.data_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        flat_h5 = os.path.join(scene_path, "inference.h5")
        if os.path.isfile(flat_h5):
            h5_entries.append((scene, flat_h5))
        else:
            for cam in sorted(os.listdir(scene_path)):
                cam_h5 = os.path.join(scene_path, cam, "inference.h5")
                if os.path.isfile(cam_h5):
                    h5_entries.append((f"{scene}/{cam}", cam_h5))

    # Collect (label, h5_path, frame_idx) candidates with valid GT
    print("Scanning for frames with valid GT...")
    candidates = []
    for label, h5_path in h5_entries:
        with h5py.File(h5_path, "r") as f:
            gt_all = f["gt_planes"][:]
            for i in range(gt_all.shape[0]):
                if (gt_all[i] > 0).sum() > 0:
                    candidates.append((label, h5_path, i))
                    break  # one candidate per entry is enough for selection

    print(f"Found {len(candidates)} entries with valid GT")

    # Pick random entries, then random frame from each
    selected = random.sample(candidates, min(args.n_frames, len(candidates)))

    # Now pick a truly random frame (not just the first valid one) per entry
    frames = []
    for label, h5_path, _ in selected:
        with h5py.File(h5_path, "r") as f:
            gt_all = f["gt_planes"][:]
            valid_indices = [i for i in range(gt_all.shape[0]) if (gt_all[i] > 0).sum() > 0]
        frame_idx = random.choice(valid_indices)
        frames.append((label, h5_path, frame_idx))

    print(f"Selected {len(frames)} frames for visualization")
    print(f"Variants: {variants}")
    print("-" * 60)

    n_variants = len(variants)
    for fi, (scene_id, h5_path, frame_idx) in enumerate(frames):
        print(f"[{fi+1}/{len(frames)}] {scene_id} frame {frame_idx}")

        with h5py.File(h5_path, "r") as f:
            depth_frame = f["depth"][frame_idx]
            normals_frame = f["normals"][frame_idx]
            planarity_frame = f["planarity"][frame_idx]
            gt_planes_frame = f["gt_planes"][frame_idx].astype(np.int64)
            intrinsics_frame = f["intrinsics"][frame_idx] if "intrinsics" in f else None

        planarity_mask = (planarity_frame > args.planarity_thresh).astype(np.uint8)
        gt_valid_mask = gt_planes_frame > 0

        # --- Run all variants ---
        results = {}
        for v in variants:
            seg_raw = run_variant(v, planarity_mask, planarity_frame,
                                 normals_frame, depth_frame, args)
            seg_merged = merge_similar_planes(
                seg_raw, normals_frame, depth_frame,
                normal_thresh_deg=args.merge_normal_thresh,
                depth_thresh=args.merge_depth_thresh)

            # Compute SC
            pred_raw = pred_to_eval_labels(seg_raw)
            pred_merged = pred_to_eval_labels(seg_merged)
            sc_raw = segmentation_covering_fast(gt_planes_frame[gt_valid_mask],
                                                pred_raw[gt_valid_mask])
            sc_merged = segmentation_covering_fast(gt_planes_frame[gt_valid_mask],
                                                   pred_merged[gt_valid_mask])
            entry = {
                "raw": seg_raw, "merged": seg_merged,
                "sc_raw": sc_raw, "sc_merged": sc_merged,
            }

            # 3D merge
            if args.merge_3d and intrinsics_frame is not None:
                seg_merged_3d = merge_planes_3d(
                    seg_raw, depth_frame, intrinsics_frame,
                    normal_thresh_deg=args.merge_normal_thresh,
                    dist_thresh=args.merge_3d_dist_thresh)
                pred_merged_3d = pred_to_eval_labels(seg_merged_3d)
                sc_merged_3d = segmentation_covering_fast(
                    gt_planes_frame[gt_valid_mask], pred_merged_3d[gt_valid_mask])
                entry["merged_3d"] = seg_merged_3d
                entry["sc_merged_3d"] = sc_merged_3d

            results[v] = entry

        # --- Create figure ---
        has_3d = args.merge_3d and "merged_3d" in next(iter(results.values()))
        n_rows = 4 if has_3d else 3
        n_cols = n_variants + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))

        # Row 0, col 0: GT segmentation
        gt_rgb = label_to_rgb(gt_planes_frame)
        axes[0, 0].imshow(gt_rgb)
        n_gt_planes = len(np.unique(gt_planes_frame[gt_planes_frame > 0]))
        axes[0, 0].set_title(f"GT ({n_gt_planes} planes)", fontsize=9, fontweight="bold")
        axes[0, 0].axis("off")

        # Row 1, col 0: planarity heatmap
        axes[1, 0].imshow(planarity_frame, cmap="hot", vmin=0, vmax=1)
        axes[1, 0].set_title("Planarity", fontsize=9)
        axes[1, 0].axis("off")

        # Row 2, col 0: depth
        valid_depth = depth_frame[depth_frame > 0]
        vmin = np.percentile(valid_depth, 2) if len(valid_depth) > 0 else 0
        vmax = np.percentile(valid_depth, 98) if len(valid_depth) > 0 else 1
        axes[2, 0].imshow(depth_frame, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[2, 0].set_title("Depth", fontsize=9)
        axes[2, 0].axis("off")

        # Row 0: GT repeated for reference (faded)
        for ci, v in enumerate(variants, start=1):
            axes[0, ci].imshow(gt_rgb, alpha=0.4)
            axes[0, ci].set_title(f"GT ref", fontsize=8, color="gray")
            axes[0, ci].axis("off")

        # Row 1: variants without merge
        for ci, v in enumerate(variants, start=1):
            seg = results[v]["raw"]
            pred_eval = pred_to_eval_labels(seg)
            rgb = label_to_rgb(pred_eval)
            axes[1, ci].imshow(rgb)
            sc = results[v]["sc_raw"]
            n_segs = len(np.unique(seg[seg >= 0]))
            axes[1, ci].set_title(f"{v} ({n_segs}seg)\nSC={sc:.3f}", fontsize=9)
            axes[1, ci].axis("off")

        # Row 2: variants with merge
        for ci, v in enumerate(variants, start=1):
            seg = results[v]["merged"]
            pred_eval = pred_to_eval_labels(seg)
            rgb = label_to_rgb(pred_eval)
            axes[2, ci].imshow(rgb)
            sc = results[v]["sc_merged"]
            n_segs = len(np.unique(seg[seg >= 0]))
            delta = results[v]["sc_merged"] - results[v]["sc_raw"]
            sign = "+" if delta >= 0 else ""
            axes[2, ci].set_title(f"{v}+merge ({n_segs}seg)\nSC={sc:.3f} ({sign}{delta:.3f})",
                                  fontsize=9)
            axes[2, ci].axis("off")

        # Row 3 (optional): variants with 3D merge
        if has_3d:
            for ci, v in enumerate(variants, start=1):
                seg = results[v]["merged_3d"]
                pred_eval = pred_to_eval_labels(seg)
                rgb = label_to_rgb(pred_eval)
                axes[3, ci].imshow(rgb)
                sc = results[v]["sc_merged_3d"]
                n_segs = len(np.unique(seg[seg >= 0]))
                delta = sc - results[v]["sc_raw"]
                sign = "+" if delta >= 0 else ""
                axes[3, ci].set_title(
                    f"{v}+merge3d ({n_segs}seg)\nSC={sc:.3f} ({sign}{delta:.3f})",
                    fontsize=9)
                axes[3, ci].axis("off")
            axes[3, 0].axis("off")

        # Row labels
        axes[0, 0].set_ylabel("Ground Truth", fontsize=10, fontweight="bold")
        axes[1, 0].set_ylabel("No merge", fontsize=10, fontweight="bold")
        axes[2, 0].set_ylabel("Merged", fontsize=10, fontweight="bold")
        if has_3d:
            axes[3, 0].set_ylabel("Merge 3D", fontsize=10, fontweight="bold")

        fig.suptitle(f"{scene_id}  frame {frame_idx}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(args.out_dir, f"{scene_id.replace('/', '_')}_frame{frame_idx:04d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone! {len(frames)} figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
