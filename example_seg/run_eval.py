"""
Self-contained evaluation script for plan2seg variants on ScanNet++ val inference H5 files.

Runs v1, v4, v5 segmentation on pre-exported inference data and computes
Segmentation Covering (SC) against ground truth plane labels.

Usage:
    python example_seg/run_eval.py --max_scenes 1 --max_frames 5   # quick test
    python example_seg/run_eval.py                                  # full run
"""

import argparse
import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

# Import plan2seg functions from same directory
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


# ---------------------------------------------------------------------------
# Inlined from metrics.py (avoids broken relative import of .planefit)
# ---------------------------------------------------------------------------
def segmentation_covering_fast(gt_seg: np.ndarray, pred_seg: np.ndarray) -> float:
    """SC metric: area-weighted best-IoU of GT segments vs predicted segments."""
    gt = gt_seg.ravel().astype(np.int64)
    pr = pred_seg.ravel().astype(np.int64)

    if gt.size == 0:
        return 0.0

    gt_labels, gt_inv = np.unique(gt, return_inverse=True)
    pr_labels, pr_inv = np.unique(pr, return_inverse=True)

    n_gt = gt_labels.size
    n_pr = pr_labels.size

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
    sc = (best_iou * gt_areas).sum() / total_area if total_area > 0 else 0.0
    return float(sc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pred_to_eval_labels(pred_seg: np.ndarray, gt_planes: np.ndarray) -> np.ndarray:
    """Map predicted labels for SC evaluation.

    plan2seg outputs 0-indexed labels (0, 1, 2, ...) for segments,
    filter_small_segments sets removed pixels to -1.
    GT uses 0 = non-planar background, >0 = plane id.

    For SC we shift pred labels: segment k -> k+1, background (-1) -> 0.
    Then we only evaluate where gt > 0.
    """
    out = np.zeros_like(pred_seg, dtype=np.int64)
    valid = pred_seg >= 0
    out[valid] = pred_seg[valid] + 1
    return out


def run_segmentation(variant, planarity_mask, planarity, normals, depth, args,
                     intrinsics=None):
    """Run a single segmentation variant and return filtered labels."""
    normal_rad = np.deg2rad(args.normal_thresh)

    if variant == "v1":
        labels, _ = compute_vectorized_planar_segments_v1(
            planarity_mask, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            neighbor_match_count_thresh=args.neighbor_thresh_v1,
        )
    elif variant == "v4":
        labels, _ = compute_vectorized_planar_segments_v4(
            planarity_mask, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device,
        )
    elif variant == "v5":
        labels, _ = compute_vectorized_planar_segments_v5(
            planarity_mask, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device,
        )
    elif variant == "v6":
        labels, _ = compute_vectorized_planar_segments_v6(
            planarity, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            weighted_thresh=args.v6_weighted_thresh,
            planarity_floor=args.v6_planarity_floor,
            device=args.device,
        )
    elif variant == "v7":
        labels, _ = compute_vectorized_planar_segments_v7(
            planarity, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            seed_thresh=args.v7_seed_thresh,
            grow_thresh=args.v7_grow_thresh,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            min_seed_frac=args.v7_min_seed_frac,
            device=args.device,
        )
    elif variant == "v8":
        labels, _ = compute_vectorized_planar_segments_v8(
            planarity, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            base_thresh=args.planarity_thresh,
            adapt_strength=args.v8_adapt_strength,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device,
        )
    elif variant == "v9":
        labels, _ = compute_vectorized_planar_segments_v9(
            planarity, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            joint_thresh=args.v9_joint_thresh,
            planarity_weight=args.v9_planarity_weight,
            planarity_floor=args.v9_planarity_floor,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device,
        )
    elif variant == "v10":
        labels, _ = compute_vectorized_planar_segments_v10(
            planarity, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            scale=args.v10_scale,
            sigma=args.v10_sigma,
            min_size=args.min_size,
            planarity_floor=args.v10_planarity_floor,
            planarity_scale=args.v10_planarity_scale,
        )
    elif variant == "v11":
        labels, _ = compute_vectorized_planar_segments_v11(
            planarity, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            seed_thresh=args.v11_seed_thresh,
            gradient_weight=args.v11_gradient_weight,
            planarity_floor=args.v11_planarity_floor,
            mask_thresh=args.v11_mask_thresh,
        )
    elif variant == "v12":
        labels, _ = compute_vectorized_planar_segments_v12(
            planarity, normals, depth,
            normal_threshold_rad=normal_rad,
            depth_threshold=args.depth_thresh,
            sigma_normal=args.v12_sigma_normal,
            sigma_depth=args.v12_sigma_depth,
            bilateral_thresh=args.v12_bilateral_thresh,
            n_iterations=args.v12_n_iterations,
            neighbor_match_count_thresh=args.neighbor_thresh_v4v5,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Ensure int32 for filter_small_segments (cc3d may return uint types)
    labels = labels.astype(np.int32)

    # All variants (scipy.ndimage.label and cc3d) return 0=background, 1+=segments.
    # Shift to -1=background, 0+=segments for filter_small_segments.
    labels = labels - 1

    filtered = filter_small_segments(labels, min_size=args.min_size)

    if args.merge:
        filtered = merge_similar_planes(
            filtered, normals, depth,
            normal_thresh_deg=args.merge_normal_thresh,
            depth_thresh=args.merge_depth_thresh,
        )

    if getattr(args, 'merge_3d', False) and intrinsics is not None:
        filtered = merge_planes_3d(
            filtered, depth, intrinsics,
            normal_thresh_deg=args.merge_normal_thresh,
            dist_thresh=args.merge_3d_dist_thresh,
        )

    return filtered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate plan2seg variants on ScanNet++ val inference")
    parser.add_argument("--data_dir", default="example_seg/scannetpp_val_inference_intrinsics",
                        help="Path to inference H5 directory")
    parser.add_argument("--planarity_thresh", type=float, default=0.5,
                        help="Binary planarity threshold")
    parser.add_argument("--normal_thresh", type=float, default=10.0,
                        help="Normal angle threshold in degrees")
    parser.add_argument("--depth_thresh", type=float, default=0.05,
                        help="Depth difference threshold in meters")
    parser.add_argument("--min_size", type=int, default=50,
                        help="Minimum segment size in pixels")
    parser.add_argument("--neighbor_thresh_v1", type=int, default=1,
                        help="Neighbor match count threshold for v1")
    parser.add_argument("--neighbor_thresh_v4v5", type=int, default=8,
                        help="Neighbor match count threshold for v4/v5")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Limit number of scenes (for quick testing)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit frames per scene (for quick testing)")
    # v6: probability-weighted connectivity
    parser.add_argument("--v6_weighted_thresh", type=float, default=3.0,
                        help="Weighted sum threshold for v6")
    parser.add_argument("--v6_planarity_floor", type=float, default=0.1,
                        help="Minimum planarity floor for v6")
    # v7: seed-and-grow
    parser.add_argument("--v7_seed_thresh", type=float, default=0.7,
                        help="High-confidence seed threshold for v7")
    parser.add_argument("--v7_grow_thresh", type=float, default=0.2,
                        help="Permissive grow threshold for v7")
    parser.add_argument("--v7_min_seed_frac", type=float, default=0.1,
                        help="Min fraction of seed pixels per segment for v7")
    # v8: adaptive threshold
    parser.add_argument("--v8_adapt_strength", type=float, default=0.3,
                        help="Adaptation strength for v8 (0=no adapt, 1=strong)")
    # v9: joint score
    parser.add_argument("--v9_joint_thresh", type=float, default=0.5,
                        help="Joint score threshold for v9")
    parser.add_argument("--v9_planarity_weight", type=float, default=0.5,
                        help="Weight for planarity in joint score (vs geometric)")
    parser.add_argument("--v9_planarity_floor", type=float, default=0.1,
                        help="Planarity floor for neighbor validity in v9")
    # v10: Felzenszwalb
    parser.add_argument("--v10_scale", type=float, default=12000.0,
                        help="Felzenszwalb scale (higher = larger segments)")
    parser.add_argument("--v10_sigma", type=float, default=0.5,
                        help="Gaussian smoothing for Felzenszwalb")
    parser.add_argument("--v10_planarity_floor", type=float, default=0.3,
                        help="Min avg planarity to keep a segment in v10")
    parser.add_argument("--v10_planarity_scale", type=float, default=5.0,
                        help="Weight for planarity channel in v10 features")
    # v11: Watershed
    parser.add_argument("--v11_seed_thresh", type=float, default=0.7,
                        help="Planarity threshold for watershed seeds")
    parser.add_argument("--v11_gradient_weight", type=float, default=2.0,
                        help="Weight for geometric gradients in elevation map")
    parser.add_argument("--v11_planarity_floor", type=float, default=0.4,
                        help="Min avg planarity to keep a segment in v11")
    parser.add_argument("--v11_mask_thresh", type=float, default=0.2,
                        help="Growth mask threshold for v11 (0=no mask)")
    # v12: Bilateral filter
    parser.add_argument("--v12_sigma_normal", type=float, default=0.1,
                        help="Normal similarity bandwidth for bilateral filter")
    parser.add_argument("--v12_sigma_depth", type=float, default=0.05,
                        help="Depth similarity bandwidth for bilateral filter")
    parser.add_argument("--v12_bilateral_thresh", type=float, default=0.5,
                        help="Threshold on bilaterally-filtered planarity")
    parser.add_argument("--v12_n_iterations", type=int, default=2,
                        help="Number of bilateral filtering iterations")
    # Post-processing: merge similar planes
    parser.add_argument("--merge", action="store_true",
                        help="Enable plane merging post-processing")
    parser.add_argument("--merge_normal_thresh", type=float, default=10.0,
                        help="Max normal angle (degrees) for merge")
    parser.add_argument("--merge_depth_thresh", type=float, default=0.05,
                        help="Max boundary depth diff (meters) for merge")
    # 3D plane merge (requires intrinsics in H5)
    parser.add_argument("--merge_3d", action="store_true",
                        help="Enable 3D plane-equation merge (requires intrinsics)")
    parser.add_argument("--merge_3d_dist_thresh", type=float, default=0.05,
                        help="Max point-to-plane distance (meters) for 3D merge")
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variants to run (default: all)")
    args = parser.parse_args()

    all_variants = ["v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"]
    variants = args.variants.split(",") if args.variants else all_variants

    # Discover H5 files: supports both flat (scene/inference.h5) and
    # nested (scene/camera/inference.h5) layouts.
    h5_entries = []  # list of (display_label, h5_path)
    for scene in sorted(os.listdir(args.data_dir)):
        scene_path = os.path.join(args.data_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        flat_h5 = os.path.join(scene_path, "inference.h5")
        if os.path.isfile(flat_h5):
            h5_entries.append((scene, flat_h5))
        else:
            # Check for camera subdirectories
            for cam in sorted(os.listdir(scene_path)):
                cam_h5 = os.path.join(scene_path, cam, "inference.h5")
                if os.path.isfile(cam_h5):
                    h5_entries.append((f"{scene}/{cam}", cam_h5))

    if args.max_scenes is not None:
        h5_entries = h5_entries[:args.max_scenes]

    print(f"Found {len(h5_entries)} H5 file(s)")
    print(f"Params: planarity={args.planarity_thresh}, normal={args.normal_thresh}°, "
          f"depth={args.depth_thresh}m, min_size={args.min_size}")
    print(f"Neighbor thresh: v1={args.neighbor_thresh_v1}, v4/v5={args.neighbor_thresh_v4v5}")
    print(f"Device: {args.device}")
    print("-" * 70)

    # Accumulate results: {variant: list of SC scores}
    all_scores = {v: [] for v in variants}
    scene_scores = {v: [] for v in variants}

    for label, h5_path in tqdm(h5_entries, desc="Scenes"):
        with h5py.File(h5_path, "r") as f:
            depth_all = f["depth"][:]          # (N, H, W)
            normals_all = f["normals"][:]      # (N, H, W, 3)
            planarity_all = f["planarity"][:]  # (N, H, W)
            gt_planes_all = f["gt_planes"][:]  # (N, H, W)
            intrinsics_all = f["intrinsics"][:] if "intrinsics" in f else None  # (N, 3, 3)

        n_frames = depth_all.shape[0]
        if args.max_frames is not None:
            n_frames = min(n_frames, args.max_frames)

        frame_scores = {v: [] for v in variants}

        for i in tqdm(range(n_frames), desc=f"  {label}", leave=False):
            depth = depth_all[i]
            normals = normals_all[i]
            planarity = planarity_all[i]
            gt_planes = gt_planes_all[i].astype(np.int64)

            # Binary planarity mask
            planarity_mask = (planarity > args.planarity_thresh).astype(np.uint8)

            # Skip frames with no GT planes
            gt_valid_mask = gt_planes > 0
            if gt_valid_mask.sum() == 0:
                continue

            intrinsics = intrinsics_all[i] if intrinsics_all is not None else None

            for variant in variants:
                pred_seg = run_segmentation(variant, planarity_mask, planarity, normals, depth, args,
                                            intrinsics=intrinsics)
                pred_eval = pred_to_eval_labels(pred_seg, gt_planes)

                # Evaluate only where GT has planar labels
                gt_masked = gt_planes[gt_valid_mask]
                pred_masked = pred_eval[gt_valid_mask]

                sc = segmentation_covering_fast(gt_masked, pred_masked)
                frame_scores[variant].append(sc)
                all_scores[variant].append(sc)

        # Per-scene summary
        for v in variants:
            if frame_scores[v]:
                mean_sc = np.mean(frame_scores[v])
                scene_scores[v].append(mean_sc)

        scene_summary = "  ".join(
            f"{v}: {np.mean(frame_scores[v]):.4f}" if frame_scores[v] else f"{v}: N/A"
            for v in variants
        )
        tqdm.write(f"  {label} ({n_frames} frames) — {scene_summary}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    for v in variants:
        if all_scores[v]:
            mean_sc = np.mean(all_scores[v])
            std_sc = np.std(all_scores[v])
            n = len(all_scores[v])
            scene_mean = np.mean(scene_scores[v]) if scene_scores[v] else 0.0
            print(f"  {v}: SC = {mean_sc:.4f} ± {std_sc:.4f}  "
                  f"(scene-avg: {scene_mean:.4f}, {n} frames)")
        else:
            print(f"  {v}: no valid frames")


if __name__ == "__main__":
    main()
