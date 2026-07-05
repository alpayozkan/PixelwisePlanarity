#!/usr/bin/env python3
"""
Evaluation metrics and functions for plane segmentation.

Provides:
- segmentation_covering: Compute SC metric
- evaluate_planarity: Main evaluation function for various methods
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import pandas as pd
import h5py
import os

from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information
from tqdm import tqdm

from shared.plane_fitting import (
    backproject_v1 as backproject,
    fit_planes_per_label,
    mark_planes_below_threshold_as_outliers,
    compute_precision_recall,
    project_labels_to_image
)


def segmentation_covering(gt_mask, pred_mask, ignore_label=0):
    """
    Compute Segmentation Covering (SC) between ground-truth instance labels and predicted instance labels.

    SC is the size-weighted average (over ground-truth regions) of best-match IoU:
        SC = (1 / sum_i |G_i|) * sum_i |G_i| * max_j IoU(G_i, P_j)

    Args:
        gt_mask (np.ndarray): 2D array of integer ground-truth instance ids (H, W).
        pred_mask (np.ndarray): 2D array of integer predicted instance ids (H, W).
        ignore_label (int): label to ignore (default 0).

    Returns:
        float: SC score in [0, 1]. If there are no GT regions (except ignored), returns 0.0.
    """
    if gt_mask.shape != pred_mask.shape:
        raise ValueError("gt_mask and pred_mask must have the same shape")

    # Flatten and filter ignored pixels
    gt = gt_mask.ravel().astype(np.int64)
    pr = pred_mask.ravel().astype(np.int64)

    valid = gt != ignore_label
    if not np.any(valid):
        return 0.0

    gt = gt[valid]
    pr = pr[valid]

    # Remap labels to compact ranges
    gt_labels, gt_inv = np.unique(gt, return_inverse=True)
    pr_labels, pr_inv = np.unique(pr, return_inverse=True)

    n_gt = gt_labels.size
    n_pr = pr_labels.size

    # Build contingency matrix
    combined = gt_inv * n_pr + pr_inv
    counts = np.bincount(combined, minlength=n_gt * n_pr).astype(np.int64)
    contingency = counts.reshape((n_gt, n_pr))

    gt_areas = contingency.sum(axis=1)
    pr_areas = contingency.sum(axis=0)

    # Compute best IoU for each GT region
    best_iou = np.zeros(n_gt, dtype=float)
    for i in range(n_gt):
        inter = contingency[i, :]
        union = gt_areas[i] + pr_areas - inter
        valid_union = union > 0
        if not np.any(valid_union):
            best_iou[i] = 0.0
        else:
            ious = np.zeros_like(inter, dtype=float)
            ious[valid_union] = inter[valid_union] / union[valid_union]
            best_iou[i] = ious.max()

    # Weighted average by GT area
    total_area = gt_areas.sum()
    sc = (best_iou * gt_areas).sum() / total_area if total_area > 0 else 0.0
    return float(sc)


def get_plane_seg_baseline_from_h5(h5_path, frame_idx):
    """Load plane segmentation from baseline method HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        planes = f["planes"][:]  # (N, H, W)
        frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]

    # Detect format
    first_id = frame_ids[0]
    if first_id.startswith("frame_") and first_id.endswith("_0"):
        key = f"frame_{frame_idx:06d}_0"
    else:
        key = f"{frame_idx:06d}"

    lookup = {fid: i for i, fid in enumerate(frame_ids)}

    if key in lookup:
        return planes[lookup[key]]
    else:
        print(f"[WARN] Frame {key} not found in {h5_path}")
        return None


def moge_planarity_infer(inference_model, rgb_path, img_res):
    """Run MoGe planarity inference and return segmentation."""
    H, W = img_res
    res = inference_model.predict(rgb_path, num_tokens=1024, return_all_heads=True)

    depth = res['points'][:, :, 2]
    normal = res['normal']  # (H, W, 3)
    planarity = res['planarity_probability']

    # Resize to target resolution
    depth = cv2.resize(depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    normal = cv2.resize(normal.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    planarity = cv2.resize(planarity.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    # Compute segmentation (canonical region-growing parameters)
    from shared.segmentation import compute_vectorized_planar_segments
    from shared.utils.label_utils import remap_labels

    planarity_mask = (planarity > 0.3).astype(np.int16)
    normal_threshold_rad = np.deg2rad(5.0)

    labels, _ = compute_vectorized_planar_segments(
        planarity_mask, normal, depth,
        normal_threshold_rad, 0.025,
        neighbor_match_count_thresh=8
    )
    filtered_segmentation = labels.copy()
    filtered_segmentation, _ = remap_labels(filtered_segmentation)

    return filtered_segmentation


def pseudo_mono_infer(
    inference_model,
    rgb_path: str,
    *,
    tau_d_ratio: float = 0.05,
    tau_theta: float = 10.0,
    min_plane_px: int = 2000,
    ransac_n: int = 3,
    num_iters: int = 5000,
    max_planes: int = 1000,
    candidate_min_px: int = 30,
) -> np.ndarray:
    """
    Runs single-image plane segmentation:
    - MoGe infer (depth, normals, intrinsics, mask)
    - Build 3D points from MoGe normalized intrinsics
    - Open3D sequential RANSAC to get plane candidates
    - Global re-label with normal gate and post size-filter

    Returns:
        labels: (H,W) int32 with 0 = non-planar, 1..K = planes
    """
    import torch
    import open3d as o3d
    from PIL import Image

    def _denorm_K(K_norm: np.ndarray, W: int, H: int):
        fx = float(K_norm[0, 0] * W)
        fy = float(K_norm[1, 1] * H)
        cx = float(K_norm[0, 2] * W)
        cy = float(K_norm[1, 2] * H)
        return fx, fy, cx, cy

    def _depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        Z = depth.astype(np.float32)
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.stack([X, Y, Z], axis=-1).astype(np.float32)

    def _o3d_candidates(P: np.ndarray, valid_mask: np.ndarray):
        H, W, _ = P.shape
        flatP = P.reshape(-1, 3)
        idx_all = np.flatnonzero(valid_mask.reshape(-1))
        pts = flatP[idx_all]

        medZ = np.median(pts[:, 2][np.isfinite(pts[:, 2])])
        dist_thresh = float(tau_d_ratio * max(1e-6, medZ))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        cur2orig = idx_all.copy()
        cands = []

        for _ in range(max_planes):
            if len(cur2orig) < ransac_n:
                break
            model, inliers = pcd.segment_plane(
                distance_threshold=dist_thresh,
                ransac_n=ransac_n,
                num_iterations=num_iters,
            )
            inliers = np.asarray(inliers, dtype=np.int64)
            if inliers.size < candidate_min_px:
                break
            cands.append((np.asarray(model, np.float32), cur2orig[inliers]))
            keep = np.ones(len(cur2orig), dtype=bool)
            keep[inliers] = False
            pcd = pcd.select_by_index(inliers.tolist(), invert=True)
            cur2orig = cur2orig[keep]

        return cands, dist_thresh, idx_all, pts

    def _global_relabel(models: np.ndarray, pts_valid: np.ndarray, idx_all: np.ndarray,
                        normals: np.ndarray, dist_thresh: float, H: int, W: int) -> np.ndarray:
        if models.shape[0] == 0:
            return np.zeros((H, W), np.int32)

        n_stack = models[:, :3]
        d_stack = models[:, 3]
        R = np.abs(pts_valid @ n_stack.T + d_stack[None, :])

        flatN = normals.reshape(-1, 3)[idx_all]
        n_norm = np.linalg.norm(flatN, axis=1, keepdims=True)
        hasN = n_norm > 1e-6
        Nunit = np.zeros_like(flatN)
        if np.any(hasN):
            Nunit[hasN[:, 0]] = flatN[hasN[:, 0]] / n_norm[hasN[:, 0]]
        cos_th = float(np.cos(np.deg2rad(tau_theta)))
        gate = (np.abs(Nunit @ n_stack.T) >= cos_th) | (~hasN)

        Rm = np.where(gate, R, np.inf)
        assign = np.argmin(Rm, axis=1)
        best = Rm[np.arange(Rm.shape[0]), assign]
        ok = best < dist_thresh

        labels = np.zeros(H * W, np.int32)
        labels[idx_all[ok]] = assign[ok] + 1
        return labels.reshape(H, W)

    def _post_size_filter(labels: np.ndarray) -> np.ndarray:
        if labels.max() <= 0:
            return labels
        for pid in range(1, int(labels.max()) + 1):
            if (labels == pid).sum() < min_plane_px:
                labels[labels == pid] = 0
        ids = [pid for pid in range(1, int(labels.max()) + 1) if (labels == pid).sum() > 0]
        remap = {old: i + 1 for i, old in enumerate(ids)}
        flat = labels.reshape(-1)
        for old, new in remap.items():
            flat[flat == old] = new
        return flat.reshape(labels.shape).astype(np.int32)

    # Run MoGe inference
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    H, W, _ = rgb.shape

    t = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    device = next(inference_model.parameters()).device if hasattr(inference_model, "parameters") else "cpu"
    t = t.to(device)
    out = inference_model.infer(t[0])
    res = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}

    if ("depth" not in res) or ("intrinsics" not in res):
        raise RuntimeError("Model must return 'depth' and 'intrinsics' keys")

    depth = res["depth"].astype(np.float32)
    normals = res.get("normal", np.zeros((H, W, 3), np.float32)).astype(np.float32)
    valid_mask = res.get("mask", (depth > 0)).astype(bool)
    K_norm = res["intrinsics"].astype(np.float32)

    # Build points via MoGe intrinsics
    fx, fy, cx, cy = _denorm_K(K_norm, W, H)
    P = _depth_to_points(depth, fx, fy, cx, cy)

    # Open3D RANSAC candidates
    cands, dist_thresh, idx_all, pts_valid = _o3d_candidates(P, valid_mask)
    models = np.stack([m for (m, _) in cands], axis=0) if len(cands) > 0 else np.zeros((0, 4), np.float32)

    # Global re-label + post size filter
    labels = _global_relabel(models, pts_valid, idx_all, normals, dist_thresh, H, W)
    labels = _post_size_filter(labels)

    return labels


def mono_planarity_infer(model, rgb_path, img_res):
    """Run monoplane inference using pseudo_mono_infer."""
    H, W = img_res
    labels = pseudo_mono_infer(
        model,
        rgb_path,
        tau_d_ratio=0.05,
        tau_theta=10.0,
        min_plane_px=2000,
        ransac_n=3,
        num_iters=5000,
        max_planes=1000,
        candidate_min_px=30,
    )
    # Resize to target resolution if needed
    if labels.shape[0] != H or labels.shape[1] != W:
        labels = cv2.resize(labels.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return labels


def evaluate_planarity(
    val_loader,
    inference_model=None,
    tag="moge",
    img_res=(480, 640),
    rgb_root=None,
    baseline_root=None,
    frame_skip=25
):
    """
    Evaluate planarity predictions on val_loader and return metrics DataFrame.

    Args:
        val_loader: PyTorch DataLoader returning batches with keys:
                    "scene_id", "frame_idx", "image", "depth", "plane", "K", "c2w", "rgb_path"
        inference_model: optional MoGe model for 'moge' inference
        tag: One of ["gt", "moge", "planercnn", "zeroplane", "monoplane"]
        img_res: (height, width) tuple for output resolution
        rgb_root: Root directory for RGB images (for moge inference)
        baseline_root: Root directory for baseline method HDF5 files
        frame_skip: Process every Nth frame (default: 25)

    Returns:
        pd.DataFrame with per-frame metrics
    """
    # Get paths from environment if not provided
    if rgb_root is None:
        rgb_root = os.environ.get("SCANNETPP_RGB_ROOT", "/cluster/project/cvg/Shared_datasets/scannet++/data")
    if baseline_root is None:
        baseline_root = os.environ.get("BASELINE_ROOT", "/cluster/scratch/ohatipoglu/dataset/scannetpp")

    results = []
    device = 'cpu'
    thresholds = [0.01, 0.02, 0.05]  # in meters

    for batch in tqdm(val_loader):
        # frame_idx is a string frame id (e.g. "frame_000025") in ScanNetPPPlaneDataset
        fid = batch["frame_idx"][0]
        scene_id = batch["scene_id"][0]
        frame_idx = int(fid.split("_")[-1])

        if frame_idx % frame_skip != 0:
            continue  # Skip non-keyframes

        # === Load image data ===
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        gt_plane = batch["plane"].to(device)
        Ks = batch["K"]
        c2ws = batch["c2w"]

        rgb_path = batch["rgb_path"][0]

        # === Convert to numpy ===
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        depths_np = depths[0][0].cpu().numpy()
        gt_plane_np = gt_plane[0][0].cpu().numpy()
        K_np = Ks[0].cpu().numpy()
        c2w_np = c2ws[0].cpu().numpy()

        # === Get plane prediction ===
        if tag == "gt":
            plane_pred = gt_plane_np
        elif tag == "moge":
            assert inference_model is not None, "MoGe model must be provided for tag='moge'"
            plane_pred = moge_planarity_infer(inference_model, rgb_path, img_res)
        elif tag == "monoplane":
            assert inference_model is not None, "MoGe model must be provided for tag='monoplane'"
            plane_pred = mono_planarity_infer(inference_model.model, rgb_path, img_res)
        elif tag in ["planercnn", "zeroplane"]:
            base_dir = "PlaneRCNN" if tag == "planercnn" else "ZeroPlane"
            h5_path = os.path.join(baseline_root, base_dir, scene_id, "rendered_v2.h5")
            plane_pred = get_plane_seg_baseline_from_h5(h5_path, frame_idx=frame_idx)
            if plane_pred is None:
                print(f"[WARN] Missing prediction for {scene_id} frame {frame_idx}")
                continue
        else:
            raise ValueError(f"Unknown tag: {tag}")

        # === Evaluate prediction ===
        H, W = depths_np.shape[:2]
        pts_world, labels = backproject(depths_np, K_np, c2w_np, plane_pred)
        label_img = project_labels_to_image(pts_world, labels, K_np[:3, :3], c2w_np, H, W)

        metric_per_threshold = {}
        for thr in thresholds:
            results_planefit, plane_df = fit_planes_per_label(
                pts_world, labels,
                ignore_labels=(0,),
                distance_threshold=thr,
                ransac_n=3,
                num_iterations=2000,
                min_support=100
            )

            results_planefit, plane_df = mark_planes_below_threshold_as_outliers(
                results_planefit, plane_df, inlier_ratio_threshold=0.5
            )

            metric_res = compute_precision_recall(plane_df, total_scene_points=pts_world.shape[0])

            prec = float(metric_res['global_precision'])
            rec = float(metric_res['global_recall'])

            metric_per_threshold[f'prec@{int(thr*100)}cm'] = prec
            metric_per_threshold[f'rec@{int(thr*100)}cm'] = rec

        # === Compute clustering-based metrics ===
        labels_true = gt_plane_np.astype(np.int32)
        labels_pred = plane_pred.astype(np.int32)

        # Rand Index
        ri = rand_score(labels_true.flatten(), labels_pred.flatten())

        # VOI: variation_of_information
        H_split, H_merge = variation_of_information(labels_true, labels_pred)
        voi_total = H_split + H_merge

        # SC: Segmentation covering
        sc = segmentation_covering(labels_true.flatten(), labels_pred.flatten())

        results.append({
            "scene_id": scene_id,
            "frame_idx": frame_idx,
            "rgb_path": rgb_path,
            "rand_index": ri,
            "voi": voi_total,
            "sc": sc,
            **metric_per_threshold
        })

    return pd.DataFrame(results)
