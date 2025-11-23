import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import numpy as np
import pandas as pd
import time
import copy
import json
import imageio
import h5py
import argparse
import torch
import torch.nn.functional as F
import os

from sklearn.metrics import rand_score
from skimage.metrics import variation_of_information
from natsort import natsorted
from tqdm import tqdm
from PIL import Image

from shared.plane_fitting import backproject_v1 as backproject, fit_planes_per_label_v1, mark_planes_below_threshold_as_outliers, compute_precision_recall_v1, project_labels_to_image
from shared.utils import remap_labels



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

    # Remap labels to compact ranges to make bincount small
    # (This avoids huge arrays if labels are large integers)
    gt_labels, gt_inv = np.unique(gt, return_inverse=True)   # gt_labels[k] -> original id, gt_inv gives 0..n_gt-1
    pr_labels, pr_inv = np.unique(pr, return_inverse=True)   # pr_inv gives 0..n_pr-1

    n_gt = gt_labels.size
    n_pr = pr_labels.size

    # Build contingency via combined index: i = gt_index * n_pr + pr_index
    combined = gt_inv * n_pr + pr_inv
    counts = np.bincount(combined, minlength=n_gt * n_pr).astype(np.int64)
    # shape into contingency matrix (n_gt, n_pr)
    contingency = counts.reshape((n_gt, n_pr))

    # area of GT regions and predicted regions (only counted on valid pixels)
    gt_areas = contingency.sum(axis=1)    # shape (n_gt,)
    pr_areas = contingency.sum(axis=0)    # shape (n_pr,)

    # For each GT region, compute IoU with all predictions and take max
    # IoU = intersection / (gt_area + pr_area - intersection)
    # Avoid division by zero (pr_area==0 won't occur because pr_labels come from present pixels)
    best_iou = np.zeros(n_gt, dtype=float)
    for i in range(n_gt):
        inter = contingency[i, :]                  # intersections with all predicted regions
        union = gt_areas[i] + pr_areas - inter
        # mask union > 0 (should be always true)
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
    with h5py.File(h5_path, "r") as f:
        planes = f["planes"][:]  # (N, H, W)
        frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]  # (N,)
    
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


def evaluate_planarity(val_loader, inference_model=None, tag="moge", img_res=(480, 640)):
    """
    Evaluate planarity predictions on val_loader and return metrics DataFrame.

    Args:
        val_loader: PyTorch DataLoader returning batches with keys:
                    "scene_id", "frame_idx", "image", "depth", "plane", "intrinsic", "pose"
        inference_model: optional MoGe model for 'moge' inference
        tag: One of ["gt", "moge", "planercnn", "zeroplane"]

    Returns:
        pd.DataFrame with per-frame precision/recall at 1cm, 2cm, 5cm
    """

    results = []
    device = 'cpu'
    thresholds = [0.01, 0.02, 0.05]  # in meters

    for batch in tqdm(val_loader):
        frame_idx = batch["frame_idx"].item()
        scene_id = batch["scene_id"][0]

        # if frame_idx % 50 != 0:
        if frame_idx % 25 != 0:
            continue  # Skip non-keyframes

        # === Load image data ===
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        gt_plane = batch["plane"].to(device)
        Ks = batch["intrinsic"]
        c2ws = batch["pose"]

        rgb_path = os.path.join(
            "/cluster/project/cvg/Shared_datasets/scannet++/data",
            scene_id, "iphone", "rgb", f"frame_{frame_idx:06d}.jpg"
        )

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
            assert inference_model is not None, "MoGe model must be provided for tag='moge'"
            plane_pred = mono_planarity_infer_v1(inference_model.model, rgb_path, img_res)
        elif tag in ["planercnn", "zeroplane"]:
            base_dir = "PlaneRCNN" if tag == "planercnn" else "ZeroPlane"
            h5_path = f"/cluster/scratch/ohatipoglu/dataset/scannetpp/{base_dir}/{scene_id}/rendered_v2.h5"
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
            results_planefit, plane_df = fit_planes_per_label_v1(
                pts_world, labels,
                ignore_labels=(0,),
                distance_threshold=thr,
                ransac_n=3,
                num_iterations=2000,
                min_support=100
            )

            results_planefit, plane_df = mark_planes_below_threshold_as_outliers(results_planefit, plane_df, inlier_ratio_threshold=0.5)
            
            # metric_res = compute_precision_recall(plane_df, total_scene_points=pts_world.shape[0])
            metric_res = compute_precision_recall_v1(plane_df, total_scene_points=pts_world.shape[0])

            prec = float(metric_res['global_precision'])
            rec = float(metric_res['global_recall'])

            metric_per_threshold[f'prec@{int(thr*100)}cm'] = prec
            metric_per_threshold[f'rec@{int(thr*100)}cm'] = rec

        # === Compute clustering-based metrics ===
        labels_true = gt_plane_np.astype(np.int32)
        labels_pred = plane_pred.astype(np.int32)
        
        # Flatten for rand_index
        ri = rand_score(labels_true.flatten(), labels_pred.flatten())
        
        # VOI: variation_of_information returns (H_split, H_merge)
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
            # "voi_merge": H_merge,
            # "voi_split": H_split,

        # if frame_idx == 250:  # optional limit for debug
        #     break

    return pd.DataFrame(results)

