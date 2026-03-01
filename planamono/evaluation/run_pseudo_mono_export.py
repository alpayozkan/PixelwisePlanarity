#!/usr/bin/env python3
"""
Run pseudo_mono_infer (sequential RANSAC on MoGe depth) on test sets and save results.

Supports 4 datasets: scannetpp, hypersim, vkitti2, synthia.
Saves per-frame plane labels as uint16 PNG files.

Usage:
    python planamono/evaluation/run_pseudo_mono_export.py \
        --checkpoint <moge_ckpt.pt> \
        --output_dir /path/to/output \
        --dataset scannetpp

    # All datasets:
    python planamono/evaluation/run_pseudo_mono_export.py \
        --checkpoint <moge_ckpt.pt> \
        --output_dir /path/to/output \
        --dataset all
"""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from planamono.inference.planarity.moge_inference import MoGePlanarityInference


# ============================================================
# pseudo_mono_infer adapted for numpy RGB input
# ============================================================

def pseudo_mono_infer(
    moge_model,
    rgb_uint8: np.ndarray,
    *,
    tau_d_ratio: float = 0.05,
    tau_theta: float = 10.0,
    min_plane_px: int = 2000,
    ransac_n: int = 3,
    num_iters: int = 5000,
    max_planes: int = 1000,
    candidate_min_px: int = 30,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run sequential RANSAC on MoGe depth output.

    Args:
        moge_model: MoGeModel (the .model attribute of MoGePlanarityInference)
        rgb_uint8: (H, W, 3) uint8 RGB image

    Returns:
        labels: (H, W) int32 with 0 = non-planar, 1..K = planes
    """
    import open3d as o3d

    H, W = rgb_uint8.shape[:2]

    t = torch.tensor(rgb_uint8 / 255.0, dtype=torch.float32).permute(2, 0, 1).to(device)
    with torch.no_grad():
        out = moge_model.infer(t)
    res = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
           for k, v in out.items()}

    depth = res["depth"].astype(np.float32)
    normals = res.get("normal", np.zeros((H, W, 3), np.float32)).astype(np.float32)
    valid_mask = res.get("mask", (depth > 0)).astype(bool)
    K_norm = res["intrinsics"].astype(np.float32)

    fx = float(K_norm[0, 0] * W)
    fy = float(K_norm[1, 1] * H)
    cx = float(K_norm[0, 2] * W)
    cy = float(K_norm[1, 2] * H)

    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    P = np.stack([(u - cx) * depth / fx, (v - cy) * depth / fy, depth], axis=-1).astype(np.float32)

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
            distance_threshold=dist_thresh, ransac_n=ransac_n, num_iterations=num_iters)
        inliers = np.asarray(inliers, dtype=np.int64)
        if inliers.size < candidate_min_px:
            break
        cands.append((np.asarray(model, np.float32), cur2orig[inliers]))
        keep = np.ones(len(cur2orig), dtype=bool)
        keep[inliers] = False
        pcd = pcd.select_by_index(inliers.tolist(), invert=True)
        cur2orig = cur2orig[keep]

    if len(cands) == 0:
        return np.zeros((H, W), np.int32)

    models = np.stack([m for (m, _) in cands], axis=0)
    n_stack, d_stack = models[:, :3], models[:, 3]
    pts_valid = flatP[idx_all]
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
    labels = labels.reshape(H, W)

    # Post size filter + compact
    for pid in range(1, int(labels.max()) + 1):
        if (labels == pid).sum() < min_plane_px:
            labels[labels == pid] = 0
    ids = [pid for pid in range(1, int(labels.max()) + 1) if (labels == pid).sum() > 0]
    remap = {old: i + 1 for i, old in enumerate(ids)}
    flat = labels.reshape(-1)
    for old, new in remap.items():
        flat[flat == old] = new
    return flat.reshape(H, W).astype(np.int32)


# ============================================================
# Dataset iterators: yield (relative_path, rgb_uint8)
# ============================================================

def iter_scannetpp_test(args):
    split_file = os.path.join(args.splits_root, "scannetpp", "nvs_sem_test_with_planes.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene_id in scenes:
        gt_h5 = os.path.join(args.scannetpp_gt_root, scene_id, "rendered.h5")
        if not os.path.exists(gt_h5):
            continue
        with h5py.File(gt_h5, "r") as hf:
            frame_ids = [fid.decode() if isinstance(fid, bytes) else str(fid)
                         for fid in hf["frame_ids"][:]]
        for fid in frame_ids:
            rgb_path = os.path.join(args.scannetpp_rgb_root, scene_id, "iphone", "rgb", f"{fid}.jpg")
            if not os.path.exists(rgb_path):
                continue
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (args.width, args.height))
            yield f"{scene_id}/{fid}", rgb


def load_hypersim_hdr_rgb(h5_path, percentile=90, target_max=0.8, gamma=2.2):
    with h5py.File(h5_path, 'r') as f:
        hdr = f['dataset'][:].astype(np.float32)
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=1e4, neginf=0.0)
    hdr = np.clip(hdr, 0, 1e4)
    brightness = hdr.mean(axis=2)
    scale_val = np.nanpercentile(brightness, percentile)
    scale_val = max(scale_val, 1e-6) if np.isfinite(scale_val) else 1.0
    img = hdr * (target_max / scale_val)
    img = np.clip(img, 0, None) ** (1.0 / gamma)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def iter_hypersim_test(args):
    split_csv = os.path.join(args.splits_root, "hypersim",
                             "metadata_images_split_with_planes_filtered.csv")
    df = pd.read_csv(split_csv)
    test_df = df[df["split_partition_name"] == "test"]
    for _, row in test_df.iterrows():
        scene, cam, fid = row["scene_name"], row["camera_name"], int(row["frame_id"])
        rgb_path = os.path.join(
            args.hypersim_data_root, scene, "images",
            f"scene_{cam}_final_hdf5", f"frame.{fid:04d}.color.hdf5")
        if not os.path.exists(rgb_path):
            continue
        rgb = load_hypersim_hdr_rgb(rgb_path)
        rgb = cv2.resize(rgb, (args.width, args.height))
        yield f"{scene}/{cam}/{fid:04d}", rgb


def iter_vkitti2_test(args):
    split_file = os.path.join(args.splits_root, "vkitti2", "test.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene in scenes:
        h5_path = os.path.join(args.vkitti2_plane_root, scene, "clone", "scene_data.h5")
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, "r") as hf:
            n = hf["rgb"].shape[0]
            for i in range(n):
                rgb = hf["rgb"][i]
                rgb = cv2.resize(rgb, (args.width, args.height))
                yield f"{scene}/clone/{i:04d}", rgb


def iter_synthia_test(args):
    split_file = os.path.join(args.splits_root, "synthia", "test.txt")
    with open(split_file) as f:
        scenes = [l.strip() for l in f if l.strip()]
    for scene in scenes:
        h5_path = os.path.join(args.synthia_plane_root, "test", scene, "scene_data.h5")
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, "r") as hf:
            n = hf["rgb"].shape[0]
            for i in range(n):
                rgb = hf["rgb"][i]
                rgb = cv2.resize(rgb, (args.width, args.height))
                yield f"{scene}/{i:04d}", rgb


DATASET_ITERS = {
    "scannetpp": iter_scannetpp_test,
    "hypersim": iter_hypersim_test,
    "vkitti2": iter_vkitti2_test,
    "synthia": iter_synthia_test,
}


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Export pseudo_mono_infer results on test sets")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=str, default=None,
                   help="Fine-tuned MoGe planarity checkpoint (.pt)")
    g.add_argument("--use_original_moge", action="store_true",
                   help="Use original MoGe v2 from HuggingFace (no planarity head needed)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output root (default: /cluster/scratch/ayavuz/dataset/pseudo_planamono_{dataset})")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["scannetpp", "hypersim", "vkitti2", "synthia", "all"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)

    # RANSAC params
    p.add_argument("--tau_d_ratio", type=float, default=0.05)
    p.add_argument("--tau_theta", type=float, default=10.0)
    p.add_argument("--min_plane_px", type=int, default=2000)
    p.add_argument("--num_iters", type=int, default=5000)
    p.add_argument("--max_planes", type=int, default=1000)

    # Dataset paths
    p.add_argument("--splits_root", type=str,
                   default=str(Path(__file__).resolve().parents[1] / "splits"))
    p.add_argument("--scannetpp_rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--scannetpp_gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--hypersim_data_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/hypersim")
    p.add_argument("--vkitti2_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/vkitti2_planes")
    p.add_argument("--synthia_plane_root", type=str,
                   default="/cluster/scratch/ayavuz/dataset/synthia_planes")
    return p.parse_args()


def export_dataset(dataset_name, moge_model, args):
    if args.output_dir:
        ds_out = os.path.join(args.output_dir, dataset_name)
    else:
        ds_out = f"/cluster/scratch/ayavuz/dataset/pseudo_planamono_{dataset_name}"
    os.makedirs(ds_out, exist_ok=True)

    count = 0
    for rel_path, rgb in tqdm(DATASET_ITERS[dataset_name](args), desc=dataset_name):
        if args.max_frames is not None and count >= args.max_frames:
            break

        labels = pseudo_mono_infer(
            moge_model, rgb,
            tau_d_ratio=args.tau_d_ratio,
            tau_theta=args.tau_theta,
            min_plane_px=args.min_plane_px,
            num_iters=args.num_iters,
            max_planes=args.max_planes,
            device=args.device,
        )

        out_path = os.path.join(ds_out, f"{rel_path}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        imageio.imwrite(out_path, labels.astype(np.uint16))
        count += 1

    print(f"  {dataset_name}: saved {count} frames to {ds_out}")


def main():
    args = parse_args()
    datasets = list(DATASET_ITERS.keys()) if args.dataset == "all" else [args.dataset]

    # Load model
    if args.use_original_moge:
        from planamono.moge.moge.model import MoGeModel
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
        moge_model = moge_model.to(args.device).eval()
        ckpt_label = "original MoGe v2 (HuggingFace)"
    else:
        wrapper = MoGePlanarityInference(args.checkpoint, device=args.device)
        moge_model = wrapper.model
        ckpt_label = args.checkpoint

    out_label = args.output_dir or "/cluster/scratch/ayavuz/dataset/pseudo_planamono_{dataset}"

    print("Pseudo-Mono Export (Sequential RANSAC on MoGe Depth)")
    print("=" * 60)
    print(f"Model:      {ckpt_label}")
    print(f"Datasets:   {', '.join(datasets)}")
    print(f"Output:     {out_label}")
    print(f"Resolution: {args.height}x{args.width}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames} per dataset")
    print("=" * 60)

    for ds in datasets:
        print(f"\n--- {ds} ---")
        export_dataset(ds, moge_model, args)

    print("\nDone!")


if __name__ == "__main__":
    main()
