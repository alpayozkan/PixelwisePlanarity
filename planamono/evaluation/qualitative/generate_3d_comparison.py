#!/usr/bin/env python3
"""
Generate 3D plane visualization comparisons: Ours vs ZeroPlane vs GT.

Reads a scene list (scene_id frame_id per line), runs MoGe inference for "Ours",
loads ZeroPlane H5 predictions, loads GT plane labels, and renders 2D segmentation
+ 3D plane-projected point clouds from multiple rotations.

Output structure:
    {output_root}/{scene_id}/
        rgb.png
        ours/   seg_2d.png  3d_rotY{angle}.png ...
        zeroplane/  seg_2d.png  3d_rotY{angle}.png ...
        gt/     seg_2d.png  3d_rotY{angle}.png ...
        compare_rotY{angle}.png   (side-by-side: GT | ZeroPlane | Ours)

Usage:
    python planamono/evaluation/qualitative/generate_3d_comparison.py \
        --scene_list scenes.txt \
        --output_root /cluster/scratch/ayavuz/3d_vis/scannetpp

scenes.txt format (one per line):
    0d2ee665be 0
    09c1414f1b 1350
"""

import os
import sys
import copy
import json
import argparse
import numpy as np
import h5py
import cv2
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from planamono.moge.moge.model.v2 import MoGeModel
from planamono.shared.segmentation.plan2seg import compute_vectorized_planar_segments_v5_relative


# ── Geometry helpers ──────────────────────────────────────────────────────────

def fit_plane_svd(pts):
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[-1]
    d = -float(n @ c)
    norm = np.linalg.norm(n)
    return np.array([*(n / norm), d / norm])


def project_onto_plane_ray(pts, p, max_displacement_factor=0.3):
    """Ray-plane intersection: project points along camera rays onto fitted plane.

    Clamps displacement to avoid overshooting when rays are nearly parallel
    to the plane (shallow viewing angles like tables/floors).
    """
    n = p[:3].astype(np.float64)
    d = float(p[3])
    rays = pts.astype(np.float64)
    denom = rays @ n
    valid = np.abs(denom) > 1e-8
    s = np.zeros(len(denom), dtype=np.float64)
    s[valid] = -d / denom[valid]
    projected = s[:, None] * rays
    bad = ~valid | (s <= 0)
    projected[bad] = rays[bad]

    # Clamp: if projected point moved too far from original, keep original
    disp = np.linalg.norm(projected - rays, axis=1)
    orig_dist = np.linalg.norm(rays, axis=1)
    too_far = disp > max_displacement_factor * (np.median(orig_dist) + 1e-8)
    projected[too_far] = rays[too_far]

    return projected


def rotation_matrix(rot_x_deg, rot_y_deg):
    rx, ry = np.deg2rad(rot_x_deg), np.deg2rad(rot_y_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    return Ry @ Rx


# ── Point cloud building ─────────────────────────────────────────────────────

def build_pointcloud(pts_3d, labels, num_planes, rgb_img, min_plane_px=100,
                     no_project=False):
    """Build (pts, colors_rgb) from 3D point map + plane labels."""
    z = pts_3d[:, :, 2]
    np.random.seed(42)
    pcolors = np.random.randint(50, 230, size=(num_planes + 1, 3), dtype=np.uint8)

    all_pts, all_colors = [], []
    for pid in range(1, num_planes + 1):
        valid = (labels == pid) & np.isfinite(z) & (z > 0)
        if valid.sum() < min_plane_px:
            continue
        seg_pts = pts_3d[valid]
        finite_mask = np.isfinite(seg_pts).all(axis=1)
        seg_pts = seg_pts[finite_mask].astype(np.float64)
        if len(seg_pts) < min_plane_px:
            continue

        if no_project:
            out_pts = seg_pts
        else:
            plane = fit_plane_svd(seg_pts)
            out_pts = project_onto_plane_ray(seg_pts, plane)

        good = np.isfinite(out_pts).all(axis=1)
        out_pts = out_pts[good].astype(np.float32)

        if rgb_img is not None:
            seg_rgb = rgb_img[valid][finite_mask][good]
        else:
            seg_rgb = np.tile(pcolors[pid], (len(out_pts), 1)).astype(np.uint8)

        all_pts.append(out_pts)
        all_colors.append(seg_rgb)

    if not all_pts:
        return None, None
    all_pts = np.concatenate(all_pts)
    all_colors = np.concatenate(all_colors)
    med = np.median(all_pts, axis=0)
    dists = np.linalg.norm(all_pts - med, axis=1)
    keep = dists < np.percentile(dists, 98)
    return all_pts[keep], all_colors[keep]


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_pointcloud_image(pts, colors, width, height, rot_x, rot_y,
                            bg_color=(255, 255, 255), point_radius=1):
    """Perspective projection with z-buffering."""
    centroid = np.median(pts, axis=0)
    R = rotation_matrix(rot_x, rot_y)
    pts_r = (R @ (pts - centroid).T).T
    x, y, z = pts_r[:, 0], pts_r[:, 1], pts_r[:, 2]

    # Shift z so all points are in front of virtual camera
    z_near = np.percentile(z, 1)
    z_cam = z - z_near + 0.1  # ensure z > 0
    z_cam = np.maximum(z_cam, 0.01)

    # Auto-compute focal length so scene fills ~80% of frame
    margin = 0.1
    z_med = np.median(z_cam)
    x_half = np.percentile(np.abs(x), 95)
    y_half = np.percentile(np.abs(y), 95)
    fx = (width / 2) * (1 - margin) * z_med / max(x_half, 1e-6)
    fy = (height / 2) * (1 - margin) * z_med / max(y_half, 1e-6)
    f = min(fx, fy)

    px = (f * x / z_cam + width / 2).astype(np.int32)
    py = (-f * y / z_cam + height / 2).astype(np.int32)

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    order = np.argsort(-z)
    px, py, z_sorted = px[order], py[order], z[order]
    cs = colors[order]

    for i in range(len(px)):
        xi, yi = px[i], py[i]
        if 0 <= xi < width and 0 <= yi < height:
            for dy in range(-point_radius, point_radius + 1):
                for dx in range(-point_radius, point_radius + 1):
                    nx_, ny_ = xi + dx, yi + dy
                    if 0 <= nx_ < width and 0 <= ny_ < height and z_sorted[i] < zbuf[ny_, nx_]:
                        zbuf[ny_, nx_] = z_sorted[i]
                        img[ny_, nx_] = cs[i]
    return img


# ── 2D segmentation visualization ────────────────────────────────────────────

def colorize_seg(labels, rgb=None, alpha=0.5):
    """Colorize plane segmentation. If rgb given, blend as overlay."""
    unique_ids = sorted([i for i in np.unique(labels) if i > 0])
    np.random.seed(42)
    cmap = {pid: np.random.randint(50, 230, 3).astype(np.uint8)
            for pid in unique_ids}

    seg_img = np.full((*labels.shape, 3), 210, dtype=np.uint8)
    for pid, color in cmap.items():
        seg_img[labels == pid] = color

    if rgb is not None:
        mask = labels > 0
        out = rgb.copy()
        out[mask] = (rgb[mask].astype(np.float32) * (1 - alpha) +
                     seg_img[mask].astype(np.float32) * alpha).astype(np.uint8)
        return out
    return seg_img


# ── Backprojection helpers ────────────────────────────────────────────────────

def backproject(depth, K, flip_y=True):
    """Depth + intrinsics → (H, W, 3) point map. flip_y for y-up convention."""
    H, W = depth.shape
    K = K.astype(np.float64)
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    z = depth.astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    if flip_y:
        y = -y
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def scale_K(K, from_wh, to_wh):
    """Scale intrinsics from one resolution to another."""
    K = K.copy()
    K[0, :] *= to_wh[0] / from_wh[0]
    K[1, :] *= to_wh[1] / from_wh[1]
    return K


def find_frame_in_h5(fids, frame_id):
    """Find frame_id in H5 frame_ids list, trying multiple formats.

    Tries: exact match, stripped leading zeros, with 'frame_' prefix.
    Returns index or None.
    """
    # Exact match
    if frame_id in fids:
        return fids.index(frame_id)
    # Strip leading zeros: "011325" → "11325"
    stripped = frame_id.lstrip("0") or "0"
    if stripped in fids:
        return fids.index(stripped)
    # With frame_ prefix
    prefixed = f"frame_{frame_id}"
    if prefixed in fids:
        return fids.index(prefixed)
    # Integer match
    try:
        fi = int(frame_id)
        for i, fid in enumerate(fids):
            try:
                if int(fid) == fi:
                    return i
            except ValueError:
                continue
    except ValueError:
        pass
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gt_data(scene_id, frame_id, args):
    """Load GT plane labels + depth + K → (pts_3d, labels, num_planes, rgb)."""
    gt_root = args.gt_root
    rgb_root = args.rgb_root

    # GT plane labels
    rendered_h5 = os.path.join(gt_root, scene_id, "rendered.h5")
    with h5py.File(rendered_h5, "r") as f:
        fids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["frame_ids"][:]]
        idx = find_frame_in_h5(fids, frame_id)
        if idx is None:
            print(f"  [GT] Frame {frame_id} not in rendered.h5 for {scene_id} (has: {fids[:3]}...)")
            return None
        labels = f["planes"][idx].astype(np.int32)

    # GT depth (mm → meters)
    depth_h5 = os.path.join(gt_root, scene_id, "rendered_depth.h5")
    with h5py.File(depth_h5, "r") as f:
        depth = f["depth"][idx].astype(np.float32) / 1000.0

    # Intrinsics from pose JSON
    pose_file = os.path.join(rgb_root, scene_id, "iphone", "pose_intrinsic_imu.json")
    with open(pose_file) as f:
        pose_data = json.load(f)
    # Try multiple key formats for pose lookup
    pose_key = None
    for candidate in [frame_id, f"frame_{frame_id}", frame_id.lstrip("0") or "0"]:
        if candidate in pose_data:
            pose_key = candidate
            break
    if pose_key is None:
        print(f"  [GT] No pose for frame {frame_id} in {scene_id}")
        return None
    K = np.array(pose_data[pose_key]["intrinsic"], dtype=np.float64)

    H, W = depth.shape
    # Scale K from original iPhone resolution to GT label resolution
    # K from JSON is at original capture resolution; GT labels may be smaller
    K_orig_w = K[0, 2] * 2
    K_orig_h = K[1, 2] * 2
    if K_orig_w > W * 1.5:
        K = scale_K(K, (K_orig_w, K_orig_h), (W, H))

    pts_3d = backproject(depth, K)
    num_planes = int(labels.max())

    # RGB
    rgb_path = os.path.join(rgb_root, scene_id, "iphone", "rgb", f"frame_{frame_id}.jpg")
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H))

    print(f"  [GT] {num_planes} planes, depth [{depth[depth>0].min():.2f}, {depth[depth>0].max():.2f}]m")
    return pts_3d, labels, num_planes, rgb


def load_zeroplane_data(scene_id, frame_id, args):
    """Load ZeroPlane H5 predictions → (pts_3d, labels, num_planes, rgb)."""
    zp_root = args.zeroplane_root

    planes_h5 = os.path.join(zp_root, "planes", scene_id, "planes.h5")
    depth_h5 = os.path.join(zp_root, "planes_depth", scene_id, "planes_depth.h5")
    intr_h5 = os.path.join(zp_root, "intrinsics", scene_id, "intrinsics.h5")

    if not all(os.path.exists(p) for p in [planes_h5, depth_h5, intr_h5]):
        print(f"  [ZP] Missing H5 files for {scene_id}")
        return None

    with h5py.File(planes_h5, "r") as f:
        fids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["frame_ids"][:]]
        idx = find_frame_in_h5(fids, frame_id)
        if idx is None:
            print(f"  [ZP] Frame {frame_id} not found in {scene_id} (has: {fids[:3]}...)")
            return None
        labels = f["planes"][idx].astype(np.int32)

    with h5py.File(depth_h5, "r") as f:
        depth = f["planes_depth"][idx].astype(np.float32)

    with h5py.File(intr_h5, "r") as f:
        K = f["intrinsics"][idx].astype(np.float64)

    H, W = depth.shape

    # Scale K from original resolution to depth resolution
    K_orig_w = K[0, 2] * 2
    K_orig_h = K[1, 2] * 2
    if K_orig_w > W * 1.5:
        K = scale_K(K, (K_orig_w, K_orig_h), (W, H))

    pts_3d = backproject(depth, K)
    num_planes = int(labels.max())

    # Load RGB
    rgb_path = os.path.join(args.rgb_root, scene_id, "iphone", "rgb", f"frame_{frame_id}.jpg")
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (W, H))

    vd = depth[depth > 0]
    print(f"  [ZP] {num_planes} planes, depth [{vd.min():.2f}, {vd.max():.2f}]m")
    return pts_3d, labels, num_planes, rgb


def load_moge_model(args, device):
    """Load MoGe model with planarity head."""
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.planarity_head = copy.deepcopy(model.normal_head).to(device)
    last_conv = None
    for name, module in model.planarity_head.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)
    name, old_conv = last_conv
    new_conv = nn.Conv2d(
        old_conv.in_channels, 1,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, dilation=old_conv.dilation,
        groups=old_conv.groups, bias=old_conv.bias is not None,
    )
    parent = model.planarity_head
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_conv)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.float().to(device).eval()


def run_moge_inference(model, rgb, device, args):
    """Run MoGe inference → (pts_3d, labels, num_planes, rgb_resized)."""
    resized = cv2.resize(rgb, (644, 476))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1).to(device)

    with torch.no_grad():
        output = model.forward(tensor.unsqueeze(0), num_tokens=args.num_tokens)

    points = output["points"][0].cpu().numpy()
    points[:, :, 1] *= -1  # y-down → y-up
    normals = output["normal"][0].cpu().numpy()
    planarity = output["planarity"][0].cpu().numpy()
    depth = points[:, :, 2]

    out_h, out_w = args.height, args.width
    depth_r = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    normals_r = cv2.resize(normals, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    norm_len = np.linalg.norm(normals_r, axis=-1, keepdims=True)
    normals_r = normals_r / (norm_len + 1e-8)
    planarity_r = cv2.resize(planarity, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    points_r = cv2.resize(points, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    rgb_r = cv2.resize(rgb, (out_w, out_h))

    planarity_binary = (planarity_r > args.planarity_threshold).astype(np.uint8)
    seg_device = "cpu" if device.type == "mps" else str(device)
    labels, num_planes = compute_vectorized_planar_segments_v5_relative(
        planarity_mask=planarity_binary,
        normal=normals_r,
        depth=depth_r,
        normal_threshold_rad=args.normal_threshold_rad,
        depth_threshold=args.depth_threshold,
        neighbor_match_count_thresh=args.neighbor_match_count,
        device=seg_device,
    )
    print(f"  [Ours] {num_planes} planes")
    return points_r, labels, num_planes, rgb_r


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_scene(scene_id, frame_id, model, device, args):
    """Process one scene: generate all outputs for GT, ZeroPlane, Ours."""
    print(f"\n{'='*60}")
    print(f"Scene: {scene_id}, Frame: {frame_id}")
    print(f"{'='*60}")

    scene_dir = os.path.join(args.output_root, scene_id)

    # Load RGB
    rgb_path = os.path.join(args.rgb_root, scene_id, "iphone", "rgb", f"frame_{frame_id}.jpg")
    if not os.path.exists(rgb_path):
        print(f"  RGB not found: {rgb_path}")
        return
    rgb_full = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    # Save RGB
    os.makedirs(scene_dir, exist_ok=True)
    rgb_save = cv2.resize(rgb_full, (args.width, args.height))
    cv2.imwrite(os.path.join(scene_dir, "rgb.png"),
                cv2.cvtColor(rgb_save, cv2.COLOR_RGB2BGR))

    # ── Process each method ──
    methods = {}

    # GT
    gt_data = load_gt_data(scene_id, frame_id, args)
    if gt_data is not None:
        methods["gt"] = gt_data

    # ZeroPlane
    zp_data = load_zeroplane_data(scene_id, frame_id, args)
    if zp_data is not None:
        methods["zeroplane"] = zp_data

    # Ours (MoGe)
    moge_data = run_moge_inference(model, rgb_full, device, args)
    if moge_data is not None:
        methods["ours"] = moge_data

    # ── Build point clouds and save per-method outputs ──
    pointclouds = {}
    for method_name, (pts_3d, labels, num_planes, rgb_img) in methods.items():
        method_dir = os.path.join(scene_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)

        # 2D segmentation
        H_seg, W_seg = labels.shape
        rgb_seg = cv2.resize(rgb_full, (W_seg, H_seg))
        seg_overlay = colorize_seg(labels, rgb=rgb_seg)
        seg_flat = colorize_seg(labels)
        cv2.imwrite(os.path.join(method_dir, "seg_2d.png"),
                    cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(method_dir, "seg_2d_flat.png"),
                    cv2.cvtColor(seg_flat, cv2.COLOR_RGB2BGR))

        # Build 3D point cloud with RGB texture
        pts, colors = build_pointcloud(pts_3d, labels, num_planes, rgb_img,
                                       min_plane_px=args.min_plane_px)
        if pts is None:
            print(f"  [{method_name}] No valid planes for 3D render")
            continue

        pointclouds[method_name] = (pts, colors)
        print(f"  [{method_name}] {len(pts):,} projected points")

        # Frontal render (rot_x=0, rot_y=0) — should match 2D camera view
        img_front = render_pointcloud_image(
            pts, colors,
            width=args.width, height=args.height,
            rot_x=0, rot_y=0,
            point_radius=args.point_radius,
        )
        cv2.imwrite(os.path.join(method_dir, "3d_frontal.png"),
                    cv2.cvtColor(img_front, cv2.COLOR_RGB2BGR))

        # 3D renders at each rotation
        for rot_y in args.rotations:
            img = render_pointcloud_image(
                pts, colors,
                width=args.width, height=args.height,
                rot_x=args.rot_x, rot_y=rot_y,
                point_radius=args.point_radius,
            )
            fname = f"3d_rotY{rot_y:.0f}.png"
            cv2.imwrite(os.path.join(method_dir, fname),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"  Saved to {scene_dir}/")


def parse_scene_list(path):
    """Parse scene list file. Each line: scene_id frame_id"""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
            elif len(parts) == 1:
                entries.append((parts[0], "0"))
    return entries


def main():
    p = argparse.ArgumentParser(description="Generate 3D plane comparison visualizations")

    # Input
    p.add_argument("--scene_list", type=str, required=True,
                   help="Text file with 'scene_id frame_id' per line")

    # Paths
    p.add_argument("--output_root", type=str,
                   default="/cluster/scratch/ayavuz/3d_vis/scannetpp")
    p.add_argument("--rgb_root", type=str,
                   default="/cluster/project/cvg/Shared_datasets/scannet++/data")
    p.add_argument("--gt_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/dataset/scannetpp")
    p.add_argument("--zeroplane_root", type=str,
                   default="/cluster/scratch/aoezkan/planeseg/scannetpp/inference/zeroplane_default_dust3r_released")
    p.add_argument("--checkpoint", type=str,
                   default="/cluster/scratch/ayavuz/moge_HIRES_4datasets/model_epoch2.pt")

    # MoGe inference
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_tokens", type=int, default=1600)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)

    # Segmentation
    p.add_argument("--planarity_threshold", type=float, default=0.5)
    p.add_argument("--normal_threshold_rad", type=float, default=0.087)
    p.add_argument("--depth_threshold", type=float, default=0.025)
    p.add_argument("--neighbor_match_count", type=int, default=8)
    p.add_argument("--min_plane_px", type=int, default=200)

    # Rendering
    p.add_argument("--rotations", type=float, nargs="+", default=[0, 1, 2, 3],
                   help="Y-axis rotation angles in degrees")
    p.add_argument("--rot_x", type=float, default=20.0)
    p.add_argument("--point_radius", type=int, default=1)

    args = p.parse_args()

    # Parse scene list
    entries = parse_scene_list(args.scene_list)
    print(f"Processing {len(entries)} scene-frame pairs")

    # Load MoGe model once
    device = torch.device(args.device)
    print(f"Loading MoGe model: {args.checkpoint}")
    model = load_moge_model(args, device)

    for scene_id, frame_id in entries:
        process_scene(scene_id, frame_id, model, device, args)

    print(f"\nDone. Output: {args.output_root}/")


if __name__ == "__main__":
    main()
