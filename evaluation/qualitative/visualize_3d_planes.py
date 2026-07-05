#!/usr/bin/env python3
"""
Render 3D reconstructed planar surfaces from a single RGB image.

Pipeline:
  RGB -> MoGe (depth/normals/planarity) -> plan2seg -> fit plane per segment
  -> project pixels onto fitted planes -> textured point cloud -> render rotated view

Usage:
    python evaluation/qualitative/visualize_3d_planes.py \
        --checkpoint /path/to/model.pt --image path/to/image.jpg
"""

import os
import sys
import argparse
import glob
import copy
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MoGe.moge.model.v2 import MoGeModel
from shared.segmentation.plan2seg import compute_vectorized_planar_segments
from shared.plane_fitting.planefit import refine_plane_least_squares


def load_model(checkpoint_path, device):
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.float().to(device).eval()
    return model


def project_points_onto_plane(pts, plane_params):
    """Project 3D points onto a fitted plane."""
    n = plane_params[:3].astype(np.float64)
    d = float(plane_params[3])
    pts64 = pts.astype(np.float64)
    dist = pts64 @ n + d
    projected = pts64 - dist[:, None] * n[None, :]
    return projected.astype(np.float32)


def rotation_matrix(rot_x_deg, rot_y_deg):
    """Create a rotation matrix from X and Y Euler angles (degrees)."""
    rx = np.deg2rad(rot_x_deg)
    ry = np.deg2rad(rot_y_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])
    return Ry @ Rx


def render_pointcloud_image(pts, colors, width, height, rot_x, rot_y,
                            bg_color=(255, 255, 255), point_radius=1):
    """Render a colored point cloud to an image via manual projection.

    Rotates the point cloud around its centroid, then does orthographic
    projection to produce a (H, W, 3) uint8 image with z-buffering.
    """
    # Center points
    centroid = np.median(pts, axis=0)
    pts_c = pts - centroid

    # Rotate
    R = rotation_matrix(rot_x, rot_y)
    pts_r = (R @ pts_c.T).T  # (N, 3)

    # Orthographic projection: x, y -> image, z -> depth (for z-buffer)
    x, y, z = pts_r[:, 0], pts_r[:, 1], pts_r[:, 2]

    # Scale to fit image with margin
    margin = 0.05
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])
    xrange = max(xmax - xmin, 1e-6)
    yrange = max(ymax - ymin, 1e-6)

    # Preserve aspect ratio
    scale = min(width * (1 - 2 * margin) / xrange,
                height * (1 - 2 * margin) / yrange)
    cx = width / 2.0
    cy = height / 2.0
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    px = ((x - xmid) * scale + cx).astype(np.int32)
    py = ((y - ymid) * scale + cy).astype(np.int32)

    # Z-buffer rendering
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    # Sort by depth (far to near) so near points overwrite far ones
    order = np.argsort(-z)
    px, py, z = px[order], py[order], z[order]
    colors_sorted = colors[order]

    for i in range(len(px)):
        xi, yi = px[i], py[i]
        if 0 <= xi < width and 0 <= yi < height:
            r = point_radius
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = xi + dx, yi + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if z[i] < zbuf[ny, nx]:
                            zbuf[ny, nx] = z[i]
                            img[ny, nx] = colors_sorted[i]

    return img


def process_image(model, image_path, output_dir, device, args):
    rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    stem = Path(image_path).stem

    resized = cv2.resize(rgb, (644, 476))
    tensor = torch.tensor(resized / 255.0, dtype=torch.float32).permute(2, 0, 1).to(device)

    with torch.no_grad():
        output = model.forward(tensor.unsqueeze(0), num_tokens=args.num_tokens)

    points = output["points"][0].cpu().numpy()
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

    # Plane segmentation
    planarity_binary = (planarity_r > args.planarity_threshold).astype(np.uint8)
    seg_device = "cpu" if device.type == "mps" else str(device)
    labels, num_planes = compute_vectorized_planar_segments(
        planarity_mask=planarity_binary,
        normal=normals_r,
        depth=depth_r,
        normal_threshold_rad=args.normal_threshold_rad,
        depth_threshold=args.depth_threshold,
        neighbor_match_count_thresh=args.neighbor_match_count,
        device=seg_device,
    )
    print(f"  {stem}: {num_planes} planes detected")

    # Fit plane per segment and project points onto fitted planes
    all_pts = []
    all_colors = []

    for pid in range(1, num_planes + 1):
        seg_mask = (labels == pid)
        finite_mask = np.isfinite(points_r).all(axis=-1) & (depth_r > 0)
        valid = seg_mask & finite_mask
        if valid.sum() < args.min_plane_px:
            continue

        pts_seg = points_r[valid].astype(np.float64)
        rgb_seg = rgb_r[valid]

        # Fit plane via SVD
        plane_params = refine_plane_least_squares(pts_seg)

        # Project points onto the fitted plane
        pts_projected = project_points_onto_plane(pts_seg, plane_params)

        # Filter any remaining non-finite values
        good = np.isfinite(pts_projected).all(axis=1)
        all_pts.append(pts_projected[good])
        all_colors.append(rgb_seg[good])

    if len(all_pts) == 0:
        print(f"    No valid planes for {stem}, skipping.")
        return

    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Remove outliers (points far from median)
    med = np.median(all_pts, axis=0)
    dists = np.linalg.norm(all_pts - med, axis=1)
    thresh = np.percentile(dists, 98)
    keep = dists < thresh
    all_pts = all_pts[keep]
    all_colors = all_colors[keep]

    print(f"    {len(all_pts)} projected points")

    for rot_y in args.rotations:
        img = render_pointcloud_image(
            all_pts, all_colors,
            width=out_w, height=out_h,
            rot_x=args.rot_x, rot_y=rot_y,
            point_radius=args.point_radius,
        )
        suffix = f"_rot{int(rot_y)}" if len(args.rotations) > 1 else ""
        out_path = os.path.join(output_dir, f"{stem}_3d_planes{suffix}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"    Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="3D visualization of reconstructed planar surfaces")
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint", type=str,
                   default=os.path.expanduser(
                       "~/Desktop/checkpoints/moge_HIRES_4datasets/model_epoch1.pt"))
    p.add_argument("--output_dir", type=str, default="3d_plane_vis")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--num_tokens", type=int, default=1024)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)

    p.add_argument("--planarity_threshold", type=float, default=0.5)
    p.add_argument("--normal_threshold_rad", type=float, default=0.087)
    p.add_argument("--depth_threshold", type=float, default=0.025)
    p.add_argument("--neighbor_match_count", type=int, default=8)
    p.add_argument("--min_plane_px", type=int, default=100)

    p.add_argument("--rotations", type=float, nargs="+", default=[30.0],
                   help="Y-axis rotation angles in degrees")
    p.add_argument("--rot_x", type=float, default=20.0)
    p.add_argument("--point_radius", type=int, default=1,
                   help="Radius of rendered points in pixels")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device != "mps" or torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, device)

    if os.path.isfile(args.image):
        image_paths = [args.image]
    elif os.path.isdir(args.image):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.image, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image, ext.upper())))
        image_paths.sort()
    else:
        raise FileNotFoundError(f"Not found: {args.image}")

    print(f"Processing {len(image_paths)} image(s) -> {args.output_dir}/")

    for path in image_paths:
        process_image(model, path, args.output_dir, device, args)

    print("Done.")


if __name__ == "__main__":
    main()
