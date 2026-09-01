#!/usr/bin/env python3
"""
Interactive 3D viewer for plane segmentation results.

Mode 1 — From H5 files (ZeroPlane, etc.):
    python view_3d_planes.py /path/to/scene_dir
    python view_3d_planes.py /path/to/scene_dir --frame 2 --rgb

Mode 2 — From MoGe inference on an RGB image (our method):
    python view_3d_planes.py --moge /path/to/image.jpg
    python view_3d_planes.py --moge /path/to/image.jpg \
        --checkpoint /path/to/model.pt

Save rendered images at multiple rotations:
    python view_3d_planes.py /path/to/scene_dir --save --rotations 0 1 2 3
    python view_3d_planes.py --moge /path/to/image.jpg --save --rotations 1 2 3

Both methods side by side:
    python view_3d_planes.py /path/to/scene_dir --moge /path/to/image.jpg \
        --save --rgb
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pxwplanar.paths import planarity_model_path  # noqa: E402


def fit_plane_svd(pts):
    c = pts.mean(axis=0)
    U, S, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n = Vt[-1]
    d = -float(n @ c)
    norm = np.linalg.norm(n)
    return np.array([*(n / norm), d / norm])


def project_onto_plane_ray(pts, p, max_displacement_factor=0.3):
    """Project points onto a plane along viewing rays (ray-plane intersection).

    For each point (which represents a viewing ray from the origin),
    find where the ray intersects the plane n·x + d = 0:
        s = -d / (n · ray),  projected = s * ray

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
    rx = np.deg2rad(rot_x_deg)
    ry = np.deg2rad(rot_y_deg)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ]
    )
    return Ry @ Rx


def render_pointcloud_image(
    pts,
    colors,
    width,
    height,
    rot_x,
    rot_y,
    bg_color=(255, 255, 255),
    point_radius=1,
):
    """Render a colored point cloud to an image via orthographic
    projection with z-buffering."""
    centroid = np.median(pts, axis=0)
    pts_c = pts - centroid

    R = rotation_matrix(rot_x, rot_y)
    pts_r = (R @ pts_c.T).T

    x, y, z = pts_r[:, 0], pts_r[:, 1], pts_r[:, 2]

    margin = 0.05
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])
    xrange = max(xmax - xmin, 1e-6)
    yrange = max(ymax - ymin, 1e-6)

    scale = min(
        width * (1 - 2 * margin) / xrange, height * (1 - 2 * margin) / yrange
    )
    cx, cy = width / 2.0, height / 2.0
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    px = ((x - xmid) * scale + cx).astype(np.int32)
    py = (-(y - ymid) * scale + cy).astype(
        np.int32
    )  # negate y: world y-up → screen y-down

    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    order = np.argsort(-z)
    px, py, z = px[order], py[order], z[order]
    colors_sorted = colors[order]

    for i in range(len(px)):
        xi, yi = px[i], py[i]
        if 0 <= xi < width and 0 <= yi < height:
            r = point_radius
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx_, ny_ = xi + dx, yi + dy
                    if (
                        0 <= nx_ < width
                        and 0 <= ny_ < height
                        and z[i] < zbuf[ny_, nx_]
                    ):
                        zbuf[ny_, nx_] = z[i]
                        img[ny_, nx_] = colors_sorted[i]

    return img


def render_side_by_side(
    pts_list,
    colors_list,
    labels_list,
    width,
    height,
    rot_x,
    rot_y,
    bg_color=(255, 255, 255),
    point_radius=1,
):
    """Render multiple point clouds side by side.

    Each point cloud is centered independently (they may be in different
    coordinate systems), but a shared scale is used so relative sizes are
    comparable across panels.
    """
    n_views = len(pts_list)
    total_width = width * n_views
    R = rotation_matrix(rot_x, rot_y)
    margin = 0.05

    # Center each point cloud independently, rotate, compute per-view ranges
    rotated_list = []
    ranges = []
    for pts in pts_list:
        centroid = np.median(pts, axis=0)
        pts_r = (R @ (pts - centroid).T).T
        rotated_list.append(pts_r)
        x, y = pts_r[:, 0], pts_r[:, 1]
        xmin, xmax = np.percentile(x, [1, 99])
        ymin, ymax = np.percentile(y, [1, 99])
        ranges.append((xmin, xmax, ymin, ymax))

    # Shared scale: use the largest extent so both panels have same zoom
    max_xrange = max(max(r[1] - r[0], 1e-6) for r in ranges)
    max_yrange = max(max(r[3] - r[2], 1e-6) for r in ranges)
    scale = min(
        width * (1 - 2 * margin) / max_xrange,
        height * (1 - 2 * margin) / max_yrange,
    )

    img = np.full((height, total_width, 3), bg_color, dtype=np.uint8)
    zbuf = np.full((height, total_width), np.inf, dtype=np.float32)

    for vi, (pts_r, colors) in enumerate(
        zip(rotated_list, colors_list, strict=False)
    ):
        offset_x = vi * width
        x, y, z = pts_r[:, 0], pts_r[:, 1], pts_r[:, 2]

        # Center each view in its own panel
        xmin, xmax, ymin, ymax = ranges[vi]
        xmid = (xmin + xmax) / 2.0
        ymid = (ymin + ymax) / 2.0
        cx = offset_x + width / 2.0
        cy = height / 2.0

        px = ((x - xmid) * scale + cx).astype(np.int32)
        py = (-(y - ymid) * scale + cy).astype(
            np.int32
        )  # negate y: world y-up → screen y-down

        order = np.argsort(-z)
        px, py, z = px[order], py[order], z[order]
        colors_sorted = colors[order]

        x_lo, x_hi = offset_x, offset_x + width
        for i in range(len(px)):
            xi, yi = px[i], py[i]
            if x_lo <= xi < x_hi and 0 <= yi < height:
                r = point_radius
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx_, ny_ = xi + dx, yi + dy
                        if (
                            x_lo <= nx_ < x_hi
                            and 0 <= ny_ < height
                            and z[i] < zbuf[ny_, nx_]
                        ):
                            zbuf[ny_, nx_] = z[i]
                            img[ny_, nx_] = colors_sorted[i]

    # Draw separator lines
    for vi in range(1, n_views):
        x_sep = vi * width
        img[:, x_sep - 1 : x_sep + 1, :] = 180

    return img


def build_pointcloud(pts_3d, labels, num_planes, rgb_img, args):
    """Build point cloud arrays from 3D points and plane labels.

    Returns (pts, colors_rgb) or None.
    """
    H, W = labels.shape
    z = pts_3d[:, :, 2]

    np.random.seed(42)
    pcolors = np.random.randint(
        50, 230, size=(num_planes + 1, 3), dtype=np.uint8
    )

    all_pts, all_colors = [], []

    for pid in range(1, num_planes + 1):
        valid = (labels == pid) & np.isfinite(z) & (z > 0)
        if valid.sum() < args.min_plane_px:
            continue

        seg_pts = pts_3d[valid]
        finite_mask = np.isfinite(seg_pts).all(axis=1)
        seg_pts = seg_pts[finite_mask].astype(np.float64)

        if len(seg_pts) < args.min_plane_px:
            continue

        if args.no_project:
            out_pts = seg_pts
        else:
            plane = fit_plane_svd(seg_pts)
            out_pts = project_onto_plane_ray(seg_pts, plane)

        good = np.isfinite(out_pts).all(axis=1)
        out_pts = out_pts[good].astype(np.float32)

        if rgb_img is not None:
            seg_rgb = rgb_img[valid][finite_mask][good]
        else:
            color = pcolors[pid]
            seg_rgb = np.tile(color, (len(out_pts), 1)).astype(np.uint8)

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


def show_interactive(pts, colors):
    """Show point cloud interactively with trimesh."""
    rgba = np.column_stack([colors, np.full(len(pts), 255, dtype=np.uint8)])
    cloud = trimesh.PointCloud(vertices=pts, colors=rgba)
    cloud.show()


def save_renders(pts, colors, args, prefix="planes"):
    """Render and save images at each rotation angle."""
    os.makedirs(args.output_dir, exist_ok=True)
    mode = "raw" if args.no_project else "projected"
    for rot_y in args.rotations:
        img = render_pointcloud_image(
            pts,
            colors,
            width=args.width,
            height=args.height,
            rot_x=args.rot_x,
            rot_y=rot_y,
            point_radius=args.point_radius,
        )
        fname = f"{prefix}_{mode}_rotY{rot_y:.0f}_rotX{args.rot_x:.0f}.png"
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out_path}")


def save_side_by_side(pts_list, colors_list, label_names, args):
    """Render both methods side by side and save."""
    os.makedirs(args.output_dir, exist_ok=True)
    mode = "raw" if args.no_project else "projected"
    for rot_y in args.rotations:
        img = render_side_by_side(
            pts_list,
            colors_list,
            label_names,
            width=args.width,
            height=args.height,
            rot_x=args.rot_x,
            rot_y=rot_y,
            point_radius=args.point_radius,
        )
        # Add text labels
        for i, name in enumerate(label_names):
            x_pos = i * args.width + 10
            cv2.putText(
                img,
                name,
                (x_pos, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        fname = f"compare_{mode}_rotY{rot_y:.0f}_rotX{args.rot_x:.0f}.png"
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out_path}")


def load_h5_data(args):
    """Load from H5 files and return (pts_3d, labels, num_planes, rgb_img)."""
    with h5py.File(f"{args.scene_dir}/planes.h5", "r") as f:
        labels = f["planes"][args.frame]
        fids = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f["frame_ids"][:]
        ]
    with h5py.File(f"{args.scene_dir}/planes_depth.h5", "r") as f:
        depth = f["planes_depth"][args.frame]
    with h5py.File(f"{args.scene_dir}/intrinsics.h5", "r") as f:
        K = f["intrinsics"][args.frame].astype(np.float64)

    fid = fids[args.frame]
    num_planes = int(labels.max())
    vd = depth[depth > 0]
    print(f"[H5] Frame: {fid} (index {args.frame})")
    print(f"[H5] Planes: {num_planes}, Depth: [{vd.min():.3f}, {vd.max():.3f}]")

    H, W = depth.shape

    # Scale intrinsics from stored resolution to actual depth resolution.
    # ZeroPlane H5 stores K at the original capture resolution (e.g.
    # 1920x1440 for ScanNet++ iPhone).  Depth/labels are at a smaller
    # resolution (480x640).
    K_orig_w = K[0, 2] * 2  # approximate original width from cx
    K_orig_h = K[1, 2] * 2  # approximate original height from cy
    if K_orig_w > W * 1.5 or K_orig_h > H * 1.5:
        scale_x = W / K_orig_w
        scale_y = H / K_orig_h
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        print(f"[H5] Scaled K from ~{int(K_orig_w)}x{int(K_orig_h)} to {W}x{H}")

    u, v = np.meshgrid(
        np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64)
    )
    z = depth.astype(np.float64)
    # Flip y: OpenCV y-down → y-up so 3D mesh matches image orientation
    pts_3d = np.stack(
        [(u - K[0, 2]) * z / K[0, 0], -(v - K[1, 2]) * z / K[1, 1], z], axis=-1
    ).astype(np.float32)

    rgb_img = None
    if args.rgb:
        for ext in [".jpg", ".png"]:
            p = os.path.join(args.scene_dir, fid + ext)
            if os.path.exists(p):
                rgb_img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                rgb_img = cv2.resize(rgb_img, (W, H))
                print(f"[H5] Using RGB from {os.path.basename(p)}")
                break

    return pts_3d, labels, num_planes, rgb_img


def load_moge_data(args):
    """Run MoGe inference and return (pts_3d, labels, num_planes, rgb_img)."""
    import torch
    import torch.nn as nn

    from MoGe.moge.model.v2 import MoGeModel
    from pxwplanar.shared.segmentation.plan2seg import compute_planar_segments

    image_path = args.moge
    print(f"[MoGe] Image: {image_path}")
    rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    device = torch.device(
        args.device
        if args.device != "mps" or torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[MoGe] Loading model: {args.checkpoint}")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    model.planarity_head = copy.deepcopy(model.normal_head).to(device)
    last_conv = None
    for name, module in model.planarity_head.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)
    name, old_conv = last_conv
    new_conv = nn.Conv2d(
        old_conv.in_channels,
        1,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
    )
    parent = model.planarity_head
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_conv)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.float().to(device).eval()

    resized = cv2.resize(rgb, (644, 476))
    tensor = (
        torch.tensor(resized / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .to(device)
    )

    with torch.no_grad():
        output = model.forward(tensor.unsqueeze(0), num_tokens=args.num_tokens)

    points = output["points"][0].cpu().numpy()
    # Flip y: MoGe y-down → y-up so 3D mesh matches image orientation
    points[:, :, 1] *= -1
    normals = output["normal"][0].cpu().numpy()
    planarity = output["planarity"][0].cpu().numpy()
    depth = points[:, :, 2]

    out_h, out_w = args.height, args.width
    depth_r = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    normals_r = cv2.resize(
        normals, (out_w, out_h), interpolation=cv2.INTER_LINEAR
    )
    norm_len = np.linalg.norm(normals_r, axis=-1, keepdims=True)
    normals_r = normals_r / (norm_len + 1e-8)
    planarity_r = cv2.resize(
        planarity, (out_w, out_h), interpolation=cv2.INTER_LINEAR
    )
    points_r = cv2.resize(
        points, (out_w, out_h), interpolation=cv2.INTER_LINEAR
    )
    rgb_r = cv2.resize(rgb, (out_w, out_h))

    planarity_binary = (planarity_r > args.planarity_threshold).astype(np.uint8)
    seg_device = "cpu" if device.type == "mps" else str(device)
    labels, num_planes = compute_planar_segments(
        planarity_mask=planarity_binary,
        normal=normals_r,
        depth=depth_r,
        normal_threshold_rad=args.normal_threshold_rad,
        depth_threshold=args.depth_threshold,
        neighbor_match_count_thresh=args.neighbor_match_count,
        device=seg_device,
    )
    print(f"[MoGe] Detected {num_planes} planes")

    rgb_img = rgb_r if args.rgb else None
    return points_r, labels, num_planes, rgb_img


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D plane viewer")

    # Mode 1: H5 files
    parser.add_argument(
        "scene_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directory with planes.h5, planes_depth.h5, intrinsics.h5",
    )
    parser.add_argument("--frame", type=int, default=0)

    # Mode 2: MoGe inference
    parser.add_argument(
        "--moge",
        type=str,
        default=None,
        help="Path to RGB image for MoGe inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=planarity_model_path,
        help="4-head MoGe checkpoint; default: paths.planarity_model_path",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_tokens", type=int, default=1024)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--planarity_threshold", type=float, default=0.3)
    parser.add_argument("--normal_threshold_rad", type=float, default=0.087)
    parser.add_argument("--depth_threshold", type=float, default=0.025)
    parser.add_argument("--neighbor_match_count", type=int, default=8)

    # Shared
    parser.add_argument("--min_plane_px", type=int, default=200)
    parser.add_argument("--no_project", action="store_true")
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Use RGB texture instead of random colors",
    )

    # Save mode
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save rendered images instead of interactive view",
    )
    parser.add_argument(
        "--rotations",
        type=float,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Y-axis rotation angles in degrees (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--rot_x",
        type=float,
        default=20.0,
        help="X-axis rotation angle in degrees",
    )
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="3d_renders")

    args = parser.parse_args()

    have_h5 = args.scene_dir is not None
    have_moge = args.moge is not None

    if not have_h5 and not have_moge:
        parser.error("Provide either scene_dir or --moge <image_path>")

    # Both methods: side-by-side comparison
    if have_h5 and have_moge:
        pts_3d_h5, labels_h5, nplanes_h5, rgb_h5 = load_h5_data(args)
        pts_h5, colors_h5 = build_pointcloud(
            pts_3d_h5, labels_h5, nplanes_h5, rgb_h5, args
        )

        pts_3d_moge, labels_moge, nplanes_moge, rgb_moge = load_moge_data(args)
        pts_moge, colors_moge = build_pointcloud(
            pts_3d_moge, labels_moge, nplanes_moge, rgb_moge, args
        )

        if pts_h5 is None or pts_moge is None:
            print("One method produced no valid planes.")
            return

        print(f"H5: {len(pts_h5):,} pts, MoGe: {len(pts_moge):,} pts")

        if args.save:
            save_side_by_side(
                [pts_h5, pts_moge],
                [colors_h5, colors_moge],
                ["ZeroPlane", "Ours"],
                args,
            )
        else:
            show_interactive(
                np.concatenate([pts_h5, pts_moge]),
                np.concatenate([colors_h5, colors_moge]),
            )
        return

    # Single method
    if have_h5:
        pts_3d, labels, num_planes, rgb_img = load_h5_data(args)
        prefix = "h5"
    else:
        pts_3d, labels, num_planes, rgb_img = load_moge_data(args)
        prefix = "moge"

    pts, colors = build_pointcloud(pts_3d, labels, num_planes, rgb_img, args)
    if pts is None:
        print("No valid planes found.")
        return

    mode = "raw depth" if args.no_project else "plane-projected"
    print(f"{len(pts):,} points ({mode})")

    if args.save:
        save_renders(pts, colors, args, prefix=prefix)
    else:
        show_interactive(pts, colors)


if __name__ == "__main__":
    main()
