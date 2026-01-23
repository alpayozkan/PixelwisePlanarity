#!/usr/bin/env python3
"""
Visualize Hypersim RGB, depth, semantics, and GT plane labels.
Saves visualizations as PNG files.

Usage:
    python visualize_hypersim_data.py --scene ai_001_001 --frame 0010 --output output/
    python visualize_hypersim_data.py --n-samples 10 --output output/
    python visualize_hypersim_data.py --scene ai_001_001 --all-frames --output output/
"""

import os
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Add planamono to path
sys.path.insert(0, '/cluster/home/aoezkan/planeseg/PixelwisePlanarity')
from planamono import paths


def load_hypersim_rgb(h5_path, percentile=90, target_max=0.8, gamma=2.2):
    """
    Load RGB from Hypersim HDR .color.hdf5 and apply tone mapping.

    Returns:
        np.ndarray: RGB image (H, W, 3) in [0,1] as float32
    """
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        hdr = f[keys[0]][:]  # (H, W, 3)

    brightness = hdr.mean(axis=2)
    scale_val = np.percentile(brightness, percentile)
    scale_val = max(scale_val, 1e-6)

    img = hdr * (target_max / scale_val)
    img = np.clip(img, 0, None)
    img = img ** (1.0 / gamma)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img


def load_hypersim_depth(h5_path):
    """Load depth from Hypersim .depth_meters.hdf5"""
    with h5py.File(h5_path, "r") as f:
        depth = f["dataset"][:].astype(np.float32)
    return depth


def load_hypersim_semantic(h5_path):
    """Load semantic labels from Hypersim .semantic.hdf5"""
    with h5py.File(h5_path, "r") as f:
        sem = f["dataset"][:].astype(np.int32)
    return sem


def load_plane_labels(h5_path, frame_id):
    """
    Load plane segmentation labels from rendered h5 file.

    Args:
        h5_path: Path to rendered_planes_cam_XX.h5
        frame_id: Frame ID as string (e.g., '0000')

    Returns:
        np.ndarray: Plane labels (H, W) as int32
    """
    with h5py.File(h5_path, "r") as f:
        frame_ids = [x.decode() if isinstance(x, bytes) else x for x in f["frame_ids"][:]]
        if frame_id not in frame_ids:
            raise ValueError(f"Frame {frame_id} not found in {h5_path}")
        idx = frame_ids.index(frame_id)
        planes = f["planes"][idx]
    return planes.astype(np.int32)


def create_label_colormap(n_labels):
    """Create a colormap for segmentation labels."""
    np.random.seed(42)
    cmap = np.random.rand(n_labels, 3)
    cmap[0] = [0, 0, 0]  # Background is black
    return cmap


def colorize_labels(labels, cmap=None):
    """Convert integer labels to RGB image."""
    n_labels = labels.max() + 1
    if cmap is None:
        cmap = create_label_colormap(max(n_labels, 256))
    return cmap[labels % len(cmap)]


def get_frame_paths(scene_id, cam_name, frame_id):
    """Get all file paths for a frame."""
    color_dir = os.path.join(paths.hypersim_merged_path, scene_id, "images", f"scene_{cam_name}_final_hdf5")
    geom_dir = os.path.join(paths.hypersim_merged_path, scene_id, "images", f"scene_{cam_name}_geometry_hdf5")

    return {
        "rgb": os.path.join(color_dir, f"frame.{frame_id}.color.hdf5"),
        "depth": os.path.join(geom_dir, f"frame.{frame_id}.depth_meters.hdf5"),
        "semantic": os.path.join(geom_dir, f"frame.{frame_id}.semantic.hdf5"),
        "plane": os.path.join(paths.hypersim_rendered_path, scene_id, f"rendered_planes_{cam_name}.h5"),
    }


def visualize_and_save_frame(scene_id, cam_name, frame_id, output_dir):
    """Visualize a single frame and save as PNG."""
    file_paths = get_frame_paths(scene_id, cam_name, frame_id)

    # Check if files exist
    for name, p in file_paths.items():
        if not os.path.exists(p):
            print(f"[SKIP] Missing {name}: {p}")
            return False

    try:
        rgb = load_hypersim_rgb(file_paths["rgb"])
        depth = load_hypersim_depth(file_paths["depth"])
        sem = load_hypersim_semantic(file_paths["semantic"])
        planes = load_plane_labels(file_paths["plane"], frame_id)
    except Exception as e:
        print(f"[ERROR] Loading {scene_id}/{cam_name}/{frame_id}: {e}")
        return False

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"RGB - {scene_id}/{cam_name}/frame.{frame_id}")
    axes[0, 0].axis("off")

    # Depth
    valid_depth = depth.copy()
    valid_depth[np.isinf(valid_depth)] = np.nan
    im_depth = axes[0, 1].imshow(valid_depth, cmap="turbo")
    axes[0, 1].set_title(f"Depth (meters) - range: [{np.nanmin(valid_depth):.2f}, {np.nanpercentile(valid_depth, 99):.2f}]")
    axes[0, 1].axis("off")
    plt.colorbar(im_depth, ax=axes[0, 1], fraction=0.046)

    # Semantics
    sem_colored = colorize_labels(sem)
    axes[1, 0].imshow(sem_colored)
    axes[1, 0].set_title(f"Semantics - {len(np.unique(sem))} classes")
    axes[1, 0].axis("off")

    # Plane labels
    plane_colored = colorize_labels(planes)
    axes[1, 1].imshow(plane_colored)
    axes[1, 1].set_title(f"GT Plane Labels - {len(np.unique(planes))} planes")
    axes[1, 1].axis("off")

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scene_id}_{cam_name}_{frame_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")
    return True


def visualize_multi_frame_grid(scene_id, cam_name, frame_ids, output_dir):
    """Visualize multiple frames in a grid and save as PNG."""
    n_frames = len(frame_ids)
    fig, axes = plt.subplots(n_frames, 4, figsize=(20, 4 * n_frames))

    if n_frames == 1:
        axes = [axes]

    for i, frame_id in enumerate(frame_ids):
        file_paths = get_frame_paths(scene_id, cam_name, frame_id)

        try:
            rgb = load_hypersim_rgb(file_paths["rgb"])
            depth = load_hypersim_depth(file_paths["depth"])
            sem = load_hypersim_semantic(file_paths["semantic"])
            planes = load_plane_labels(file_paths["plane"], frame_id)
        except Exception as e:
            print(f"[ERROR] Loading {scene_id}/{cam_name}/{frame_id}: {e}")
            continue

        # RGB
        axes[i][0].imshow(rgb)
        axes[i][0].set_title(f"{scene_id}/{frame_id}")
        axes[i][0].axis("off")

        # Depth
        valid_depth = depth.copy()
        valid_depth[np.isinf(valid_depth)] = np.nan
        axes[i][1].imshow(valid_depth, cmap="turbo")
        axes[i][1].set_title("Depth")
        axes[i][1].axis("off")

        # Semantics
        axes[i][2].imshow(colorize_labels(sem))
        axes[i][2].set_title(f"Sem ({len(np.unique(sem))} cls)")
        axes[i][2].axis("off")

        # Planes
        axes[i][3].imshow(colorize_labels(planes))
        axes[i][3].set_title(f"Planes ({len(np.unique(planes))})")
        axes[i][3].axis("off")

    # Add column titles
    col_titles = ["RGB", "Depth", "Semantics", "GT Planes"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{scene_id}_{cam_name}_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {out_path}")


def get_available_frames(scene_id, cam_name):
    """Get available frame IDs for a scene/camera."""
    plane_path = os.path.join(paths.hypersim_rendered_path, scene_id, f"rendered_planes_{cam_name}.h5")
    if not os.path.exists(plane_path):
        return []

    with h5py.File(plane_path, "r") as f:
        frame_ids = [x.decode() if isinstance(x, bytes) else x for x in f["frame_ids"][:]]
    return frame_ids


def get_available_cameras(scene_id):
    """Get available cameras for a scene."""
    rendered_dir = os.path.join(paths.hypersim_rendered_path, scene_id)
    if not os.path.exists(rendered_dir):
        return []

    cameras = []
    for f in os.listdir(rendered_dir):
        if f.startswith("rendered_planes_") and f.endswith(".h5"):
            cam_name = f.replace("rendered_planes_", "").replace(".h5", "")
            cameras.append(cam_name)
    return sorted(cameras)


def main():
    parser = argparse.ArgumentParser(description="Visualize Hypersim data")
    parser.add_argument("--scene", type=str, help="Scene ID (e.g., ai_001_001)")
    parser.add_argument("--cam", type=str, default="cam_00", help="Camera name (default: cam_00)")
    parser.add_argument("--frame", type=str, help="Frame ID (e.g., 0010)")
    parser.add_argument("--all-frames", action="store_true", help="Visualize all frames for a scene")
    parser.add_argument("--n-samples", type=int, help="Number of random samples to visualize")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--grid", action="store_true", help="Save as multi-frame grid (requires --scene)")
    parser.add_argument("--grid-frames", type=int, default=5, help="Number of frames in grid")
    args = parser.parse_args()

    # Get available scenes
    rendered_scenes = sorted(os.listdir(paths.hypersim_rendered_path))
    print(f"Found {len(rendered_scenes)} scenes with rendered plane labels")

    if args.scene and args.frame:
        # Single frame visualization
        visualize_and_save_frame(args.scene, args.cam, args.frame, args.output)

    elif args.scene and args.grid:
        # Multi-frame grid for a scene
        frame_ids = get_available_frames(args.scene, args.cam)
        if not frame_ids:
            print(f"[ERROR] No frames found for {args.scene}/{args.cam}")
            return

        # Sample evenly spaced frames
        indices = np.linspace(0, len(frame_ids) - 1, args.grid_frames, dtype=int)
        selected_frames = [frame_ids[i] for i in indices]
        print(f"Creating grid with frames: {selected_frames}")

        visualize_multi_frame_grid(args.scene, args.cam, selected_frames, args.output)

    elif args.scene and args.all_frames:
        # All frames for a scene
        frame_ids = get_available_frames(args.scene, args.cam)
        if not frame_ids:
            print(f"[ERROR] No frames found for {args.scene}/{args.cam}")
            return

        print(f"Visualizing {len(frame_ids)} frames for {args.scene}/{args.cam}")
        for frame_id in frame_ids:
            visualize_and_save_frame(args.scene, args.cam, frame_id, args.output)

    elif args.n_samples:
        # Random samples from different scenes
        np.random.seed(42)
        sample_scenes = np.random.choice(rendered_scenes, min(args.n_samples, len(rendered_scenes)), replace=False)

        for scene_id in sample_scenes:
            cameras = get_available_cameras(scene_id)
            if not cameras:
                continue
            cam_name = cameras[0]

            frame_ids = get_available_frames(scene_id, cam_name)
            if not frame_ids:
                continue

            frame_id = np.random.choice(frame_ids)
            visualize_and_save_frame(scene_id, cam_name, frame_id, args.output)

    else:
        # Default: show help and list some scenes
        parser.print_help()
        print(f"\nAvailable scenes (first 10): {rendered_scenes[:10]}")

        if rendered_scenes:
            sample_scene = rendered_scenes[0]
            cameras = get_available_cameras(sample_scene)
            if cameras:
                frame_ids = get_available_frames(sample_scene, cameras[0])
                print(f"\nExample: {sample_scene}/{cameras[0]} has {len(frame_ids)} frames")
                print(f"  Run: python {sys.argv[0]} --scene {sample_scene} --frame {frame_ids[0]} --output output/")


if __name__ == "__main__":
    main()
