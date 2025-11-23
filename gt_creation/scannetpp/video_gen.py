import sys
sys.path.append('/cluster/home/aoezkan/planeseg/3d_vision/planarity_2_segmentation')
from visualize import visualize_top_components_v1

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2
from tqdm import tqdm
import argparse


def generate_video_from_plane_h5(scene_id, h5_path, save_video_path, fps=10):
    # === Load HDF5 ===
    with h5py.File(h5_path, "r") as f:
        planes = f["rendered_planes"][:]         # (N, H, W), dtype=uint16
        frame_ids = [fid.decode("utf-8") for fid in f["frame_ids"][:]]  # (N,)

    print(f"[INFO] Loaded {len(planes)} frames from {scene_id}")

    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
    writer = imageio.get_writer(save_video_path, fps=fps)

    # rgb_base_dir = f"/cluster/scratch/aoezkan/dataset/scannetpp/data/{scene_id}/iphone/rgb"
    rgb_base_dir = f"/cluster/project/cvg/Shared_datasets/scannet++/data/{scene_id}/iphone/rgb"

    N = len(planes)
    # N = 10
    for i in tqdm(range(N), desc=f"[{scene_id}] Writing video"):
        plane_img = planes[i]
        frame_id = frame_ids[i]

        # === Load and process RGB ===
        rgb_path = os.path.join(rgb_base_dir, f"{frame_id}.jpg")
        rgb_img = None
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (plane_img.shape[1], plane_img.shape[0]))
        else:
            print(f"[WARN] Missing RGB: {rgb_path}")
            rgb_img = np.zeros((plane_img.shape[0], plane_img.shape[1], 3), dtype=np.uint8)

        # === Visualize plane ===
        plane_vis = visualize_top_components_v1(plane_img, top_n=20, visualize=False)

        # === Plot RGB + Plane side-by-side ===
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(rgb_img)
        axs[0].set_title("RGB")
        axs[0].axis("off")

        im = axs[1].imshow(plane_vis, cmap="tab20", interpolation="nearest")
        axs[1].set_title("Plane Segmentation")
        axs[1].axis("off")

        fig.suptitle(f"Scene: {scene_id} | Frame: {frame_id}", fontsize=12)
        fig.tight_layout()

        # Render to image
        fig.canvas.draw()
        img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(img_np)

        plt.close(fig)

    writer.close()
    print(f"[DONE] Saved video: {save_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_id", type=str)
    args_cli = parser.parse_args()
    scene_id = args_cli.scene_id

    base_dir = "/cluster/scratch/aoezkan/dataset/scannetpp/"
    h5_path = os.path.join(base_dir, "plane_ours_gt", scene_id, "rendered_planes.h5")
    video_dir = os.path.join(base_dir, "visual", "plane_ours_gt")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{scene_id}.mp4")

    if not os.path.exists(h5_path):
        print(f"[ERROR] HDF5 not found: {h5_path}")
        exit(1)

    generate_video_from_plane_h5(scene_id, h5_path, video_path, fps=5)
