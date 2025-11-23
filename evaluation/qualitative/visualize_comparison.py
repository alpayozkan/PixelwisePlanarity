import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
from tqdm import tqdm
import glob

from PIL import Image
from natsort import natsorted
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from shared.utils import visualize_top_components_v1


def merge_plane_masks(seg_pred):
    """
    Convert multi-channel binary plane masks into a single-channel instance mask.

    Args:
        seg_pred (np.ndarray): shape (C, H, W), each channel is a binary mask for a plane.

    Returns:
        np.ndarray: shape (H, W), each pixel has the plane instance ID (0 = background).
    """
    C, H, W = seg_pred.shape
    instance_mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(C):
        mask = seg_pred[i] > 0  # convert to boolean if needed
        instance_mask[mask] = i + 1  # plane IDs start from 1

    return instance_mask


# Update for scannet: camera + depth separate, dont violate the resolution etc
#########################################
# fx = 886.810
# fy = 886.810
# cx = 512.0
# cy = 384.0

apply_billateral=True
threshold_planarity = 0.6
neighbor_match_count_thresh = 24
normal_threshold_deg = 10.0
depth_threshold = 0.05


root_results = '/cluster/scratch/aoezkan/results/scannet'
root_dir = '/cluster/scratch/aoezkan/dataset/scannet_new/scans'
root_gt_dir = '/cluster/scratch/aoezkan/dataset/planercnn/scannet_planeseg/'




scene_id_list = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
scene_id_list = natsorted(scene_id_list)
assert len(scene_id_list) > 0, "No scene IDs found in the root directory."

# print("Scene IDs:", scene_id_list)
# print(len(scene_id_list))

# scene_id = 'scene0000_00'
# scene_id = scene_id_list[0]
# scene_id_list = scene_id_list[:3]
# scene_id_list = scene_id_list[:5]
for scene_id in tqdm(scene_id_list):
    output_dir = os.path.join(root_results, 'seg_vis_moge')
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, "{}_baseline.mp4".format(scene_id))
    
    fps = 1
    frame_size = (1600, 400)  # width, height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    
    color_dir = os.path.join(root_dir, scene_id, 'color')
    image_list = natsorted(glob.glob(os.path.join(color_dir, '*.jpg')))

    plane_gt_dir = os.path.join(root_gt_dir, scene_id)
    plane_gt_list = natsorted(glob.glob(os.path.join(plane_gt_dir, '*.png')))

    # plane_ours = os.path.join(root_dir, scene_id, 'seg_pred', 'ourseg')
    plane_ours_dir = os.path.join(root_results, 'moge', 'seg_pred', scene_id)
    plane_ours_list = natsorted(glob.glob(os.path.join(plane_ours_dir, '*.npy')))

    plane_rcnn = os.path.join(root_dir, scene_id, 'seg_pred', 'planercnn')
    plane_rcnn_list = natsorted(glob.glob(os.path.join(plane_rcnn, '*.npy')))

    plane_planezero_dir = os.path.join(root_dir, scene_id, 'seg_pred', 'zeroplane')
    plane_planezero_list = natsorted(glob.glob(os.path.join(plane_planezero_dir, '*.npy')))
    
    # for idx in range(0, len(image_list), 50):
    for subindx, idx in enumerate(range(0, len(image_list), 50)):
        image_path = image_list[idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # plane_gt = Image.open(plane_gt_list[idx%50]).convert('L')
        plane_gt = Image.open(plane_gt_list[subindx]).convert('L')
        plane_gt_arr = np.array(plane_gt).astype(np.uint8)

        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        plane_ours = np.load(plane_ours_list[subindx])
        plane_rcnn = merge_plane_masks(np.load(plane_rcnn_list[subindx]))
        plane_zero = np.load(plane_planezero_list[subindx])

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # seg_save_path = os.path.join(seg_dir, f"{base_name}_seg_pred.npy")
        # np.save(seg_save_path, filtered_segmentation)
        # segvis_save_path = os.path.join(segvis_dir, f"{base_name}_seg_vis.png")
        

        fig, axs = plt.subplots(1, 5, figsize=(16, 4))
        fig.suptitle(f"Scannet Image: {scene_id}_{base_name}", fontsize=16)

        indx_fig = 0
        im0 = axs[indx_fig].imshow(img)
        axs[indx_fig].set_title("Image")
        axs[indx_fig].axis('off')
        # fig.colorbar(im0, ax=axs[indx_fig], orientation='horizontal')

        indx_fig+=1
        n = len(np.unique(plane_gt_arr))
        n1 = min(10, n)
        seg_top_gt = visualize_top_components_v1(
            plane_gt_arr, k=n1, return_colors=True
        )
        axs[indx_fig].imshow(seg_top_gt)
        axs[indx_fig].set_title("plane GT: Top-{}".format(n1))
        axs[indx_fig].axis('off')

        indx_fig+=1
        n = len(np.unique(plane_ours))
        n1 = min(10, n)
        seg_pred = visualize_top_components_v1(
            plane_ours, k=n1, return_colors=True
        )
        axs[indx_fig].imshow(seg_pred)
        axs[indx_fig].set_title("plane_ours: Top-{}".format(n1))
        axs[indx_fig].axis('off')
        
        indx_fig+=1
        n = len(np.unique(plane_rcnn))
        n1 = min(10, n)
        seg_pred = visualize_top_components_v1(
            plane_rcnn, k=n1, return_colors=True
        )
        axs[indx_fig].imshow(seg_pred)
        axs[indx_fig].set_title("plane_rcnn: Top-{}".format(n1))
        axs[indx_fig].axis('off')

        indx_fig+=1
        n = len(np.unique(plane_zero))
        n1 = min(10, n)
        seg_pred = visualize_top_components_v1(
            plane_zero, k=n1, return_colors=True
        )
        axs[indx_fig].imshow(seg_pred)
        axs[indx_fig].set_title("plane_zero: Top-{}".format(n1))
        axs[indx_fig].axis('off')
        

        plt.tight_layout()
        # plt.show()
        # fig.savefig(segvis_save_path)
        # fig.savefig(f"{base_name}_visualization.png")
        # plt.close(fig)
        # plt.show()
        time.sleep(1)
        # Draw figure to canvas
        canvas = FigureCanvas(fig)
        canvas.draw()
        fig_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        fig_img = fig_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)  # avoid memory leak

        # Resize if needed
        fig_img = cv2.resize(fig_img, frame_size)
        video_writer.write(cv2.cvtColor(fig_img, cv2.COLOR_RGB2BGR))
        # break
    video_writer.release()
    print("✅ Video saved to", output_video_path)
    # break