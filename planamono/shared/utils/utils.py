import open3d as o3d
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
from tqdm import tqdm

from plyfile import PlyData
import numpy as np

from scipy import stats

import imageio
import torch
import torch.nn.functional as F

def save_label_image_sem(path, label_img):
    label_img = np.asarray(label_img)
    dtype = np.uint16
    
    max_val = label_img.max()
    if abs(max_val) > 65535:
        raise ValueError(f"Label value too large: {max_val}, use .npy instead")
    # Replace -1 (optional) to reserved 0
    label_img += 1 # unannotated -100, -1 no raycast, 0 wall =>  -99, 0, 1, ...
    label_img[label_img < 0] = 0
    imageio.imwrite(path, label_img.astype(dtype))

def save_label_image(path, label_img):
    label_img = np.asarray(label_img)

    # Decide dtype based on max value
    max_val = label_img.max()
    if max_val <= 255:
        dtype = np.uint8
    elif max_val <= 65535:
        dtype = np.uint16
    else:
        raise ValueError(f"Label value too large: {max_val}, use .npy instead")
    # Replace -1 (optional) to reserved 0
    label_img[label_img == -1] = 0

    imageio.imwrite(path, label_img.astype(dtype))


def load_label_image(path):
    label_img = imageio.imread(path)
    # Convert to int32 for downstream consistency
    return label_img.astype(np.int32)

def remap_semantic(semantic_img):
    unique_labels = np.unique(semantic_img)
    # print("Original labels:", unique_labels)
    label_remap_dict = {old_label: new_index for new_index, old_label in enumerate(unique_labels)}
    remapped_semantic_img = np.vectorize(label_remap_dict.get)(semantic_img)
    return remapped_semantic_img


def load_semantic_id_to_name_list(txt_path):
    """
    Loads semantic_classes.txt where each line is a class name.
    Returns a dict: {id: class_name}, with line number as id.
    """
    id_to_name = {}
    with open(txt_path, "r") as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            if class_name:
                id_to_name[idx] = class_name
    return id_to_name

def labels_to_names(label_array, id_to_name):
    # Vectorized map using np.vectorize
    map_func = np.vectorize(lambda x: id_to_name.get(x, "unknown"))
    return map_func(label_array)




def get_random_cmap(num_classes, seed=42):
    import numpy as np
    np.random.seed(seed)
    colors = np.random.rand(num_classes, 3)
    return mcolors.ListedColormap(colors)


def resize_moge_outputs_fast(
    depth: np.ndarray,
    normal: np.ndarray,
    planarity: np.ndarray,
    target_h: int,
    target_w: int,
    device: str = "cuda"
) -> tuple:
    """
    Fast GPU-accelerated resize of MoGe outputs.

    ~7-15x faster than separate cv2.resize calls by:
    - Keeping data on GPU
    - Batching all resizes into single interpolate call
    - Avoiding CPU-GPU transfers

    Args:
        depth: (H, W) depth map from MoGe
        normal: (3, H, W) or (H, W, 3) surface normals
        planarity: (H, W) planarity mask (will use nearest interpolation)
        target_h: Target height
        target_w: Target width
        device: Torch device

    Returns:
        depth_resized: (H, W) numpy array
        normal_resized: (3, H, W) numpy array
        planarity_resized: (H, W) numpy array
    """
    # Handle normal shape: convert to (3, H, W) if needed
    if normal.ndim == 3 and normal.shape[2] == 3:
        normal = normal.transpose(2, 0, 1)

    # Convert to torch tensors on GPU
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity), device=device, dtype=torch.float32
    )

    # Batch depth and normal together for single bilinear resize
    # depth: (1, 1, H, W), normal: (1, 3, H, W) -> combined: (1, 4, H, W)
    combined = torch.cat([
        depth_t.unsqueeze(0).unsqueeze(0),
        normal_t.unsqueeze(0)
    ], dim=1)

    # Single bilinear interpolation for depth + normal
    combined_resized = F.interpolate(
        combined,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )

    # Nearest interpolation for planarity (labels)
    planarity_resized = F.interpolate(
        planarity_t.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode='nearest'
    )

    # Extract results
    depth_resized = combined_resized[0, 0].cpu().numpy()
    normal_resized = combined_resized[0, 1:4].cpu().numpy()  # (3, H, W)
    planarity_resized = planarity_resized[0, 0].cpu().numpy()

    return depth_resized, normal_resized, planarity_resized


def resize_moge_outputs_fast_gpu(
    depth_t: torch.Tensor,
    normal_t: torch.Tensor,
    planarity_t: torch.Tensor,
    target_h: int,
    target_w: int
) -> tuple:
    """
    Fast GPU resize when inputs are already torch tensors on GPU.

    Even faster than resize_moge_outputs_fast as it skips numpy conversion.
    Use this when data is already on GPU from MoGe inference.

    Args:
        depth_t: (H, W) or (1, 1, H, W) depth tensor on GPU
        normal_t: (3, H, W) or (1, 3, H, W) normal tensor on GPU
        planarity_t: (H, W) or (1, 1, H, W) planarity tensor on GPU
        target_h: Target height
        target_w: Target width

    Returns:
        depth_resized: (H, W) tensor on GPU
        normal_resized: (3, H, W) tensor on GPU
        planarity_resized: (H, W) tensor on GPU
    """
    # Ensure 4D shape for interpolate
    if depth_t.dim() == 2:
        depth_t = depth_t.unsqueeze(0).unsqueeze(0)
    elif depth_t.dim() == 3:
        depth_t = depth_t.unsqueeze(0)

    if normal_t.dim() == 3:
        normal_t = normal_t.unsqueeze(0)

    if planarity_t.dim() == 2:
        planarity_t = planarity_t.unsqueeze(0).unsqueeze(0)
    elif planarity_t.dim() == 3:
        planarity_t = planarity_t.unsqueeze(0)

    # Batch depth and normal for single bilinear resize
    combined = torch.cat([depth_t, normal_t], dim=1)  # (1, 4, H, W)

    combined_resized = F.interpolate(
        combined,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )

    planarity_resized = F.interpolate(
        planarity_t.float(),
        size=(target_h, target_w),
        mode='nearest'
    )

    return (
        combined_resized[0, 0],       # depth: (H, W)
        combined_resized[0, 1:4],     # normal: (3, H, W)
        planarity_resized[0, 0]       # planarity: (H, W)
    )

