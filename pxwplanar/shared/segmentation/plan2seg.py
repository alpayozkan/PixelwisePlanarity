"""
Planar segmentation via region growing on normal and depth consistency.

`compute_vectorized_planar_segments` is the
single region-growing algorithm kept in this repo. It is GPU-accelerated
(PyTorch) and uses a Sobel-based normal-similarity gate combined with a
relative depth gate, followed by connected-component labeling.

Canonical parameters (used across inference and evaluation):
    normal_threshold_rad = np.deg2rad(5.0)
    depth_threshold      = 0.025   (relative: 2.5% of center depth)
    neighbor_match_count_thresh = 8
    planarity mask threshold upstream: probability > 0.3
"""

import numpy as np
from typing import Tuple

import torch
import torch.nn.functional as F
import cc3d


def compute_vectorized_planar_segments(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    Segment planar regions using Sobel-based normal similarity and a relative
    depth gate: |d_center - d_neighbor| < depth_threshold * d_center
    (depth_threshold is a fraction, e.g. 0.025 = 2.5% of center depth).

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Relative depth threshold as fraction of center depth
        neighbor_match_count_thresh: Minimum matching neighbors in 5x5 window (default 8)
        device: Torch device ("cuda" or "cpu")

    Returns:
        labels: (H,W) int array with segment IDs
        num_components: Number of segments found
    """
    H, W = planarity_mask.shape

    planarity_t = torch.as_tensor(np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool)
    normal_t = torch.as_tensor(np.ascontiguousarray(normal), device=device, dtype=torch.float32)
    depth_t = torch.as_tensor(np.ascontiguousarray(depth), device=device, dtype=torch.float32)

    # Sobel filters for normal gradient magnitude
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32)
    sobel_x_batch = sobel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    sobel_y_batch = sobel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()

    normal_batch = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_dx = F.conv2d(normal_batch, sobel_x_batch, padding=1, groups=3)
    normal_dy = F.conv2d(normal_batch, sobel_y_batch, padding=1, groups=3)
    normal_grad_mag = torch.sqrt((normal_dx ** 2 + normal_dy ** 2).sum(dim=1)).squeeze(0)

    grad_mag_threshold = torch.sqrt(
        torch.tensor(2.0, device=device) - 2 * torch.cos(torch.tensor(normal_threshold_rad, device=device))
    )
    normal_similar = (normal_grad_mag <= grad_mag_threshold)

    kernel_size = 5
    pad = kernel_size // 2
    center_idx = (kernel_size * kernel_size) // 2
    neighbor_indices = [i for i in range(kernel_size * kernel_size) if i != center_idx]

    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode='constant', value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size).view(25, H, W)

    mask_padded = F.pad(planarity_t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size).view(25, H, W).bool()

    normal_sim_padded = F.pad(normal_similar[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    normal_sim_patches = F.unfold(normal_sim_padded, kernel_size=kernel_size).view(25, H, W).bool()

    neighbor_depths = depth_patches[neighbor_indices]
    neighbor_masks = mask_patches[neighbor_indices]
    neighbor_normals = normal_sim_patches[neighbor_indices]

    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks

    # Relative depth check: |d_c - d_n| < threshold * d_c
    center_depth = depth_t.unsqueeze(0)
    depth_diff = torch.abs(center_depth - neighbor_depths)
    depth_close = depth_diff < (depth_threshold * center_depth)

    matches = valid_pair & neighbor_normals & depth_close
    connected = (matches.sum(dim=0) >= neighbor_match_count_thresh)

    labels = cc3d.connected_components(connected.cpu().numpy())
    num_components = int(labels.max())

    return labels, num_components


def filter_small_segments(
    segmentation: np.ndarray,
    min_size: int = 50
) -> np.ndarray:
    """
    Remove small segments and relabel remaining ones.

    Args:
        segmentation: (H,W) segment labels (>=0 for valid, <0 for background)
        min_size: Minimum number of pixels for a valid segment

    Returns:
        new_seg: (H,W) filtered and relabeled segmentation
    """
    seg = segmentation.copy()
    unique_labels = np.unique(seg[seg >= 0])
    new_seg = np.full_like(seg, -1)
    next_label = 0

    for label_val in unique_labels:
        mask = seg == label_val
        if np.sum(mask) < min_size:
            continue  # Skip small regions
        new_seg[mask] = next_label
        next_label += 1

    return new_seg
