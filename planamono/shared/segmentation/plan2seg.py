"""
Planar segmentation algorithms using normal and depth consistency.

This module provides vectorized algorithms for segmenting planar regions
based on surface normal similarity and depth proximity.
"""

import numpy as np
from scipy.ndimage import label
from typing import Tuple

import torch
import torch.nn.functional as F
import cc3d


def compute_vectorized_planar_segments_v1(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 1
) -> Tuple[np.ndarray, int]:
    """
    Compute planar segments using 8-connected neighbor matching.

    This is the recommended version with neighbor match count thresholding
    for robustness against noise.

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Depth difference threshold in meters
        neighbor_match_count_thresh: Minimum number of matching neighbors (default 1)

    Returns:
        labeled: (H,W) int array with segment IDs (0-indexed)
        num_components: Number of segments found
    """
    H, W = planarity_mask.shape
    mask = planarity_mask.astype(bool)

    # Pad arrays for neighbor access
    pad = ((1, 1), (1, 1))
    padded_mask = np.pad(mask, pad, constant_values=0)
    padded_depth = np.pad(depth, pad, constant_values=0)
    padded_normal = np.pad(normal, pad + ((0, 0),), mode='constant')

    # 8-connected neighbor offsets
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

    def get_shifted(arr, shift):
        dy, dx = shift
        return arr[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]

    # Stack shifted versions of neighbors
    neighbor_masks = np.stack([get_shifted(padded_mask, s) for s in shifts], axis=2)  # (H, W, 8)
    neighbor_depths = np.stack([get_shifted(padded_depth, s) for s in shifts], axis=2)  # (H, W, 8)
    neighbor_normals = np.stack([get_shifted(padded_normal, s) for s in shifts], axis=3)  # (H, W, 3, 8)

    # Prepare center values
    center_depth = depth[:, :, None]  # (H, W, 1)
    center_normal = normal[:, :, :, None]  # (H, W, 3, 1)

    # Valid planar neighbor pairs
    valid_pair = mask[:, :, None] & neighbor_masks

    # Normal similarity check (angular difference)
    dot = np.sum(center_normal * neighbor_normals, axis=2)  # (H, W, 8)
    norm_center = np.linalg.norm(center_normal, axis=2, keepdims=True)  # (H, W, 1, 1)
    norm_center = norm_center[..., 0]  # (H, W, 1)
    norm_neighbor = np.linalg.norm(neighbor_normals, axis=2)  # (H, W, 8)

    cos_angle = np.clip(dot / (norm_center * norm_neighbor + 1e-8), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    normal_similar = angle < normal_threshold_rad  # (H, W, 8)

    # Depth closeness check
    depth_diff = np.abs(center_depth - neighbor_depths)
    depth_close = depth_diff < depth_threshold  # (H, W, 8)

    # Count matching neighbors
    neighbor_match_count = np.sum(valid_pair & normal_similar & depth_close, axis=2)
    connected = neighbor_match_count >= neighbor_match_count_thresh  # (H, W)

    # Connected component labeling with 8-connectivity
    structure = np.ones((3, 3), dtype=bool)
    labeled, num_components = label(connected & mask, structure=structure)

    return labeled, num_components


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


def compute_vectorized_planar_segments_v4(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    GPU-accelerated planar segmentation using 5x5 neighborhood.

    Uses Sobel filters for gradient computation and GPU tensors for
    faster processing on large images.

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Depth difference threshold in meters
        neighbor_match_count_thresh: Minimum matching neighbors in 5x5 window (default 8)
        device: Torch device ("cuda" or "cpu")

    Returns:
        labels: (H,W) int array with segment IDs
        num_components: Number of segments found
    """
    # Convert to torch tensors on GPU
    planarity_mask = torch.as_tensor(planarity_mask, device=device, dtype=torch.bool)
    normal = torch.as_tensor(normal, device=device, dtype=torch.float32)
    depth = torch.as_tensor(depth, device=device, dtype=torch.float32)

    H, W = planarity_mask.shape

    # === 1. Gradients (Sobel via conv2d) ===
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], device=device, dtype=torch.float32).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]], device=device, dtype=torch.float32).view(1,1,3,3)

    def sobel_filter(img2d):
        img = img2d[None,None]  # (1,1,H,W)
        dx = F.conv2d(img, sobel_x, padding=1)
        dy = F.conv2d(img, sobel_y, padding=1)
        return dx[0,0], dy[0,0]

    depth_dx, depth_dy = sobel_filter(depth)
    normal_dx = []
    normal_dy = []
    for i in range(3):
        dx, dy = sobel_filter(normal[..., i])
        normal_dx.append(dx)
        normal_dy.append(dy)
    normal_dx = torch.stack(normal_dx, dim=-1)
    normal_dy = torch.stack(normal_dy, dim=-1)

    normal_grad_mag = torch.sqrt(torch.sum(normal_dx**2 + normal_dy**2, dim=-1))
    grad_mag_threshold = torch.sqrt(torch.tensor(2.0, device=device) -
                                    2*torch.cos(torch.tensor(normal_threshold_rad, device=device)))

    normal_similar = (normal_grad_mag <= grad_mag_threshold)

    # === 2. Pad arrays ===
    n = 2
    padded_mask   = F.pad(planarity_mask[None,None].float(), (n,n,n,n)).squeeze(0).squeeze(0).bool()
    padded_depth  = F.pad(depth[None,None], (n,n,n,n)).squeeze(0).squeeze(0)
    padded_normal = F.pad(normal_similar[None,None].float(), (n,n,n,n)).squeeze(0).squeeze(0).bool()

    # === 3. Neighbor stacking ===
    shifts = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if not (dy == 0 and dx == 0)]

    def get_shifted(arr, dy, dx):
        return arr[n+dy:n+dy+H, n+dx:n+dx+W]

    neighbor_masks   = torch.stack([get_shifted(padded_mask, *s) for s in shifts], dim=-1)
    neighbor_depths  = torch.stack([get_shifted(padded_depth, *s) for s in shifts], dim=-1)
    neighbor_normals = torch.stack([get_shifted(padded_normal, *s) for s in shifts], dim=-1)

    center_depth = depth.unsqueeze(-1)

    valid_pair = planarity_mask.unsqueeze(-1) & neighbor_masks
    depth_diff = (center_depth - neighbor_depths).abs()
    depth_close = (depth_diff < depth_threshold)

    neighbor_match_count = (valid_pair & neighbor_normals & depth_close).sum(dim=-1)
    connected = (neighbor_match_count >= neighbor_match_count_thresh)

    # === 4. Connected components ===
    labels = cc3d.connected_components(connected.detach().cpu().numpy())
    num_components = labels.max()

    return labels, int(num_components)
