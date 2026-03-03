"""
Planar segmentation algorithms using normal and depth consistency.

This module provides vectorized algorithms for segmenting planar regions
based on surface normal similarity and depth proximity.
"""

import numpy as np
import cv2
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


def compute_vectorized_planar_segments_v5(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    Optimized GPU-accelerated planar segmentation using 5x5 neighborhood.

    ~3-5x faster than v4 through:
    - Batched Sobel filters (grouped convolution)
    - F.unfold for efficient neighbor extraction
    - Minimized GPU-CPU transfers
    - Pre-allocated tensors

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
    H, W = planarity_mask.shape

    # Convert to torch tensors on GPU - ensure contiguous
    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )

    # === 1. Batched Sobel filters for all 3 normal channels ===
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], device=device, dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], device=device, dtype=torch.float32)

    # Expand for grouped conv: (3, 1, 3, 3) for 3 input channels
    sobel_x_batch = sobel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    sobel_y_batch = sobel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()

    # Normal: (H, W, 3) -> (1, 3, H, W)
    normal_batch = normal_t.permute(2, 0, 1).unsqueeze(0)

    # Grouped convolution: each channel processed independently
    normal_dx = F.conv2d(normal_batch, sobel_x_batch, padding=1, groups=3)  # (1, 3, H, W)
    normal_dy = F.conv2d(normal_batch, sobel_y_batch, padding=1, groups=3)  # (1, 3, H, W)

    # Gradient magnitude across all channels
    normal_grad_mag = torch.sqrt((normal_dx ** 2 + normal_dy ** 2).sum(dim=1))  # (1, H, W)
    normal_grad_mag = normal_grad_mag.squeeze(0)  # (H, W)

    grad_mag_threshold = torch.sqrt(
        torch.tensor(2.0, device=device) -
        2 * torch.cos(torch.tensor(normal_threshold_rad, device=device))
    )
    normal_similar = (normal_grad_mag <= grad_mag_threshold)  # (H, W)

    # === 2. Use unfold for efficient 5x5 neighbor extraction ===
    kernel_size = 5
    pad = kernel_size // 2

    # Pad and unfold depth
    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode='constant', value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size)  # (1, 25, H*W)
    depth_patches = depth_patches.view(kernel_size * kernel_size, H, W)  # (25, H, W)

    # Pad and unfold mask
    mask_padded = F.pad(planarity_t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size)  # (1, 25, H*W)
    mask_patches = mask_patches.view(kernel_size * kernel_size, H, W).bool()  # (25, H, W)

    # Pad and unfold normal_similar
    normal_sim_padded = F.pad(normal_similar[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    normal_sim_patches = F.unfold(normal_sim_padded, kernel_size=kernel_size)  # (1, 25, H*W)
    normal_sim_patches = normal_sim_patches.view(kernel_size * kernel_size, H, W).bool()  # (25, H, W)

    # === 3. Compute matches excluding center pixel (index 12 in 5x5) ===
    center_idx = (kernel_size * kernel_size) // 2  # = 12

    # Create neighbor indices (exclude center)
    neighbor_indices = [i for i in range(kernel_size * kernel_size) if i != center_idx]

    # Extract neighbor data
    neighbor_depths = depth_patches[neighbor_indices]  # (24, H, W)
    neighbor_masks = mask_patches[neighbor_indices]  # (24, H, W)
    neighbor_normals = normal_sim_patches[neighbor_indices]  # (24, H, W)

    # Center values
    center_depth = depth_t  # (H, W)

    # Valid pairs: center is planar AND neighbor is planar
    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks  # (24, H, W)

    # Depth check
    depth_diff = torch.abs(center_depth.unsqueeze(0) - neighbor_depths)  # (24, H, W)
    depth_close = (depth_diff < depth_threshold)  # (24, H, W)

    # Count matches
    matches = valid_pair & neighbor_normals & depth_close  # (24, H, W)
    neighbor_match_count = matches.sum(dim=0)  # (H, W)

    # Final connected mask
    connected = (neighbor_match_count >= neighbor_match_count_thresh)  # (H, W)

    # === 4. Connected components on CPU ===
    labels = cc3d.connected_components(connected.cpu().numpy())
    num_components = int(labels.max())

    return labels, num_components


def compute_vectorized_planar_segments_v5_relative(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    v5 variant with relative depth threshold.

    Same Sobel-based normal check as v5, but depth check is relative:
    |d_center - d_neighbor| < threshold * d_center
    (threshold is a fraction, e.g. 0.025 = 2.5% of center depth).

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

    # Sobel filters (same as v5)
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


def compute_vectorized_planar_segments_v5_no_sobel(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    v5 variant replacing Sobel with pairwise dot-product normal check.

    Uses per-neighbor dot-product comparison (like v6) but keeps v5's
    absolute depth threshold and cc3d connected components.

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Absolute depth difference threshold in meters
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

    kernel_size = 5
    pad = kernel_size // 2
    center_idx = (kernel_size * kernel_size) // 2
    neighbor_indices = [i for i in range(kernel_size * kernel_size) if i != center_idx]

    # Pairwise normal dot-product
    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode='constant', value=0)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size).view(3, 25, H, W)
    neighbor_normals = normal_patches[:, neighbor_indices]

    center_normal = normal_t.permute(2, 0, 1)
    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0)
    dot = torch.clamp(dot, -1.0, 1.0)
    normal_similar = torch.acos(dot) < normal_threshold_rad

    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode='constant', value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size).view(25, H, W)

    mask_padded = F.pad(planarity_t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size).view(25, H, W).bool()

    neighbor_depths = depth_patches[neighbor_indices]
    neighbor_masks = mask_patches[neighbor_indices]

    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks

    # Absolute depth check
    depth_diff = torch.abs(depth_t.unsqueeze(0) - neighbor_depths)
    depth_close = (depth_diff < depth_threshold)

    matches = valid_pair & normal_similar & depth_close
    connected = (matches.sum(dim=0) >= neighbor_match_count_thresh)

    labels = cc3d.connected_components(connected.cpu().numpy())
    num_components = int(labels.max())

    return labels, num_components


def compute_vectorized_planar_segments_v5_dotprod_relative(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    v5 variant with both dot-product normals and relative depth.

    Combines pairwise dot-product normal check with relative depth threshold.
    Uses cc3d for connected components.

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

    kernel_size = 5
    pad = kernel_size // 2
    center_idx = (kernel_size * kernel_size) // 2
    neighbor_indices = [i for i in range(kernel_size * kernel_size) if i != center_idx]

    # Pairwise normal dot-product
    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode='constant', value=0)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size).view(3, 25, H, W)
    neighbor_normals = normal_patches[:, neighbor_indices]

    center_normal = normal_t.permute(2, 0, 1)
    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0)
    dot = torch.clamp(dot, -1.0, 1.0)
    normal_similar = torch.acos(dot) < normal_threshold_rad

    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode='constant', value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size).view(25, H, W)

    mask_padded = F.pad(planarity_t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size).view(25, H, W).bool()

    neighbor_depths = depth_patches[neighbor_indices]
    neighbor_masks = mask_patches[neighbor_indices]

    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks

    # Relative depth check
    center_depth = depth_t.unsqueeze(0)
    depth_diff = torch.abs(center_depth - neighbor_depths)
    depth_close = depth_diff < (depth_threshold * center_depth)

    matches = valid_pair & normal_similar & depth_close
    connected = (matches.sum(dim=0) >= neighbor_match_count_thresh)

    labels = cc3d.connected_components(connected.cpu().numpy())
    num_components = int(labels.max())

    return labels, num_components


def compute_vectorized_planar_segments_v6(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 18,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    GPU-accelerated planar segmentation using pairwise normal comparison.

    Based on Shaohui's design (limap moge_planarity) with v5-level GPU optimizations.
    Key differences from v5:
    - Pairwise dot-product normal check (per neighbor pair, not Sobel gradient)
    - Relative depth threshold (fraction of center depth, not absolute meters)
    - Default neighbor_match_count_thresh=18 (75% of 24, less boundary erosion)

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Relative depth threshold as fraction of center depth
            (e.g., 0.02 means 2% of depth at that pixel)
        neighbor_match_count_thresh: Minimum matching neighbors in 5x5 window (default 18)
        device: Torch device ("cuda" or "cpu")

    Returns:
        labels: (H,W) int array with segment IDs
        num_components: Number of segments found
    """
    H, W = planarity_mask.shape

    # Convert to torch tensors on GPU - ensure contiguous
    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )

    # === 1. Use unfold for efficient 5x5 neighbor extraction ===
    kernel_size = 5
    pad = kernel_size // 2

    # Pad and unfold depth
    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode='constant', value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size)  # (1, 25, H*W)
    depth_patches = depth_patches.view(kernel_size * kernel_size, H, W)  # (25, H, W)

    # Pad and unfold mask
    mask_padded = F.pad(planarity_t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size)  # (1, 25, H*W)
    mask_patches = mask_patches.view(kernel_size * kernel_size, H, W).bool()  # (25, H, W)

    # Pad and unfold normals (3 channels)
    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode='constant', value=0)
    # Unfold each channel: (1, 3*25, H*W)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size)
    normal_patches = normal_patches.view(3, kernel_size * kernel_size, H, W)  # (3, 25, H, W)

    # === 2. Exclude center pixel (index 12 in 5x5) ===
    center_idx = (kernel_size * kernel_size) // 2  # = 12
    neighbor_indices = [i for i in range(kernel_size * kernel_size) if i != center_idx]

    neighbor_depths = depth_patches[neighbor_indices]  # (24, H, W)
    neighbor_masks = mask_patches[neighbor_indices]  # (24, H, W)
    neighbor_normals = normal_patches[:, neighbor_indices]  # (3, 24, H, W)

    # === 3. Pairwise normal similarity (dot product per neighbor) ===
    center_normal = normal_t.permute(2, 0, 1)  # (3, H, W)
    # Dot product: sum over channel dim
    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0)  # (24, H, W)
    dot = torch.clamp(dot, -1.0, 1.0)
    angle = torch.acos(dot)  # (24, H, W)
    normal_similar = angle < normal_threshold_rad  # (24, H, W)

    # === 4. Relative depth threshold ===
    center_depth = depth_t.unsqueeze(0)  # (1, H, W)
    depth_diff = torch.abs(center_depth - neighbor_depths)  # (24, H, W)
    depth_close = depth_diff < (depth_threshold * center_depth)  # (24, H, W)

    # === 5. Valid pairs and match counting ===
    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks  # (24, H, W)
    matches = valid_pair & normal_similar & depth_close  # (24, H, W)
    neighbor_match_count = matches.sum(dim=0)  # (H, W)

    connected = (neighbor_match_count >= neighbor_match_count_thresh)  # (H, W)

    # === 6. Connected components on CPU ===
    labels = cc3d.connected_components(connected.cpu().numpy())
    num_components = int(labels.max())

    return labels, num_components


@torch.no_grad()
def compute_vectorized_planar_segments_v9(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 18,
    device: str = "cuda",
) -> Tuple[np.ndarray, int]:
    """
    Planar segmentation v9: pairwise normals + symmetric relative depth + hard voting.

    Same structure as v6 but uses symmetric relative depth (avg of center and
    neighbor) and scipy.ndimage.label for connected components.

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar (int32)
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Relative depth threshold (fraction of avg depth)
        neighbor_match_count_thresh: Minimum matching neighbors in 5x5 window (default 18)
        device: Torch device ("cuda" or "cpu")

    Returns:
        labels: (H,W) int32 array with segment IDs (0 = background)
        num_components: Number of segments found
    """
    H, W = planarity_mask.shape

    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )
    normal_t = F.normalize(normal_t, dim=-1, eps=1e-6)

    kernel_size = 5
    pad = kernel_size // 2
    center_idx = kernel_size ** 2 // 2
    neighbor_indices = [i for i in range(kernel_size ** 2) if i != center_idx]

    # === 1. Pairwise normal comparison ===
    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode="constant", value=0)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size)
    normal_patches = normal_patches.view(3, kernel_size ** 2, H, W)

    neighbor_normals = normal_patches[:, neighbor_indices]
    neighbor_normals = F.normalize(neighbor_normals, dim=0, eps=1e-6)

    center_normal = normal_t.permute(2, 0, 1)
    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0)
    dot = torch.clamp(dot, -1.0, 1.0)
    cos_thresh = torch.cos(torch.tensor(normal_threshold_rad, device=device))
    normal_similar = dot > cos_thresh

    # === 2. Unfold depth and planarity mask ===
    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode="constant", value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size)
    neighbor_depths = depth_patches.view(kernel_size ** 2, H, W)[neighbor_indices]

    mask_padded = F.pad(
        planarity_t[None, None].float(), (pad, pad, pad, pad), mode="constant", value=0
    )
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size)
    neighbor_masks = mask_patches.view(kernel_size ** 2, H, W)[neighbor_indices].bool()

    # === 3. Symmetric relative depth comparison ===
    center_depth = depth_t.unsqueeze(0)
    depth_diff = torch.abs(center_depth - neighbor_depths)
    avg_depth = ((center_depth + neighbor_depths) * 0.5).clamp(min=0.1)
    depth_close = depth_diff < (depth_threshold * avg_depth)

    # === 4. Valid pairs and hard voting ===
    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks
    matches = valid_pair & normal_similar & depth_close
    neighbor_match_count = matches.sum(dim=0)

    connected = (neighbor_match_count >= neighbor_match_count_thresh)

    # === 5. Connected components on CPU ===
    structure = np.ones((3, 3), dtype=bool)
    labels, num_components = label(connected.detach().cpu().numpy(), structure=structure)
    labels = labels.astype(np.int32)

    return labels, num_components


@torch.no_grad()
def compute_vectorized_planar_segments_v9_vote(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 18,
    planarity_threshold: float = 0.6,
    planarity_ratio: float = 0.5,
    device: str = "cuda",
) -> Tuple[np.ndarray, int]:
    """
    Planar segmentation v9_vote: v9 with planarity-based segment voting.

    Algorithm:
    1. Run v9 with ALL pixels treated as planar (planarity_mask = ones)
    2. Per-segment voting: compute ratio of planar pixels (above threshold)
       to total pixels in each segment
    3. Zero out segments where ratio < planarity_ratio
    4. Relabel contiguously

    This defers the planarity decision to the segment level: individual pixels
    may be below threshold, but if enough of the segment's pixels are planar,
    the whole segment is kept.

    Args:
        planarity: (H,W) raw planarity probability in [0,1] (NOT binary mask)
        normal: (H,W,3) surface normals (unit vectors)
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Relative depth threshold (fraction of avg depth)
        neighbor_match_count_thresh: Minimum matching neighbors in 5x5 window
        planarity_threshold: Threshold for counting a pixel as "planar" in voting
        planarity_ratio: Minimum fraction of planar pixels to keep a segment
        device: Torch device

    Returns:
        labels: (H,W) int32 array with segment IDs (0 = background)
        num_components: Number of segments found
    """
    H, W = planarity.shape

    # Step 1: Run v9 with all pixels as planar
    all_planar = np.ones((H, W), dtype=np.int32)
    labels, num_components = compute_vectorized_planar_segments_v9(
        all_planar, normal, depth,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh=neighbor_match_count_thresh,
        device=device,
    )

    if num_components == 0:
        return labels, 0

    # Step 2: Per-segment planarity voting
    planar_mask = (planarity > planarity_threshold)
    flat_labels = labels.ravel()
    flat_planar = planar_mask.ravel()

    total_counts = np.bincount(flat_labels, minlength=num_components + 1)
    planar_counts = np.bincount(flat_labels, weights=flat_planar.astype(np.float64),
                                minlength=num_components + 1)

    # Avoid division by zero for label 0 (background)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(total_counts > 0, planar_counts / total_counts, 0.0)

    # Step 3: Zero out segments below ratio threshold
    reject_labels = np.where(ratios < planarity_ratio)[0]
    reject_labels = reject_labels[reject_labels > 0]  # keep background as-is

    if len(reject_labels) > 0:
        lut = np.arange(num_components + 1, dtype=np.int32)
        lut[reject_labels] = 0
        labels = lut[flat_labels].reshape(H, W)

    # Step 4: Relabel contiguously
    remaining = np.unique(labels)
    remaining = remaining[remaining > 0]
    num_components = len(remaining)

    if num_components > 0:
        relabel_lut = np.zeros(labels.max() + 1, dtype=np.int32)
        for new_id, old_id in enumerate(remaining, start=1):
            relabel_lut[old_id] = new_id
        labels = relabel_lut[labels.ravel()].reshape(H, W)

    return labels, num_components


@torch.no_grad()
def compute_vectorized_planar_segments_v10(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 18,
    device: str = "cuda",
    # v10 additions
    adaptive_frac: float = 0.75,
    min_valid_neighbors: int = 3,
    min_segment_pixels: int = 50,
) -> Tuple[np.ndarray, int]:
    """
    Planar segmentation v10: adaptive threshold + small-segment filter.

    Improvements over v6:
      1. Adaptive voting threshold — instead of requiring a fixed count of
         matching neighbors, require `adaptive_frac` (e.g. 75%) of *valid*
         neighbors (those that are planar AND have positive depth). At plane
         interiors where all 24 neighbors are valid, 75% of 24 = 18 (same as v6).
         At boundaries where only 8 neighbors are valid, 75% of 8 = 6, so boundary
         pixels are preserved instead of eroded.
         `min_valid_neighbors` sets an absolute floor: pixels with fewer valid
         neighbors than this are rejected regardless (avoids noise in isolated pixels).

      2. Minimum segment size filter — after connected components, segments smaller
         than `min_segment_pixels` are set to background (label 0). Tiny fragments
         are too small for reliable RANSAC plane fitting and hurt precision.

    Uses pairwise normal dot-product comparison, symmetric relative depth,
    5x5 F.unfold neighborhoods, and scipy.ndimage.label for 8-connected CC.

    Args:
        planarity_mask: (H,W) binary mask where 1 = planar, 0 = non-planar
        normal: (H,W,3) surface normals
        depth: (H,W) depth map in meters
        normal_threshold_rad: Angular threshold in radians for normal similarity
        depth_threshold: Relative depth threshold (fraction, e.g. 0.025 = 2.5%)
        neighbor_match_count_thresh: Ignored when adaptive_frac > 0 (kept for API compat)
        device: Torch device
        adaptive_frac: Fraction of valid neighbors required to match (0.75 = 75%)
        min_valid_neighbors: Minimum valid neighbor count to consider a pixel (absolute floor)
        min_segment_pixels: Segments smaller than this are removed (set 0 to disable)

    Returns:
        labels: (H,W) int32 array with segment IDs (0 = background)
        num_components: Number of segments found
    """
    H, W = planarity_mask.shape

    # --- Transfer to GPU ---
    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )

    # Defensive normalization
    normal_t = F.normalize(normal_t, dim=-1, eps=1e-6)

    kernel_size = 5
    pad = kernel_size // 2
    center_idx = kernel_size ** 2 // 2  # = 12
    neighbor_indices = [i for i in range(kernel_size ** 2) if i != center_idx]

    # === 1. Pairwise normal comparison ===
    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode="constant", value=0)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size)
    normal_patches = normal_patches.view(3, kernel_size ** 2, H, W)

    neighbor_normals = normal_patches[:, neighbor_indices]
    neighbor_normals = F.normalize(neighbor_normals, dim=0, eps=1e-6)

    center_normal = normal_t.permute(2, 0, 1)

    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0)
    dot = torch.clamp(dot, -1.0, 1.0)
    cos_thresh = torch.cos(torch.tensor(normal_threshold_rad, device=device))
    normal_similar = dot > cos_thresh

    # === 2. Unfold depth and planarity mask ===
    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode="constant", value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size)
    neighbor_depths = depth_patches.view(kernel_size ** 2, H, W)[neighbor_indices]

    mask_padded = F.pad(
        planarity_t[None, None].float(), (pad, pad, pad, pad), mode="constant", value=0
    )
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size)
    neighbor_masks = mask_patches.view(kernel_size ** 2, H, W)[neighbor_indices].bool()

    # === 3. Symmetric relative depth comparison ===
    center_depth = depth_t.unsqueeze(0)
    depth_diff = torch.abs(center_depth - neighbor_depths)
    avg_depth = ((center_depth + neighbor_depths) * 0.5).clamp(min=0.1)
    depth_close = depth_diff < (depth_threshold * avg_depth)

    # === 4. Valid pairs and matching ===
    depth_valid = (center_depth > 0) & (neighbor_depths > 0)
    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks & depth_valid

    matches = valid_pair & normal_similar & depth_close
    match_count = matches.sum(dim=0)          # (H, W)
    valid_count = valid_pair.sum(dim=0)       # (H, W)

    # === 5. Adaptive threshold (v10) ===
    adaptive_thresh = (valid_count.float() * adaptive_frac).clamp(min=float(min_valid_neighbors))
    connected = (match_count >= adaptive_thresh) & (valid_count >= min_valid_neighbors)

    # === 6. Connected components on CPU ===
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labels, num_components = label(connected.detach().cpu().numpy(), structure=structure)
    labels = labels.astype(np.int32)

    # === 7. Small segment filter (v10) ===
    if min_segment_pixels > 0:
        flat = labels.ravel()
        counts = np.bincount(flat, minlength=num_components + 1)
        small_labels = np.where(counts < min_segment_pixels)[0]
        small_labels = small_labels[small_labels > 0]
        if len(small_labels) > 0:
            lut = np.arange(num_components + 1, dtype=np.int32)
            lut[small_labels] = 0
            labels = lut[flat].reshape(H, W)
            num_components = len(np.unique(labels)) - 1  # subtract background

    return labels, num_components


@torch.no_grad()
def compute_vectorized_planar_segments_v11(
    planarity_mask: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 18,
    device: str = "cuda",
    # v10 params
    adaptive_frac: float = 0.75,
    min_valid_neighbors: int = 3,
    min_segment_pixels: int = 50,
    # v11: decoupled voting (None = fallback to adaptive_frac)
    normal_adaptive_frac: float = None,
    depth_adaptive_frac: float = None,
    # v11: post-merge
    merge_enabled: bool = False,
    merge_normal_deg: float = 10.0,
    merge_offset_m: float = 0.02,
    merge_min_pixels: int = 50,
    merge_gap_px: int = 5,
) -> Tuple[np.ndarray, int]:
    """
    Planar segmentation v11: v10 with decoupled voting + optional post-merge.

    When normal_adaptive_frac and depth_adaptive_frac are set, uses DECOUPLED
    voting: normals and depth are evaluated independently against their own
    fraction thresholds, then ANDed. This lets you be lenient on noisy normals
    while keeping depth strict.

    When both are None, falls back to the original v10 behavior (single
    adaptive_frac applied to the joint normal+depth match count).

    Post-merge (merge_enabled=True):
      After connected components + small segment filter, iteratively merge
      adjacent segments whose mean normals and plane offsets are similar.

    Args:
        planarity_mask: (H,W) binary mask (1 = planar)
        normal: (H,W,3) surface normals
        depth: (H,W) depth in meters
        normal_threshold_rad: Angular threshold in radians for pixel-level normal check
        depth_threshold: Relative depth threshold (fraction, e.g. 0.025)
        neighbor_match_count_thresh: Ignored (kept for API compat)
        device: Torch device
        adaptive_frac: Fraction of valid neighbors required (v10)
        min_valid_neighbors: Minimum valid neighbor count (v10)
        min_segment_pixels: Minimum segment size (v10)
        normal_adaptive_frac: Separate fraction for normal voting (None = use adaptive_frac)
        depth_adaptive_frac: Separate fraction for depth voting (None = use adaptive_frac)
        merge_enabled: Whether to run post-merge (default False)
        merge_normal_deg: Max angle between mean normals for merge
        merge_offset_m: Max mean-depth difference for merge in meters
        merge_min_pixels: Ignore segments smaller than this during merge
        merge_gap_px: Dilation radius in pixels to bridge gaps

    Returns:
        labels: (H,W) int32 segment IDs (0 = background)
        num_components: Number of segments
    """
    H, W = planarity_mask.shape
    decoupled = (normal_adaptive_frac is not None) or (depth_adaptive_frac is not None)
    if decoupled:
        naf = normal_adaptive_frac if normal_adaptive_frac is not None else adaptive_frac
        daf = depth_adaptive_frac if depth_adaptive_frac is not None else adaptive_frac

    # --- Transfer to GPU ---
    planarity_t = torch.as_tensor(
        np.ascontiguousarray(planarity_mask), device=device, dtype=torch.bool
    )
    normal_t = torch.as_tensor(
        np.ascontiguousarray(normal), device=device, dtype=torch.float32
    )
    depth_t = torch.as_tensor(
        np.ascontiguousarray(depth), device=device, dtype=torch.float32
    )
    normal_t = F.normalize(normal_t, dim=-1, eps=1e-6)

    kernel_size = 5
    pad = kernel_size // 2
    center_idx = kernel_size ** 2 // 2
    neighbor_indices = [i for i in range(kernel_size ** 2) if i != center_idx]

    # === 1. Pairwise normal comparison ===
    normal_nchw = normal_t.permute(2, 0, 1).unsqueeze(0)
    normal_padded = F.pad(normal_nchw, (pad, pad, pad, pad), mode="constant", value=0)
    normal_patches = F.unfold(normal_padded, kernel_size=kernel_size).view(3, kernel_size**2, H, W)
    neighbor_normals = F.normalize(normal_patches[:, neighbor_indices], dim=0, eps=1e-6)
    center_normal = normal_t.permute(2, 0, 1)
    dot = (center_normal.unsqueeze(1) * neighbor_normals).sum(dim=0).clamp(-1.0, 1.0)
    cos_thresh = torch.cos(torch.tensor(normal_threshold_rad, device=device))
    normal_similar = dot > cos_thresh

    # === 2. Unfold depth and planarity mask ===
    depth_padded = F.pad(depth_t[None, None], (pad, pad, pad, pad), mode="constant", value=0)
    depth_patches = F.unfold(depth_padded, kernel_size=kernel_size)
    neighbor_depths = depth_patches.view(kernel_size**2, H, W)[neighbor_indices]

    mask_padded = F.pad(
        planarity_t[None, None].float(), (pad, pad, pad, pad), mode="constant", value=0
    )
    mask_patches = F.unfold(mask_padded, kernel_size=kernel_size)
    neighbor_masks = mask_patches.view(kernel_size**2, H, W)[neighbor_indices].bool()

    # === 3. Symmetric relative depth comparison ===
    center_depth = depth_t.unsqueeze(0)
    depth_diff = torch.abs(center_depth - neighbor_depths)
    avg_depth = ((center_depth + neighbor_depths) * 0.5).clamp(min=0.1)
    depth_close = depth_diff < (depth_threshold * avg_depth)

    # === 4. Valid pairs ===
    depth_valid = (center_depth > 0) & (neighbor_depths > 0)
    valid_pair = planarity_t.unsqueeze(0) & neighbor_masks & depth_valid
    valid_count = valid_pair.sum(dim=0)

    # === 5. Connectivity decision ===
    if decoupled:
        # Decoupled voting: normal and depth evaluated independently
        normal_match_count = (valid_pair & normal_similar).sum(dim=0)
        depth_match_count = (valid_pair & depth_close).sum(dim=0)

        normal_thresh = (valid_count.float() * naf).clamp(min=float(min_valid_neighbors))
        depth_thresh = (valid_count.float() * daf).clamp(min=float(min_valid_neighbors))

        connected = (
            (normal_match_count >= normal_thresh)
            & (depth_match_count >= depth_thresh)
            & (valid_count >= min_valid_neighbors)
        )
    else:
        # Joint voting (original v10 behavior)
        matches = valid_pair & normal_similar & depth_close
        match_count = matches.sum(dim=0)
        adaptive_thresh = (valid_count.float() * adaptive_frac).clamp(min=float(min_valid_neighbors))
        connected = (match_count >= adaptive_thresh) & (valid_count >= min_valid_neighbors)

    # === 6. Connected components on CPU ===
    structure = np.ones((3, 3), dtype=bool)
    labels, num_components = label(connected.detach().cpu().numpy(), structure=structure)
    labels = labels.astype(np.int32)

    # === 7. Small segment filter ===
    if min_segment_pixels > 0:
        flat = labels.ravel()
        counts = np.bincount(flat, minlength=num_components + 1)
        small_labels = np.where(counts < min_segment_pixels)[0]
        small_labels = small_labels[small_labels > 0]
        if len(small_labels) > 0:
            lut = np.arange(num_components + 1, dtype=np.int32)
            lut[small_labels] = 0
            labels = lut[flat].reshape(H, W)
            num_components = len(np.unique(labels)) - 1

    # === 8. Optional post-merge ===
    if merge_enabled:
        from planamono.shared.segmentation.postprocess import postmerge_adjacent_segments
        labels = postmerge_adjacent_segments(
            labels, normal, depth,
            merge_normal_deg=merge_normal_deg,
            merge_offset_m=merge_offset_m,
            merge_min_pixels=merge_min_pixels,
            merge_gap_px=merge_gap_px,
        )
        num_components = len(np.unique(labels)) - 1

    return labels, num_components


# ──────────────────────────────────────────────────────────────────
# v12: low-threshold v10 + segment voting + gap-bridging merge
# ──────────────────────────────────────────────────────────────────


def _find_adjacency_with_gap_bridging(
    labels: np.ndarray,
    depth: np.ndarray,
    gap_px: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find adjacent segment pairs, bridging small gaps via dilation.

    Returns boundary info for both direct adjacencies (label-to-label borders)
    and gap-bridged adjacencies (segments separated by up to gap_px background
    pixels). Gap-bridged pairs use depth from the nearest labeled pixels on
    each side of the gap.

    Uses cv2.dilate (C++ backend) for fast gap bridging instead of iterative
    scipy binary_dilation.

    Args:
        labels: (H, W) int32, 0 = background
        depth: (H, W) depth in meters
        gap_px: dilation radius in pixels to bridge gaps

    Returns:
        pair_a, pair_b: (N,) int32 arrays of segment ID pairs (a < b)
        boundary_ddiff: (N,) float64 depth differences at boundary
        boundary_dmean: (N,) float64 mean depths at boundary
    """
    H, W = labels.shape

    pairs_list, ddiff_list, dmean_list = [], [], []

    # ── Direct adjacencies (1-pixel border) ──
    for l_a, l_b, d_a, d_b in [
        (labels[:, :-1], labels[:, 1:], depth[:, :-1], depth[:, 1:]),
        (labels[:-1, :], labels[1:, :], depth[:-1, :], depth[1:, :]),
    ]:
        la_flat, lb_flat = l_a.ravel(), l_b.ravel()
        da_flat, db_flat = d_a.ravel(), d_b.ravel()
        mask = (la_flat != lb_flat) & (la_flat > 0) & (lb_flat > 0)
        if mask.any():
            p = np.column_stack([la_flat[mask], lb_flat[mask]])
            swap = p[:, 0] > p[:, 1]
            p[swap] = p[swap, ::-1]
            pairs_list.append(p)
            ddiff_list.append(np.abs(da_flat[mask] - db_flat[mask]))
            dmean_list.append(0.5 * (da_flat[mask] + db_flat[mask]))

    # ── Gap-bridged adjacencies (segments separated by background) ──
    if gap_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1)
        )
        bg_mask = labels == 0

        # Get unique segment IDs (excluding 0)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        # Dilate all segments at once: create a "nearest label" map
        # by dilating from labeled pixels into background
        # Use cv2.dilate on the label map with max filter behavior
        # Instead, dilate each segment's mask and check overlaps
        # Optimization: dilate the full label map, but that doesn't work
        # because max-dilation doesn't give nearest-label semantics.

        # Efficient approach: dilate the binary foreground mask, then for
        # each gap pixel, find which segment is closest.
        # Faster: use distance transform to find nearest labeled pixel.

        if bg_mask.any() and len(unique_labels) > 1:
            # Distance transform from labeled regions
            fg_mask_u8 = (labels > 0).astype(np.uint8)
            dist, nearest_idx = cv2.distanceTransformWithLabels(
                1 - fg_mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                labelType=cv2.DIST_LABEL_PIXEL,
            )

            # nearest_idx maps each pixel to the index of the nearest
            # foreground pixel (1-indexed, flattened). We need to convert
            # to segment labels.
            # Build a map: nearest_idx -> segment label
            fg_indices = np.where(fg_mask_u8.ravel())[0]  # 0-indexed
            # nearest_idx is 1-indexed, and maps to sequential foreground pixels
            # Create lookup: nearest_idx_value -> label
            fg_labels = labels.ravel()[fg_indices]
            fg_depths = depth.ravel()[fg_indices]

            # The distanceTransformWithLabels assigns each pixel a label
            # corresponding to the nearest foreground connected component.
            # nearest_idx values are 1-indexed, sequential per CC.
            # We need a LUT from nearest_idx -> our segment label.
            max_nearest = int(nearest_idx.max())
            idx_to_label = np.zeros(max_nearest + 1, dtype=np.int32)
            idx_to_depth = np.zeros(max_nearest + 1, dtype=np.float64)

            # For each foreground pixel, record its nearest_idx -> label mapping
            fg_nearest = nearest_idx.ravel()[fg_indices]
            # Use the first occurrence (all pixels with same nearest_idx
            # should have the same label since they're in the same CC)
            idx_to_label[fg_nearest] = fg_labels
            idx_to_depth[fg_nearest] = fg_depths

            # Now find gap pixels within gap_px distance
            gap_mask = bg_mask & (dist <= gap_px)

            # For gap boundaries: check horizontal and vertical neighbors
            # that are in different nearest-label regions
            nearest_labels = idx_to_label[nearest_idx.ravel()].reshape(H, W)
            nearest_depths = idx_to_depth[nearest_idx.ravel()].reshape(H, W)

            for nl_a, nl_b, nd_a, nd_b, gm_a, gm_b in [
                (nearest_labels[:, :-1], nearest_labels[:, 1:],
                 nearest_depths[:, :-1], nearest_depths[:, 1:],
                 gap_mask[:, :-1], gap_mask[:, 1:]),
                (nearest_labels[:-1, :], nearest_labels[1:, :],
                 nearest_depths[:-1, :], nearest_depths[1:, :],
                 gap_mask[:-1, :], gap_mask[1:, :]),
            ]:
                nla_f, nlb_f = nl_a.ravel(), nl_b.ravel()
                nda_f, ndb_f = nd_a.ravel(), nd_b.ravel()
                gma_f, gmb_f = gm_a.ravel(), gm_b.ravel()

                # At least one side must be a gap pixel, labels must differ
                mask = ((gma_f | gmb_f) &
                        (nla_f != nlb_f) &
                        (nla_f > 0) & (nlb_f > 0))
                if mask.any():
                    p = np.column_stack([nla_f[mask], nlb_f[mask]])
                    swap = p[:, 0] > p[:, 1]
                    p[swap] = p[swap, ::-1]
                    pairs_list.append(p)
                    ddiff_list.append(np.abs(nda_f[mask] - ndb_f[mask]))
                    dmean_list.append(0.5 * (nda_f[mask] + ndb_f[mask]))

    if not pairs_list:
        return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                np.array([], dtype=np.float64), np.array([], dtype=np.float64))

    all_pairs = np.vstack(pairs_list)
    all_ddiff = np.concatenate(ddiff_list)
    all_dmean = np.concatenate(dmean_list)

    return all_pairs[:, 0], all_pairs[:, 1], all_ddiff, all_dmean


def _boundary_merge(
    labels: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    planarity: np.ndarray = None,
    merge_normal_deg: float = 10.0,
    merge_depth_thresh: float = 0.03,
    merge_min_boundary: int = 5,
    min_segment_pixels: int = 50,
    gap_px: int = 3,
    planarity_vote_thresh: float = 0.0,
) -> np.ndarray:
    """Fast boundary-aware merge with gap bridging and planarity voting.

    Combines three mechanisms:
    1. Direct boundary merge — segments sharing a border with compatible
       normals and smooth depth transition are merged.
    2. Gap bridging — segments separated by up to gap_px background pixels
       are also considered for merging (uses distance transform to find
       nearest segment across gaps).
    3. Planarity voting — after merge, segments with mean planarity below
       planarity_vote_thresh are rejected.

    Args:
        labels: (H, W) int32, 0 = background
        normal: (H, W, 3) surface normals
        depth: (H, W) depth in meters
        planarity: (H, W) planarity probability [0,1]. If provided and
            planarity_vote_thresh > 0, segments with low mean planarity
            are removed after merge.
        merge_normal_deg: max angle between mean normals to merge
        merge_depth_thresh: max relative boundary depth difference to merge
        merge_min_boundary: minimum boundary pixel count to consider a merge
        min_segment_pixels: segments smaller than this are removed after merge
        gap_px: dilation radius for gap bridging (0 = direct adjacency only)
        planarity_vote_thresh: reject segments with mean planarity below this

    Returns:
        labels_merged: (H, W) int32 relabeled 1..N
    """
    H, W = labels.shape
    max_label = int(labels.max())
    if max_label <= 1:
        return labels

    flat = labels.ravel()

    # ── Per-segment mean normals (vectorized via bincount) ──
    counts = np.bincount(flat, minlength=max_label + 1).astype(np.float64)
    nsum = np.zeros((max_label + 1, 3), dtype=np.float64)
    for c in range(3):
        nsum[:, c] = np.bincount(
            flat, weights=normal[:, :, c].ravel(), minlength=max_label + 1
        )

    seg_normal = np.zeros_like(nsum)
    valid_mask = counts > 0
    seg_normal[valid_mask] = nsum[valid_mask] / counts[valid_mask, None]
    norms = np.maximum(np.linalg.norm(seg_normal, axis=1, keepdims=True), 1e-8)
    seg_normal = seg_normal / norms

    # ── Adjacency detection (direct + gap-bridged) ──
    pa, pb, all_ddiff, all_dmean = _find_adjacency_with_gap_bridging(
        labels, depth, gap_px=gap_px,
    )

    if len(pa) == 0:
        return labels

    all_pairs = np.column_stack([pa, pb])

    # ── Group by unique pair: compute mean boundary depth diff ──
    pair_key = all_pairs[:, 0].astype(np.int64) * (max_label + 1) + all_pairs[:, 1]
    unique_keys, inverse = np.unique(pair_key, return_inverse=True)
    n_unique = len(unique_keys)

    sum_ddiff = np.bincount(inverse, weights=all_ddiff, minlength=n_unique)
    sum_dmean = np.bincount(inverse, weights=all_dmean, minlength=n_unique)
    pair_counts = np.bincount(inverse, minlength=n_unique)

    mean_ddiff = sum_ddiff / np.maximum(pair_counts, 1)
    mean_dmean = sum_dmean / np.maximum(pair_counts, 1)

    # Decode unique pairs
    pair_a = (unique_keys // (max_label + 1)).astype(np.int32)
    pair_b = (unique_keys % (max_label + 1)).astype(np.int32)

    # ── Union-Find merge (largest boundary first = most confident) ──
    parent = np.arange(max_label + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cos_thresh = np.cos(np.deg2rad(merge_normal_deg))
    order = np.argsort(-pair_counts)

    for idx in order:
        a, b = int(pair_a[idx]), int(pair_b[idx])
        n_boundary = int(pair_counts[idx])
        if n_boundary < merge_min_boundary:
            continue

        ra, rb = find(a), find(b)
        if ra == rb:
            continue

        # Normal similarity
        cos_sim = abs(np.dot(seg_normal[ra], seg_normal[rb]))
        if cos_sim < cos_thresh:
            continue

        # Boundary depth continuity (relative)
        rel_ddiff = mean_ddiff[idx] / max(mean_dmean[idx], 0.1)
        if rel_ddiff > merge_depth_thresh:
            continue

        # Merge rb into ra, update stats
        parent[rb] = ra
        total = counts[ra] + counts[rb]
        merged_nsum = nsum[ra] + nsum[rb]
        nsum[ra] = merged_nsum
        counts[ra] = total
        nm = merged_nsum / total
        nn = np.linalg.norm(nm)
        seg_normal[ra] = nm / max(nn, 1e-8)

    # ── Relabel ──
    root_lut = np.array([find(i) for i in range(max_label + 1)], dtype=np.int32)
    labels = root_lut[flat].reshape(H, W)

    # ── Planarity voting: reject segments with low mean planarity ──
    if planarity is not None and planarity_vote_thresh > 0:
        flat_merged = labels.ravel()
        ml = int(labels.max())
        plan_sum = np.bincount(
            flat_merged, weights=planarity.ravel(), minlength=ml + 1
        )
        seg_counts = np.bincount(flat_merged, minlength=ml + 1).astype(np.float64)
        mean_plan = np.zeros(ml + 1)
        valid = seg_counts > 0
        mean_plan[valid] = plan_sum[valid] / seg_counts[valid]

        reject = np.where((mean_plan < planarity_vote_thresh))[0]
        reject = reject[reject > 0]
        if len(reject) > 0:
            lut = np.arange(ml + 1, dtype=np.int32)
            lut[reject] = 0
            labels = lut[flat_merged].reshape(H, W)

    # ── Size filter + contiguous relabeling ──
    flat_final = labels.ravel()
    ml = int(labels.max()) if labels.max() > 0 else 0
    final_counts = np.bincount(flat_final, minlength=ml + 1)

    remaining = np.unique(flat_final)
    remaining = remaining[remaining > 0]
    relut = np.zeros(ml + 1, dtype=np.int32)
    new_id = 1
    for old_id in remaining:
        if final_counts[old_id] >= min_segment_pixels:
            relut[old_id] = new_id
            new_id += 1
    labels = relut[flat_final].reshape(H, W)

    return labels


@torch.no_grad()
def compute_vectorized_planar_segments_v12(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    neighbor_match_count_thresh: int = 18,
    device: str = "cuda",
    # v10 core params
    adaptive_frac: float = 0.75,
    min_valid_neighbors: int = 3,
    min_segment_pixels: int = 50,
    # v12: planarity thresholds (low for mask, high for voting)
    mask_planarity_thresh: float = 0.1,
    vote_planarity_thresh: float = 0.3,
    # v12: merge params
    merge_normal_deg: float = 10.0,
    merge_depth_thresh: float = 0.03,
    merge_min_boundary: int = 5,
    merge_gap_px: int = 3,
) -> Tuple[np.ndarray, int]:
    """v12: low-threshold v10 + segment-level planarity voting + gap-bridging merge.

    Three key improvements over v10:

    1. LOW PLANARITY THRESHOLD for initial mask (mask_planarity_thresh=0.1):
       Uses a permissive threshold so that pixels where planarity temporarily
       dips (e.g., at texture boundaries on the same plane) are still included
       in the initial segmentation. This prevents the connected component
       splitting that creates most of v10's over-segmentation.

    2. SEGMENT-LEVEL PLANARITY VOTING (vote_planarity_thresh=0.3):
       After segmentation and merging, computes mean planarity per segment.
       Segments where mean planarity < vote_planarity_thresh are rejected.
       This filters out non-planar regions that leaked in through the low
       initial threshold, without losing planar regions that had local dips.

    3. GAP-BRIDGING BOUNDARY MERGE (merge_gap_px=3):
       Uses cv2.distanceTransform to find segments separated by up to
       merge_gap_px background pixels. These gap-bridged pairs are also
       considered for merging (with the same normal + boundary depth checks).
       This recovers from planarity gaps that the low threshold couldn't bridge.

    Merge criteria (same as before):
      - Mean normals similar (within merge_normal_deg)
      - Depth continuous at boundary (relative diff < merge_depth_thresh)
      - Shared boundary >= merge_min_boundary pixels

    Runtime: ~25-30ms total (v10 GPU ~15ms + merge ~8-12ms CPU).

    Args:
        planarity: (H,W) raw planarity probability in [0, 1] (NOT binary mask)
        normal: (H,W,3) surface normals
        depth: (H,W) depth in meters
        normal_threshold_rad: Angular threshold in radians for v10 pixel-level check
        depth_threshold: Relative depth threshold for v10 pixel-level check
        neighbor_match_count_thresh: Ignored (API compat)
        device: Torch device
        adaptive_frac: v10 adaptive voting fraction
        min_valid_neighbors: v10 minimum valid neighbors
        min_segment_pixels: Minimum segment size
        mask_planarity_thresh: Low threshold for initial binary mask (bridges gaps)
        vote_planarity_thresh: High threshold for segment-level planarity voting
        merge_normal_deg: Max angle between mean normals to merge (degrees)
        merge_depth_thresh: Max relative boundary depth diff to merge
        merge_min_boundary: Min boundary pixels to consider merge
        merge_gap_px: Dilation radius for gap bridging (0 = direct only)

    Returns:
        labels: (H,W) int32 segment IDs (0 = background)
        num_components: Number of segments
    """
    H, W = planarity.shape

    # Phase 1: v10 GPU core with LOW planarity threshold (~15ms)
    planarity_mask = (planarity > mask_planarity_thresh).astype(np.int32)
    labels, n_seg = compute_vectorized_planar_segments_v10(
        planarity_mask, normal, depth,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh=neighbor_match_count_thresh,
        device=device,
        adaptive_frac=adaptive_frac,
        min_valid_neighbors=min_valid_neighbors,
        min_segment_pixels=min_segment_pixels,
    )

    if n_seg <= 1:
        return labels, n_seg

    # Phase 2: Gap-bridging merge + planarity voting (~8-12ms CPU)
    labels = _boundary_merge(
        labels, normal, depth,
        planarity=planarity,
        merge_normal_deg=merge_normal_deg,
        merge_depth_thresh=merge_depth_thresh,
        merge_min_boundary=merge_min_boundary,
        min_segment_pixels=min_segment_pixels,
        gap_px=merge_gap_px,
        planarity_vote_thresh=vote_planarity_thresh,
    )
    n_seg = len(np.unique(labels)) - 1

    return labels, n_seg
