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


# ============================================================
# Shared helpers for v6–v9
# ============================================================

def _sobel_normal_similarity(normal_t, normal_threshold_rad, device):
    """Compute per-pixel normal similarity via Sobel gradients. Returns (H, W) bool tensor."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=device, dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=device, dtype=torch.float32)
    sobel_x_b = sobel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()
    sobel_y_b = sobel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3).contiguous()

    normal_batch = normal_t.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    dx = F.conv2d(normal_batch, sobel_x_b, padding=1, groups=3)
    dy = F.conv2d(normal_batch, sobel_y_b, padding=1, groups=3)
    grad_mag = torch.sqrt((dx ** 2 + dy ** 2).sum(dim=1)).squeeze(0)  # (H,W)

    thresh = torch.sqrt(torch.tensor(2.0, device=device) -
                        2 * torch.cos(torch.tensor(normal_threshold_rad, device=device)))
    return grad_mag <= thresh


def _unfold_5x5_neighbors(tensor_2d, H, W):
    """Unfold (H,W) tensor into 5x5 neighbor patches excluding center. Returns (24, H, W)."""
    padded = F.pad(tensor_2d[None, None].float(), (2, 2, 2, 2), mode='constant', value=0)
    patches = F.unfold(padded, kernel_size=5).view(25, H, W)
    indices = [i for i in range(25) if i != 12]
    return patches[indices]


def _run_v5_core(mask_np, normal_np, depth_np,
                 normal_threshold_rad, depth_threshold,
                 neighbor_match_count_thresh, device):
    """Core v5 pipeline with a given binary mask. Returns (labels, num_components) from cc3d."""
    H, W = mask_np.shape
    mask_t = torch.as_tensor(np.ascontiguousarray(mask_np), device=device, dtype=torch.bool)
    normal_t = torch.as_tensor(np.ascontiguousarray(normal_np), device=device, dtype=torch.float32)
    depth_t = torch.as_tensor(np.ascontiguousarray(depth_np), device=device, dtype=torch.float32)

    normal_similar = _sobel_normal_similarity(normal_t, normal_threshold_rad, device)

    depth_patches = _unfold_5x5_neighbors(depth_t, H, W)
    mask_patches = _unfold_5x5_neighbors(mask_t, H, W).bool()
    normal_sim_patches = _unfold_5x5_neighbors(normal_similar, H, W).bool()

    valid_pair = mask_t.unsqueeze(0) & mask_patches
    depth_close = (depth_t.unsqueeze(0) - depth_patches).abs() < depth_threshold
    matches = valid_pair & normal_sim_patches & depth_close
    connected = matches.sum(dim=0) >= neighbor_match_count_thresh

    labels = cc3d.connected_components(connected.cpu().numpy())
    return labels, int(labels.max())


# ============================================================
# v6: Probability-weighted connectivity
# ============================================================

def compute_vectorized_planar_segments_v6(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    weighted_thresh: float = 3.0,
    planarity_floor: float = 0.1,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    Probability-weighted planar segmentation.

    Instead of binary mask gating, uses continuous planarity as weights:
    match_strength = planarity_center * planarity_neighbor * geometric_ok.
    Pixels pass if their weighted sum exceeds weighted_thresh.
    """
    H, W = planarity.shape
    planarity_t = torch.as_tensor(np.ascontiguousarray(planarity), device=device, dtype=torch.float32)
    normal_t = torch.as_tensor(np.ascontiguousarray(normal), device=device, dtype=torch.float32)
    depth_t = torch.as_tensor(np.ascontiguousarray(depth), device=device, dtype=torch.float32)

    normal_similar = _sobel_normal_similarity(normal_t, normal_threshold_rad, device)

    depth_patches = _unfold_5x5_neighbors(depth_t, H, W)
    planarity_patches = _unfold_5x5_neighbors(planarity_t, H, W)
    normal_sim_patches = _unfold_5x5_neighbors(normal_similar, H, W).bool()

    depth_close = (depth_t.unsqueeze(0) - depth_patches).abs() < depth_threshold
    geometric_ok = normal_sim_patches & depth_close  # (24, H, W)

    weights = planarity_t.unsqueeze(0) * planarity_patches  # (24, H, W)
    weighted_sum = (weights * geometric_ok.float()).sum(dim=0)  # (H, W)

    connected = (weighted_sum >= weighted_thresh) & (planarity_t > planarity_floor)

    labels = cc3d.connected_components(connected.cpu().numpy())
    return labels, int(labels.max())


# ============================================================
# v7: Seed-and-grow
# ============================================================

def compute_vectorized_planar_segments_v7(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    seed_thresh: float = 0.7,
    grow_thresh: float = 0.2,
    neighbor_match_count_thresh: int = 8,
    min_seed_frac: float = 0.1,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    Seed-and-grow planar segmentation.

    Uses a permissive grow_thresh to include boundary pixels, then
    post-filters to keep only segments anchored by high-confidence seeds.
    """
    grow_mask = (planarity > grow_thresh).astype(np.uint8)
    labels, _ = _run_v5_core(
        grow_mask, normal, depth,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh, device
    )

    max_label = labels.max()
    if max_label == 0:
        return labels, 0

    seed_mask = planarity > seed_thresh
    seed_count = np.bincount(labels.ravel(),
                             weights=seed_mask.ravel().astype(float),
                             minlength=max_label + 1)
    total_count = np.bincount(labels.ravel(), minlength=max_label + 1)
    seed_frac = np.where(total_count > 0, seed_count / total_count, 0.0)

    # Remove segments without enough seeds (keep label 0 = background)
    remove = seed_frac < min_seed_frac
    remove[0] = False
    mapping = np.arange(max_label + 1)
    mapping[remove] = 0
    labels = mapping[labels]

    return labels, int(labels.max())


# ============================================================
# v8: Adaptive per-pixel threshold
# ============================================================

def compute_vectorized_planar_segments_v8(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    base_thresh: float = 0.5,
    adapt_strength: float = 0.3,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    Adaptive-threshold planar segmentation.

    Lowers the planarity threshold for pixels whose neighbors are confident,
    allowing boundary pixels to be included when surrounded by strong evidence.
    """
    H, W = planarity.shape
    planarity_t = torch.as_tensor(np.ascontiguousarray(planarity), device=device, dtype=torch.float32)

    avg_planarity = F.avg_pool2d(
        planarity_t[None, None], kernel_size=5, stride=1, padding=2
    ).squeeze()  # (H, W)

    offset = adapt_strength * torch.clamp(avg_planarity - base_thresh, min=0)
    effective_thresh = base_thresh - offset

    adaptive_mask = (planarity_t > effective_thresh).cpu().numpy().astype(np.uint8)

    return _run_v5_core(
        adaptive_mask, normal, depth,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh, device
    )


# ============================================================
# v9: Joint planarity + geometric consistency score
# ============================================================

def compute_vectorized_planar_segments_v9(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    joint_thresh: float = 0.5,
    planarity_weight: float = 0.5,
    planarity_floor: float = 0.1,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    Joint-score planar segmentation.

    Combines planarity probability with local geometric consistency into a
    single score. Pixels with low planarity but strong geometric agreement
    with neighbors can still be included.
    """
    H, W = planarity.shape
    planarity_t = torch.as_tensor(np.ascontiguousarray(planarity), device=device, dtype=torch.float32)
    normal_t = torch.as_tensor(np.ascontiguousarray(normal), device=device, dtype=torch.float32)
    depth_t = torch.as_tensor(np.ascontiguousarray(depth), device=device, dtype=torch.float32)

    normal_similar = _sobel_normal_similarity(normal_t, normal_threshold_rad, device)

    # Use floor mask for neighbor validity
    floor_mask = planarity_t > planarity_floor

    depth_patches = _unfold_5x5_neighbors(depth_t, H, W)
    floor_patches = _unfold_5x5_neighbors(floor_mask, H, W).bool()
    normal_sim_patches = _unfold_5x5_neighbors(normal_similar, H, W).bool()

    depth_close = (depth_t.unsqueeze(0) - depth_patches).abs() < depth_threshold
    valid_geometric = floor_patches & normal_sim_patches & depth_close  # (24, H, W)
    geom_fraction = valid_geometric.float().sum(dim=0) / 24.0  # (H, W)

    pw = planarity_weight
    joint_score = pw * planarity_t + (1 - pw) * geom_fraction

    joint_mask = (joint_score > joint_thresh).cpu().numpy().astype(np.uint8)

    return _run_v5_core(
        joint_mask, normal, depth,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh, device
    )


# ============================================================
# v10: Felzenszwalb graph-based segmentation
# ============================================================

def compute_vectorized_planar_segments_v10(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    scale: float = 12000.0,
    sigma: float = 0.5,
    min_size: int = 50,
    planarity_floor: float = 0.3,
    planarity_scale: float = 5.0,
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Felzenszwalb graph-based segmentation with geometric features.

    Pre-masks non-planar pixels (planarity < planarity_floor) with sentinel
    feature values, preventing segments from crossing into non-planar regions.
    Segments planar pixels by normal + depth similarity. Post-filters to keep
    only segments with sufficient average planarity.
    """
    from skimage.segmentation import felzenszwalb

    H, W = planarity.shape
    planar_mask = planarity > planarity_floor

    # Scale features so threshold-level differences ≈ 1.0
    normal_edge_at_thresh = np.sqrt(2 - 2 * np.cos(normal_threshold_rad))
    normal_s = 1.0 / max(normal_edge_at_thresh, 1e-8)
    depth_s = 1.0 / max(depth_threshold, 1e-8)

    features = np.concatenate([
        normal * normal_s,
        depth[:, :, None] * depth_s,
    ], axis=2).astype(np.float64)

    # Sentinel: non-planar pixels get extreme values → huge edge weights
    # at planar/non-planar boundaries prevent cross-merging
    features[~planar_mask] = 1e6

    labels = felzenszwalb(features, scale=scale, sigma=sigma, min_size=min_size,
                          channel_axis=2)

    # Zero out non-planar pixels
    labels[~planar_mask] = 0

    # Post-filter: keep segments with sufficient average planarity
    max_label = labels.max()
    if max_label <= 0:
        return np.zeros((H, W), dtype=np.int32), 0

    plan_sum = np.bincount(labels.ravel(), weights=planarity.ravel(),
                           minlength=max_label + 1)
    counts = np.bincount(labels.ravel(), minlength=max_label + 1)
    avg_plan = np.where(counts > 0, plan_sum / counts, 0.0)

    keep = avg_plan > planarity_floor
    keep[0] = False
    mapping = np.zeros(max_label + 1, dtype=np.int32)
    next_id = 1
    for i in range(1, max_label + 1):
        if keep[i]:
            mapping[i] = next_id
            next_id += 1
    labels = mapping[labels]

    return labels, int(labels.max())


# ============================================================
# v11: Watershed segmentation
# ============================================================

def compute_vectorized_planar_segments_v11(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    seed_thresh: float = 0.7,
    gradient_weight: float = 2.0,
    planarity_floor: float = 0.4,
    mask_thresh: float = 0.2,
    min_seed_size: int = 20,
    **kwargs
) -> Tuple[np.ndarray, int]:
    """
    Watershed segmentation on planarity + geometric gradient elevation.

    Treats inverted planarity as a landscape: high-planarity pixels are valleys
    (seeds), normal/depth edges are ridges (barriers). Regions grow from
    high-confidence seeds until they hit geometric boundaries.
    A growth mask (planarity > mask_thresh) prevents filling non-planar regions.
    """
    from scipy.ndimage import label as scipy_label, sobel as scipy_sobel
    from skimage.segmentation import watershed

    H, W = planarity.shape

    # Normal gradient magnitude
    normal_grad_sq = np.zeros((H, W))
    for c in range(3):
        gx = scipy_sobel(normal[:, :, c], axis=1)
        gy = scipy_sobel(normal[:, :, c], axis=0)
        normal_grad_sq += gx ** 2 + gy ** 2
    normal_grad = np.sqrt(normal_grad_sq)

    # Depth gradient magnitude
    dgx = scipy_sobel(depth, axis=1)
    dgy = scipy_sobel(depth, axis=0)
    depth_grad = np.sqrt(dgx ** 2 + dgy ** 2)

    # Normalize to [0, 1]
    ng_max = normal_grad.max()
    dg_max = depth_grad.max()
    if ng_max > 0:
        normal_grad /= ng_max
    if dg_max > 0:
        depth_grad /= dg_max

    # Elevation: low = plane interior, high = boundary
    elevation = (1 - planarity) + gradient_weight * (normal_grad + depth_grad)

    # Seeds from high-planarity connected components
    seed_mask = planarity > seed_thresh
    structure = np.ones((3, 3), dtype=bool)
    seeds, n_seeds = scipy_label(seed_mask, structure=structure)

    if n_seeds == 0:
        return np.zeros((H, W), dtype=np.int32), 0

    # Remove tiny seed components
    for sid in range(1, n_seeds + 1):
        if (seeds == sid).sum() < min_seed_size:
            seeds[seeds == sid] = 0

    if seeds.max() == 0:
        return np.zeros((H, W), dtype=np.int32), 0

    # Relabel seeds contiguously
    unique_seeds = np.unique(seeds)
    unique_seeds = unique_seeds[unique_seeds > 0]
    seed_mapping = np.zeros(seeds.max() + 1, dtype=np.int32)
    for new_id, old_id in enumerate(unique_seeds, start=1):
        seed_mapping[old_id] = new_id
    seeds = seed_mapping[seeds]

    # Watershed (restrict growth to plausible planar pixels)
    growth_mask = planarity > mask_thresh if mask_thresh > 0 else None
    labels = watershed(elevation, markers=seeds, mask=growth_mask)

    # Post-filter by average planarity
    max_label = labels.max()
    plan_sum = np.bincount(labels.ravel(), weights=planarity.ravel(),
                           minlength=max_label + 1)
    counts = np.bincount(labels.ravel(), minlength=max_label + 1)
    avg_plan = np.where(counts > 0, plan_sum / counts, 0.0)

    keep = avg_plan > planarity_floor
    mapping = np.zeros(max_label + 1, dtype=np.int32)
    next_id = 1
    for i in range(max_label + 1):
        if keep[i]:
            mapping[i] = next_id
            next_id += 1
    labels = mapping[labels]

    return labels, int(labels.max())


# ============================================================
# v12: Bilateral-filtered planarity + v5 connectivity
# ============================================================

def compute_vectorized_planar_segments_v12(
    planarity: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    normal_threshold_rad: float,
    depth_threshold: float,
    sigma_normal: float = 0.1,
    sigma_depth: float = 0.05,
    bilateral_thresh: float = 0.5,
    n_iterations: int = 2,
    neighbor_match_count_thresh: int = 8,
    device: str = "cuda"
) -> Tuple[np.ndarray, int]:
    """
    Bilateral-filtered planarity segmentation.

    Smooths the planarity map using a joint bilateral filter weighted by
    normal and depth similarity. Geometrically consistent neighbors reinforce
    each other's planarity, while boundaries are preserved. The refined
    planarity is thresholded and fed into the v5 connectivity pipeline.
    """
    H, W = planarity.shape
    planarity_t = torch.as_tensor(np.ascontiguousarray(planarity), device=device, dtype=torch.float32)
    normal_t = torch.as_tensor(np.ascontiguousarray(normal), device=device, dtype=torch.float32)
    depth_t = torch.as_tensor(np.ascontiguousarray(depth), device=device, dtype=torch.float32)

    pad = 2

    def unfold_25(t):
        padded = F.pad(t[None, None].float(), (pad, pad, pad, pad), mode='constant', value=0)
        return F.unfold(padded, kernel_size=5).view(25, H, W)

    # Unfold normals and depth for weight computation
    nx_p = unfold_25(normal_t[:, :, 0])
    ny_p = unfold_25(normal_t[:, :, 1])
    nz_p = unfold_25(normal_t[:, :, 2])
    depth_p = unfold_25(depth_t)

    center = 12  # center pixel in 5x5

    # Normal similarity: 1 - cos(angle)
    dot = (nx_p * nx_p[center:center + 1] +
           ny_p * ny_p[center:center + 1] +
           nz_p * nz_p[center:center + 1])
    dot = torch.clamp(dot, -1.0, 1.0)
    normal_dist = 1.0 - dot
    normal_w = torch.exp(-normal_dist ** 2 / (2 * sigma_normal ** 2))

    # Depth similarity
    depth_diff = (depth_p - depth_p[center:center + 1]).abs()
    depth_w = torch.exp(-depth_diff ** 2 / (2 * sigma_depth ** 2))

    # Combined bilateral weight (constant across iterations)
    weight = normal_w * depth_w  # (25, H, W)

    # Iterative bilateral filtering of planarity
    current = planarity_t.clone()
    for _ in range(n_iterations):
        plan_p = unfold_25(current)
        current = (weight * plan_p).sum(dim=0) / weight.sum(dim=0).clamp(min=1e-8)

    # Threshold refined planarity
    mask = (current > bilateral_thresh).cpu().numpy().astype(np.uint8)

    return _run_v5_core(
        mask, normal, depth,
        normal_threshold_rad, depth_threshold,
        neighbor_match_count_thresh, device
    )
