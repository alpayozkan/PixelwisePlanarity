"""
Label manipulation and mapping utilities.

This module provides functions for:
- Filtering and remapping segmentation labels
- Matching planes across views
- Label inpainting and hole filling
"""

import numpy as np
import cv2
from typing import Dict, Tuple


def keep_top_k_planes(
    plane_segmentation: np.ndarray,
    k: int = 5,
    ignore_idx: int = 0
) -> np.ndarray:
    """
    Keep only the top-k largest plane segments.

    Args:
        plane_segmentation: (H,W) int array with plane IDs per pixel
        k: Number of largest planes to keep
        ignore_idx: Background/non-planar label to ignore (default=0)

    Returns:
        filtered_segmentation: (H,W) with only top-k planes, others set to ignore_idx
    """
    seg = plane_segmentation.copy()
    unique, counts = np.unique(seg, return_counts=True)

    # Remove ignore index
    valid_mask = unique != ignore_idx
    unique, counts = unique[valid_mask], counts[valid_mask]

    # Sort by size (descending)
    top_k_ids = unique[np.argsort(-counts)[:k]]

    # Mask out all others
    filtered = np.where(np.isin(seg, top_k_ids), seg, ignore_idx)
    return filtered


def remap_labels(
    seg: np.ndarray,
    start: int = 0
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Remap segmentation labels to compact range [start..start+N-1].

    Args:
        seg: (H,W) segmentation map with arbitrary integer labels
        start: Starting label index (default=0)

    Returns:
        seg_remapped: (H,W) remapped segmentation
        mapping: Dict mapping old labels -> new labels
    """
    # unique_labels = np.unique(seg)
    # mapping = {old: new for new, old in enumerate(unique_labels, start=start)}
    # seg_remapped = np.full_like(seg, fill_value=-1)

    # for old, new in mapping.items():
    #     seg_remapped[seg == old] = new

    # return seg_remapped, mapping
    unique_labels = np.unique(seg)
    planar_labels = unique_labels[unique_labels > 0]

    mapping = {int(old): int(new)
               for new, old in enumerate(planar_labels, start=1)}

    seg_remapped = np.zeros(seg.shape, dtype=np.int32)

    for old, new in mapping.items():
        seg_remapped[seg == old] = new

    return seg_remapped, mapping


def fill_holes_inpaint(warped: np.ndarray) -> np.ndarray:
    """
    Fill black pixels in a warped image using OpenCV inpainting.

    Useful for filling holes in homography-warped images.

    Args:
        warped: (H,W,3) warped RGB image with holes (black pixels)

    Returns:
        filled: (H,W,3) image with holes filled via inpainting
    """
    # Create mask of holes (where all channels == 0)
    mask = (np.sum(warped, axis=-1) == 0).astype(np.uint8) * 255

    # Inpaint using TELEA algorithm (fast marching)
    filled = cv2.inpaint(warped, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return filled


def match_planes_by_overlap(
    uv2: np.ndarray,
    valid: np.ndarray,
    plane_seg1: np.ndarray,
    plane_seg2: np.ndarray
) -> Dict[int, int]:
    """
    Match plane IDs between two views based on reprojection overlap.

    Args:
        uv2: (N,2) projected pixel coordinates from view1 to view2
        valid: (N,) boolean mask of valid projections
        plane_seg1: (H,W) plane segmentation of view1
        plane_seg2: (H,W) plane segmentation of view2

    Returns:
        matches: Dict mapping plane_id_in_view1 -> plane_id_in_view2
    """
    H, W = plane_seg2.shape
    plane_ids1 = plane_seg1.reshape(-1)[valid]

    # Clip projected coordinates to image bounds
    uv2_valid = np.round(uv2[valid]).astype(int)
    uv2_valid[:, 0] = np.clip(uv2_valid[:, 0], 0, W - 1)
    uv2_valid[:, 1] = np.clip(uv2_valid[:, 1], 0, H - 1)
    plane_ids2 = plane_seg2[uv2_valid[:, 1], uv2_valid[:, 0]]

    matches = {}
    for pid in np.unique(plane_ids1):
        if pid <= 0:
            continue  # Skip background
        mask = plane_ids1 == pid
        if np.sum(mask) < 10:  # Too few pixels
            continue

        # Find most overlapping plane in view2
        target_ids, counts = np.unique(plane_ids2[mask], return_counts=True)
        best = target_ids[np.argmax(counts)]
        matches[pid] = int(best)

    return matches


def remap_labels_fast(
    seg: np.ndarray,
    start: int = 1
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Fast remap segmentation labels using vectorized LUT lookup.

    ~10-20x faster than remap_labels for images with many labels.

    Args:
        seg: (H,W) segmentation map with arbitrary integer labels
        start: Starting label index for planar regions (default=1, 0 reserved for background)

    Returns:
        seg_remapped: (H,W) remapped segmentation
        mapping: Dict mapping old labels -> new labels
    """
    unique_labels = np.unique(seg)
    planar_labels = unique_labels[unique_labels > 0]

    if len(planar_labels) == 0:
        return np.zeros_like(seg, dtype=np.int32), {}

    # Build lookup table
    max_label = int(seg.max())
    lut = np.zeros(max_label + 1, dtype=np.int32)

    mapping = {}
    for new, old in enumerate(planar_labels, start=start):
        lut[old] = new
        mapping[int(old)] = int(new)

    # Single vectorized lookup - O(pixels) not O(labels × pixels)
    seg_remapped = lut[seg]

    return seg_remapped, mapping


def map_array(
    arr: np.ndarray,
    matches: Dict[int, int],
    default: int = 0
) -> np.ndarray:
    """
    Map values in array according to a dictionary (fast LUT-based).

    Args:
        arr: Input array with integer values
        matches: Dict mapping old_value -> new_value
        default: Default value for unmapped entries

    Returns:
        mapped: Array with values mapped according to dict
    """
    # Use LUT for speed (only works for non-negative integers)
    if np.issubdtype(arr.dtype, np.integer) and len(matches) > 0 and min(matches.keys()) >= 0:
        lut = np.full(max(arr.max(), max(matches.keys())) + 1, default, dtype=int)
        for k, v in matches.items():
            if k < len(lut):
                lut[k] = v
        return lut[arr]

    # Fallback for other cases
    result = np.full_like(arr, default)
    for k, v in matches.items():
        result[arr == k] = v
    return result
