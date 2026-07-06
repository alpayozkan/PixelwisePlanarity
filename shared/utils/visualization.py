"""
Visualization utilities for plane segmentation and analysis.

This module provides functions for visualizing:
- Plane segmentations with color coding
- Top-k largest planes
- Surface normals
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Tuple, List


def visualize_top_components(
    segmentation: np.ndarray,
    k: int = 10,
    ignore_label: int = -1,
    cmap: str = 'tab20',
    return_colors: bool = False
) -> Optional[np.ndarray]:
    """
    Visualize top-k largest plane segments with distinct colors.

    Args:
        segmentation: (H,W) segmentation map with plane IDs
        k: Number of top planes to show
        ignore_label: Label to treat as background
        cmap: Matplotlib colormap name
        return_colors: If True, return colored image instead of displaying

    Returns:
        colored_seg: (H,W,3) RGB image if return_colors=True, else None
    """
    # Find top-k largest components
    unique_labels, counts = np.unique(segmentation, return_counts=True)

    # Filter out ignore label
    valid_mask = unique_labels != ignore_label
    unique_labels = unique_labels[valid_mask]
    counts = counts[valid_mask]

    # Sort by size and take top-k
    sorted_indices = np.argsort(-counts)
    top_k_labels = unique_labels[sorted_indices[:k]]

    # Create visualization
    colored_seg = np.zeros((*segmentation.shape, 3), dtype=np.uint8)

    # Get colormap
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Assign colors to top-k planes
    for i, label in enumerate(top_k_labels):
        color = np.array(cmap_obj(i / max(1, k - 1))[:3]) * 255
        colored_seg[segmentation == label] = color.astype(np.uint8)

    if return_colors:
        return colored_seg

    # Display
    plt.figure(figsize=(10, 8))
    plt.imshow(colored_seg)
    plt.title(f'Top-{k} Largest Plane Segments')
    plt.axis('off')
    plt.show()



def generate_plane_colors(
    n_planes: int,
    seed: int = 42
) -> np.ndarray:
    """
    Generate distinct colors for plane visualization.

    Args:
        n_planes: Number of planes/colors needed
        seed: Random seed for reproducibility

    Returns:
        colors: (n_planes,3) RGB colors in range [0,255]
    """
    np.random.seed(seed)

    if n_planes <= 20:
        # Use matplotlib tab20 for small numbers
        cmap = plt.get_cmap('tab20')
        colors = np.array([cmap(i / 20)[:3] for i in range(n_planes)]) * 255
    else:
        # Generate random distinct colors
        colors = np.random.randint(0, 256, size=(n_planes, 3))

    return colors.astype(np.uint8)


def visualize_segmentation_comparison(
    rgb: np.ndarray,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    title: str = "Segmentation Comparison"
) -> None:
    """
    Visualize RGB, ground truth, and predicted segmentations side-by-side.

    Args:
        rgb: (H,W,3) RGB image
        gt_seg: (H,W) ground truth segmentation
        pred_seg: (H,W) predicted segmentation
        title: Figure title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')

    axes[1].imshow(gt_seg, cmap='tab20')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(pred_seg, cmap='tab20')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_normals(
    normal_map: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert surface normals to RGB visualization.

    Args:
        normal_map: (H,W,3) normal vectors (range [-1,1])
        mask: (H,W) optional binary mask

    Returns:
        vis: (H,W,3) RGB visualization in range [0,255]
    """
    if mask is not None:
        mask = np.expand_dims(mask, axis=2)
        vis = normal_map * mask + mask - 1
    else:
        vis = normal_map

    # Transform [-1,1] -> [0,1]
    vis = (1 - vis) / 2
    vis = (vis * 255).astype(np.uint8)

    return vis
