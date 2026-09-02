"""
Visualization utilities for plane fitting results.

Consolidated from plane_fitting/*visualize.py
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_plane_segmentation(
    rgb_image: np.ndarray,
    segmentation: np.ndarray,
    title: str = "Plane Segmentation",
    cmap: str = "tab20",
) -> None:
    """
    Visualize plane segmentation overlaid on RGB image.

    Args:
        rgb_image: (H,W,3) RGB image
        segmentation: (H,W) plane labels
        title: Plot title
        cmap: Colormap for plane labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")

    axes[1].imshow(segmentation, cmap=cmap)
    axes[1].set_title(title)
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
