"""
Post-processing utilities for plane segmentations.

This module provides functions for cleaning and refining segmentation results.
"""

import numpy as np
from scipy import ndimage


def remove_small_components(
    label_map: np.ndarray,
    min_size: int
) -> np.ndarray:
    """
    Remove small connected components from segmentation.

    Args:
        label_map: (H,W) labeled segmentation map
        min_size: Minimum number of pixels for a component to be kept

    Returns:
        cleaned_label_map: (H,W) cleaned segmentation with small components removed
    """
    cleaned_label_map = np.zeros_like(label_map)
    current_label = 1

    unique_labels = np.unique(label_map)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background

        mask = (label_map == label)
        labeled_mask, num_features = ndimage.label(mask)
        component_sizes = np.bincount(labeled_mask.ravel())

        # Keep only components >= min_size
        for i in range(1, num_features + 1):
            if component_sizes[i] >= min_size:
                cleaned_label_map[labeled_mask == i] = current_label
                current_label += 1

    return cleaned_label_map
