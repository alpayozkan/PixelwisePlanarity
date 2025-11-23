"""
I/O utilities for saving and loading data.

This module provides functions for:
- HDF5 file I/O
- Image saving with labels
- Semantic label remapping
"""

import numpy as np
import h5py
import cv2
from typing import Dict, Optional


def save_label_image(
    label_img: np.ndarray,
    output_path: str,
    remap_dict: Optional[Dict[int, int]] = None
) -> None:
    """
    Save integer label image as PNG.

    Args:
        label_img: (H,W) int32 label map
        output_path: Path to save PNG file
        remap_dict: Optional mapping to remap labels before saving
    """
    if remap_dict is not None:
        remapped = np.full_like(label_img, fill_value=0)
        for old_val, new_val in remap_dict.items():
            remapped[label_img == old_val] = new_val
        label_img = remapped

    # Ensure valid range for PNG
    label_img = np.clip(label_img, 0, 65535).astype(np.uint16)
    cv2.imwrite(output_path, label_img)


def load_h5_dataset(
    h5_path: str,
    dataset_name: str
) -> np.ndarray:
    """
    Load dataset from HDF5 file.

    Args:
        h5_path: Path to HDF5 file
        dataset_name: Name of dataset within file

    Returns:
        data: Loaded numpy array
    """
    with h5py.File(h5_path, 'r') as f:
        data = f[dataset_name][:]
    return data


def save_h5_dataset(
    h5_path: str,
    dataset_name: str,
    data: np.ndarray,
    compression: str = 'gzip'
) -> None:
    """
    Save numpy array to HDF5 file.

    Args:
        h5_path: Path to HDF5 file
        dataset_name: Name for dataset within file
        data: Numpy array to save
        compression: Compression method ('gzip', 'lzf', or None)
    """
    with h5py.File(h5_path, 'a') as f:
        if dataset_name in f:
            del f[dataset_name]  # Overwrite if exists
        f.create_dataset(dataset_name, data=data, compression=compression)


def remap_semantic(
    semantic_img: np.ndarray,
    id_to_name: Dict[int, str],
    name_to_target: Dict[str, int]
) -> np.ndarray:
    """
    Remap semantic labels using two-stage lookup.

    Args:
        semantic_img: (H,W) semantic label image
        id_to_name: Mapping from original ID to semantic class name
        name_to_target: Mapping from class name to target ID

    Returns:
        remapped: (H,W) remapped semantic image
    """
    remapped = np.zeros_like(semantic_img)

    for original_id, class_name in id_to_name.items():
        if class_name in name_to_target:
            target_id = name_to_target[class_name]
            remapped[semantic_img == original_id] = target_id

    return remapped
