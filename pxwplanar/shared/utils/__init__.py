"""
Shared utilities for depth/normal processing, visualization, and I/O.
"""

from .depth_normal import (
    depth_to_3d,
    depth_to_normal_remi,
    extract_zdepth,
    vector_normalization,
)
from .io_utils import (
    load_h5_dataset,
    remap_semantic,
    save_h5_dataset,
    save_label_image,
)
from .label_utils import (
    fill_holes_inpaint,
    keep_top_k_planes,
    map_array,
    match_planes_by_overlap,
    remap_labels,
    remap_labels_fast,
)
from .visualization import (
    generate_plane_colors,
    visualize_normals,
    visualize_segmentation_comparison,
    visualize_top_components,
)

__all__ = [
    # depth_normal
    "depth_to_normal_remi",
    "extract_zdepth",
    "depth_to_3d",
    "vector_normalization",
    # label_utils
    "keep_top_k_planes",
    "remap_labels",
    "remap_labels_fast",
    "fill_holes_inpaint",
    "match_planes_by_overlap",
    "map_array",
    # visualization
    "visualize_top_components",
    "generate_plane_colors",
    "visualize_segmentation_comparison",
    "visualize_normals",
    # io_utils
    "save_label_image",
    "load_h5_dataset",
    "save_h5_dataset",
    "remap_semantic",
]
