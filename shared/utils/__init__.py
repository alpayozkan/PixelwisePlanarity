"""
Shared utilities for depth/normal processing, visualization, and I/O.
"""

from .depth_normal import (
    depth_to_normal_remi,
    extract_zdepth,
    depth_to_3d,
    vector_normalization
)
from .label_utils import (
    keep_top_k_planes,
    remap_labels,
    fill_holes_inpaint,
    match_planes_by_overlap,
    map_array
)
from .visualization import (
    visualize_top_components_v1,
    generate_plane_colors,
    visualize_segmentation_comparison,
    visualize_normals
)
from .io_utils import (
    save_label_image,
    load_h5_dataset,
    save_h5_dataset,
    remap_semantic
)

__all__ = [
    # depth_normal
    'depth_to_normal_remi',
    'extract_zdepth',
    'depth_to_3d',
    'vector_normalization',
    # label_utils
    'keep_top_k_planes',
    'remap_labels',
    'fill_holes_inpaint',
    'match_planes_by_overlap',
    'map_array',
    # visualization
    'visualize_top_components_v1',
    'generate_plane_colors',
    'visualize_segmentation_comparison',
    'visualize_normals',
    # io_utils
    'save_label_image',
    'load_h5_dataset',
    'save_h5_dataset',
    'remap_semantic',
]
