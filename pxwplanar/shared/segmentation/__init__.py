"""
Planar segmentation algorithms and post-processing.
"""

from .merge import merge
from .plan2seg import compute_planar_segments, filter_small_segments
from .postprocess import (
    plane_merge,
    postmerge_adjacent_segments,
    remove_small_components,
)

__all__ = [
    "compute_planar_segments",
    "filter_small_segments",
    "remove_small_components",
    "plane_merge",
    "postmerge_adjacent_segments",
    "merge",
]
