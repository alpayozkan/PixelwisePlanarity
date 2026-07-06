"""
ScanNet++ ground truth generation for plane segmentation.
"""

from .plane_extraction import run as extract_planes

__all__ = ['extract_planes']
