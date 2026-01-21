"""Bounding box study project: IoU + NMS + visualization utilities."""

from .boxes import Bbox
from .iou import iou
from .nms import hybrid_nms

__all__ = [
    "Bbox",
    "hybrid_nms",
    "iou",
]