from __future__ import annotations

from .boxes import Bbox


def iou(bbox_1: Bbox, bbox_2: Bbox) -> float:
    """Intersection over Union (IoU) for two bboxes in xyxy format."""
    x1_min, x1_max = bbox_1.x_min, bbox_1.x_max
    y1_min, y1_max = bbox_1.y_min, bbox_1.y_max

    if x1_max <= x1_min or y1_max <= y1_min:
        return 0.0

    bbox_1_area = (x1_max - x1_min) * (y1_max - y1_min)

    x2_min, x2_max = bbox_2.x_min, bbox_2.x_max
    y2_min, y2_max = bbox_2.y_min, bbox_2.y_max

    if x2_max <= x2_min or y2_max <= y2_min:
        return 0.0

    bbox_2_area = (x2_max - x2_min) * (y2_max - y2_min)

    i_x_min = max(x2_min, x1_min)
    i_y_min = max(y2_min, y1_min)
    i_x_max = min(x2_max, x1_max)
    i_y_max = min(y2_max, y1_max)

    width = max(i_x_max - i_x_min, 0)
    height = max(i_y_max - i_y_min, 0)
    i_area = width * height

    denom = bbox_1_area + bbox_2_area - i_area
    if denom <= 0:
        return 0.0

    return i_area / denom

