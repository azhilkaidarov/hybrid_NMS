from __future__ import annotations

import numpy as np

from .boxes import Bbox


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "OpenCV is required for visualization. Install dependencies (e.g. `uv sync`) "
            "or run `pip install opencv-python`."
        ) from e
    return cv2


def draw_bbox(image: np.ndarray, bbox: Bbox, thickness: int = 3) -> np.ndarray:
    """Draw a single bbox on an image (in-place) and return the image."""
    cv2 = _require_cv2()
    cv2.rectangle(
        img=image,
        pt1=(bbox.x_min, bbox.y_min),
        pt2=(bbox.x_max, bbox.y_max),
        color=bbox.color,
        thickness=thickness,
    )

    text = f"{bbox.label} conf=({round(bbox.confidence, 2)})"
    font = cv2.FONT_ITALIC
    font_scale = 0.6
    text_org = (bbox.x_min, max(0, bbox.y_min - 8))

    cv2.putText(
        img=image,
        text=text,
        org=text_org,
        fontFace=font,
        fontScale=font_scale,
        color=bbox.color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return image


def draw_bboxes(image: np.ndarray, bboxes: list[Bbox], thickness: int = 3) -> np.ndarray:
    """Draw multiple bboxes on an image (in-place) and return the image."""
    for bbox in bboxes:
        draw_bbox(image, bbox, thickness=thickness)
    return image

