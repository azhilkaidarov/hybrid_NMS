from dataclasses import dataclass


@dataclass
class Bbox:
    """Bounding box in (x_min, y_min, x_max, y_max)
        format + confidence/label/color.

    Notes:
    - `color` is in **BGR** order (OpenCV convention).
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    label: str = "object"
    color: tuple[int, int, int] = (0, 255, 0)
