from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "OpenCV is required for image I/O. Install dependencies (e.g. `uv sync`) "
            "or run `pip install opencv-python`."
        ) from e
    return cv2


def load_image(image_path: str | Path) -> np.ndarray:
    cv2 = _require_cv2()
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

    return image


def save_image(output_path: str | Path, image: np.ndarray) -> None:
    cv2 = _require_cv2()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise OSError(f"Failed to write image: {output_path}")

