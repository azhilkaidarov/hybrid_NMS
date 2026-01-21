from __future__ import annotations

import numpy as np

from .boxes import Bbox


def random_bboxes(
    image: np.ndarray,
    n: int = 10,
    *,
    seed: int | None = None,
    min_wh: tuple[int, int] = (20, 20),
    max_wh: tuple[int, int] | None = None,
) -> list[Bbox]:
    """Generate N random bboxes for a given image (for demos/tests)."""
    h, w = image.shape[:2]
    rng = np.random.default_rng(seed)

    if max_wh is None:
        max_wh = (max(min_wh[0], w // 2), max(min_wh[1], h // 2))

    max_w = max(1, min(max_wh[0], w - 1))
    max_h = max(1, min(max_wh[1], h - 1))
    min_w = min(min_wh[0], max_w)
    min_h = min(min_wh[1], max_h)

    bboxes: list[Bbox] = []
    used_sizes: set[tuple[int, int]] = set()

    while len(bboxes) < n:
        bw = int(rng.integers(min_w, max_w + 1))
        bh = int(rng.integers(min_h, max_h + 1))

        # Keep sizes unique (purely for nicer visualization)
        if (bw, bh) in used_sizes:
            continue
        used_sizes.add((bw, bh))

        x_min = int(rng.integers(0, w - bw))
        y_min = int(rng.integers(0, h - bh))
        x_max = x_min + bw
        y_max = y_min + bh

        confidence = float(rng.uniform(0.1, 1.0))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))  # BGR for OpenCV

        bboxes.append(
            Bbox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=confidence,
                label=f"rand_{len(bboxes)}",
                color=color,
            )
        )

    return bboxes

