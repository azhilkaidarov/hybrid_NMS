from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bounding_box_project import Bbox  # noqa: E402
from bounding_box_project import iou  # noqa: E402


class TestIoU(unittest.TestCase):
    def test_identical_is_one(self) -> None:
        b = Bbox(0, 0, 10, 10, confidence=1.0)
        self.assertAlmostEqual(iou(b, b), 1.0)

    def test_no_overlap_is_zero(self) -> None:
        b1 = Bbox(0, 0, 10, 10, confidence=1.0)
        b2 = Bbox(20, 20, 30, 30, confidence=1.0)
        self.assertAlmostEqual(iou(b1, b2), 0.0)

    def test_partial_overlap(self) -> None:
        b1 = Bbox(0, 0, 10, 10, confidence=1.0)
        b2 = Bbox(5, 5, 15, 15, confidence=1.0)
        # Intersection area: 5*5=25, union: 100+100-25=175
        self.assertAlmostEqual(iou(b1, b2), 25 / 175)


if __name__ == "__main__":
    unittest.main()
