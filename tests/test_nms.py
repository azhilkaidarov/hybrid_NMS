from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bounding_box_project import Bbox  # noqa: E402
from bounding_box_project import hybrid_nms  # noqa: E402


class TestHybridNms(unittest.TestCase):
    def test_deletes_lower_confidence_when_overlapping(self) -> None:
        # IoU ~ 0.47 (below merge threshold 0.60, above delete threshold 0.2)
        hi = Bbox(0, 0, 10, 10, confidence=0.9, label="hi")
        lo = Bbox(2, 2, 12, 12, confidence=0.1, label="lo")

        out = hybrid_nms([hi, lo], threshold=0.2, verbose=False)
        labels = {b.label for b in out}

        self.assertIn("hi", labels)
        self.assertNotIn("lo", labels)

    def test_merges_high_iou_cluster(self) -> None:
        # IoU = 64/100 = 0.64 >= merge threshold
        b1 = Bbox(0, 0, 10, 10, confidence=0.4, label="a")
        b2 = Bbox(1, 1, 9, 9, confidence=0.9, label="b")

        out = hybrid_nms(
            [b1, b2],
            threshold=0.2,
            merge_iou_threshold=0.60,
            verbose=False,
        )

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].label, "MERGED OBJ")


if __name__ == "__main__":
    unittest.main()
