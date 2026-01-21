from __future__ import annotations

import argparse
from pathlib import Path


def _repo_root() -> Path:
    # <repo>/src/bounding_box_project/__main__.py  -> parents[2] == <repo>
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    repo_root = _repo_root()

    parser = argparse.ArgumentParser(description="Bounding box NMS demo")
    parser.add_argument(
        "--image",
        type=Path,
        default=repo_root / "data" / "sample" / "sample_image.png",
        help="Path to input image",
    )
    parser.add_argument("--seed", type=int, default=4, help="RNG seed")
    parser.add_argument("--n", type=int, default=10, help="Number of random boxes")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="IoU threshold for suppression (stage 2)",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.60,
        help="IoU threshold for merging (stage 1)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show OpenCV windows (requires GUI)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print NMS debug output")

    args = parser.parse_args(argv)

    # Delay imports so `--help` works even if optional deps (opencv) aren't installed yet.
    from .image_io import load_image, save_image
    from .nms import hybrid_nms
    from .synthetic import random_bboxes
    from .viz import draw_bboxes

    # Load once to generate boxes (we'll reload for the second visualization)
    image = load_image(args.image)
    bboxes = random_bboxes(image, n=args.n, seed=args.seed)

    before = draw_bboxes(image.copy(), bboxes)
    before_path = args.out_dir / "before_nms_output.jpg"
    save_image(before_path, before)

    # Reload original image for the "after"
    image2 = load_image(args.image)
    bboxes_after = hybrid_nms(
        bboxes,
        args.threshold,
        merge_iou_threshold=args.merge_threshold,
        verbose=args.verbose,
    )
    after = draw_bboxes(image2, bboxes_after)
    after_path = args.out_dir / "after_nms_output.jpg"
    save_image(after_path, after)

    if args.show:
        import cv2  # type: ignore

        cv2.imshow("Bounding Boxes before NMS", before)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imshow("Bounding Boxes after NMS", after)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Saved: {before_path}")
    print(f"Saved: {after_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

