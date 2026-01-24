from collections import defaultdict

from boxes import Bbox
from iou import iou


def union(parents: list[int], a: int, b: int) -> None:
    root_a = find(parents, a)
    root_b = find(parents, b)

    if root_a != root_b:
        parents[root_b] = root_a


def find(parents: list[int], x: int) -> int:
    if parents[x] != x:
        parents[x] = find(parents, parents[x])
    return parents[x]


def delete_bboxes(bboxes: list[Bbox], to_del: set[int]) -> list[Bbox]:
    return [bboxes[i] for i in range(len(bboxes)) if i not in to_del]


def merge(bboxes: list[Bbox], color=(180, 105, 255)) -> Bbox:
    x_min = min(box.x_min for box in bboxes)
    y_min = min(box.y_min for box in bboxes)
    x_max = max(box.x_max for box in bboxes)
    y_max = max(box.y_max for box in bboxes)
    confidence = max(box.confidence for box in bboxes)

    # BGR for OpenCV
    return Bbox(x_min, y_min, x_max, y_max, confidence, "MERGED OBJ", color)


def hybrid_nms(
    bboxes: list[Bbox],
    threshold: float,
    *,
    merge_iou_threshold: float = 0.60,
    verbose: bool = False,
) -> list[Bbox]:
    """Two-stage NMS:
    - Stage 1: merge bboxes with IoU >= merge_iou_threshold (Union-Find)
    - Stage 2: for remaining overlaps (IoU > threshold), keep only higher-confidence bbox
    """

    if verbose:
        print("\n-----START NMS-----\n")
        print(f"--We got {len(bboxes)} bboxes.\n")

    length = len(bboxes)
    parents = list(range(length))

    # Stage 1: merge clusters
    if verbose:
        print("--Start STEP №1")

    for i in range(length - 1):
        for j in range(i + 1, length):
            if iou(bboxes[i], bboxes[j]) >= merge_iou_threshold:
                union(parents, i, j)

    if verbose:
        print(f"-parents: list[int] = {parents}")
        if len(set(parents)) == len(parents):
            print("-No bboxes to merge!")
        else:
            print("-There's some bboxes to merge!")

    group_bboxes: dict[int, list[int]] = defaultdict(list)
    for i in range(length):
        root = find(parents, i)
        group_bboxes[root].append(i)

    # Merge each group into one bbox (if group size > 1)
    for group in group_bboxes.values():
        if len(group) > 1:
            merged = merge([bboxes[i] for i in group], )
            bboxes = delete_bboxes(bboxes, set(group))
            bboxes.append(merged)

    # Stage 2: delete lower-confidence overlaps
    if verbose:
        print("\n--Start STEP №2")

    length = len(bboxes)
    to_die: list[int] = []

    for i in range(length - 1):
        for j in range(i + 1, length):
            if iou(bboxes[i], bboxes[j]) > threshold:
                to_die.append(j if bboxes[i].confidence >= bboxes[j].confidence else i)

    to_die_set = sorted(set(to_die))
    if verbose:
        print(f"These bboxes {[bboxes[idx].label for idx in to_die_set]} will be deleted!\n")

    return delete_bboxes(bboxes, set(to_die_set))

