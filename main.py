from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from collections import defaultdict

import cv2
import numpy as np


# СОЗДАНИЕ РАНДОМНЫХ БОКСОВ
@dataclass
class Bbox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    label: str = "object"
    color: tuple[int, int, int] = (0, 255, 0)


# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Image not found or cannot be read: {image_path}"
        )

    return image


# РИСУЕМ БОКС
def draw_bbox(image: np.ndarray, bbox: Bbox, thickness: int = 3) -> np.ndarray:
    cv2.rectangle(
        img=image,
        pt1=(bbox.x_min, bbox.y_min),
        pt2=(bbox.x_max, bbox.y_max),
        color=bbox.color,
        thickness=thickness,
    )

    # ДОБАВЛЯЕМ ПОДПИСЬ
    text = bbox.label
    font = cv2.FONT_ITALIC
    font_scale = 1

    # КООРДИНАТЫ ТЕКСТА ЧУТЬ ВЫШЕ ВЕРХНЕГО ЛЕВОГО УГЛА
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


# РИСУЕМ МНОЖЕСТВО БОКСОВ
def draw_bboxes(image: np.ndarray, bboxes: list[Bbox]) -> np.ndarray:
    for bbox in bboxes:
        image = draw_bbox(image, bbox)
    return image


# ВЫЧИСЛЕНИЕ IoU
def iou(bbox_1: Bbox, bbox_2: Bbox) -> float:
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

    # i - stands for intesection
    i_x_min = max(x2_min, x1_min)
    i_y_min = max(y2_min, y1_min)
    i_x_max = min(x2_max, x1_max)
    i_y_max = min(y2_max, y1_max)

    width = max(i_x_max - i_x_min, 0)
    heigth = max(i_y_max - i_y_min, 0)
    i_area = width * heigth

    # denom - from zero division error
    denom = bbox_1_area + bbox_2_area - i_area
    if denom <= 0:
        return 0.0

    iou_value = i_area / denom

    return iou_value


# UNION-FIND ALGORITHM
def union(parents: list[int], a: int, b: int):
    root_a = find(parents, a)
    root_b = find(parents, b)

    if root_a != root_b:
        parents[root_b] = root_a


# UNION-FIND ALGORITHM
def find(parents: list[int], x: int):
    if parents[x] != x:
        parents[x] = find(parents, parents[x])
    return parents[x]


# УДАЛЯЕМ СТАРЫЕ БОКСЫ ПОСЛЕ СЛИЯНИЯ ИЛИ В РЕЗУЛЬТАТЕ МЕНЬШЕГО CONFIDENCE
def delete_bboxes(bboxes: list[Bbox], to_del: set[int]) -> list[Bbox]:
    return [bboxes[i] for i in range(len(bboxes)) if i not in to_del]


# СЛИТЬ В ОДИН БОКС ВСЕ ПЕРЕДАННЫЕ БОКСЫ И ВЕРНУТЬ ЕГО
def merge(bboxes: list[Bbox]) -> Bbox:
    x_min = min([box.x_min for box in bboxes])
    y_min = min([box.y_min for box in bboxes])
    x_max = max([box.x_max for box in bboxes])
    y_max = max([box.y_max for box in bboxes])
    confidence = max([box.confidence for box in bboxes])
    color = (180, 105, 255)

    new_bbox = Bbox(x_min, y_min, x_max, y_max, confidence, "MERGED OBJ", color)
    return new_bbox


# РУЧНОЙ NMS
def hybrid_nms(bboxes: list[Bbox], treshold: float) -> list[Bbox]:
    length = len(bboxes)
    # INDEX - ИНДЕКС ОПРЕДЕЛЕННОГО BBOX в BBOXES, VALUE - КОРЕНЬ, КУДА ОН БУДЕТ СЛИВАТЬСЯ ПОЗЖЕ
    parents = list(range(length))

    # ЭТАП №1 - MERGE BBOXES WITH IoU > 0.75 ----------------------------------
    # ШАГ 1. ПОПАРНО ВСЕ ПРОВЕРИМ. У СЛИВАЮЩИХСЯ BOX'ов в PARENTS ДОЛЖЕН БЫТЬ УКАЗАН КОРЕНЬ СЛИЯНИЯ
    for i in range(length-1):
        for j in range(i+1, length):
            if iou(bboxes[i], bboxes[j]) >= 0.75:
                union(parents, i, j)
    # parents = [0,0,0,0,0,0,6] ПОЛУЧИЛИ ВСЕ КОРНИ СЛИЯНИЯ

    # ШАГ 2. РАЗДЕЛИМ ВСЕ СЛИЯНИЯ НА ОТДЕЛЬНЫЕ ГРУППЫ
    group_bboxes = defaultdict(list)

    for i in range(length):
        root = find(parents, i)
        group_bboxes[root].append(i)
    # print(group_bboxes) -> defaultdict(<class 'list'>, {0: [0], 1: [1, 2]})

    # ШАГ 3. ПЕРЕДАЕМ MERGE() КАЖДУЮ ГРУППУ. СТАРЫЕ ББОКСЫ УДАЛЯЕМ И ДОБАВЛЯЕМ НОВЫЙ
    for group in group_bboxes.values():
        if len(group) > 1:
            new_box = merge(group)
            bboxes = delete_bboxes(bboxes, set(group))
            bboxes.append(new_box)

    length = len(bboxes)

    # ЭТАП №2 - DELETE BBOX WITH LESS CONFIDENCE VALUE ------------------------
    to_die = []
    for i in range(length-1):
        for j in range(i + 1, length):
            if iou(bboxes[i], bboxes[j]) > treshold:
                print("SOMETHING IS HERE")
                if bboxes[i].confidence >= bboxes[j].confidence:
                    l_c_bbox = j
                else:
                    l_c_bbox = i
                to_die.append(l_c_bbox)
    bboxes = delete_bboxes(bboxes, set(to_die))

    return bboxes
    # ЭТАП №3 - РАДОВАТЬСЯ! ---------------------------------------------------


# ОТ CHATGPT
def random_bboxes(
    image: np.ndarray,
    n: int = 10,
    *,
    seed: int | None = None,
    min_wh: tuple[int, int] = (20, 20),
    max_wh: tuple[int, int] | None = None,
) -> list[Bbox]:
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
        if (bw, bh) in used_sizes:
            continue
        used_sizes.add((bw, bh))

        x_min = int(rng.integers(0, w - bw))
        y_min = int(rng.integers(0, h - bh))
        x_max = x_min + bw
        y_max = y_min + bh

        confidence = float(rng.uniform(0.1, 1.0))
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))  # BGR для OpenCV

        bboxes.append(Bbox(x_min, y_min, x_max, y_max, confidence, label=f"rand_{len(bboxes)}", color=color))

    return bboxes


def main():
    image_path = "sample_image.png"
    image = load_image(image_path)
    # ВЫБЕРИ РАЗМЕР И СИД <----------------------------------------
    bboxes = random_bboxes(image, n=10, seed=42)

    draw_bboxes(image, bboxes)

    cv2.imshow("Bounding Boxes before NMS", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("before_nms_output.jpg", image)

    image_path = "sample_image.png"
    image = load_image(image_path)
    new_bboxes = hybrid_nms(bboxes, 0.4)
    draw_bboxes(image, new_bboxes)
    cv2.imshow("Bounding Boxes after NMS", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("after_nms_output.jpg", image)


if __name__ == "__main__":
    main()
