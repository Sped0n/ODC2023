from pathlib import Path

import cv2
import numpy as np
from treasure.locate import find_locating_boxes, get_locating_coords

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_find_locating_boxes():
    img = cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg", cv2.IMREAD_GRAYSCALE)
    (h, w) = img.shape[:2]
    r: float = 480 / float(h)
    dim = (int(w * r), 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    boxes = find_locating_boxes(img)
    for box in boxes:
        assert box.all()


def test_find_locating_boxes_debug_enable():
    img = cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg", cv2.IMREAD_GRAYSCALE)
    (h, w) = img.shape[:2]
    r: float = 480 / float(h)
    dim = (int(w * r), 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 1)
    boxes, length_pack = find_locating_boxes(img, debug=True)
    for box in boxes:
        assert box.all()
    assert len(length_pack) == 4
    for length in length_pack:
        assert length > 0


def test_get_locating_coords():
    input_data = np.load(f"{TEST_DATA_DIR}/raw_boxes.npy", allow_pickle=True)
    ref = [(176, 104), (389, 150), (141, 364), (405, 368)]
    res = get_locating_coords(input_data, 10)
    assert sorted(res) == sorted(ref)
