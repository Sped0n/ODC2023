from pathlib import Path

import cv2
import numpy as np
from treasure.locate import find_locating_boxes, filter_locating_boxes

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_find_locating_boxes():
    img = cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg", cv2.IMREAD_GRAYSCALE)
    # resize image
    (h, w) = img.shape[:2]
    r: float = 480 / float(h)
    dim = (int(w * r), 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # blur image
    img = cv2.GaussianBlur(img, (3, 3), 1)
    # find locating boxes
    for box in find_locating_boxes(img).boxes:
        assert box.all()


def test_find_locating_boxes_debug_enable():
    img = cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg", cv2.IMREAD_GRAYSCALE)
    # resize image
    (h, w) = img.shape[:2]
    r: float = 480 / float(h)
    dim = (int(w * r), 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # blur image
    img = cv2.GaussianBlur(img, (3, 3), 1)
    # find locating boxes
    result = find_locating_boxes(img)
    boxes = result.boxes
    cnt_quantity_list = result.debug
    for box in boxes:
        assert box.all()
    assert len(cnt_quantity_list) == 4
    last_cnt_quantity = float("inf")
    for quantity in cnt_quantity_list:
        assert 0 < quantity <= last_cnt_quantity
        last_cnt_quantity = quantity


def test_filter_locating_boxes():
    input_data = np.load(f"{TEST_DATA_DIR}/raw_boxes.npy", allow_pickle=True)
    ref = [(176, 104), (389, 150), (141, 364), (405, 368)]
    res = filter_locating_boxes(input_data, 10)
    assert sorted(res) == sorted(ref)
