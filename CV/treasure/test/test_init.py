from pathlib import Path

import cv2
from treasure import find_treasure
from utils import image_resize

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_find_treasure():
    img = image_resize(cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg"), height=480)
    assert sorted(find_treasure(img)) == sorted(
        [
            (1, 8),
            (10, 3),
            (2, 9),
            (1, 4),
            (8, 8),
            (9, 2),
            (3, 3),
            (10, 7),
        ]
    )


def test_find_treasure_debug_enable():
    img = image_resize(cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg"), height=480)
    treasures, o_frame, c_frame = find_treasure(img, debug=True)
    assert sorted(treasures) == sorted(
        [
            (1, 8),
            (10, 3),
            (2, 9),
            (1, 4),
            (8, 8),
            (9, 2),
            (3, 3),
            (10, 7),
        ]
    )
    assert o_frame.shape[-1] == 3
    assert c_frame.shape == (800, 800, 3)
