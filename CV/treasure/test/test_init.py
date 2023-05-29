from pathlib import Path

import cv2
from treasure import find_treasure
from translate import coord_to_index
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


def test_find_treasure_deployment():
    cap = cv2.VideoCapture(f"{TEST_DATA_DIR}/test.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    accumulator = 0
    last_coords, coords = None, None
    while ret:
        ret, frame = cap.read()
        raw_coords = find_treasure(frame)
        if not raw_coords:
            continue
        coords = coord_to_index(raw_coords)
        if not coords:
            continue
        if last_coords is None:
            last_coords = coords
            continue
        if coords == last_coords:
            accumulator += 1
        else:
            accumulator = 0
        if accumulator > 10:
            break
    cap.release()
    assert accumulator > 10
    assert sorted(coords) == sorted(
        [(13, 1), (5, 1), (3, 3), (15, 5), (5, 15), (17, 17), (15, 19), (7, 19)]
    )
