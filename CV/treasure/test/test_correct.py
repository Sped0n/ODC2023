from pathlib import Path

import cv2
import numpy as np
from treasure.correct import img_correction, ricd

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_img_correction():
    img = cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg")
    # resize image
    (h, w) = img.shape[:2]
    r: float = 480 / float(h)
    dim = (int(w * r), 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # correct image
    c_img = img_correction(img, [(176, 104), (389, 150), (141, 364), (405, 368)]).result
    assert c_img.shape == (800, 800, 3)


def test_img_correction_tmap_enable():
    img = cv2.imread(f"{TEST_DATA_DIR}/test_pattern.jpg")
    # resize image
    (h, w) = img.shape[:2]
    r: float = 480 / float(h)
    dim = (int(w * r), 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # correct image
    t_map = img_correction(
        img, [(176, 104), (389, 150), (141, 364), (405, 368)]
    ).transform_map
    assert len(t_map) == 3
    for vec in t_map:
        assert len(vec) == 3


def test_ricd():
    original_img = cv2.imread(f"{TEST_DATA_DIR}/corrected.jpg")
    test_img_list = [original_img]
    # get 4 different rotated images
    for i in range(1, 4):
        method: int = -1
        match i:
            case 1:
                method = cv2.ROTATE_90_CLOCKWISE
            case 2:
                method = cv2.ROTATE_180
            case 3:
                method = cv2.ROTATE_90_COUNTERCLOCKWISE
        test_img_list.append(cv2.rotate(original_img, method))
    for img in test_img_list:
        img = ricd(img)
        # convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # use blue mask to find the blue block
        blue_mask: np.ndarray = cv2.inRange(hsv, (100, 80, 46), (124, 255, 255))
        eroded_blue_mask: np.ndarray = cv2.erode(blue_mask, None, iterations=2)
        cnts = cv2.findContours(
            eroded_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        # find the largest contour
        cnt: np.ndarray = max(cnts, key=cv2.contourArea)
        # find the center of the contour
        m: dict[str, float] = cv2.moments(cnt)
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        assert cx < 400 < cy
