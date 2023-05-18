from pathlib import Path

import cv2
import numpy as np
from treasure.correct import img_correction, ricd
from utils import image_resize

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_img_correction():
    img = image_resize(cv2.imread(f"{TEST_DATA_DIR}/test_pattern2.jpg"), width=480)
    c_img = img_correction(img, [(176, 104), (389, 150), (141, 364), (405, 368)])
    assert c_img.shape == (800, 800, 3)


def test_ricd():
    img = ricd(cv2.imread(f"{TEST_DATA_DIR}/corrected.jpg"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask: np.ndarray = cv2.inRange(hsv, (100, 80, 46), (124, 255, 255))
    eroded_blue_mask: np.ndarray = cv2.erode(blue_mask, None, iterations=2)
    cnts: tuple[np.ndarray] = cv2.findContours(
        eroded_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    cnt: np.ndarray = max(cnts, key=cv2.contourArea)
    m: dict[str, float] = cv2.moments(cnt)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    assert cx < 400 < cy
