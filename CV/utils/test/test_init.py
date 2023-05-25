from pathlib import Path

import cv2
from utils import image_resize

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_image_resize_width():
    image = cv2.imread(f"{TEST_DATA_DIR}/test.jpg")
    resized = image_resize(image, width=400)
    assert resized.shape[1] == 400
    assert resized.shape[2] == image.shape[2]


def test_image_resize_height():
    image = cv2.imread(f"{TEST_DATA_DIR}/test.jpg")
    resized = image_resize(image, height=400)
    assert resized.shape[0] == 400
    assert resized.shape[2] == image.shape[2]
