from pathlib import Path

import cv2
from treasure.dots import treasure_identification

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_treasure_identification():
    img = cv2.imread(f"{TEST_DATA_DIR}/corrected.jpg", cv2.IMREAD_GRAYSCALE)
    treasures = treasure_identification(img)
    assert len(treasures) == 8
