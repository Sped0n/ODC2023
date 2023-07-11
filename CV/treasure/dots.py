import cv2
import numpy as np

from ctyper import Image, QuantityMismatch, Treasure


def treasure_identification(
    frame: Image,
    min_radius: int = 8,
    max_radius: int = 22,
    min_dist: int = 40,
    param1: int = 50,
    param2: int = 16,
) -> list[Treasure]:
    """
    find all treasures
    :param frame: grayscale and gaussian blur processed input image
    :param min_radius: minimum radius of the circle
    :param max_radius: maximum radius of the circle
    :param min_dist: minimum distance between two circles
    :param param1: threshold for canny edge detection
    :param param2: threshold for accumulator
    :return: when dots number is valid, return treasures of all treasures
    """
    if frame.ndim != 2:
        raise ValueError("frame must be a grayscale image")
    circles = cv2.HoughCircles(
        frame,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    treasures: list[Treasure] = []
    if circles is not None:
        # convert the (x, y) coordiantes and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            # take the circles that in region of interest
            if 150 < x < 650 and 150 < y < 650:
                treasures.append((x, y, r))
    if len(treasures) != 8:
        raise QuantityMismatch("treasure number is not 8")
    return treasures
