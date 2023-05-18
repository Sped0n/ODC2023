import cv2
import numpy as np


def treasure_identification(
    frame, min_radius=8, max_radius=22, min_dist=40, param1=50, param2=16
) -> list[tuple[int, int, int]]:
    """
    find all treasures
    :param frame: grayscale and gaussian blur processed input image
    :param min_radius: minimum radius of the circle
    :param max_radius: maximum radius of the circle
    :param min_dist: minimum distance between two circles
    :param param1: threshold for canny edge detection
    :param param2: threshold for accumulator
    :return: coordinates of all treasures
    """
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
    coordinates: list[tuple[int, int, int]] = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            if 150 < x < 650 and 150 < y < 650:
                coordinates.append((x, y, r))
    return coordinates
