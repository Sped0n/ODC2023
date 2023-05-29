import math

import cv2
import numpy as np


def p2p_distance(point1: tuple[int, int], point2: tuple[int, int]) -> float:
    """
    calculate the distance between two points

    >>> p2p_distance((0, 0), (3, 4))
    5.0

    :param point1: point1 coordinate
    :param point2: point2 coordinate
    :return: distance between two points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def filter_points(
    points: list[tuple[int, int]], threshold: int | float
) -> list[tuple[int, int]]:
    """
    For each point, iterate through each point following it, and if there is a followed point whose
    distance from the current coordinate is less than the threshold, discard the current point.

    >>> filter_points([(0, 0), (1, 1), (2, 2), (3, 3)], 1)
    [(0, 0), (1, 1), (2, 2), (3, 3)]
    >>> filter_points([(0, 0), (1, 1), (2, 2), (3, 3)], 2)
    [(3, 3)]
    >>> filter_points([(0, 0), (1, 1), (2, 2), (3, 3)], 4)
    [(3, 3)]

    :param points: list of points
    :param threshold: distance threshold
    :return: list of selected points
    """
    selected_points: list[tuple[int, int]] = []
    for idx, point in enumerate(points):
        valid: bool = True
        for followed_point in points[idx + 1 :]:
            if p2p_distance(point, followed_point) < threshold:
                valid = False
                break
        if valid:
            selected_points.append(point)
    return selected_points


def m2c(moments: dict[str, float]) -> tuple[int, int]:
    """
    calculate the center coordinate from cv2.moments

    >>> m2c({"m00": 2, "m10": 3, "m01": 4})
    (1, 2)

    :param moments: list of moments
    :return: center coordinate
    """
    if not moments["m00"] > 0:
        return -1, -1
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def area_compare(
    area1: int | float, area2: int | float, threshold: int | float
) -> bool:
    """
    compare the ratio of two areas, if the ratio is greater than the threshold, return True

    >>> area_compare(1, 1, 2)
    False
    >>> area_compare(1, 3, 2)
    True

    :param area1: area1
    :param area2: area2
    :param threshold: minimum value of the ratio between two areas
    :return: area ratio (always greater than one)
    """
    return max([area1, area2]) / min([area1, area2]) > threshold


def img_preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    preprocess the input image
    :param img: input image (RGB)
    :return: preprocessed images: gray, blur after gray
    """
    if img.shape[-1] != 3:
        raise ValueError("img must be a RGB image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    return gray, blur


def coord_scale(coord: tuple[int, int]) -> tuple[int, int]:
    """
    scale coordinates from 800x800 to a 10x10 map (2d array index)

    >>> coord_scale((400, 400))
    (6, 5)

    :param coord: coordinate
    :return: scaled coordinate
    """
    scaled_x = round((coord[0] - 125) / 50)
    scaled_y = 11 - round((coord[1] - 125) / 50)
    return scaled_x, scaled_y
