import math
import cv2


def p2p_distance(point1, point2):
    """
    calculate the distance between two points
    :param point1: point1 coordinate
    :param point2: point2 coordinate
    :return: distance between two points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def filter_points(points, threshold):
    """
    For each point, iterate through each point following it, and if there is a followed point whose
    distance from the current coordinate is less than the threshold, discard the current point.
    :param points: list of points
    :param threshold: distance threshold
    :return: list of selected points
    """
    selected_points = []
    for idx, point in enumerate(points):
        valid = True
        for followed_point in points[idx + 1 :]:
            if p2p_distance(point, followed_point) < threshold:
                valid = False
                break
        if valid:
            selected_points.append(point)
    return selected_points


def m2c(moments):
    """
    calculate the center coordinate from cv2.moments
    :param moments: list of moments
    :return: center coordinate
    """
    return [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]


def area_compare(area1, area2, threshold):
    """
    calculate the area ratio
    :param area1: area1
    :param area2: area2
    :param threshold: minimum value of the ratio between two areas
    :return: area ratio (always greater than one)
    """
    return max([area1, area2]) / min([area1, area2]) > threshold


def img_preprocess(img):
    """
    preprocess the input image
    :param img: input image (RGB)
    :return: preprocessed images: gray, blur after gray
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    return gray, blur


def coord_scale(coord):
    """
    scale coordinates from 800x800 to a 21x21 map
    :param coord: coordinate
    :return: scaled coordinate
    """
    scaled_x = round((coord[0] - 125) / 25)
    scaled_y = 22 - round((coord[1] - 125) / 25)
    return scaled_x, scaled_y
