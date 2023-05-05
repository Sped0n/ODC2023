import math


def p2p_distance(point1, point2):
    """
    - point1: point coordinate
    - point2: point coordinate
    - return: distance between two points
    - function: calculate the distance between two points
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def filter_points(points, threshold):
    """
    - points: list of points
    - threshold: distance threshold
    - return: list of selected points
    - function: For each point, iterate through each point following it, and if there is a followed point whose
    distance from the current coordinate is less than the threshold, discard the current point.
    """
    selected_points = []
    for idx, point in enumerate(points):
        valid = True
        for followed_point in points[idx + 1 :]:
            if p2p_distance(point, followed_point) > threshold:
                valid = False
                break
        if valid:
            selected_points.append(point)
    return selected_points


def m2c(moments):
    """
    - moments: list of moments
    - return: center coordinate
    - function: calculate the center coordinate from a list of moments
    """
    return [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]
