from itertools import permutations

import numpy as np

from .astar import a_star
from ctyper import Coordinate, Distance


class a_star_data_pack:
    def __init__(self):
        self.length: int | None = None
        self.path: list[Coordinate] | None = None


def precompute(
    maze: np.ndarray,
    start: Coordinate,
    end: Coordinate,
    mid_points: list[Coordinate],
    eager: bool = True,
) -> dict[Coordinate, dict[Coordinate, Distance]]:
    """
    precompute all distances between any two points and load them all into a dict
    :param maze: A two-dimensional list that represents a maze map. 0 indicates a path
    and 1 indicates an obstacle.
    :param start: a tuple that represents the start point
    :param end: a tuple that represents the end point
    :param mid_points: a list of points between start and end
    :param eager: if true, the distance between the two points is considered unique
    (only calculate once), instead of
    considering each point as a different starting point to calculate the distance
    independently
    :return: precomputed data
    """
    # make a copy of the middle points
    dots: list[Coordinate] = mid_points[:]
    dots.append(end)
    dots.append(start)
    to_precompute: list[tuple[Coordinate, ...]] = list(permutations(dots, 2))
    # dict initialize
    precomputed: dict[Coordinate, dict[Coordinate, Distance]] = {}
    for dot in dots:
        precomputed[dot] = {}
    # fill in the distance
    for dot1, dot2 in to_precompute:
        if eager:
            try:
                precomputed[dot1][dot2] = precomputed[dot2][dot1]
            except KeyError:
                precomputed[dot1][dot2] = a_star(maze, dot1, dot2)[1]
        else:
            precomputed[dot1][dot2] = a_star(maze, dot1, dot2)[1]
    return precomputed
