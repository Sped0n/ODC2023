from itertools import permutations

from .astar import a_star


def precompute(
    maze: list[list[int]],
    start: tuple[int, int],
    end: tuple[int, int],
    mid_points: list[tuple[int, int]],
    eager: bool = True,
) -> dict[tuple[int, int], dict[tuple[int, int], int]]:
    """
    precompute all distances between any two points and load them all into a dict
    :param maze: A two-dimensional list that represents a maze map. 0 indicates a path and 1 indicates an obstacle.
    :param start: a tuple that represents the start point
    :param end: a tuple that represents the end point
    :param mid_points: a list of points between start and end
    :param eager: if true, the distance between the two points is considered unique (only calculate once), instead of
    considering each point as a different starting point to calculate the distance twice
    :return: precomputed data
    """
    # make a copy of the middle points
    dots: list[tuple[int, int]] = mid_points[:]
    dots.append(end)
    dots.append(start)
    to_precompute: list[tuple[tuple[int, int], ...]] = list(permutations(dots, 2))
    # dict initialize
    precomputed: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
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
