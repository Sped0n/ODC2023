__all__ = ["get_shortest_path", "accelerate", "astar"]

from itertools import permutations

import numpy as np

from .accelerate import precompute
from .astar import a_star
from ctyper import Coordinate, Distance


def get_shortest_path(
    maze: np.ndarray,
    start: Coordinate,
    end: Coordinate,
    mid_points: list[Coordinate],
) -> tuple[list[Coordinate], Distance | float]:
    """
    get the shortest path from start to end
    :param maze: the maze
    :param start: the start point
    :param end: the end point
    :param mid_points: the middle points
    :return: the shortest path
    """
    precompute_dict = precompute(maze, start, end, mid_points)
    # generate all possible paths
    paths: list[tuple[Coordinate, ...]] = list(permutations(mid_points))
    shortest_length: Distance | float = float("inf")
    shortest_path: list[Coordinate] = []
    final_dots_path: list[Coordinate] = []
    for path in paths:
        # insert start and the end to the path
        path = list(path)
        path.append(end)
        path.insert(0, start)
        path_length: Distance = 0
        for idx, i in enumerate(path[:-1]):
            length: Distance = precompute_dict[i][path[idx + 1]]
            if length < 0:
                break
            path_length += length
        else:
            if path_length < shortest_length:
                shortest_length = path_length
                shortest_path = path
    for idx0, i0 in enumerate(shortest_path[:-1]):
        for i1 in a_star(maze, i0, shortest_path[idx0 + 1])[0][:-1]:
            final_dots_path.append(i1)
    final_dots_path.append(end)
    return final_dots_path, shortest_length
