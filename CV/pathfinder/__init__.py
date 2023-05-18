__all__ = ["get_shortest_path", "accelerate", "astar", "utils"]

from itertools import permutations

from .accelerate import precompute
from .astar import a_star


def get_shortest_path(
    maze: list[list[int]],
    start: tuple[int, int],
    end: tuple[int, int],
    mid_points: list[tuple[int, int], ...],
) -> tuple[list[tuple[int, int]], int]:
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
    paths: list[tuple[tuple[int, int], ...]] = list(permutations(mid_points))
    shortest_length: int | float = float("inf")
    shortest_path: list[tuple[int, int], ...] | None = None
    final_dots_path: list[tuple[int, int]] = []
    for path in paths:
        # insert start and the end to the path
        path: list[tuple[int, int], ...] = list(path)
        path.append(end)
        path.insert(0, start)
        path_length: int = 0
        for idx, i in enumerate(path[:-1]):
            length: int = precompute_dict[i][path[idx + 1]]
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
