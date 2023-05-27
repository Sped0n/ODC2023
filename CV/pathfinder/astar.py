from heapq import heappush, heappop

import numpy as np

from .utils import manhattan_distance


def a_star(
    maze: np.ndarray, start: tuple[int, int], end: tuple[int, int]
) -> tuple[list[tuple[int, int]], int]:
    """
    use astar algorithm to find the shortest path from start to end
    :param maze: A two-dimensional array that represents a maze map. 0 indicates a path and 1 indicates an obstacle.
    :param start: a tuple that represents the start point
    :param end: a tuple that represents the end point
    :return: path (represents the shortest path from the start point to the end point, where each element is a
    coordinate tuple), the length of the path
    """
    if maze.ndim != 2:
        raise ValueError("maze must be a two-dimensional array")
    open_list: list[tuple[int, tuple[int, int]]] = [(0, start)]
    closed_list: set[tuple[int, int]] = set()
    g_cost_dict: dict[tuple[int, int], int] = {start: 0}
    parent_node_dict: dict[
        tuple[int, int], tuple[int, int]
    ] = {}  # parent node of each node

    while open_list:
        min_f_cost_node: tuple[int, int] = heappop(open_list)[1]
        if min_f_cost_node == end:
            path: list[tuple[int, int]] = [min_f_cost_node]
            # reverse trace of the path from end node to start node
            while min_f_cost_node in parent_node_dict:
                min_f_cost_node = parent_node_dict[min_f_cost_node]
                path.append(min_f_cost_node)
            path_length: int = int((len(path) - 1) / 2)
            return path[::-1], path_length

        closed_list.add(min_f_cost_node)

        for neighbor in [
            (min_f_cost_node[0] - 1, min_f_cost_node[1]),
            (min_f_cost_node[0] + 1, min_f_cost_node[1]),
            (min_f_cost_node[0], min_f_cost_node[1] - 1),
            (min_f_cost_node[0], min_f_cost_node[1] + 1),
        ]:
            # make sure within range
            if not (0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0])):
                continue
            # make sure walkable terrain
            if maze[neighbor[0]][neighbor[1]] != 0:
                continue
            tentative_g_cost: int = g_cost_dict[min_f_cost_node] + 1
            # get the best possible "neighbor" to go
            # Update g_cost and parent node if neighbor do not have g cost, or if the new g cost is better
            if neighbor not in g_cost_dict or tentative_g_cost < g_cost_dict[neighbor]:
                g_cost_dict[neighbor] = tentative_g_cost
                # f cost = g cost + h cost
                f_cost: int = tentative_g_cost + manhattan_distance(neighbor, end)
                # add valid neighbor to open_list
                heappush(open_list, (f_cost, neighbor))
                # parent node of neighbor is the minimum f cost node(in the closed list)
                parent_node_dict[neighbor] = min_f_cost_node
    # no path found
    return [], -1
