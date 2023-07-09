from heapq import heapify, heappush, heappop

import numpy as np

from .utils import manhattan_distance
from ctyper import Coordinate, Distance


class Node:
    def __init__(self, position: Coordinate, parent=None) -> None:
        self.parent = parent
        self.position = position
        self.g = 0  # distance to start node
        self.f = 0  # total cost

    def __eq__(self, other) -> bool:
        return self.position == other.position

    def __lt__(self, other) -> bool:
        return self.f < other.f

    def __gt__(self, other) -> bool:
        return self.f > other.f

    def refresh_f(self, end: Coordinate) -> None:
        """
        refresh f cost
        :param end: coordinate of end node
        :return: None
        """
        self.f = self.g + manhattan_distance(self.position, end)

    def neighbors(self) -> list:
        """
        get neighbors of current node
        :return: list of neighbor nodes
        """
        return [
            Node((self.position[0] - 1, self.position[1])),
            Node((self.position[0] + 1, self.position[1])),
            Node((self.position[0], self.position[1] - 1)),
            Node((self.position[0], self.position[1] + 1)),
        ]


def a_star(
    maze: np.ndarray, start: Coordinate, end: Coordinate
) -> tuple[list[Coordinate], Distance]:
    """
    use astar algorithm to find the shortest path from start to end
    :param maze: A two-dimensional array that represents a maze map. 0 indicates a path
    and 1 indicates an obstacle.
    :param start: a tuple that represents the start point
    :param end: a tuple that represents the end point
    :return: path (represents the shortest path from the start point to the end
    point, where each element is a
    coordinate tuple), the length of the path
    """
    if maze.ndim != 2:
        raise ValueError("maze must be a two-dimensional array")
    # initialize start node
    start_node = Node(start)
    start_node.g = 0
    start_node.refresh_f(end)
    # initialize open list
    open_list: list[Node] = []
    heapify(open_list)
    heappush(open_list, start_node)
    # initialize g_list (list of nodes whose g has been calculated)
    g_list: list[Node] = [start_node]

    while open_list:
        # pop up the minimum f cost node as current node
        current_node = heappop(open_list)

        # reach the end
        if current_node.position == end:
            path: list[Coordinate] = [current_node.position]
            # reverse trace of the path from end node to start node
            while current_node.parent:
                current_node = current_node.parent
                path.append(current_node.position)
            path_length: Distance = int((len(path) - 1) / 2)
            return path[::-1], path_length

        for neighbor in current_node.neighbors():
            # make sure within range
            if not (
                0 <= neighbor.position[0] < len(maze)
                and 0 <= neighbor.position[1] < len(maze[0])
            ):
                continue
            # make sure walkable terrain
            if maze[neighbor.position[0]][neighbor.position[1]] != 0:
                continue
            tentative_g_cost: int = current_node.g + 1
            # check if we have ever calculated the value of g
            idx: int = -1
            try:
                idx = g_list.index(neighbor)
            except ValueError:
                pass
            # get the best possible "neighbor" to go
            # if we haven't calculated g or new g is better
            # update g, f and parent node
            if idx == -1 or tentative_g_cost < g_list[idx].g:
                # update g_cost
                neighbor.g = tentative_g_cost
                # f cost = g cost + h cost
                neighbor.refresh_f(end)
                # parent node of neighbor is the minimum f cost node(in the closed list)
                neighbor.parent = current_node
                # add valid neighbor to open_list
                heappush(open_list, neighbor)
                # if we haven't calculated the g, add it to g list
                if idx == -1:
                    g_list.append(neighbor)
    # no path found
    return [], -1
