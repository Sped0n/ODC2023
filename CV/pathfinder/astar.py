import heapq
from .utils import manhattan_distance


def a_star(maze, start, end):
    """
    use astar algorithm to find the shortest path from start to end
    :param maze: A two-dimensional list that represents a maze map. 0 indicates a path and 1 indicates an obstacle.
    :param start: a tuple that represents the start point
    :param end: a tuple that represents the end point
    :return: path (represents the shortest path from the start point to the end point, where each element is a
    coordinate tuple), the length of the path
    """
    open_list = [(0, start)]
    closed_list = set()
    g_cost = {start: 0}
    parent_node = {}  # parent node of each node
    path_length = 0

    while open_list:
        # get the node with the lowest f cost
        current = heapq.heappop(open_list)[1]
        if current == end:
            path = [current]
            # reverse trace of the path from end node to start node
            while current in parent_node:
                current = parent_node[current]
                path.append(current)
            path_length = int((len(path) - 1) / 2)
            return path[::-1], path_length

        # add lowest f cost node to closed_list
        closed_list.add(current)

        for neighbor in [
            (current[0] - 1, current[1]),
            (current[0] + 1, current[1]),
            (current[0], current[1] - 1),
            (current[0], current[1] + 1),
        ]:
            # make sure within range
            if not (0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0])):
                continue
            # make sure walkable terrain
            if maze[neighbor[0]][neighbor[1]] != 0:
                continue
            tentative_g_cost = g_cost[current] + 1
            # get the best possible "neighbor" to go
            # Update g_cost and parent node if neighbor do not have g cost, or if the new g cost is better
            if neighbor not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g_cost
                # f cost = g cost + h cost
                f_cost = tentative_g_cost + manhattan_distance(neighbor, end)
                # add valid neighbor to open_list
                heapq.heappush(open_list, (f_cost, neighbor))
                # parent node of neighbor is the current node(in the closed list)
                parent_node[neighbor] = current
    # no path found
    return [], path_length
