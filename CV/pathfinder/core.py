from itertools import permutations
from .accelerate import precompute
from .astar import a_star


def get_shortest_path(maze, start, end, mid_points):
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
    paths = list(permutations(mid_points))
    shortest_length = float("inf")
    shortest_path = None
    final_dots_path = []
    for path in paths:
        # insert start and the end to the path
        path = list(path)
        path.append(end)
        path.insert(0, start)
        path_length = 0
        for idx, i in enumerate(path[:-1]):
            length = precompute_dict[i][path[idx + 1]]
            if length < 0:
                break
            path_length += length
        else:
            if path_length < shortest_length:
                shortest_length = path_length
                shortest_path = path
    print(shortest_path)
    for idx0, i0 in enumerate(shortest_path[:-1]):
        for i1 in a_star(maze, i0, shortest_path[idx0 + 1])[0][:-1]:
            final_dots_path.append(i1)
    final_dots_path.append(end)
    return final_dots_path, shortest_length
