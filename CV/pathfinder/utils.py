def manhattan_distance(coord1, coord2):
    """
    calculate the manhattan distance
    :param coord1: coordinate of the first point
    :param coord2: coordinate of the second point
    :return: heuristic based on manhattan distance
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
