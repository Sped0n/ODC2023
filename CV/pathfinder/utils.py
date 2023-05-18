def manhattan_distance(coord1: tuple[int, int], coord2: tuple[int, int]) -> int:
    """
    calculate the manhattan distance

    >>> manhattan_distance((0, 0), (5, 5))
    10

    :param coord1: coordinate of the first point
    :param coord2: coordinate of the second point
    :return: heuristic based on manhattan distance
    """
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
