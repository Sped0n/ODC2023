__all__ = ["coord_to_index", "seq_to_motion"]

import numpy as np


def coord_to_index(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    convert coordinates to 2d array index
    :param coords: 10x10 coordinates
    :return: 2d array index (21x21)
    """
    indexes = []
    for coord in coords:
        tmp: list[int, int] = list(coord)
        tmp[0] = tmp[0] * 2 - 1
        tmp[1] = (11 - tmp[1]) * 2 - 1
        indexes.append(tuple((tmp[1], tmp[0])))
    return indexes


def get_walkable(
    curr_coord: tuple[int, int], last_vector: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    get walkable coordinates with current position and last vector

    >>> get_walkable((1, 1), (1, 0))
    [(1, 2), (2, 1), (1, 0)]

    :param curr_coord: current coordinate
    :param last_vector: last vector
    :return: a list of walkable coordinates (max length = 3)
    """
    walkable: list = []
    for neighbor in [
        (curr_coord[0] - 1, curr_coord[1]),
        (curr_coord[0] + 1, curr_coord[1]),
        (curr_coord[0], curr_coord[1] - 1),
        (curr_coord[0], curr_coord[1] + 1),
    ]:
        if neighbor == tuple(np.subtract(curr_coord, last_vector)):
            continue
        walkable.append(neighbor)
    return walkable


def seq_to_motion(maze: list[list[int]], seq: list[tuple[int, int]]):
    """
    convert the coordinate sequence to motion sequence
    :param maze:
    :param seq:
    :return:
    """
    motion = []
    last_vector = None
    for idx, coord in enumerate(seq[:-1]):
        curr_vector = (seq[idx + 1][0] - coord[0], seq[idx + 1][1] - coord[1])
        # skip if last_vector is None
        if last_vector is None:
            last_vector = curr_vector
            continue
