from ctyper import Coordinate, Vector
import numpy as np


def direction_prediction(
    curr_coord: Coordinate, last_vector: Vector
) -> list[Coordinate]:
    """
    get walkable coordinates with current position and last vector

    >>> sorted(direction_prediction((1, 1), (1, 0)))
    [(1, 0), (1, 2), (2, 1)]

    :param curr_coord: current coordinate
    :param last_vector: last vector
    :return: a list of walkable coordinates (max length = 3)
    """
    walkable: list[Coordinate] = []
    for neighbor in [
        (curr_coord[0] - 1, curr_coord[1]),
        (curr_coord[0] + 1, curr_coord[1]),
        (curr_coord[0], curr_coord[1] - 1),
        (curr_coord[0], curr_coord[1] + 1),
    ]:
        # skip if neighbor is the last coordinate
        if neighbor == tuple(np.subtract(curr_coord, last_vector)):
            continue
        walkable.append(neighbor)
    return walkable


def vector_to_direction(last_vector: Vector, curr_vector: Vector) -> str:
    """
    get direction from two vectors

    >>> vector_to_direction((1, 0), (0, 1))
    'left'
    >>> vector_to_direction((1, 0), (0, -1))
    'right'
    >>> vector_to_direction((1, 0), (1, 0))
    'align'
    >>> vector_to_direction((0, 1), (0, -1))
    'align'

    :param last_vector: last vector
    :param curr_vector: current vector
    :return: direction
    """
    match last_vector[0] * curr_vector[1] - last_vector[1] * curr_vector[0]:
        case 1:
            return "left"
        case -1:
            return "right"
        case 0:
            return "align"
        case _:
            raise ValueError(
                "invalid direction, please check whether input is an integer vector"
            )


def mod_output_string(
    content: str,
    intersection_cond: str,
    curr_coord: Coordinate,
    debug: bool = False,
) -> str:
    """
    modify the output string
    :param content:
    :param intersection_cond:
    :param curr_coord:
    :param debug:
    :return:
    """
    if not debug:
        return content
    return f"{intersection_cond}_{content}{curr_coord}"
