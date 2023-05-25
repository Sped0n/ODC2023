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


def direction_prediction(
    curr_coord: tuple[int, int], last_vector: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    get walkable coordinates with current position and last vector

    >>> sorted(direction_prediction((1, 1), (1, 0)))
    [(1, 0), (1, 2), (2, 1)]

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
        curr_vector = tuple(np.subtract(seq[idx + 1], coord))
        # skip if last_vector is None
        if last_vector is None:
            last_vector = curr_vector
            continue
        predictions = direction_prediction(coord, last_vector)
        walkable_predictions = []
        for prediction in predictions:
            if maze[prediction[0]][prediction[1]] == 0:
                walkable_predictions.append(prediction)
        # straight scenario
        follow_last_motion_vector = curr_vector == last_vector
        if len(walkable_predictions) == 1 and follow_last_motion_vector:
            pass
        # turn scenario
        elif len(walkable_predictions) == 1 and not follow_last_motion_vector:
            if curr_vector[0] * last_vector[1] - curr_vector[1] * last_vector[0] == -1:
                motion.append(f"s_left{coord}{walkable_predictions}")
            elif curr_vector[0] * last_vector[1] - curr_vector[1] * last_vector[0] == 1:
                motion.append(f"s_right{coord}{walkable_predictions}")
        # crossroad scenario
        elif len(walkable_predictions) > 1 and follow_last_motion_vector:
            motion.append(f"c_straight{coord}{walkable_predictions}")
        elif len(walkable_predictions) > 1 and not follow_last_motion_vector:
            if curr_vector[0] * last_vector[1] - curr_vector[1] * last_vector[0] == -1:
                motion.append(f"c_left{coord}{walkable_predictions}")
            elif curr_vector[0] * last_vector[1] - curr_vector[1] * last_vector[0] == 1:
                motion.append(f"c_right{coord}{walkable_predictions}")
        # U-turn scenario
        elif len(walkable_predictions) == 0:
            pass
        last_vector = curr_vector
    return motion
