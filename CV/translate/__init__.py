__all__ = ["coord_to_index", "seq_to_motions"]

import numpy as np

from ctyper import Coordinate, Vector

from .utils import direction_prediction, vector_to_direction, mod_output_string


def coord_to_index(coords: list[Coordinate]) -> list[Coordinate]:
    """
    convert coordinates to 2d array index

    >>> coord_to_index([(1, 1), (10, 10)])
    [(19, 1), (1, 19)]

    :param coords: 10x10 coordinates
    :return: 2d array index (21x21)
    """
    indexes: list[Coordinate] = []
    for coord in coords:
        indexes.append(((11 - coord[1]) * 2 - 1, coord[0] * 2 - 1))
    return indexes


def seq_to_motions(
    maze: np.ndarray, seq: list[Vector], debug: bool = False
) -> list[str]:
    """
    convert the coordinate sequence to motion sequence
    :param maze: maze
    :param seq: moving coordinate sequence
    :param debug: whether to print debug info (the coordinates when
    turning and the direction)
    :return:
    """
    motions: list[str] = []
    last_vector: Vector | None = None
    for idx, coord in enumerate(seq[:-1]):
        motion: str | None = None
        curr_vector: Vector = tuple(np.subtract(seq[idx + 1], coord))
        # skip if last_vector is None
        if last_vector is None:
            last_vector = curr_vector
            continue
        predictions: list[Coordinate] = direction_prediction(coord, last_vector)
        walkable_predictions: list[Coordinate] = []
        for prediction in predictions:
            if maze[prediction[0]][prediction[1]] == 0:
                walkable_predictions.append(prediction)
        follow_last_motion_vector: bool = curr_vector == last_vector
        match len(walkable_predictions):
            # no walkable predictions, dead end
            case 0:
                # U-turn scenario
                pass
            # one walkable prediction, only one way to go
            case 1:
                # go straight without intersection
                if follow_last_motion_vector:
                    pass
                # one-way turn scenario
                else:
                    match vector_to_direction(last_vector, curr_vector):
                        case "left":
                            motion = mod_output_string("left", "single", coord, debug)
                        case "right":
                            motion = mod_output_string("right", "single", coord, debug)
                        # it should not happen, but just in case
                        case _:
                            raise ValueError("invalid planning")
            case _:
                # crossroad scenario
                if follow_last_motion_vector:
                    motion = mod_output_string("straight", "cross", coord, debug)
                else:
                    match vector_to_direction(last_vector, curr_vector):
                        case "left":
                            motion = mod_output_string("left", "cross", coord, debug)
                        case "right":
                            motion = mod_output_string("right", "cross", coord, debug)
                        # it should not happen, but just in case
                        case _:
                            raise ValueError("invalid planning")
        # update last_vector
        if motion is not None:
            motions.append(motion)
        last_vector = curr_vector
    return motions
