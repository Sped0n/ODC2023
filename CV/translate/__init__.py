__all__ = ['coord_to_index', 'seq_to_motion']


def coord_to_index(coords):
    """
    convert coordinates to 2d array index
    :param coords: 10x10 coordinates
    :return: 2d array index (21x21)
    """
    indexes = []
    for coord in coords:
        tmp = list(coord)
        tmp[0] = tmp[0] * 2 - 1
        tmp[1] = (11 - tmp[1]) * 2 - 1
        indexes.append(tuple([tmp[1], tmp[0]]))


def seq_to_motion(seq):
    pass
