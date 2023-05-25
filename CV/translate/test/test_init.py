from translate import seq_to_motion


def test_seq_to_motion():
    map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    path = [
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (2, 6),
        (3, 6),
        (2, 6),
        (2, 7),
        (2, 8),
        (1, 8),
    ]
    for coord in path:
        print(map[coord[0]][coord[1]])
    motion = seq_to_motion(map, path)
    print(motion)
    assert len(motion) > 0


def test_seq_to_motion_single_turn():
    map = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    path = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
    motion = seq_to_motion(map, path)
    print(motion)
    assert len(motion) > 0
