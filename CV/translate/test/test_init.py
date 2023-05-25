from translate import seq_to_motions


def test_seq_to_motions_cross_straight_plus_dead_end():
    maze = [
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
    motions = seq_to_motions(maze, path)
    expected_motions = ["straight", "right", "right", "left"]
    for a, b in zip(motions, expected_motions):
        assert a == b


def test_seq_to_motions_single_right_turn():
    maze = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    path = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]
    motions = seq_to_motions(maze, path)
    expected_motions = ["right", "right"]
    for a, b in zip(motions, expected_motions):
        assert a == b


def test_seq_to_motions_crosses():
    # multiple crosses
    maze = [
        [1, 1, 1, 1, 1, 1, 1, 1],  # 0
        [1, 1, 0, 1, 1, 0, 1, 1],  # 1
        [1, 1, 0, 1, 1, 0, 1, 1],  # 2
        [1, 1, 0, 0, 0, 0, 0, 1],  # 3
        [1, 1, 1, 1, 1, 0, 1, 1],  # 4
        [1, 1, 1, 1, 1, 0, 1, 1],  # 5
        [1, 1, 0, 1, 1, 0, 1, 1],  # 6
        [1, 0, 0, 0, 0, 0, 0, 1],  # 7
        [1, 1, 0, 1, 1, 0, 1, 1],  # 8
        [1, 1, 1, 1, 1, 1, 1, 1],  # 9
    ]
    path = [
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 5),
        (6, 5),
        (5, 5),
        (4, 5),
        (3, 5),
        (3, 4),
        (3, 3),
        (3, 2),
        (2, 2),
        (1, 2),
    ]
    motions = seq_to_motions(maze, path)
    expected_motions = ["straight", "straight", "right", "left", "right"]
    for a, b in zip(motions, expected_motions):
        assert a == b


def test_seq_to_motions_single_right_turn_debug_enable():
    maze = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    path = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1)]
    motions = seq_to_motions(maze, path, debug=True)
    print(motions)
    expected_motions = ["single_right(1, 3)", "single_right(3, 3)"]
    for a, b in zip(motions, expected_motions):
        assert a == b


def test_seq_to_motions_cross_straight_plus_dead_end_debug_enable():
    maze = [
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
    motions = seq_to_motions(maze, path, debug=True)
    print(motions)
    expected_motions = [
        "cross_straight(2, 3)",
        "cross_right(2, 6)",
        "cross_right(2, 6)",
        "single_left(2, 8)",
    ]
    for a, b in zip(motions, expected_motions):
        assert a == b
