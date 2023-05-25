from pathlib import Path

import numpy as np
from pathfinder.astar import a_star

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_a_star_found_path():
    start = (6, 1)
    end = (3, 3)
    maze = np.load(f"{TEST_DATA_DIR}/v_maze.npy", allow_pickle=True)
    path, length = a_star(maze, start, end)
    # validate path and length
    assert path == [
        (6, 1),
        (5, 1),
        (4, 1),
        (3, 1),
        (2, 1),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert length == 4


def test_a_star_no_path():
    start = (6, 1)
    end = (3, 3)
    maze = np.load(f"{TEST_DATA_DIR}/v_maze.npy", allow_pickle=True)
    maze[2][3] = 1
    maze[2][4] = 1
    path, length = a_star(maze, start, end)
    # no path situation
    assert path == []
    assert length == -1


def test_a_star_complicate_maze():
    start = (19, 1)
    end = (1, 19)
    maze = np.load(f"{TEST_DATA_DIR}/complicate_maze.npy", allow_pickle=True)
    path, length = a_star(maze, start, end)
    valid_path = np.load(
        f"{TEST_DATA_DIR}/complicate_valid_path.npy", allow_pickle=True
    )
    assert length == 22
    # validate path
    for a, b in zip(path, valid_path):
        diff = a[0] + a[1] - b[0] - b[1]
        assert diff == 0
