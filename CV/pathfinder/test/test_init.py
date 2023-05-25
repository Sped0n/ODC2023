from pathlib import Path

import numpy as np
from pathfinder import get_shortest_path

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_get_shortest_path():
    start = (19, 1)
    end = (1, 19)
    maze = np.load(f"{TEST_DATA_DIR}/complicate_maze.npy", allow_pickle=True)
    mids = [(1, 1), (19, 19)]
    path, length = get_shortest_path(maze, start, end, mids)
    shortest_path = np.load(
        f"{TEST_DATA_DIR}/complicate_shortest_path.npy", allow_pickle=True
    )
    assert length == 56
    # validate path
    for a, b in zip(path, shortest_path):
        diff = a[0] + a[1] - b[0] - b[1]
        assert diff == 0
