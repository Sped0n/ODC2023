from pathlib import Path

import numpy as np
from pathfinder.accelerate import precompute

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def test_precompute_eager():
    start = (19, 1)
    end = (1, 19)
    maze = np.load(f"{TEST_DATA_DIR}/complicate_maze.npy", allow_pickle=True)
    mids = [(1, 1), (19, 19)]
    precomputed = precompute(maze, start, end, mids, eager=True)
    # generate dots list
    dots = mids[:]
    dots.append(end)
    dots.append(start)
    for dot1 in dots:
        for dot2 in dots:
            if dot1 == dot2:
                continue
            assert precomputed[dot1][dot2] == precomputed[dot2][dot1]


def test_precompute_not_eager():
    start = (19, 1)
    end = (1, 19)
    maze = np.load(f"{TEST_DATA_DIR}/complicate_maze.npy", allow_pickle=True)
    mids = [(1, 1), (19, 19)]
    precomputed = precompute(maze, start, end, mids, eager=False)
    # generate dots list
    dots = mids[:]
    dots.append(end)
    dots.append(start)
    for dot1 in dots:
        for dot2 in dots:
            if dot1 == dot2:
                continue
            assert precomputed[dot1][dot2] == precomputed[dot2][dot1]
