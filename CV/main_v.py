import cv2

# /opt/homebrew/bin/ffmpeg
import numpy as np

from pathfinder import get_shortest_path
from translate import coord_to_index, seq_to_motions
from treasure import find_treasure
from ctyper import TreasureNull

cap = cv2.VideoCapture("./test.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()
accumulator = 0
last_coords = None
maze = np.load("./maze.npy", allow_pickle=True)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
accumulator = 0
last_coords, coords = [], []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        try:
            raw_coords = find_treasure(frame).dots_coords
            coords = coord_to_index(raw_coords)
        except TreasureNull:
            continue
        if last_coords == []:
            last_coords = coords
            continue
        if coords == last_coords:
            accumulator += 1
        else:
            accumulator = 0
        last_coords = coords
        if accumulator > 10:
            print("FOUND")
            print(coords)
            x1 = get_shortest_path(maze, (19, 1), (1, 19), coords)
            dots1 = list(x1)[0]
            print(dots1)
            print(seq_to_motions(maze, dots1))
            break
cap.release()
cv2.destroyAllWindows()
