import cv2

# /opt/homebrew/bin/ffmpeg
import numpy as np

from pathfinder import get_shortest_path
from translate import coord_to_index, seq_to_motions
from treasure import find_treasure

cap = cv2.VideoCapture("/Users/spedon/PycharmProjects/ODC2023/CV/test.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()
accumulator = 0
last_coords = None
maze = np.load("/Users/spedon/PycharmProjects/ODC2023/CV/maze.npy", allow_pickle=True)
while ret:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    raw_coords = find_treasure(frame)
    if not raw_coords:
        continue
    coords = coord_to_index(raw_coords)
    if not coords:
        continue
    if last_coords is None:
        last_coords = coords
        continue
    if coords == last_coords:
        accumulator += 1
    else:
        accumulator = 0
    print(accumulator)
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
