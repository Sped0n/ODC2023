import time

import cv2

# /opt/homebrew/bin/ffmpeg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from pathfinder import get_shortest_path
from translate import coord_to_index, seq_to_motions
from treasure import find_treasure
from utils import image_resize

plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"

matplotlib.use("TkAgg")

img = cv2.imread("/Users/spedon/eden/python/ODC2023/CV/mazeExp/test_pattern2.jpg")
img = image_resize(img, height=480)
raw_coords, a, b = find_treasure(img, debug=True)
plt.subplot(121)
plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))

print("raw coords\n", raw_coords)  # col, row
coords = coord_to_index(raw_coords)
maze = np.load("/Users/spedon/eden/python/ODC2023/CV/maze.npy", allow_pickle=True)
print(maze)
t0 = time.time()
x1 = get_shortest_path(maze, (19, 1), (1, 19), coords)
t1 = time.time()
print(f"took {(t1 - t0):.4f}s")

invert_maze = np.array(np.ones(21) - maze, dtype=int)
invert_maze[19][0] = 5
invert_maze[1][20] = 6
fig = plt.figure()
axes = plt.axes()
axes.invert_yaxis()
axes.set_aspect("equal")
plt.yticks([])
plt.xticks([])
dots = list(x1)[0]
frame_count = len(dots)
print(
    "motion sequence from 🟦 to 🟥:\n",
    seq_to_motions(maze, dots),
    "\n by 9201080N0133 Wen ZhiBin",
)
last0 = None
last1 = None


def a_init():
    return plt.pcolormesh(invert_maze), plt.title(f"0/{len(dots)}")


def func(f):
    global last0, last1, invert_maze, dots
    dot = dots[f]
    invert_maze[dot[0]][dot[1]] = 4
    if last0 is not None:
        invert_maze[last0[0]][last0[1]] = 3
    if last1 is not None:
        invert_maze[last1[0]][last1[1]] = 2
    if last0 is not None:
        last1 = last0
    last0 = dot
    # plt.savefig(f"/Users/spedon/Desktop/anim/{f}.png")
    return plt.pcolormesh(invert_maze), plt.title(f"{f}/{len(dots)}")


anim = FuncAnimation(
    fig,
    func,
    frames=len(dots),
    interval=60,
    blit=False,
    repeat=False,
    repeat_delay=1000,
    init_func=a_init,
)
# writer = matplotlib.animation.FFMpegWriter(
#     fps=15, metadata=dict(artist="Me"), bitrate=1800
# )
# anim.save("/Users/spedon/Desktop/anim.mp4", writer=writer)
plt.show()
