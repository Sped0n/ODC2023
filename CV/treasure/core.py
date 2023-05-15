from .utils import img_preprocess, coord_scale
from .locate import *
from .correct import *
from .dots import *


def find_treasure(frame):
    _, blur = img_preprocess(frame)
    founds = find_locating_box(blur)
    pos = get_locating_coords(founds, 10)
    pos = rearrange_locating_coords(pos)
    cframe, _ = img_correction(frame, pos)
    cframe = img_rotate(cframe)
    _, cblur = img_preprocess(cframe)
    dots = treasure_identification(cblur)
    scaled_dots_coords = []
    print(pos)
    for found in founds:
        cv2.drawContours(frame, [found], -1, (0, 255, 0), 3)
    for dot in dots:
        scaled_dots_coords.append(coord_scale(dot[:-1]))
        cv2.circle(cframe, dot[:-1], dot[-1], (0, 255, 0), 5)
    print(scaled_dots_coords)
    cv2.imshow("image", frame)
    cv2.imshow("result", cframe)
    cv2.waitKey(0)
