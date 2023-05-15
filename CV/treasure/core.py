from .utils import img_preprocess
from .locate import *
from .correct import *
from .dots import *


def find_treasure(frame):
    blur = img_preprocess(frame)
    founds = find_locating_box(blur)
    pos = get_locating_points(founds, 10)
    cframe, _ = img_correction(frame, pos)
    cblur = img_preprocess(cframe)
    dots = treasure_identification(cblur)
    print(pos)
    print(dots)
    for found in founds:
        cv2.drawContours(frame, [found], -1, (0, 255, 0), 3)
    for dot in dots:
        cv2.circle(cframe, dot[:-1], dot[-1], (0, 255, 0), 5)
    cv2.imshow("image", frame)
    cv2.imshow("result", cframe)
    cv2.waitKey(0)
