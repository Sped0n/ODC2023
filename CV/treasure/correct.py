import cv2
import numpy as np


def img_correction(img, locating_points):
    """
    - img: input image
    - locating_points: list of coordinates of the box (tl, tr, bl, br)
    - return: corrected image(800x800), transform array
    - function: correct the image based on the coordinates of the top left and bottom right points of the box
    """
    dst = [(75, 75), (725, 75), (75, 725), (725, 725)]
    dst = np.array(dst, dtype=np.float32)
    src = np.array(locating_points, dtype=np.float32)
    tmap = cv2.getPerspectiveTransform(src, dst)
    res = cv2.warpPerspective(img, tmap, (800, 800))
    return res, tmap
