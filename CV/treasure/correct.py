import cv2
import numpy as np


def img_correction(img, locating_points):
    """
    correct the image based on the coordinates of the top left and bottom right points of the box
    :param img: input image
    :param locating_points: list of coordinates of the box (tl, tr, bl, br)
    :return: corrected image(800x800), transform array
    """
    dst = [(75, 75), (725, 75), (75, 725), (725, 725)]
    dst = np.array(dst, dtype=np.float32)
    src = np.array(locating_points, dtype=np.float32)
    tmap = cv2.getPerspectiveTransform(src, dst)
    res = cv2.warpPerspective(img, tmap, (800, 800))
    return res, tmap


def img_rotate(img):
    """
    always keep the blue square in the bottom left corner
    :param img: RGB image after correction
    :return: correctly oriented image
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 80, 46), (124, 255, 255))
    eroded_blue_mask = cv2.erode(blue_mask, None, iterations=2)
    cnts = cv2.findContours(
        eroded_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    cnt = max(cnts, key=cv2.contourArea)
    m = cv2.moments(cnt)
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    # bottom left
    if cx < 400 < cy:
        return img
    # top left
    elif cx < 400 and cy < 400:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # top right
    elif cx > 400 > cy:
        return cv2.rotate(img, cv2.ROTATE_180)
    # bottom right
    else:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
