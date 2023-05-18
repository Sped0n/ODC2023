import cv2
import numpy as np

from .utils import m2c


def img_correction(
    img: np.ndarray,
    locating_points: list[tuple[int, int], ...],
    tmap_enable: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    correct the image based on the coordinates of the top left and bottom right points of the box
    :param img: input image
    :param locating_points: list of coordinates of the box (tl, tr, bl, br)
    :param tmap_enable: whether to return the transform array
    :return: corrected image(800x800), transform array
    """
    dst: np.ndarray = np.array(
        [(75, 75), (725, 75), (75, 725), (725, 725)], dtype=np.float32
    )
    src: np.ndarray = np.array(locating_points, dtype=np.float32)
    tmap: np.ndarray = cv2.getPerspectiveTransform(src, dst)
    res: np.ndarray = cv2.warpPerspective(img, tmap, (800, 800))
    if tmap_enable:
        return res, tmap
    return res


def ricd(img: np.ndarray) -> np.ndarray:
    """
    rotate the image to the correct direction to map the maze, always keep the blue
    square in the bottom left corner
    :param img: RGB image after correction
    :return: correctly oriented image
    """
    blur: np.ndarray = cv2.GaussianBlur(img, (5, 5), 0)
    hsv: np.ndarray = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    blue_mask: np.ndarray = cv2.inRange(hsv, (100, 80, 46), (124, 255, 255))
    eroded_blue_mask: np.ndarray = cv2.erode(blue_mask, None, iterations=2)
    cnts: tuple[np.ndarray] = cv2.findContours(
        eroded_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    cnt: np.ndarray = max(cnts, key=cv2.contourArea)
    m: dict[str, float] = cv2.moments(cnt)
    cx, cy = m2c(m)
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
