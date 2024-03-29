import cv2
import numpy as np

from .utils import m2c

from ctyper import Image, Array, PixelCoordinate, NoAnchorFound, NullArea


class img_correction:
    def __init__(
        self,
        img: Image,
        locating_points: list[PixelCoordinate],
    ) -> None:
        """
        correct the image based on the coordinates of the top left and bottom
        right points of the box
        :param img: input image
        :param locating_points: list of coordinates of the box (tl, tr, bl, br)
        :param tmap_enable: whether to return the transform array
        :return: corrected image(800x800), transform array
        """
        # set dst array
        dst: Array = np.array(
            [(75, 75), (725, 75), (75, 725), (725, 725)], dtype=np.float32
        )
        # transform list of coordinates to a src array
        src: Array = np.array(locating_points, dtype=np.float32)
        # get transform mapping
        self.tmap: Array = cv2.getPerspectiveTransform(src, dst)
        # transform the image
        self.res: Image = cv2.warpPerspective(img, self.tmap, (800, 800))

    @property
    def result(self) -> Image:
        return self.res

    @property
    def transform_map(self) -> Array:
        return self.tmap


def ricd(img: Image) -> Image:
    """
    rotate the image to the correct direction to map the maze, always keep the blue
    square in the bottom left corner
    :param img: RGB image after correction
    :return: correctly oriented image
    """
    if img.shape[-1] != 3:
        raise ValueError("img must be a RGB image")
    # preprocess the image
    blur: np.ndarray = cv2.GaussianBlur(img, (5, 5), 0)
    hsv: np.ndarray = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # find the blue square
    blue_mask: np.ndarray = cv2.inRange(hsv, (100, 80, 46), (124, 255, 255))
    eroded_blue_mask: np.ndarray = cv2.erode(blue_mask, None, iterations=2)
    cnts = cv2.findContours(
        eroded_blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]
    # if no blue square found, raise an exception
    if not cnts:
        raise NoAnchorFound("No anchor found")
    cnt: np.ndarray = max(cnts, key=cv2.contourArea)
    m: dict[str, float] = cv2.moments(cnt)
    try:
        # center x, center y
        cx, cy = m2c(m)
    except NullArea:
        # if the center is not in the image, raise an exception
        raise NoAnchorFound("Anchor don't have valid area")
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
