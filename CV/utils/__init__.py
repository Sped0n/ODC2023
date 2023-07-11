__all__ = ["image_resize"]

import cv2
import numpy as np


def image_resize(
    image: np.ndarray,
    width: int | None = None,
    height: int | None = None,
    inter=cv2.INTER_AREA,
):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim: tuple[int, int] | None = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # if both the width and height are not None, then raise an exception
    if width is not None and height is not None:
        raise ValueError("Can only resize by width or height, not both")

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        assert height is not None
        r: float = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r: float = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized: np.ndarray = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
