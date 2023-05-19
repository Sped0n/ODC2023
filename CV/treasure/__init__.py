__all__ = ["correct", "utils", "locate", "dots", "find_treasure"]

from .utils import img_preprocess, coord_scale
from .locate import *
from .correct import *
from .dots import *


def find_treasure(frame: np.ndarray, debug: bool = False):
    """
    find all treasure dots, if debug is True, will return the locating boxes image
    and the corrected image
    :param frame: image frame (height: 480px !!!)
    :param debug: debug mode (will return image results)
    :return: scaled treasure dots coordinates (in 10x10), locating boxes image (debug only),
    corrected image (debug only)
    """
    if frame.shape[0] != 480:
        raise ValueError("frame height must be 480px")
    if frame.shape[-1] != 3:
        raise ValueError("frame must be a RGB image")
    _, blur = img_preprocess(frame)
    raw_locating_boxes = find_locating_boxes(blur)
    pos = get_locating_coords(raw_locating_boxes, 10)
    print(pos)
    corrected_frame = ricd(img_correction(frame, pos))
    cv2.imwrite("corrected.jpg", corrected_frame)
    _, cf_blur = img_preprocess(corrected_frame)
    treasure_dots = treasure_identification(cf_blur)
    scaled_dots_coords = []
    if debug:
        frame_copy: np.ndarray = frame.copy()
        for found in raw_locating_boxes:
            cv2.drawContours(frame_copy, [found], -1, (0, 255, 0), 3)
        for dot in treasure_dots:
            scaled_dots_coords.append(coord_scale(dot[:-1]))
            cv2.circle(corrected_frame, dot[:-1], dot[-1], (0, 255, 0), 5)
        return scaled_dots_coords, frame_copy, corrected_frame
    for dot in treasure_dots:
        scaled_dots_coords.append(coord_scale(dot[:-1]))
    return scaled_dots_coords
