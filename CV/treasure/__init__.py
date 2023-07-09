__all__ = ["correct", "utils", "locate", "dots", "find_treasure"]

from .utils import img_preprocess, coord_scale
from .locate import *
from .correct import *
from .dots import *


def find_treasure(
    frame: np.ndarray, debug: bool = False
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
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
    frame_copy: np.ndarray = frame.copy()
    _, blur = img_preprocess(frame_copy)
    raw_locating_boxes = find_locating_boxes(blur)
    locating_box_coords = get_locating_coords_from_contours(raw_locating_boxes, 10)
    # if no locating boxes found or locating boxes are more than 4, return empty list
    if not locating_box_coords:
        return []
    corrected_frame = ricd(img_correction(frame_copy, locating_box_coords))
    # if no corrected frame, return empty list
    if corrected_frame is None:
        return []
    _, cf_blur = img_preprocess(corrected_frame)
    treasure_dots = treasure_identification(cf_blur)
    # if no treasure dots or treasure dots are more than 8, return empty list
    if not treasure_dots:
        return []
    scaled_dots_coords = []
    if debug:
        for found in raw_locating_boxes:
            cv2.drawContours(frame_copy, [found], -1, (0, 255, 0), 3)
        for dot in treasure_dots:
            scaled_dots_coords.append(coord_scale(dot[:-1]))
            cv2.circle(corrected_frame, dot[:-1], dot[-1], (0, 255, 0), 5)
        return sorted(scaled_dots_coords), frame_copy, corrected_frame
    for dot in treasure_dots:
        scaled_dots_coords.append(coord_scale(dot[:-1]))
    return sorted(scaled_dots_coords)
