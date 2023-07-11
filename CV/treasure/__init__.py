__all__ = ["correct", "utils", "locate", "dots", "find_treasure"]

import cv2
from .utils import img_preprocess, coord_scale
from .locate import find_locating_boxes, filter_locating_boxes
from .correct import ricd, img_correction
from .dots import treasure_identification
from ctyper import Coordinate, Image, Box, NoAnchorFound, QuantityMismatch, TreasureNull


class find_treasure:
    def __init__(self, frame: Image) -> None:
        """
        find all treasure dots, if debug is True, will return the locating boxes image
        and the corrected image
        :param frame: image frame (height: 480px !!!)
        :param debug: debug mode (will return image results)
        :return: scaled treasure dots coordinates (in 10x10), locating boxes image
        (debug only), corrected image (debug only)
        """
        if frame.shape[0] != 480:
            raise ValueError("frame height must be 480px")
        if frame.shape[-1] != 3:
            raise ValueError("frame must be a RGB image")
        # don't modify the original frame
        self.frame_copy: Image = frame.copy()
        # preprocess the image
        _, blur = img_preprocess(self.frame_copy)

        # find locating boxes
        self.raw_locating_boxes: list[Box] = find_locating_boxes(blur).boxes
        try:
            locating_box_coords = filter_locating_boxes(self.raw_locating_boxes)
        except QuantityMismatch:
            # if no locating boxes or more than 4 locating boxes, raise error
            raise TreasureNull

        # correct the image
        try:
            self.corrected_frame = ricd(
                img_correction(self.frame_copy, locating_box_coords).result
            )
        except NoAnchorFound:
            # if no anchor found, return empty list
            raise TreasureNull

        # find treasure dots
        _, cf_blur = img_preprocess(self.corrected_frame)
        try:
            self.treasure_dots = treasure_identification(cf_blur)
        except QuantityMismatch:
            # if no treasure dots or treasure dots are more than 8, raise error
            raise TreasureNull

        # scale the coordinates
        self.scaled_dots_coords = []
        for dot in self.treasure_dots:
            self.scaled_dots_coords.append(coord_scale(dot[:-1]))

    @property
    def dots_coords(self) -> list[Coordinate]:
        return sorted(self.scaled_dots_coords)

    @property
    def debug_locating_box(self) -> Image:
        for found in self.raw_locating_boxes:
            cv2.drawContours(self.frame_copy, [found], -1, (0, 255, 0), 3)
        return self.frame_copy

    @property
    def debug_corrected_frame(self) -> Image:
        for dot in self.treasure_dots:
            cv2.circle(self.corrected_frame, dot[:-1], dot[-1], (0, 255, 0), 5)
        return self.corrected_frame
