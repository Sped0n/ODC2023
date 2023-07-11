import cv2
import numpy as np

from .utils import p2p_distance, m2c, filter_points, area_compare

from ctyper import Box, Image, PixelCoordinate, QuantityMismatch


class find_locating_boxes:
    def __init__(
        self,
        frame: Image,
        min_area: int = 20,
        max_area: int = 4000,
        apd_epsilon: float = 0.043,
        wh_rate: float = 0.5,
        min_center_distance: int = 5,
    ) -> None:
        """
        find all locating boxes, including duplicate identification
        :param frame: grayscale and gaussian blur processed input image
        :param min_area: minimum area of the locating box
        :param max_area: maximum area of the locating box
        :param apd_epsilon: epsilon for cv2.approxPolyDP
        :param wh_rate: width height rate
        :param min_center_distance: minimum center distance
        :return: coordinates of the top left and bottom right points of the box
        """
        if frame.ndim != 2:
            raise ValueError("frame must be a grayscale image")
        # edge detection
        edges: Image = cv2.Canny(frame, 50, 150, apertureSize=3)
        # find contours
        raw_contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # filter based on area
        self.contours1: list[Box] = []
        for contour in raw_contours:
            area: int | float = cv2.contourArea(contour)
            if min_area < area < max_area:
                self.contours1.append(contour)

        # filter based on shape (quads)
        self.contours2: list[Box] = []
        for contour in self.contours1:
            perimeter: float = cv2.arcLength(contour, True)
            approx: np.ndarray = cv2.approxPolyDP(
                contour, apd_epsilon * perimeter, True
            )
            if len(approx) == 4:
                self.contours2.append(contour)

        # filter based on aspect ratio (square)
        self.contours3: list[Box] = []
        for contour in self.contours2:
            _, (w, h), _ = cv2.minAreaRect(contour)
            if abs(w - h) / (w + h) < wh_rate:
                self.contours3.append(contour)

        # filter based on inclusion relationship
        self.contours4: list[Box] = []
        for idx, contour in enumerate(self.contours3[:-1]):
            m1: dict[str, float] = cv2.moments(contour)
            # skip if area is zero
            if m1["m00"] == 0:
                continue
            # center of outer contour
            c1: PixelCoordinate = m2c(m1)
            for followed_contour in self.contours3[idx + 1 :]:
                m2: dict[str, float] = cv2.moments(followed_contour)
                # skip if area is zero
                if m2["m00"] == 0:
                    continue
                # center of inner contour
                c2: PixelCoordinate = m2c(m2)
                # confirm the inclusion relationship
                res1: float = cv2.pointPolygonTest(contour, c2, False)
                res2: float = cv2.pointPolygonTest(followed_contour, c1, False)
                # two contours are contained within each other and not similar in size
                if (
                    not res1 > 0
                    and res2 > 0
                    and area_compare(m1["m00"], m2["m00"], 1.3)
                ):
                    continue
                # skip if two contours are too close
                if not p2p_distance(c1, c2) < min_center_distance:
                    continue
                self.contours4.append(contour)
                break

    @property
    def boxes(self) -> list[Box]:
        return self.contours4

    @property
    def debug(self) -> list[int]:
        return [
            len(self.contours1),
            len(self.contours2),
            len(self.contours3),
            len(self.contours4),
        ]


def filter_locating_boxes(
    boxes: list[Box], center_distance_threshold: int = 10
) -> list[PixelCoordinate]:
    """
    get locating coordinates of 4 locating boxes
    :param boxes: list of locating boxes
    :param center_distance_threshold: minimum center distance, prevent overlapping
    :return: when locating boxes number is valid, return center coordinates of
    locating boxes
    """
    coordinates: list[PixelCoordinate] = []
    for box in boxes:
        # get center coordinates of all locating boxes
        coordinates.append(m2c(cv2.moments(box)))
    # filter over similar coordinates
    coordinates = filter_points(coordinates, center_distance_threshold)
    # only return when number of locating boxes is valid(4)
    if len(coordinates) != 4:
        raise QuantityMismatch
    return rearrange_locating_coords(coordinates)


def rearrange_locating_coords(
    raw_coords: list[PixelCoordinate],
) -> list[PixelCoordinate]:
    """
    rearrange locating boxes coordinates

    >>> rearrange_locating_coords([(0, 0), (2, 2), (0, 2), (2, 0)])
    [(0, 0), (2, 0), (0, 2), (2, 2)]

    :param raw_coords: list of coordinates
    :return: rearranged coordinates
    """
    avg_x: int | float = sum(c[0] for c in raw_coords) / len(raw_coords)
    avg_y: int | float = sum(c[1] for c in raw_coords) / len(raw_coords)
    tl: PixelCoordinate = (-1, -1)
    tr: PixelCoordinate = (-1, -1)
    bl: PixelCoordinate = (-1, -1)
    br: PixelCoordinate = (-1, -1)
    tl_flag, tr_flag, bl_flag, br_flag = False, False, False, False
    for c in raw_coords:
        if c[0] <= avg_x and c[1] <= avg_y:
            tl = c
            tl_flag = True
        elif c[0] > avg_x and c[1] <= avg_y:
            tr = c
            tr_flag = True
        elif c[0] <= avg_x and c[1] > avg_y:
            bl = c
            bl_flag = True
        else:
            br = c
            br_flag = True
    if not (tl_flag and tr_flag and bl_flag and br_flag):
        return []
    return list((tl, tr, bl, br))
