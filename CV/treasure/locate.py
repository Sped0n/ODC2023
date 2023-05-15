import cv2
from .utils import p2p_distance, m2c, filter_points, area_compare


def find_locating_box(
    frame,
    min_area=20,
    max_area=4000,
    apd_epsilon=0.043,
    wh_rate=0.5,
    min_center_distance=5,
    debug=False,
):
    """
    - frame: grayscale processed input image
    - min_area: minimum area of the locating box
    - max_area: maximum area of the locating box
    - apd_epsilon: epsilon for cv2.approxPolyDP
    - wh_rate: width height rate
    - min_center_distance: minimum center distance
    - debug: debug mode
    - return: coordinates of the top left and bottom right points of the box
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 2)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    raw_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter based on area
    contours1 = []
    for contour in raw_contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            contours1.append(contour)

    # filter based on shape (quads)
    contours2 = []
    for contour in contours1:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, apd_epsilon * perimeter, True)
        if len(approx) == 4:
            contours2.append(contour)

    # filter based on aspect ratio (square)
    contours3 = []
    for contour in contours2:
        _, (w, h), _ = cv2.minAreaRect(contour)
        if abs(w - h) / (w + h) < wh_rate:
            contours3.append(contour)

    # filter based on inclusion relationship
    contours4 = []
    for idx, contour in enumerate(contours3[:-1]):
        m1 = cv2.moments(contour)
        # skip if area is zero
        if m1["m00"] == 0:
            continue
        c1 = m2c(m1)
        for followed_contour in contours3[idx + 1 :]:
            m2 = cv2.moments(followed_contour)
            # skip if area is zero
            if m2["m00"] == 0:
                continue
            c2 = m2c(m2)

            res1 = cv2.pointPolygonTest(contour, c2, False)
            res2 = cv2.pointPolygonTest(followed_contour, c1, False)

            # two contours are contained within each other and not similar in size
            if res1 > 0 and res2 > 0 and area_compare(m1["m00"], m2["m00"], 1.3):
                if p2p_distance(c1, c2) < min_center_distance:
                    contours4.append(contour)
    # debug
    if debug:
        print(len(contours1), len(contours2), len(contours3), len(contours4))
    return contours4


def get_locating_points(boxes, center_distance_threshold=10):
    """
    - boxes: list of locating boxes
    - center_distance_threshold: minimum center distance, prevent overlapping
    - return: Center coordinates of locating boxes
    """
    coordinates = []
    for box in boxes:
        coordinates.append(m2c(cv2.moments(box)))
    coordinates = filter_points(coordinates, center_distance_threshold)
    return coordinates
