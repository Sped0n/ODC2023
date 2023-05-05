"""
Descripttion:
version:
Author: 肖发博
Date: 2023-04-11 03:15:34
LastEditors: 肖发博
LastEditTime: 2023-04-13 08:59:23
email 2285273839@qq.com
"""
import time
import cv2
import numpy as np
import math

approxPolyDP_epslion = 0.043  # 多边形近似参数，越小越精准
normal_std = 0.8  # 判断四条边的归一标准差
min_center_distance = 10  # 中心距离
wh_rate = 0.67  # 长宽比系数，越小越接近正方形
min_area = 20
max_area = 4000

min_ridus = 8  # 最小半径
max_ridus = 22  # 最大半径
minDist = 40  # 第四个参数是两个圆之间的最小距离。
param1 = 50  # 第五个参数是Canny边缘检测的高阈值，它用于检测图像中的边缘。
param2 = 16  # 第六个参数是累加器阈值。值越小，检测到的圆形越少。


def pose2pixel(pose_x, pose_y):
    """
    函数接受两个参数，分别为机器人的位置坐标pose_x和pose_y，返回值为该位置对应的图像像素坐标pixel_x和pixel_y。
    """
    pixel_x = 125 + 50 * pose_x
    pixel_y = 675 - 50 * pose_y
    return pixel_x, pixel_y


def standardize(data):
    """
    对数据进行离差标准化
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def std_dev(data):
    """
    计算标准差
    """
    mean_val = np.mean(data)
    deviation = data - mean_val
    return np.sqrt(np.sum(deviation**2) / len(data))


def distance(point1, point2):
    """计算距离"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def filter_points(points, threshold):
    """
    - points：包含坐标信息的列表
    - threshold：距离阈值，当两个坐标之间的距离小于此值时，剔除其中的一个坐标
    返回值：selected_points，包含经过筛选后的坐标信息的列表
    功能：对传入的坐标列表进行筛选，去掉距离过近的坐标，返回经过筛选后的坐标信息列表
    实现思路：对于每一个坐标，遍历其后面的每一个坐标，如果有一个坐标与当前坐标距离小于阈值，则剔除当前坐标，
    继续遍历下一个坐标，如果所有后面的坐标与当前坐标的距离都大于等于阈值，则将当前坐标添加到
    筛选后的坐标列表中，遍历完所有坐标后返回筛选后的坐标列表。
    """
    selected_points = []  # 定义一个空列表，存储筛选出来的坐标
    for i in range(len(points)):
        is_selected = True
        for j in range(i + 1, len(points)):
            if distance(points[i], points[j]) < threshold:
                is_selected = False
                break
        if is_selected:
            selected_points.append(points[i])
    return selected_points


def find_locating_box(frame, display=False):
    """
    该函数主要功能是在给定图像中寻找包含目标物体的定位框（location box）。函数接收两个参数，第一个是待处理的图像帧，第二个是一个可选的布尔类型参数，用于指定是否显示处理过程的可视化结果。
    该函数具体的处理流程如下：
    将图像帧转换为灰度图像，再使用Canny边缘检测算法获得边缘图像。
    在边缘图像中寻找轮廓。
    根据预设的筛选条件对轮廓进行筛选，得到一组可能的四边形。
    根据预设的条件，从可能的四边形中筛选出正方形。
    根据预设的条件，从正方形中筛选出符合包含关系的正方形，即为定位框。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选条件0 面积
    contours2 = []
    for i in range(len(contours)):
        M1 = cv2.moments(contours[i])
        if M1["m00"] < min_area or M1["m00"] > max_area:
            continue
        contours2.append(contours[i])
        print("ju:", M1["m00"], "a:", cv2.contourArea(contours[i]))

    # 筛选条件1 四边形
    sibianxing = []
    for c in contours2:
        # 对轮廓进行多边形逼近
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approxPolyDP_epslion * peri, True)
        hull = approx
        if len(hull) == 4:
            sibianxing.append(hull)

    # 筛选条件2 正方形
    # 判断正方形方案1，判断长宽比
    zhengfangxing = []
    for hull in sibianxing:
        rect = cv2.minAreaRect(hull)
        (x, y), (w, h), angle = rect
        abs(w - h)
        if abs(w - h) / (w + h) < wh_rate:  # 边长归一标准差
            zhengfangxing.append(hull)

    # 筛选条件3 正方形的包含关系
    dingweikuang = []
    for i in range(len(zhengfangxing) - 1):
        M1 = cv2.moments(zhengfangxing[i])
        if M1["m00"] == 0:
            continue
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        for j in range(i + 1, len(zhengfangxing)):
            M2 = cv2.moments(zhengfangxing[j])
            if M2["m00"] == 0:  # 两者面积为0则跳过
                continue
            cx2 = int(M2["m10"] / M2["m00"])
            cy2 = int(M2["m01"] / M2["m00"])

            # 1二者中心互相包含
            res1 = cv2.pointPolygonTest(zhengfangxing[i], (cx2, cy2), False)
            res2 = cv2.pointPolygonTest(zhengfangxing[j], (cx1, cy1), False)
            if res1 > 0 and res2 > 0:  # 如果两个轮廓中心具有包含关系
                # 2计算中心距离
                distance_ = distance((cx1, cy1), (cx2, cy2))
                if distance_ < min_center_distance:  # 中心距里小于5个像素
                    dingweikuang += [zhengfangxing[i], zhengfangxing[j]]
    if display:
        cv2.imshow("step1_edges", edges)

        contours_frame = frame.copy()
        cv2.drawContours(contours_frame, contours, -1, (0, 0, 255), 1)
        cv2.putText(
            contours_frame,
            "counts:{}".format(len(contours)),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("step2_all_contours", contours_frame)

        contours2_frame = frame.copy()
        cv2.drawContours(contours2_frame, contours2, -1, (0, 0, 255), 1)
        "FPS: {:.2f}".format(fps)
        cv2.putText(
            contours2_frame,
            "min_area:{},max_area:{},counts:{}".format(
                min_area, max_area, len(contours2)
            ),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("step3_Area_Selected_contours", contours2_frame)

        sibianxing_frame = frame.copy()
        cv2.drawContours(sibianxing_frame, sibianxing, -1, (0, 0, 255), 1)
        cv2.putText(
            sibianxing_frame,
            "approxPolyDP_epslion:{:.2f},counts:{}".format(
                approxPolyDP_epslion, len(sibianxing)
            ),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("step4_quadrilateral_selection", sibianxing_frame)

        zhengfangxing_frame = frame.copy()
        cv2.drawContours(zhengfangxing_frame, zhengfangxing, -1, (0, 0, 255), 1)
        cv2.putText(
            zhengfangxing_frame,
            "wh_rate:{:.2f},counts:{}".format(wh_rate, len(zhengfangxing)),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("step5_Square_selection", zhengfangxing_frame)

        dingweikuang_frame = frame.copy()
        cv2.drawContours(dingweikuang_frame, dingweikuang, -1, (0, 0, 255), 1)
        cv2.putText(
            dingweikuang_frame,
            "min_center_distance:{:.2f},counts:{}".format(
                min_center_distance, len(dingweikuang)
            ),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("step6_Positioning_box_selection", dingweikuang_frame)

        # 返回帧
    # print(len(contours),len(sibianxing),len(zhengfangxing),len(dingweikuang))
    return dingweikuang


def get_corners_direction(corners):
    """判断四个角点的方向"""
    x_avg = sum(p[0] for p in corners) / 4  # 计算四个角点的x坐标平均值
    y_avg = sum(p[1] for p in corners) / 4  # 计算四个角点的y坐标平均值
    tl, tr, bl, br = None, None, None, None
    for p in corners:
        if p[0] < x_avg and p[1] < y_avg:
            tl = p
        elif p[0] > x_avg and p[1] < y_avg:
            tr = p
        elif p[0] < x_avg and p[1] > y_avg:
            bl = p
        elif p[0] > x_avg and p[1] > y_avg:
            br = p
    return tl, tr, bl, br


def get_locating_point(locating_boxs):
    """
    这个函数名为 get_locating_point，它的作用是获取定位框的中心点。传入的参数 locating_boxs 是一个包含若干个定位框的列表，每个定位框是一个由点坐标构成的列表。
    该函数首先定义了一个空列表 dingweidian 和一个空列表 locating_points。然后遍历传入的所有定位框，通过 OpenCV 的 cv2.moments 函数计算每个定位框的中心点坐标，并将其添加到 dingweidian 列表中。
    接着调用了 filter_points 函数对 dingweidian 中的点进行筛选，把距离较近的点合并成一个点，最终将得到的筛选后的点添加到 locating_points 列表中。
    最后返回 locating_points 列表，其中包含了所有定位框的中心点坐标。
    """
    # 定义两个空列表
    dingweidian = []
    locating_points = []
    # 循环定位框列表中的每一个框
    for i in range(len(locating_boxs)):
        # 计算当前框的矩（moments）
        M1 = cv2.moments(locating_boxs[i])
        # 计算当前框的重心坐标
        cx1 = int(M1["m10"] / M1["m00"])
        cy1 = int(M1["m01"] / M1["m00"])
        # 将当前框的重心坐标添加到列表中
        dingweidian.append((cx1, cy1))
    # 将中心距里较近的点合并成一个点
    locating_points = filter_points(dingweidian, 10)
    # 返回合并后的点列表
    return locating_points


def image_correction(img, locating_points):
    """tl, tr, bl, br"""
    """该函数名为image_correction，用于对输入的图像进行校正，使其四个角点的位置分别接近预设的目标位置。该函数需要输入两个参数：img为待校正的图像，locating_points为检测出的定位框中心点坐标列表。
    函数会将定位框中心点坐标列表locating_points与预设目标位置坐标列表goal_box_centers进行透视变换，以得到透视矫正后的图像。如果矫正失败，函数会返回两个None值。
    函数返回两个值：warped_img为矫正后的图像，M为变换矩阵。
    """
    real_box_centers = locating_points
    goal_box_centers = [(75, 75), (725, 75), (75, 725), (725, 725)]
    if None in real_box_centers:  # 矫正失败
        return None, None
    dst_pts = np.array(goal_box_centers, dtype=np.float32)
    src_pts = np.array(real_box_centers, dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(img, M, (800, 800))
    return warped_img, M


def generate_border(img, color=(0, 0, 0), line_scale=3):
    """generate_border 要求传入的输入图片为800*800"""
    x_start = 400 - (10 // 2) * 50
    y_start = 400 - (10 // 2) * 50
    x_end = 400 + (10 // 2) * 50
    y_end = 400 + (10 // 2) * 50

    # 绘制水平方向的网格线
    for i in range(11):
        y = y_start + i * 50
        cv2.line(img, (x_start, y), (x_end, y), color, line_scale)

    # 绘制垂直方向的网格线
    for j in range(11):
        x = x_start + j * 50
        cv2.line(img, (x, y_start), (x, y_end), color, line_scale)

    cv2.rectangle(img, (0, 580), (170, 800), (0, 0, 0), -1)  # z左下遮挡
    cv2.rectangle(img, (630, 0), (800, 220), (0, 0, 0), -1)  # 右上遮挡
    cv2.rectangle(img, (0, 0), (170, 170), (0, 0, 0), -1)  # 左上遮挡
    cv2.rectangle(img, (630, 630), (800, 800), (0, 0, 0), -1)
    return img


def pixel2pose(pixel_x, pixel_y):
    """将像素坐标中的点转换到10*10坐标中"""
    pose_x = round((pixel_x - 125) / 50)
    pose_y = 11 - round((pixel_y - 125) / 50)
    return pose_x, pose_y


def treasure_Identification2(img, display=False):
    """掩膜轮廓法返回宝藏的像素坐标"""
    """函数treasure_Identification2是用掩膜轮廓法从图像中获取宝藏的像素坐标。它的参数是一个图像和一个布尔值，控制是否显示绘制有宝藏像素点的图像。函数首先对图像进行了一些处理，包括将其转换为灰度图像、对其进行二值化操作和消除边框。然后，使用cv2.findContours函数找到图像中所有的轮廓，并对这些轮廓进行循环。对于每个轮廓，函数计算出其中心坐标，并检查其是否在特定区域内。如果是，则将其中心坐标添加到列表centers0中。最后，如果display为True，则将绘制有宝藏像素点的图像显示出来，并返回centers0列表。否则，仅返回centers0列表。"""
    warped_img = img.copy()
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, tresh_warped = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    without_border_image = generate_border(tresh_warped, line_scale=15)
    kernel = np.ones((5, 5), np.uint8)
    # 进行开运算操作
    opening = cv2.morphologyEx(without_border_image, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('opening Picture after removing border', opening)

    contours, hierarchy = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    centers0 = []

    for hull in contours:
        M = cv2.moments(hull)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        if 150 < cx < 650 and 150 < cy < 650:  # 特定区域
            centers0.append((cx, cy))
    if display:
        for i in centers0:
            cv2.circle(warped_img, i, 10, color=(0, 0, 255), thickness=3)
        cv2.imshow("treasure_Identification2", warped_img)

    return centers0


def treasure_Identification1(img, display=False):
    """霍夫圆检测法返回宝藏的像素坐标"""
    """这个函数使用了霍夫圆检测法来检测宝藏的像素坐标，并返回宝藏中心的像素坐标列表。其中，使用cv2.HoughCircles函数进行霍夫圆检测，设置了一系列参数（minDist、param1、param2、min_ridus和max_ridus）来控制检测效果。如果检测到宝藏，就将中心点坐标加入到centers0列表中，并可以选择在图像上绘制出圆形和参数信息。
    函数的参数为img和display（默认为False），img是输入的图像，display用于控制是否在图像上绘制圆形和参数信息。函数返回centers0，即宝藏中心的像素坐标列表。"""
    # min_ridus=8#最小半径
    # max_ridus=22#最大半径
    # minDist =40#第四个参数是两个圆之间的最小距离。
    # param1  =50#第五个参数是Canny边缘检测的高阈值，它用于检测图像中的边缘。
    # param2  =16#第六个参数是累加器阈值。值越小，检测到的圆形越少。
    warped_img = img.copy()
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=min_ridus,
        maxRadius=max_ridus,
    )
    centers0 = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            if 150 < x < 650 and 150 < y < 650:
                if display:
                    cv2.circle(warped_img, (x, y), r, (0, 255, 0), 1)
                centers0.append((x, y))
    if display:
        cv2.putText(
            warped_img,
            "minDist,param1,param2,min_ridus,max_ridus={},{},{},{},{}".format(
                minDist, param1, param2, min_ridus, max_ridus
            ),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("step7_treasure_Identification", warped_img)
    return centers0


def on_trackbar(val):
    global approxPolyDP_epslion, normal_std, min_center_distance, wh_rate, min_area, max_area, min_ridus, max_ridus, minDist, param1, param2
    approxPolyDP_epslion = cv2.getTrackbarPos("approxPolyDP_epslion", "image") * 0.001
    wh_rate = cv2.getTrackbarPos("wh_rate", "image") * 0.01
    # normal_std = cv2.getTrackbarPos('normal_std', 'image')*0.01
    min_center_distance = cv2.getTrackbarPos("min_center_distance", "image")
    min_area = cv2.getTrackbarPos("min_area", "image")
    max_area = cv2.getTrackbarPos("max_area", "image")

    minDist = cv2.getTrackbarPos("minDist", "image")
    param1 = cv2.getTrackbarPos("param1", "image")
    param2 = cv2.getTrackbarPos("param2", "image")
    min_ridus = cv2.getTrackbarPos("min_ridus", "image")
    max_ridus = cv2.getTrackbarPos("max_ridus", "image")


def find_symmetric_point(x1, y1, x2, y2, x3, y3):
    """find_symmetric_point 已知矩形三个点求第四个点"""

    # Calculate the length of each side
    AB = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    BC = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    AC = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

    # Find the longest side and its endpoints
    if AB > BC and AB > AC:
        x_long1, y_long1 = x1, y1
        x_long2, y_long2 = x2, y2
        x_other, y_other = x3, y3
    elif BC > AB and BC > AC:
        x_long1, y_long1 = x2, y2
        x_long2, y_long2 = x3, y3
        x_other, y_other = x1, y1
    else:
        x_long1, y_long1 = x1, y1
        x_long2, y_long2 = x3, y3
        x_other, y_other = x2, y2

    # Find the midpoint of the longest side
    x_mid = (x_long1 + x_long2) / 2
    y_mid = (y_long1 + y_long2) / 2

    d_x = x_mid - x_other
    d_y = y_mid - y_other

    x_sym = x_long1 + x_long2 - x_other
    y_sym = y_long1 + y_long2 - y_other
    # Return the coordinates of the symmetric point
    return x_sym, y_sym


def get_maze_map_pose(img, display=False):
    """输入图像，输出宝藏坐标和标记后的图像"""
    map_maze_loaction = []
    frame = img.copy()
    dingweikuang = find_locating_box(frame, display=display)
    locating_points = get_locating_point(dingweikuang)
    if len(locating_points) == 4:
        locating_points = get_corners_direction(locating_points)
        warped_img, M = image_correction(frame, locating_points)
        if warped_img is not None:
            M_inv = cv2.invert(M)[1]
            maze_piexl_location = treasure_Identification1(
                warped_img, display=display
            )  # 霍夫圆检测法
            for i in range(len(maze_piexl_location)):
                cv2.circle(
                    warped_img,
                    maze_piexl_location[i],
                    10,
                    color=(0, 0, 255),
                    thickness=3,
                )
                map_maze_loaction.append(pixel2pose(*maze_piexl_location[i]))
            # print("宝藏位置",map_maze_loaction)

            origin_maze_pixel = []
            for pt in maze_piexl_location:
                p_prime = np.array([pt[0], pt[1], 1])
                p = np.dot(M_inv, p_prime)
                # 将齐次坐标转换为笛卡尔坐标
                x, y, w = p
                x /= w
                y /= w
                x = int(x)
                y = int(y)
                origin_maze_pixel.append((x, y))
            for i in range(len(origin_maze_pixel)):
                cv2.circle(
                    frame, origin_maze_pixel[i], 5, color=(0, 0, 255), thickness=2
                )
            cv2.drawContours(frame, dingweikuang, -1, (0, 255, 0), 1)
    return map_maze_loaction, frame


def online_parameter():
    cv2.namedWindow("image")
    cv2.createTrackbar("min_area", "image", 1, 100, on_trackbar)
    cv2.setTrackbarPos("min_area", "image", 10)
    cv2.createTrackbar("max_area", "image", 200, 10000, on_trackbar)
    cv2.setTrackbarPos("max_area", "image", 4000)
    cv2.createTrackbar("approxPolyDP_epslion", "image", 1, 100, on_trackbar)
    cv2.setTrackbarPos("approxPolyDP_epslion", "image", 43)
    cv2.createTrackbar("wh_rate", "image", 1, 100, on_trackbar)
    cv2.setTrackbarPos("wh_rate", "image", 67)
    # cv2.createTrackbar('normal_std', 'image', 1, 100, on_trackbar)
    # cv2.setTrackbarPos("normal_std",'image',50)
    cv2.createTrackbar("min_center_distance", "image", 1, 100, on_trackbar)
    cv2.setTrackbarPos("min_center_distance", "image", 10)

    cv2.createTrackbar("min_ridus", "image", 1, 100, on_trackbar)
    cv2.setTrackbarPos("min_ridus", "image", 8)
    cv2.createTrackbar("max_ridus", "image", 30, 100, on_trackbar)
    cv2.setTrackbarPos("max_ridus", "image", 22)
    cv2.createTrackbar("minDist", "image", 1, 200, on_trackbar)
    cv2.setTrackbarPos("minDist", "image", 40)
    cv2.createTrackbar("param1", "image", 1, 100, on_trackbar)
    cv2.setTrackbarPos("param1", "image", 50)
    cv2.createTrackbar("param2", "image", 1, 100, on_trackbar)
    cv2.setTrackbarPos("param2", "image", 16)


if __name__ == "__main__":
    # 是否展示中间过程调整参数
    change_parameter = True
    # True

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    if change_parameter:
        online_parameter()

    last_time = time.time()
    last_loction = []
    frame_count = 0
    fps = 0
    success_count = 0
    success_rate = 0
    while True:
        # 获取一帧视频
        ret, frame = cap.read()
        frame_count += 1
        if ret:
            location, img = get_maze_map_pose(frame, display=change_parameter)
            location = set(location)
            if last_loction and location == last_loction:
                success_count += 1
            last_loction = location

            if frame_count == 20:
                now_time = time.time()
                during = now_time - last_time
                last_time = now_time
                fps = 20 / during
                frame_count = 0
                success_rate = success_count / 20
                success_count = 0
            print(location, round(fps, 2), success_rate)

            cv2.putText(
                img,
                "FPS: {:.2f},Success_rate:{:.2f}".format(fps, success_rate),
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow("result", img)
            # 按下q键退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
