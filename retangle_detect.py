import cv2
import time
import numpy as np
import serial

MEAN_INNER_VAL = 100
MEAN_BORDER_VAL = 80

last_dx = 100
last_dy = 100

is_tarking = False
track_cnt = 0

ser = serial.Serial(
    port='/dev/serial/by-id/usb-ATK_ATK-HSWL-CMSIS-DAP_ATK_20190528-if00', # 串口设备名称
    baudrate=115200, # 波特率
)

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def find_a4_paper(frame):
    """
    在图像帧中检测带有黑边的白色A4纸。

    Args:
        frame: 来自视频流或静态图像的输入图像帧。

    Returns:
        一个高亮显示检测到的A4纸的帧。
        变换后的矩形图像。
    """
    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 为改善在不同光照条件下的鲁棒性，应用CLAHE（对比度有限的自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 使用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用自适应阈值处理以获得二值图像
    # 这有助于处理光照不均的场景
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 在阈值图像中查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积对轮廓进行排序，并保留最大的10个
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    detected_rect = None
    warped_image = None

    for contour in contours:
        # 将轮廓近似为多边形
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # 如果近似的轮廓有四个点，我们可以假设找到了A4纸
        if len(approx) == 4:
            # 检查面积以避免非常小的矩形
            if cv2.contourArea(approx) < 1000:
                continue

            # 检查轮廓是否为凸形
            if not cv2.isContourConvex(approx):
                continue

            # 现在，我们来验证颜色属性。
            # 我们需要检查内部是否为白色，边框是否为黑色。

            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(approx)

            # 为矩形的内部区域创建一个掩码
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, (255), -1) # 填充轮廓

            # 稍微缩小掩码尺寸以获取内部部分，避免黑色胶带
            kernel = np.ones((10, 10), np.uint8)
            inner_mask = cv2.erode(mask, kernel, iterations=1)

            # 计算内部区域的平均颜色
            mean_inner_val = cv2.mean(gray, mask=inner_mask)[0]

            # 为边框区域创建一个掩码
            outer_mask = cv2.dilate(mask, kernel, iterations=1)
            border_mask = cv2.subtract(outer_mask, mask)

            # 计算边框区域的平均颜色
            mean_border_val = cv2.mean(gray, mask=border_mask)[0]

            # 内部应为亮色（白纸），边框应为暗色（黑色胶带）
            # 根据您的光照条件调整这些阈值
            if mean_inner_val > MEAN_INNER_VAL and mean_border_val < MEAN_BORDER_VAL:
                detected_rect = approx
                break # 找到了

    # 在原始帧上绘制检测到的矩形
    global last_dx, last_dy
    if detected_rect is not None:
        cv2.drawContours(frame, [detected_rect], -1, (0, 255, 0), 3)
        # 添加文本
        cv2.putText(frame, "A4 Paper Detected", (detected_rect.ravel()[0], detected_rect.ravel()[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 进行透视变换
        # 整理顶点顺序
        rect_pts = order_points(detected_rect.reshape(4, 2))
        (tl, tr, br, bl) = rect_pts

        # 计算新图像的宽度和高度，但按照26:18的比例进行调整
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        detected_width = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        detected_height = max(int(heightA), int(heightB))

        # 按照26:18的标准比例调整尺寸
        # 选择较大的尺寸作为基准，然后按比例计算另一个尺寸
        aspect_ratio = 26.0 / 18.0  # 1.444...
        
        if detected_width / detected_height > aspect_ratio:
            # 宽度相对更大，以宽度为基准
            maxWidth = detected_width
            maxHeight = int(detected_width / aspect_ratio)
        else:
            # 高度相对更大，以高度为基准
            maxHeight = detected_height
            maxWidth = int(detected_height * aspect_ratio)

        # 定义变换后的目标点
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(rect_pts, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

        # --- 在这里添加画圆的代码 ---
        # 定义物理尺寸 (cm)
        PHYSICAL_WIDTH_CM = 26.0
        PHYSICAL_HEIGHT_CM = 18.0

        # 现在maxWidth和maxHeight已经按照26:18的比例调整过了
        # 直接使用这些值来计算像素到厘米的转换比例
        pixels_per_cm_w = maxWidth / PHYSICAL_WIDTH_CM
        pixels_per_cm_h = maxHeight / PHYSICAL_HEIGHT_CM
        
        # 由于我们已经强制按比例调整，这两个值应该相等或非常接近
        pixels_per_cm = (pixels_per_cm_w + pixels_per_cm_h) / 2.0

        # 计算圆的半径 (像素)
        CIRCLE_RADIUS_CM = 6.0
        radius_px = int(CIRCLE_RADIUS_CM * pixels_per_cm)

        # 在变换后的图像中心画圆
        center_px = (maxWidth // 2, maxHeight // 2)
        cv2.circle(warped, center_px, radius_px, (255, 0, 0), 1) 

        # --- 在原图上画出变换后的圆 ---
        # 使用逆变换矩阵
        inv_M = np.linalg.inv(M)

        # 生成圆周上的点
        num_points = 100
        circle_points_warped = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center_px[0] + radius_px * np.cos(angle)
            y = center_px[1] + radius_px * np.sin(angle)
            circle_points_warped.append([x, y])
        
        circle_points_warped = np.array([circle_points_warped], dtype=np.float32)

        # 将点变换回原始图像坐标
        original_circle_points = cv2.perspectiveTransform(circle_points_warped, inv_M)

        # 在原始图像上绘制轮廓
        cv2.polylines(frame, [np.int32(original_circle_points)], True, (255, 0, 0), 1)
        
        # 拓展并显示变换后的图像
        padding = 20
        warped_image = cv2.copyMakeBorder(warped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        # 寻找正中心
        center_warped = (warped.shape[1] // 2, warped.shape[0] // 2)
        
        # 将中心点坐标变换回原始图像
        center_warped_homogeneous = np.array([center_warped[0], center_warped[1], 1], dtype=np.float32)
        original_center_homogeneous = inv_M.dot(center_warped_homogeneous)
        original_center = (original_center_homogeneous[0] / original_center_homogeneous[2], original_center_homogeneous[1] / original_center_homogeneous[2])
        
        # 在原始图像上标记中心点
        center_x, center_y = int(original_center[0]), int(original_center[1])
        cv2.circle(frame, (center_x, center_y), 1, (0, 0, 255), -1)
        cv2.putText(frame, "Center", (center_x - 30, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 计算屏幕中心点到圆心的距离
        screen_center_x = frame.shape[1] // 2 +20
        screen_center_y = frame.shape[0] // 2 -13
        dx = center_x - screen_center_x
        dy = center_y - screen_center_y
        distance_to_center = np.sqrt(dx**2 + dy**2)
        
        # 在屏幕上标记屏幕中心点
        cv2.circle(frame, (screen_center_x, screen_center_y), 3, (0, 255, 255), -1)
        cv2.putText(frame, "Screen Center", (screen_center_x - 50, screen_center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if abs(dx) + abs(dy) < 6 and abs(last_dx) + abs(last_dy) < 6 :
            # 如果dx和dy都很小，认为已经对准
            cv2.putText(frame, "Aligned", (screen_center_x - 50, screen_center_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            global is_tarking, track_cnt
            if not is_tarking:
                track_cnt += 1
                if track_cnt > 5:
                    is_tarking = True
                    track_cnt = 0
                    time.sleep(0.01) 
                    ser.write(b"1\n")
                    time.sleep(0.01)
        else:
            is_tarking = False
            track_cnt = 0

        last_dx = dx
        last_dy = dy
        # 打印距离信息
        print(f"{-dx},{dy}")
        ser.write(f"{-dx},{dy}\n".encode())
        # 在图像上显示距离信息
        # cv2.putText(frame, f"dx:{dx}, dy:{dy}", (10, 30), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    return frame, warped_image

def main():
    # 打开摄像头（0通常是默认摄像头）
    # 如果您有多个摄像头，可以尝试更改为1、2等
    # 或者，您可以使用视频文件路径代替：cap = cv2.VideoCapture('your_video.mp4')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误：无法打开视频流。")
        return

    # 创建一个用于显示变换后图像的空图像
    transformed_view = np.zeros((100, 100, 3), np.uint8)

    while True:
        #逐帧捕获
        ret, frame = cap.read()
        if not ret:
            print("错误：无法接收帧（视频流结束？）。正在退出...")
            break

        # 检测A4纸
        processed_frame, warped_image = find_a4_paper(frame)

        # 显示结果帧
        cv2.imshow('A4 Paper Detection', processed_frame)

        # 如果有变换后的图像，则显示它
        if warped_image is not None:
            transformed_view = warped_image
        
        cv2.imshow('Transformed Rectangle', transformed_view)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 完成后，释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
