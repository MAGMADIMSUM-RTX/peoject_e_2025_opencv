import cv2
import time
import numpy as np
import math

# 串口控制宏定义
ENABLE_SERIAL = True  # 设置为 False 可以禁用串口功能

if ENABLE_SERIAL:
    import serial

# A4纸的实际尺寸
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# 原有的检测参数（不修改）
MEAN_INNER_VAL = 100
MEAN_BORDER_VAL = 80

# 原有的跟踪参数（不修改）
last_dx = 100
last_dy = 100
is_tarking = False
track_cnt = 0

# 固定的已校准距离测量参数
CAMERA_PARAMS = {
    "focal_length_mm": 2.8,
    "sensor_width_mm": 5.37,
    "sensor_height_mm": 4.04,
    "calibration_factor": 2.0776785714285713
}

# 串口初始化（保持原有逻辑）
ser = None
if ENABLE_SERIAL:
    try:
        ser = serial.Serial(
            port='/dev/serial/by-id/usb-ATK_ATK-HSWL-CMSIS-DAP_ATK_20190528-if00',
            baudrate=115200,
        )
        print("串口已成功连接")
    except Exception as e:
        print(f"串口连接失败: {e}")
        print("程序将在无串口模式下运行")
        ENABLE_SERIAL = False
        ser = None
else:
    print("串口功能已禁用")

def serial_write(data):
    """安全的串口写入函数（保持不变）"""
    if ENABLE_SERIAL and ser is not None:
        try:
            ser.write(data)
        except Exception as e:
            print(f"串口写入错误: {e}")

class DistanceOffsetCalculator:
    """基于距离的屏幕中心偏移计算器"""
    def __init__(self):
        # 根据提供的数据点进行多项式拟合
        # 数据点: (距离mm, offset_x, offset_y)
        self.calibration_points = [
            (600, -3, 0),    # 600mm: offset_x=-3, offset_y=0
            (1000, 12, -3),  # 1000mm: offset_x=12, offset_y=-3
            (1300, 19, -5)   # 1300mm: offset_x=19, offset_y=-5
        ]
        
        # 提取数据进行拟合
        distances = [point[0] for point in self.calibration_points]
        offset_x_values = [point[1] for point in self.calibration_points]
        offset_y_values = [point[2] for point in self.calibration_points]
        
        # 使用二次多项式拟合 y = ax² + bx + c
        self.poly_coeffs_x = np.polyfit(distances, offset_x_values, 2)
        self.poly_coeffs_y = np.polyfit(distances, offset_y_values, 2)
        
        print("屏幕中心偏移拟合完成:")
        print(f"X偏移拟合系数: {self.poly_coeffs_x}")
        print(f"Y偏移拟合系数: {self.poly_coeffs_y}")
        
        # 验证拟合精度
        print("\n验证拟合精度:")
        for dist, expected_x, expected_y in self.calibration_points:
            calc_x, calc_y = self.calculate_screen_center_offset(dist)
            print(f"距离{dist}mm: 期望({expected_x},{expected_y}) vs 计算({calc_x:.1f},{calc_y:.1f})")
    
    def calculate_screen_center_offset(self, distance_mm):
        """根据距离计算屏幕中心偏移量"""
        # 限制距离范围在500-1500mm之间
        distance_mm = max(500, min(1500, distance_mm))
        
        # 使用多项式计算偏移量
        offset_x = np.polyval(self.poly_coeffs_x, distance_mm)
        offset_y = np.polyval(self.poly_coeffs_y, distance_mm)
        
        return int(round(offset_x)), int(round(offset_y))

class SimpleDistanceCalculator:
    """简化的距离计算器（使用固定参数）"""
    def __init__(self):
        self.distance_history = []
        self.max_history = 5
        
    def calculate_distance_from_width(self, pixel_width, frame_width):
        """基于宽度计算距离（使用固定参数）"""
        focal_length = CAMERA_PARAMS["focal_length_mm"]
        sensor_width = CAMERA_PARAMS["sensor_width_mm"]
        calibration_factor = CAMERA_PARAMS["calibration_factor"]
        
        fov_horizontal_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        mm_per_pixel_at_1m = (1000 * math.tan(fov_horizontal_rad / 2) * 2) / frame_width
        
        if pixel_width > 0 and mm_per_pixel_at_1m > 0:
            distance_mm = (A4_WIDTH_MM * 1000) / (pixel_width * mm_per_pixel_at_1m)
            return distance_mm * calibration_factor
        return None
    
    def update_distance_history(self, distance):
        """更新距离历史"""
        if distance:
            self.distance_history.append(distance)
            if len(self.distance_history) > self.max_history:
                self.distance_history.pop(0)
    
    def get_averaged_distance(self):
        """获取平均距离"""
        if self.distance_history:
            return np.mean(self.distance_history)
        return None

# 原有的透视变换函数（完全不修改）
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# 原有的A4纸检测函数（在屏幕中心计算部分使用动态偏移）
def find_a4_paper(frame, distance_calculator, offset_calculator):
    """
    在图像帧中检测带有黑边的白色A4纸。
    使用动态屏幕中心偏移来适应不同距离。
    """
    # === 原有的检测逻辑（完全不修改） ===
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    detected_rect = None
    warped_image = None
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            if cv2.contourArea(approx) < 1000:
                continue
            
            if not cv2.isContourConvex(approx):
                continue
            
            x, y, w, h = cv2.boundingRect(approx)
            
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [approx], -1, (255), -1)
            
            kernel = np.ones((10, 10), np.uint8)
            inner_mask = cv2.erode(mask, kernel, iterations=1)
            mean_inner_val = cv2.mean(gray, mask=inner_mask)[0]
            
            outer_mask = cv2.dilate(mask, kernel, iterations=1)
            border_mask = cv2.subtract(outer_mask, mask)
            mean_border_val = cv2.mean(gray, mask=border_mask)[0]
            
            if mean_inner_val > MEAN_INNER_VAL and mean_border_val < MEAN_BORDER_VAL:
                detected_rect = approx
                break
    
    # === 原有的透视变换和绘制逻辑（完全不修改） ===
    global last_dx, last_dy
    if detected_rect is not None:
        cv2.drawContours(frame, [detected_rect], -1, (0, 255, 0), 3)
        cv2.putText(frame, "A4 Paper Detected", (detected_rect.ravel()[0], detected_rect.ravel()[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        rect_pts = order_points(detected_rect.reshape(4, 2))
        (tl, tr, br, bl) = rect_pts
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        detected_width = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        detected_height = max(int(heightA), int(heightB))
        
        aspect_ratio = 26.0 / 18.0
        
        if detected_width / detected_height > aspect_ratio:
            maxWidth = detected_width
            maxHeight = int(detected_width / aspect_ratio)
        else:
            maxHeight = detected_height
            maxWidth = int(detected_height * aspect_ratio)
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        
        # === 原有的画圆逻辑（完全不修改） ===
        PHYSICAL_WIDTH_CM = 26.0
        PHYSICAL_HEIGHT_CM = 18.0
        
        pixels_per_cm_w = maxWidth / PHYSICAL_WIDTH_CM
        pixels_per_cm_h = maxHeight / PHYSICAL_HEIGHT_CM
        pixels_per_cm = (pixels_per_cm_w + pixels_per_cm_h) / 2.0
        
        CIRCLE_RADIUS_CM = 6.0
        radius_px = int(CIRCLE_RADIUS_CM * pixels_per_cm)
        
        center_px = (maxWidth // 2, maxHeight // 2)
        cv2.circle(warped, center_px, radius_px, (255, 0, 0), 1) 
        
        inv_M = np.linalg.inv(M)
        
        num_points = 100
        circle_points_warped = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center_px[0] + radius_px * np.cos(angle)
            y = center_px[1] + radius_px * np.sin(angle)
            circle_points_warped.append([x, y])
        
        circle_points_warped = np.array([circle_points_warped], dtype=np.float32)
        original_circle_points = cv2.perspectiveTransform(circle_points_warped, inv_M)
        cv2.polylines(frame, [np.int32(original_circle_points)], True, (255, 0, 0), 1)
        
        padding = 20
        warped_image = cv2.copyMakeBorder(warped, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        center_warped = (warped.shape[1] // 2, warped.shape[0] // 2)
        center_warped_homogeneous = np.array([center_warped[0], center_warped[1], 1], dtype=np.float32)
        original_center_homogeneous = inv_M.dot(center_warped_homogeneous)
        original_center = (original_center_homogeneous[0] / original_center_homogeneous[2], original_center_homogeneous[1] / original_center_homogeneous[2])
        
        center_x, center_y = int(original_center[0]), int(original_center[1])
        cv2.circle(frame, (center_x, center_y), 1, (0, 0, 255), -1)
        cv2.putText(frame, "Center", (center_x - 30, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # === 距离测量 ===
        distance_mm = distance_calculator.calculate_distance_from_width(detected_width, frame.shape[1])
        distance_calculator.update_distance_history(distance_mm)
        avg_distance = distance_calculator.get_averaged_distance()
        
        # === 使用动态屏幕中心偏移（新的核心逻辑） ===
        if avg_distance:
            offset_x, offset_y = offset_calculator.calculate_screen_center_offset(avg_distance)
        else:
            offset_x, offset_y = 0, 0  # 默认偏移
        
        # 计算动态屏幕中心
        screen_center_x = frame.shape[1] // 2 + offset_x
        screen_center_y = frame.shape[0] // 2 + offset_y
        
        # 计算偏移量
        dx = center_x - screen_center_x
        dy = center_y - screen_center_y
        distance_to_center = np.sqrt(dx**2 + dy**2)
        
        cv2.circle(frame, (screen_center_x, screen_center_y), 3, (0, 255, 255), -1)
        cv2.putText(frame, "Screen Center", (screen_center_x - 50, screen_center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # === 显示信息 ===
        info_y = 30
        if distance_mm:
            cv2.putText(frame, f"Distance: {distance_mm:.1f}mm", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
        
        if avg_distance:
            cv2.putText(frame, f"Avg Dist: {avg_distance:.1f}mm", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
        
        cv2.putText(frame, f"Screen Offset: ({offset_x}, {offset_y})", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        info_y += 20
        
        cv2.putText(frame, f"Target Offset: ({dx}, {dy})", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if abs(dx) + abs(dy) < 6 and abs(last_dx) + abs(last_dy) < 6 :
            cv2.putText(frame, "Aligned", (screen_center_x - 50, screen_center_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            global is_tarking, track_cnt
            if not is_tarking:
                track_cnt += 1
                if track_cnt > 4:
                    is_tarking = True
                    track_cnt = 0
                    time.sleep(0.001) 
                    serial_write(b"1\n")
                    time.sleep(0.001)
        else:
            is_tarking = False
            track_cnt = 0

        last_dx = dx
        last_dy = dy
        
        print(f"{-dx},{dy}")
        serial_write(f"{-dx},{dy}\n".encode())
        
        # === 状态显示 ===
        status_text = "Serial: ON" if ENABLE_SERIAL and ser is not None else "Serial: OFF"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ENABLE_SERIAL and ser is not None else (0, 0, 255), 2)
        
        cv2.putText(frame, "Dynamic Center: ON", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, warped_image, distance_mm, avg_distance

    return frame, None, None, None

def main():
    # 初始化组件
    distance_calculator = SimpleDistanceCalculator()
    offset_calculator = DistanceOffsetCalculator()
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开视频流。")
        return
    
    print("=== 简化A4纸跟踪系统（固定参数+拟合偏移） ===")
    print("距离计算参数:", CAMERA_PARAMS)
    print("距离范围: 500-1500mm")
    print("\n操作:")
    print("- 'q': 退出程序")
    print("- 'h': 清除距离历史记录")
    print("- 'd': 显示当前距离和偏移信息")
    
    transformed_view = np.zeros((100, 100, 3), np.uint8)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法接收帧（视频流结束？）。正在退出...")
            break
        
        # 主处理函数
        processed_frame, warped_image, distance, avg_distance = \
            find_a4_paper(frame, distance_calculator, offset_calculator)
        
        cv2.imshow('Simplified A4 Paper Tracking System', processed_frame)
        
        if warped_image is not None:
            transformed_view = warped_image
        
        cv2.imshow('Transformed Rectangle', transformed_view)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            distance_calculator.distance_history.clear()
            print("距离历史记录已清除")
        elif key == ord('d'):
            if avg_distance:
                offset_x, offset_y = offset_calculator.calculate_screen_center_offset(avg_distance)
                print(f"当前平均距离: {avg_distance:.1f}mm")
                print(f"计算的屏幕中心偏移: ({offset_x}, {offset_y})")
    
    # 完成后，释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
    
    # 关闭串口连接
    if ENABLE_SERIAL and ser is not None:
        ser.close()
        print("串口已关闭")
    
    # 显示最终统计信息
    if distance_calculator.distance_history:
        avg_distance = np.mean(distance_calculator.distance_history)
        std_distance = np.std(distance_calculator.distance_history)
        print(f"\n=== 最终统计信息 ===")
        print(f"平均距离: {avg_distance:.1f} ± {std_distance:.1f} mm")
        
        # 显示最终的屏幕中心偏移
        offset_x, offset_y = offset_calculator.calculate_screen_center_offset(avg_distance)
        print(f"最终屏幕中心偏移: ({offset_x}, {offset_y})")

if __name__ == '__main__':
    main()
