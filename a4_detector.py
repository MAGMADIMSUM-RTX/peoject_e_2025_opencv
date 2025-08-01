import cv2
import numpy as np
import time
from dynamic_config import config

def order_points(pts):
    """对四个点进行排序"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

class A4PaperDetector:
    """A4纸检测器"""
    
    def __init__(self):
        self.last_dx = 100
        self.last_dy = 100
        self.is_tracking = False
        self.track_cnt = 0
    
    def detect_a4_paper(self, frame):
        """检测A4纸轮廓"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE)
        gray = clahe.apply(gray)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_KERNEL, 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, config.ADAPTIVE_THRESH_BLOCK_SIZE, config.ADAPTIVE_THRESH_C)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        detected_rect = None
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, config.APPROX_EPSILON_FACTOR * peri, True)
            
            if len(approx) == 4:
                if cv2.contourArea(approx) < config.CONTOUR_AREA_THRESHOLD:
                    continue
                
                if not cv2.isContourConvex(approx):
                    continue
                
                if self._validate_a4_paper(gray, approx):
                    detected_rect = approx
                    break
        
        return detected_rect
    
    def _validate_a4_paper(self, gray, approx):
        """验证是否为A4纸（白纸黑边）"""
        # 创建掩膜
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [approx], -1, (255), -1)
        
        # 内部区域（白纸部分）
        kernel = np.ones(config.MORPHOLOGY_KERNEL_SIZE, np.uint8)
        inner_mask = cv2.erode(mask, kernel, iterations=1)
        mean_inner_val = cv2.mean(gray, mask=inner_mask)[0]
        
        # 边框区域（黑边部分）
        outer_mask = cv2.dilate(mask, kernel, iterations=1)
        border_mask = cv2.subtract(outer_mask, mask)
        mean_border_val = cv2.mean(gray, mask=border_mask)[0]
        
        return mean_inner_val > config.MEAN_INNER_VAL and mean_border_val < config.MEAN_BORDER_VAL
    
    def calculate_perspective_corrected_dimensions(self, detected_rect, frame_width, frame_height):
        """计算透视校正后的A4纸尺寸"""
        if detected_rect is None:
            return None, None, None
        
        rect_pts = order_points(detected_rect.reshape(4, 2))
        (tl, tr, br, bl) = rect_pts
        
        # 计算四条边的长度
        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
        height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
        
        # 计算实际观测到的宽高
        observed_width = (width_top + width_bottom) / 2
        observed_height = (height_left + height_right) / 2
        
        # 防止除零错误：确保高度不为零
        if observed_height <= 1e-8:
            print("警告: 观测高度接近零，使用默认值")
            observed_height = 1.0
        
        # A4纸的标准宽高比 (210/297 ≈ 0.707)
        standard_ratio = config.A4_WIDTH_MM / config.A4_HEIGHT_MM
        observed_ratio = observed_width / observed_height
        
        # 透视校正策略：使用单应性变换计算真实尺寸
        # 定义A4纸在标准位置的四个点
        standard_width = 210  # 标准宽度，单位：像素（任意单位）
        standard_height = 297  # 标准高度
        
        dst_points = np.array([
            [0, 0],
            [standard_width, 0],
            [standard_width, standard_height],
            [0, standard_height]
        ], dtype=np.float32)
        
        # 计算单应性矩阵
        try:
            homography = cv2.getPerspectiveTransform(rect_pts.astype(np.float32), dst_points)
        except cv2.error as e:
            print(f"警告: 单应性变换计算失败: {e}")
            # 使用默认值
            homography = np.eye(3, dtype=np.float32)
        
        # 计算透视校正后的面积比例
        # 原始四边形面积
        original_area = cv2.contourArea(rect_pts)
        # 标准四边形面积
        standard_area = standard_width * standard_height
        
        # 防止除零错误：确保标准面积不为零
        if standard_area <= 1e-8:
            print("警告: 标准面积接近零，使用默认值")
            standard_area = 1.0
        
        # 面积缩放因子
        area_scale = np.sqrt(original_area / standard_area)
        
        # 透视失真校正
        # 计算透视变换的行列式，用于估计面积变化
        det = np.linalg.det(homography[:2, :2])
        perspective_scale = np.sqrt(abs(det)) if abs(det) > 1e-8 else 1.0
        
        # 计算倾斜角度（基于矩形的变形程度）
        # 防止除零错误
        width_max = max(width_top, width_bottom)
        height_max = max(height_left, height_right)
        
        width_diff = abs(width_top - width_bottom) / width_max if width_max > 1e-8 else 0
        height_diff = abs(height_left - height_right) / height_max if height_max > 1e-8 else 0
        
        # 倾斜程度指标
        tilt_factor = 1.0 + config.CAMERA_PARAMS.get("tilt_sensitivity", 0.5) * (width_diff + height_diff)
        
        # 最终校正
        # 基于几何原理：当A4纸倾斜时，投影宽度会减小
        # 校正因子应该根据观测比例与标准比例的偏差来调整
        if observed_ratio > standard_ratio:
            # 宽度相对过大，说明纸张向后倾斜（远端变窄）
            correction_factor = standard_ratio / observed_ratio if observed_ratio > 1e-8 else 1.0
            corrected_width = observed_width * correction_factor * tilt_factor
            corrected_height = observed_height
        else:
            # 高度相对过大，说明纸张左右倾斜
            correction_factor = observed_ratio / standard_ratio if standard_ratio > 1e-8 else 1.0
            corrected_width = observed_width
            corrected_height = observed_height * correction_factor * tilt_factor
        
        # 应用透视缩放校正
        corrected_width *= perspective_scale
        corrected_height *= perspective_scale
        
        # 计算A4纸中心点
        center_x = (tl[0] + tr[0] + br[0] + bl[0]) / 4
        center_y = (tl[1] + tr[1] + br[1] + bl[1]) / 4
        
        return corrected_width, corrected_height, (center_x, center_y)
    
    def create_warped_image(self, frame, detected_rect):
        """创建透视变换后的图像"""
        if detected_rect is None:
            return None, None, None, None
        
        rect_pts = order_points(detected_rect.reshape(4, 2))
        (tl, tr, br, bl) = rect_pts
        
        # 计算宽度和高度
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        detected_width = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        detected_height = max(int(heightA), int(heightB))
        
        # 防止除零错误：确保检测到的高度不为零
        if detected_height <= 0:
            print("警告: 检测到的高度为零，使用默认值")
            detected_height = 1
        
        # 计算目标尺寸
        aspect_ratio = 26.0 / 18.0
        
        if detected_width / detected_height > aspect_ratio:
            maxWidth = detected_width
            maxHeight = int(detected_width / aspect_ratio)
        else:
            maxHeight = detected_height
            maxWidth = int(detected_height * aspect_ratio)
        
        # 确保最小尺寸
        maxWidth = max(maxWidth, 1)
        maxHeight = max(maxHeight, 1)
        
        # 透视变换
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        try:
            M = cv2.getPerspectiveTransform(rect_pts, dst)
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        except cv2.error as e:
            print(f"警告: 透视变换失败: {e}")
            return None, None, None, None
        
        return warped, M, detected_width, (maxWidth, maxHeight)
    
    def draw_circle_on_warped(self, warped, frame, M):
        """在变换后的图像上绘制圆形，并投影回原图"""
        maxWidth, maxHeight = warped.shape[1], warped.shape[0]
        
        # 计算像素/厘米比例
        # 防止除零错误
        pixels_per_cm_w = maxWidth / config.PHYSICAL_WIDTH_CM if config.PHYSICAL_WIDTH_CM > 1e-8 else 1.0
        pixels_per_cm_h = maxHeight / config.PHYSICAL_HEIGHT_CM if config.PHYSICAL_HEIGHT_CM > 1e-8 else 1.0
        pixels_per_cm = (pixels_per_cm_w + pixels_per_cm_h) / 2.0
        
        # 绘制圆形
        radius_px = int(config.CIRCLE_RADIUS_CM * pixels_per_cm)
        center_px = (maxWidth // 2, maxHeight // 2)
        cv2.circle(warped, center_px, radius_px, (255, 0, 0), 1)
        
        # 投影圆形到原图
        try:
            inv_M = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            print("警告: 矩阵不可逆，使用单位矩阵")
            inv_M = np.eye(3, dtype=np.float32)
            # 在这种情况下，直接返回变换图像的中心点
            return center_px[0], center_px[1]
        
        num_points = 100
        circle_points_warped = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center_px[0] + radius_px * np.cos(angle)
            y = center_px[1] + radius_px * np.sin(angle)
            circle_points_warped.append([x, y])
        
        circle_points_warped = np.array([circle_points_warped], dtype=np.float32)
        
        try:
            original_circle_points = cv2.perspectiveTransform(circle_points_warped, inv_M)
            cv2.polylines(frame, [np.int32(original_circle_points)], True, (255, 0, 0), 1)
        except cv2.error as e:
            print(f"警告: 透视变换圆形失败: {e}")
        
        # 计算中心点在原图中的位置
        center_warped_homogeneous = np.array([center_px[0], center_px[1], 1], dtype=np.float32)
        original_center_homogeneous = inv_M.dot(center_warped_homogeneous)
        
        # 防止除零错误
        if abs(original_center_homogeneous[2]) < 1e-8:  # 检查是否接近零
            print("警告: 透视变换除数接近零，使用默认中心点")
            return center_px[0], center_px[1]  # 返回变换图像的中心点
        
        original_center = (
            original_center_homogeneous[0] / original_center_homogeneous[2],
            original_center_homogeneous[1] / original_center_homogeneous[2]
        )
        
        return int(original_center[0]), int(original_center[1])
    
    def calculate_offset_and_check_alignment(self, center_x, center_y, screen_center_x, screen_center_y, serial_controller):
        """计算偏移量并检查对齐状态"""
        dx = center_x - screen_center_x
        dy = center_y - screen_center_y
        
        # 检查对齐状态
        if abs(dx) + abs(dy) < config.ALIGNMENT_THRESHOLD and abs(self.last_dx) + abs(self.last_dy) < config.ALIGNMENT_THRESHOLD:
            if not self.is_tracking:
                self.track_cnt += 1
                if self.track_cnt > config.TRACK_COUNT_THRESHOLD:
                    self.is_tracking = True
                    self.track_cnt = 0
                    time.sleep(0.001)
                    try:
                        serial_controller.write(b"1\n")
                    except Exception as e:
                        print(f"串口写入失败: {e}")
                    time.sleep(0.001)
            alignment_status = "Aligned"
        else:
            self.is_tracking = False
            self.track_cnt = 0
            alignment_status = None
        
        # 发送偏移数据
        print(f"{-dx},{dy}")
        try:
            serial_controller.write(f"{-dx},{dy}\n")
        except Exception as e:
            print(f"串口写入失败: {e}")
        
        # 可选：读取串口返回的数据
        try:
            if serial_controller.in_waiting() > 0:
                response = serial_controller.read_line(timeout=0.1)
                if response:
                    print(f"串口返回: {response}")
        except Exception as e:
            print(f"串口读取失败: {e}")
        
        self.last_dx = dx
        self.last_dy = dy
        
        return dx, dy, alignment_status