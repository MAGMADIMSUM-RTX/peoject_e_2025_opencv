import cv2
import numpy as np
import time
from config import (
    MEAN_INNER_VAL, MEAN_BORDER_VAL, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    GAUSSIAN_BLUR_KERNEL, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C,
    CONTOUR_AREA_THRESHOLD, APPROX_EPSILON_FACTOR, MORPHOLOGY_KERNEL_SIZE,
    CIRCLE_RADIUS_CM, PHYSICAL_WIDTH_CM, PHYSICAL_HEIGHT_CM,
    ALIGNMENT_THRESHOLD, TRACK_COUNT_THRESHOLD
)

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
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        gray = clahe.apply(gray)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        detected_rect = None
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, APPROX_EPSILON_FACTOR * peri, True)
            
            if len(approx) == 4:
                if cv2.contourArea(approx) < CONTOUR_AREA_THRESHOLD:
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
        kernel = np.ones(MORPHOLOGY_KERNEL_SIZE, np.uint8)
        inner_mask = cv2.erode(mask, kernel, iterations=1)
        mean_inner_val = cv2.mean(gray, mask=inner_mask)[0]
        
        # 边框区域（黑边部分）
        outer_mask = cv2.dilate(mask, kernel, iterations=1)
        border_mask = cv2.subtract(outer_mask, mask)
        mean_border_val = cv2.mean(gray, mask=border_mask)[0]
        
        return mean_inner_val > MEAN_INNER_VAL and mean_border_val < MEAN_BORDER_VAL
    
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
        
        # 计算目标尺寸
        aspect_ratio = 26.0 / 18.0
        
        if detected_width / detected_height > aspect_ratio:
            maxWidth = detected_width
            maxHeight = int(detected_width / aspect_ratio)
        else:
            maxHeight = detected_height
            maxWidth = int(detected_height * aspect_ratio)
        
        # 透视变换
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect_pts, dst)
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
        
        return warped, M, detected_width, (maxWidth, maxHeight)
    
    def draw_circle_on_warped(self, warped, frame, M):
        """在变换后的图像上绘制圆形，并投影回原图"""
        maxWidth, maxHeight = warped.shape[1], warped.shape[0]
        
        # 计算像素/厘米比例
        pixels_per_cm_w = maxWidth / PHYSICAL_WIDTH_CM
        pixels_per_cm_h = maxHeight / PHYSICAL_HEIGHT_CM
        pixels_per_cm = (pixels_per_cm_w + pixels_per_cm_h) / 2.0
        
        # 绘制圆形
        radius_px = int(CIRCLE_RADIUS_CM * pixels_per_cm)
        center_px = (maxWidth // 2, maxHeight // 2)
        cv2.circle(warped, center_px, radius_px, (255, 0, 0), 1)
        
        # 投影圆形到原图
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
        
        # 计算中心点在原图中的位置
        center_warped_homogeneous = np.array([center_px[0], center_px[1], 1], dtype=np.float32)
        original_center_homogeneous = inv_M.dot(center_warped_homogeneous)
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
        if abs(dx) + abs(dy) < ALIGNMENT_THRESHOLD and abs(self.last_dx) + abs(self.last_dy) < ALIGNMENT_THRESHOLD:
            if not self.is_tracking:
                self.track_cnt += 1
                if self.track_cnt > TRACK_COUNT_THRESHOLD:
                    self.is_tracking = True
                    self.track_cnt = 0
                    time.sleep(0.001)
                    serial_controller.write(b"1\n")
                    time.sleep(0.001)
            alignment_status = "Aligned"
        else:
            self.is_tracking = False
            self.track_cnt = 0
            alignment_status = None
        
        # 发送偏移数据
        print(f"{-dx},{dy}")
        serial_controller.write(f"{-dx},{dy}\n")
        
        self.last_dx = dx
        self.last_dy = dy
        
        return dx, dy, alignment_status
