import cv2
import numpy as np
import math
import json
import os

# A4纸的实际尺寸 (单位: 毫米)
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# 检测阈值参数
MEAN_INNER_VAL = 100
MEAN_BORDER_VAL = 80

# 校准文件路径
CALIBRATION_FILE = "camera_calibration.json"

class CalibratedDistanceEstimator:
    def __init__(self):
        # 默认摄像头参数（需要校准）
        self.camera_params = {
            "focal_length_mm": 2.8,      # 焦距
            "sensor_width_mm": 5.37,     # 传感器宽度  
            "sensor_height_mm": 4.04,    # 传感器高度
            "calibration_factor": 1.0     # 校准因子
        }
        
        # 加载校准参数
        self.load_calibration()
        
        # 历史记录
        self.distance_history = []
        self.max_history = 10
        
        # 校准相关
        self.calibration_mode = False
        self.calibration_data = []
        
    def load_calibration(self):
        """加载校准参数"""
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r') as f:
                    saved_params = json.load(f)
                    self.camera_params.update(saved_params)
                print(f"已加载校准参数: {self.camera_params}")
            except Exception as e:
                print(f"加载校准参数失败: {e}")
    
    def save_calibration(self):
        """保存校准参数"""
        try:
            with open(CALIBRATION_FILE, 'w') as f:
                json.dump(self.camera_params, f, indent=2)
            print(f"校准参数已保存到 {CALIBRATION_FILE}")
        except Exception as e:
            print(f"保存校准参数失败: {e}")
    
    def order_points(self, pts):
        """整理四个顶点的顺序"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        return rect
    
    def calculate_distance_from_width(self, pixel_width, frame_width):
        """
        基于宽度计算距离（推荐方法）
        使用相似三角形原理和摄像头FOV
        """
        focal_length = self.camera_params["focal_length_mm"]
        sensor_width = self.camera_params["sensor_width_mm"]
        calibration_factor = self.camera_params["calibration_factor"]
        
        # 计算水平视场角
        fov_horizontal_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        
        # 计算在1米距离处每像素对应的实际距离
        mm_per_pixel_at_1m = (1000 * math.tan(fov_horizontal_rad / 2) * 2) / frame_width
        
        # 计算距离
        if pixel_width > 0 and mm_per_pixel_at_1m > 0:
            distance_mm = (A4_WIDTH_MM * 1000) / (pixel_width * mm_per_pixel_at_1m)
            return distance_mm * calibration_factor
        return None
    
    def calculate_distance_simple_pinhole(self, pixel_width, frame_width):
        """
        简化的针孔相机模型
        更直观的计算方法
        """
        focal_length = self.camera_params["focal_length_mm"]
        sensor_width = self.camera_params["sensor_width_mm"]
        calibration_factor = self.camera_params["calibration_factor"]
        
        # 针孔相机模型: distance = (real_width * focal_length * image_width) / (pixel_width * sensor_width)
        if pixel_width > 0:
            distance_mm = (A4_WIDTH_MM * focal_length * frame_width) / (pixel_width * sensor_width)
            return distance_mm * calibration_factor
        return None
    
    def auto_calibrate_from_known_distance(self, measured_distance, known_distance):
        """
        根据已知距离自动校准
        """
        if measured_distance and measured_distance > 0:
            new_factor = known_distance / measured_distance
            self.camera_params["calibration_factor"] *= new_factor
            self.save_calibration()
            print(f"校准因子已更新为: {self.camera_params['calibration_factor']:.4f}")
            return True
        return False
    
    def manual_calibrate_focal_length(self, pixel_width, frame_width, known_distance):
        """
        手动校准焦距
        """
        sensor_width = self.camera_params["sensor_width_mm"]
        
        # 反向计算焦距
        # distance = (A4_WIDTH * focal_length * frame_width) / (pixel_width * sensor_width)
        # focal_length = (distance * pixel_width * sensor_width) / (A4_WIDTH * frame_width)
        
        if pixel_width > 0:
            new_focal_length = (known_distance * pixel_width * sensor_width) / (A4_WIDTH_MM * frame_width)
            self.camera_params["focal_length_mm"] = new_focal_length
            self.camera_params["calibration_factor"] = 1.0  # 重置校准因子
            self.save_calibration()
            print(f"焦距已校准为: {new_focal_length:.4f} mm")
            return True
        return False
    
    def detect_and_measure_a4(self, frame):
        """检测A4纸并测量距离"""
        original_frame = frame.copy()
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用CLAHE改善对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        detected_rect = None
        distance_mm = None
        pixel_width = 0
        
        for contour in contours:
            # 轮廓近似
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                # 检查面积
                area = cv2.contourArea(approx)
                if area < 1000:
                    continue
                
                # 检查是否为凸形
                if not cv2.isContourConvex(approx):
                    continue
                
                # 颜色验证
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
        
        if detected_rect is not None:
            # 绘制检测到的矩形
            cv2.drawContours(frame, [detected_rect], -1, (0, 255, 0), 3)
            
            # 透视变换准备
            rect_pts = self.order_points(detected_rect.reshape(4, 2))
            (tl, tr, br, bl) = rect_pts
            
            # 计算像素宽度（更精确的方法）
            width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            pixel_width = max(int(width_a), int(width_b))
            
            # 计算距离
            distance_fov = self.calculate_distance_from_width(pixel_width, frame.shape[1])
            distance_pinhole = self.calculate_distance_simple_pinhole(pixel_width, frame.shape[1])
            
            # 使用FOV方法作为主要结果
            distance_mm = distance_fov
            
            # 更新历史记录
            if distance_mm:
                self.distance_history.append(distance_mm)
                if len(self.distance_history) > self.max_history:
                    self.distance_history.pop(0)
            
            # 显示信息
            cv2.putText(frame, "A4 Paper Detected", (detected_rect.ravel()[0], detected_rect.ravel()[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示距离信息
            y_offset = 30
            if distance_fov:
                cv2.putText(frame, f"Distance (FOV): {distance_fov:.1f} mm", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            if distance_pinhole:
                cv2.putText(frame, f"Distance (Pinhole): {distance_pinhole:.1f} mm", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                y_offset += 25
            
            cv2.putText(frame, f"Pixel Width: {pixel_width}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            # 显示平均距离
            if self.distance_history:
                avg_distance = np.mean(self.distance_history)
                cv2.putText(frame, f"Avg Distance: {avg_distance:.1f} mm", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
            
            # 显示校准参数
            cv2.putText(frame, f"Focal: {self.camera_params['focal_length_mm']:.2f}mm", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_offset += 20
            cv2.putText(frame, f"Cal Factor: {self.camera_params['calibration_factor']:.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            return frame, pixel_width, distance_mm
        
        return frame, 0, None

def main():
    estimator = CalibratedDistanceEstimator()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    print("=== 可校准距离测量程序 ===")
    print("当前校准参数:")
    for key, value in estimator.camera_params.items():
        print(f"  {key}: {value}")
    print("\n操作说明:")
    print("- 'q': 退出程序")
    print("- 'c': 进入校准模式")
    print("- 'r': 重置校准参数")
    print("- 's': 保存当前校准参数")
    print("- 'h': 清除历史记录")
    print("\n校准模式操作:")
    print("- '1': 已知距离300mm，按1校准")
    print("- '2': 已知距离500mm，按2校准") 
    print("- '3': 已知距离800mm，按3校准")
    print("- 'f': 使用当前测量校准焦距")
    
    calibration_mode = False
    known_distances = {ord('1'): 300, ord('2'): 500, ord('3'): 800}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测A4纸并测量距离
        processed_frame, pixel_width, distance = estimator.detect_and_measure_a4(frame)
        
        # 显示校准模式状态
        if calibration_mode:
            cv2.putText(processed_frame, "CALIBRATION MODE", (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Calibrated Distance Measurement', processed_frame)
        
        # 键盘输入处理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration_mode = not calibration_mode
            print(f"校准模式: {'开启' if calibration_mode else '关闭'}")
        elif key == ord('r'):
            # 重置校准参数
            estimator.camera_params = {
                "focal_length_mm": 4.0,
                "sensor_width_mm": 3.68,
                "sensor_height_mm": 2.76,
                "calibration_factor": 1.0
            }
            estimator.save_calibration()
            print("校准参数已重置")
        elif key == ord('s'):
            estimator.save_calibration()
        elif key == ord('h'):
            estimator.distance_history.clear()
            print("历史记录已清除")
        elif calibration_mode and key in known_distances:
            # 校准距离
            if distance:
                known_dist = known_distances[key]
                estimator.auto_calibrate_from_known_distance(distance, known_dist)
                print(f"使用已知距离 {known_dist}mm 进行校准")
        elif calibration_mode and key == ord('f'):
            # 校准焦距
            if distance and pixel_width > 0:
                known_dist = float(input("请输入实际距离(mm): "))
                estimator.manual_calibrate_focal_length(pixel_width, frame.shape[1], known_dist)
                print("焦距校准完成")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 显示最终统计
    if estimator.distance_history:
        avg_distance = np.mean(estimator.distance_history)
        std_distance = np.std(estimator.distance_history)
        print(f"\n最终统计:")
        print(f"平均距离: {avg_distance:.1f} ± {std_distance:.1f} mm")
        print(f"最终校准参数: {estimator.camera_params}")

if __name__ == '__main__':
    main()