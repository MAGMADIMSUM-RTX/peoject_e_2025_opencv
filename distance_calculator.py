import math
import numpy as np
from config import (
    CAMERA_PARAMS, A4_WIDTH_MM, MAX_DISTANCE_HISTORY
)

class SimpleDistanceCalculator:
    """简化的距离计算器（使用固定参数）"""
    
    def __init__(self, max_history=None):
        self.distance_history = []
        self.max_history = max_history or MAX_DISTANCE_HISTORY
        
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
    
    def clear_history(self):
        """清除距离历史记录"""
        self.distance_history.clear()
    
    def get_distance_statistics(self):
        """获取距离统计信息"""
        if not self.distance_history:
            return None, None
        
        mean_distance = np.mean(self.distance_history)
        std_distance = np.std(self.distance_history)
        return mean_distance, std_distance
