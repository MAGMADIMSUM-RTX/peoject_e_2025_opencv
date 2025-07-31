import numpy as np
from config import CALIBRATION_POINTS, MIN_DISTANCE_MM, MAX_DISTANCE_MM

class DistanceOffsetCalculator:
    """基于距离的屏幕中心偏移计算器"""
    
    def __init__(self, calibration_points=None):
        # 使用提供的校准点或默认校准点
        self.calibration_points = calibration_points or CALIBRATION_POINTS
        
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
        self._validate_fitting()
    
    def _validate_fitting(self):
        """验证拟合精度"""
        print("\n验证拟合精度:")
        for dist, expected_x, expected_y in self.calibration_points:
            calc_x, calc_y = self.calculate_screen_center_offset(dist)
            print(f"距离{dist}mm: 期望({expected_x},{expected_y}) vs 计算({calc_x:.1f},{calc_y:.1f})")
    
    def calculate_screen_center_offset(self, distance_mm):
        """根据距离计算屏幕中心偏移量"""
        # 限制距离范围
        distance_mm = max(MIN_DISTANCE_MM, min(MAX_DISTANCE_MM, distance_mm))
        
        # 使用多项式计算偏移量
        offset_x = np.polyval(self.poly_coeffs_x, distance_mm)
        offset_y = np.polyval(self.poly_coeffs_y, distance_mm)
        
        return int(round(offset_x)), int(round(offset_y))
    
    def update_calibration_points(self, new_points):
        """更新校准点并重新拟合"""
        self.calibration_points = new_points
        self.__init__(new_points)
