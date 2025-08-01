import math
import numpy as np
import cv2
from dynamic_config import config

class SimpleDistanceCalculator:
    """简化的距离计算器（使用透视校正和3D模型）"""
    
    def __init__(self, max_history=None):
        self.distance_history = []
        self.max_history = max_history or config.MAX_DISTANCE_HISTORY
        
    def calculate_distance_from_corrected_width(self, corrected_pixel_width, frame_width):
        """基于透视校正后的宽度计算距离"""
        focal_length = config.CAMERA_PARAMS["focal_length_mm"]
        sensor_width = config.CAMERA_PARAMS["sensor_width_mm"]
        calibration_factor = config.CAMERA_PARAMS["calibration_factor"]
        perspective_correction_weight = config.CAMERA_PARAMS.get("perspective_correction_weight", 1.0)
        
        # 计算水平视场角
        fov_horizontal_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        
        # 计算每像素在1米距离处的毫米数
        mm_per_pixel_at_1m = (1000 * math.tan(fov_horizontal_rad / 2) * 2) / frame_width
        
        if corrected_pixel_width > 0 and mm_per_pixel_at_1m > 0:
            # 基础距离计算
            distance_mm = (config.A4_WIDTH_MM * 1000) / (corrected_pixel_width * mm_per_pixel_at_1m)
            
            # 应用校准因子和透视校正权重
            final_distance = distance_mm * calibration_factor * perspective_correction_weight
            
            return final_distance
        return None
    
    def calculate_distance_from_width(self, pixel_width, frame_width):
        """保持原有接口兼容（未校正的宽度计算）"""
        focal_length = config.CAMERA_PARAMS["focal_length_mm"]
        sensor_width = config.CAMERA_PARAMS["sensor_width_mm"]
        calibration_factor = config.CAMERA_PARAMS["calibration_factor"]
        
        fov_horizontal_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        mm_per_pixel_at_1m = (1000 * math.tan(fov_horizontal_rad / 2) * 2) / frame_width
        
        if pixel_width > 0 and mm_per_pixel_at_1m > 0:
            distance_mm = (config.A4_WIDTH_MM * 1000) / (pixel_width * mm_per_pixel_at_1m)
            return distance_mm * calibration_factor
        return None
    
    def calculate_3d_distance_with_homography(self, detected_rect, frame_width, frame_height):
        """使用单应性矩阵计算3D距离"""
        if detected_rect is None:
            return None
        
        try:
            # 获取A4纸四个角点
            rect_pts = detected_rect.reshape(4, 2)
            
            # A4纸实际尺寸（毫米）
            real_width = config.A4_WIDTH_MM
            real_height = config.A4_HEIGHT_MM
            
            # 定义A4纸在世界坐标系中的四个角点（Z=0平面）
            object_points = np.array([
                [0, 0, 0],
                [real_width, 0, 0],
                [real_width, real_height, 0],
                [0, real_height, 0]
            ], dtype=np.float32)
            
            # 图像坐标系中的四个角点
            image_points = rect_pts.astype(np.float32)
            
            # 相机内参矩阵
            focal_length_px = self._mm_to_pixels(config.CAMERA_PARAMS["focal_length_mm"], frame_width)
            camera_matrix = np.array([
                [focal_length_px, 0, frame_width / 2],
                [0, focal_length_px, frame_height / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # 畸变系数（假设无畸变）
            dist_coeffs = np.zeros(4, dtype=np.float32)
            
            # 使用PnP算法计算位姿
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            
            if success:
                # tvec包含了A4纸中心到相机的平移向量
                distance_mm = np.linalg.norm(tvec)
                
                # 应用3D校准因子
                distance_3d_factor = config.CAMERA_PARAMS.get("distance_3d_calibration_factor", 1.0)
                return distance_mm * distance_3d_factor
        except Exception as e:
            print(f"3D距离计算失败: {e}")
        
        return None
    
    def calculate_advanced_perspective_distance(self, detected_rect, frame_width, frame_height):
        """高级透视距离计算方法"""
        if detected_rect is None:
            return None
        
        try:
            from a4_detector import order_points
            
            rect_pts = order_points(detected_rect.reshape(4, 2))
            (tl, tr, br, bl) = rect_pts
            
            # 计算A4纸的透视变换参数
            # 真实世界中的A4纸尺寸
            real_width_mm = config.A4_WIDTH_MM
            real_height_mm = config.A4_HEIGHT_MM
            
            # 图像中观测到的四条边长度
            width_top = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
            width_bottom = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
            height_left = np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
            height_right = np.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
            
            # 计算透视失真程度
            width_ratio = min(width_top, width_bottom) / max(width_top, width_bottom)
            height_ratio = min(height_left, height_right) / max(height_left, height_right)
            
            # 透视失真校正因子
            perspective_distortion = (width_ratio + height_ratio) / 2
            
            # 使用较短的边作为基准（更接近真实投影）
            effective_width = min(width_top, width_bottom)
            effective_height = min(height_left, height_right)
            
            # 计算相机参数
            focal_length = config.CAMERA_PARAMS["focal_length_mm"]
            sensor_width = config.CAMERA_PARAMS["sensor_width_mm"]
            
            # 计算视场角
            fov_horizontal_rad = 2 * math.atan(sensor_width / (2 * focal_length))
            mm_per_pixel_at_1m = (1000 * math.tan(fov_horizontal_rad / 2) * 2) / frame_width
            
            # 基于有效宽度计算距离
            if effective_width > 0:
                base_distance = (real_width_mm * 1000) / (effective_width * mm_per_pixel_at_1m)
                
                # 透视校正
                perspective_correction = 1.0 / max(0.3, perspective_distortion)  # 防止除零，限制校正范围
                
                # 应用校正
                corrected_distance = base_distance * perspective_correction
                
                # 应用配置的校准因子
                calibration_factor = config.CAMERA_PARAMS["calibration_factor"]
                advanced_correction = config.CAMERA_PARAMS.get("advanced_perspective_factor", 1.0)
                
                final_distance = corrected_distance * calibration_factor * advanced_correction
                
                return final_distance
                
        except Exception as e:
            print(f"高级透视距离计算失败: {e}")
        
        return None
    
    def _mm_to_pixels(self, focal_length_mm, frame_width):
        """将焦距从毫米转换为像素"""
        sensor_width = config.CAMERA_PARAMS["sensor_width_mm"]
        return (focal_length_mm * frame_width) / sensor_width
    
    def calculate_distance_hybrid(self, corrected_pixel_width, detected_rect, frame_width, frame_height):
        """混合距离计算方法"""
        # 方法1：基于透视校正的宽度计算
        distance_width = self.calculate_distance_from_corrected_width(corrected_pixel_width, frame_width)
        
        # 方法2：基于3D单应性的距离计算
        distance_3d = self.calculate_3d_distance_with_homography(detected_rect, frame_width, frame_height)
        
        # 方法3：高级透视距离计算
        distance_advanced = self.calculate_advanced_perspective_distance(detected_rect, frame_width, frame_height)
        
        # 混合权重
        width_weight = config.CAMERA_PARAMS.get("distance_width_weight", 0.3)
        d3d_weight = config.CAMERA_PARAMS.get("distance_3d_weight", 0.3)
        advanced_weight = config.CAMERA_PARAMS.get("distance_advanced_weight", 0.4)
        
        # 归一化权重
        total_weight = width_weight + d3d_weight + advanced_weight
        if total_weight > 0:
            width_weight /= total_weight
            d3d_weight /= total_weight
            advanced_weight /= total_weight
        
        # 可用的距离值
        distances = []
        weights = []
        
        if distance_width:
            distances.append(distance_width)
            weights.append(width_weight)
        
        if distance_3d:
            distances.append(distance_3d)
            weights.append(d3d_weight)
            
        if distance_advanced:
            distances.append(distance_advanced)
            weights.append(advanced_weight)
        
        if distances:
            # 加权平均
            if len(distances) == 1:
                return distances[0]
            else:
                # 重新归一化权重
                total_available_weight = sum(weights)
                if total_available_weight > 0:
                    normalized_weights = [w / total_available_weight for w in weights]
                    final_distance = sum(d * w for d, w in zip(distances, normalized_weights))
                    return final_distance
        
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