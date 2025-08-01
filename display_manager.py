import cv2
import numpy as np
import os
from dynamic_config import config

class DisplayManager:
    """显示管理器 - 负责所有的图像显示和信息绘制"""
    
    def __init__(self):
        self.transformed_view = np.zeros((100, 100, 3), np.uint8)
        self.display_enabled = config.ENABLE_DISPLAY
        print(f"配置中的 ENABLE_DISPLAY: {self.display_enabled}")
        
        # 检测是否在SSH环境下
        self.is_ssh = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ or os.environ.get('DISPLAY', '') == ''
        
        if self.is_ssh and self.display_enabled:
            print("检测到SSH环境，将禁用图像显示窗口")
            self.display_enabled = False
        elif self.display_enabled:
            print("显示功能已启用")
        else:
            print("显示功能已禁用")
    
    def draw_detection_info(self, frame, detected_rect, center_x, center_y):
        """绘制检测信息"""
        if detected_rect is not None:
            # 绘制检测到的轮廓
            cv2.drawContours(frame, [detected_rect], -1, (0, 255, 0), 3)
            cv2.putText(frame, "A4 Paper Detected", 
                       (detected_rect.ravel()[0], detected_rect.ravel()[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(frame, (center_x, center_y), 1, (0, 0, 255), -1)
            cv2.putText(frame, "Center", (center_x - 30, center_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    def draw_screen_center(self, frame, screen_center_x, screen_center_y):
        """绘制屏幕中心"""
        cv2.circle(frame, (screen_center_x, screen_center_y), 3, (0, 255, 255), -1)
        cv2.putText(frame, "Screen Center", (screen_center_x - 50, screen_center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def draw_distance_info(self, frame, distance_mm, avg_distance):
        """绘制距离信息"""
        info_y = 30
        if distance_mm:
            cv2.putText(frame, f"Distance: {distance_mm:.1f}mm", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 25
        
        if avg_distance:
            cv2.putText(frame, f"Avg Dist: {avg_distance:.1f}mm", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
        
        return info_y
    
    def draw_offset_info(self, frame, offset_x, offset_y, dx, dy, alignment_status, info_y):
        """绘制偏移信息"""
        cv2.putText(frame, f"Screen Offset: ({offset_x}, {offset_y})", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        info_y += 20
        
        cv2.putText(frame, f"Target Offset: ({dx}, {dy})", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if alignment_status:
            cv2.putText(frame, alignment_status, (dx + 320 - 50, dy + 240 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def draw_status_info(self, frame, serial_connected):
        """绘制状态信息"""
        # 串口状态
        status_text = "Serial: ON" if serial_connected else "Serial: OFF"
        color = (0, 255, 0) if serial_connected else (0, 0, 255)
        cv2.putText(frame, status_text, (10, frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 动态中心状态
        cv2.putText(frame, "Dynamic Center: ON", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def update_transformed_view(self, warped_image):
        """更新变换视图"""
        if warped_image is not None:
            # 添加边框
            padding = 20
            self.transformed_view = cv2.copyMakeBorder(
                warped_image, padding, padding, padding, padding, 
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )
    
    def show_frames(self, processed_frame):
        """显示所有窗口"""
        if not self.display_enabled:
            return
            
        try:
            cv2.imshow('Simplified A4 Paper Tracking System', processed_frame)
            cv2.imshow('Transformed Rectangle', self.transformed_view)
        except cv2.error as e:
            if "can't open display" in str(e).lower() or "no display" in str(e).lower():
                print("无法打开显示器，切换到无头模式")
                self.display_enabled = False
            else:
                raise e
    
    def handle_keyboard_input(self, distance_calculator, offset_calculator):
        """处理键盘输入"""
        if not self.display_enabled:
            # 无显示模式下，添加简单的延时避免CPU占用过高
            import time
            time.sleep(0.01)
            return None
            
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('h'):
            distance_calculator.clear_history()
            print("距离历史记录已清除")
        elif key == ord('d'):
            avg_distance = distance_calculator.get_averaged_distance()
            if avg_distance:
                offset_x, offset_y = offset_calculator.calculate_screen_center_offset(avg_distance)
                print(f"当前平均距离: {avg_distance:.1f}mm")
                print(f"计算的屏幕中心偏移: ({offset_x}, {offset_y})")
        
        return None
    
    def cleanup(self):
        """清理资源"""
        if self.display_enabled:
            try:
                cv2.destroyAllWindows()
            except:
                pass  # 忽略清理时的错误
