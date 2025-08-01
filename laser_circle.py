#!/usr/bin/env python3
"""
激光圆形轨迹跟踪系统
基于laser_tracking_system.py，添加激光光斑沿圆形轨迹运动功能
主文件 - 包含主要的控制逻辑和激光光斑检测
"""

import cv2
import numpy as np
import math
import time

# 导入模块化组件
from dynamic_config import config
from serial_controller import SerialController
from distance_offset_calculator import DistanceOffsetCalculator
from distance_calculator import SimpleDistanceCalculator
from a4_detector import A4PaperDetector
from display_manager import DisplayManager

dx_from_uart = 0
dy_from_uart = 0  # 用于从串口接收偏移量

class LaserSpotDetector:
    """激光光斑检测器"""
    
    def __init__(self):
        # 基于测试数据的多组HSV范围（更鲁棒的检测）
        self.hsv_ranges = [
            # 测试数据1: H(0-6) S(0-178) V(238-255) - 高亮度蓝紫色
            ([0, 0, 238], [6, 178, 255]),
            # 测试数据2: H(2-161) S(0-34) V(155-255) - 低饱和度高亮度
            ([2, 0, 155], [161, 34, 255]),
            # 测试数据3: H(12-122) S(0-241) V(215-255) - 宽色相范围
            ([12, 0, 215], [122, 241, 255]),
            # 测试数据4: H(15-121) S(0-230) V(192-255) - 中等亮度范围
            ([15, 0, 192], [121, 230, 255]),
        ]
        
        # 将HSV范围转换为numpy数组
        self.hsv_ranges = [(np.array(lower), np.array(upper)) for lower, upper in self.hsv_ranges]
        
        # 额外的过曝白色光斑检测（作为备用）
        self.lower_overexposed = np.array([0, 0, 245])  # 提高亮度阈值
        self.higher_overexposed = np.array([180, 30, 255])  # 降低饱和度阈值
        
        # 光斑检测参数
        self.min_area = 3          # 最小面积
        self.max_area = 300        # 最大面积
        self.min_circularity = 0.3  # 最小圆形度
        
        # 形态学操作核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 调试模式：是否显示各个掩膜
        self.debug_mode = False
        
    def detect_laser_spot(self, frame):
        """检测激光光斑位置（使用多组HSV范围）"""
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 初始化组合掩膜
        mask_combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # 应用多组HSV范围检测
        for i, (lower_hsv, upper_hsv) in enumerate(self.hsv_ranges):
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            mask_combined = cv2.bitwise_or(mask_combined, mask)
            
            if self.debug_mode:
                cv2.imshow(f'Laser Mask {i+1}', mask)
        
        # 备用检测：过曝白色掩膜
        mask_overexposed = cv2.inRange(hsv, self.lower_overexposed, self.higher_overexposed)
        mask_combined = cv2.bitwise_or(mask_combined, mask_overexposed)
        
        if self.debug_mode:
            cv2.imshow('Overexposed Mask', mask_overexposed)
            cv2.imshow('Combined Laser Mask', mask_combined)
        
        # 形态学操作去噪
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, self.kernel)
        mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, self.kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_spot = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < self.min_area or area > self.max_area:
                continue
            
            # 计算圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.min_circularity:
                continue
            
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # 获取该点的HSV值用于验证
            h, s, v = hsv[cy, cx]
            
            # 综合评分：面积适中，圆形度高，亮度高
            area_score = min(area / 50.0, 1.0)  # 归一化面积权重
            brightness_score = v / 255.0        # 亮度权重
            score = circularity * area_score * brightness_score
            
            if score > best_score:
                best_score = score
                best_spot = (cx, cy, area, circularity, h, s, v)  # 添加HSV信息
        
        return best_spot, mask_combined
    
    def set_debug_mode(self, enable):
        """设置调试模式"""
        self.debug_mode = enable
        if not enable:
            # 关闭调试窗口
            debug_windows = ['Laser Mask 1', 'Laser Mask 2', 'Laser Mask 3', 'Laser Mask 4', 
                           'Overexposed Mask', 'Combined Laser Mask']
            for window in debug_windows:
                try:
                    cv2.destroyWindow(window)
                except:
                    pass
    
    def draw_laser_spot(self, frame, spot_info):
        """在图像上绘制检测到的激光光斑"""
        if spot_info is None:
            return
        
        if len(spot_info) == 7:  # 新格式包含HSV信息
            cx, cy, area, circularity, h, s, v = spot_info
            
            # 绘制光斑中心
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)  # 紫色实心圆
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), 2)   # 紫色圆环
            
            # 添加标签和详细信息
            cv2.putText(frame, f"Laser ({cx},{cy})", 
                       (cx + 15, cy - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # 显示光斑属性信息
            cv2.putText(frame, f"Area: {area:.0f}", 
                       (cx + 15, cy - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            cv2.putText(frame, f"Circ: {circularity:.2f}", 
                       (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # 显示HSV值
            cv2.putText(frame, f"HSV: ({h},{s},{v})", 
                       (cx + 15, cy + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        else:  # 兼容旧格式
            cx, cy, area, circularity = spot_info
            
            # 绘制光斑中心
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)  # 紫色实心圆
            cv2.circle(frame, (cx, cy), 10, (255, 0, 255), 2)   # 紫色圆环
            
            # 添加标签
            cv2.putText(frame, f"Laser ({cx},{cy})", 
                       (cx + 15, cy - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # 显示光斑信息
            cv2.putText(frame, f"Area: {area:.0f}", 
                       (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            cv2.putText(frame, f"Circ: {circularity:.2f}", 
                       (cx + 15, cy + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

class CircleTrajectoryController:
    """圆形轨迹控制器 - 控制激光沿圆形轨迹运动"""
    
    def __init__(self):
        # 轨迹参数
        self.circle_center = None       # 圆心坐标 (x, y)
        self.circle_radius_px = 0       # 圆半径（像素）
        self.current_angle = 0.0        # 当前角度（弧度）
        self.angle_step = 0.05          # 角度步长（弧度，约2.9度）
        self.is_tracking_circle = False # 是否正在进行圆形轨迹跟踪
        
        # 轨迹状态
        self.trajectory_started = False # 轨迹是否已开始
        self.trajectory_completed = False # 轨迹是否已完成
        self.start_time = None          # 开始时间
        self.total_revolutions = 1      # 总圈数
        self.current_revolution = 0     # 当前圈数
        
        # PID控制参数（专用于圆形轨迹跟踪）
        self.pid_kp = 0.8
        self.pid_ki = 0.0
        self.pid_kd = 0.15
        self.error_integral_x = 0
        self.error_integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        
        # 轨迹点历史（用于绘制轨迹）
        self.trajectory_points = []     # 目标轨迹点
        self.laser_trail = []           # 激光实际轨迹
        self.max_trail_length = 200     # 最大轨迹长度
        
    def set_circle_parameters(self, center, radius_px):
        """设置圆形参数"""
        self.circle_center = center
        self.circle_radius_px = radius_px
        print(f"设置圆形轨迹: 中心{center}, 半径{radius_px}px")
        
    def start_circle_tracking(self, total_revolutions=1):
        """开始圆形轨迹跟踪"""
        if self.circle_center is None or self.circle_radius_px <= 0:
            print("错误: 圆形参数未设置，无法开始轨迹跟踪")
            return False
            
        self.is_tracking_circle = True
        self.trajectory_started = True
        self.trajectory_completed = False
        self.current_angle = 0.0
        self.current_revolution = 0
        self.total_revolutions = total_revolutions
        self.start_time = time.time()
        
        # 重置PID控制
        self.error_integral_x = 0
        self.error_integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        
        # 清空轨迹记录
        self.trajectory_points.clear()
        self.laser_trail.clear()
        
        print(f"开始圆形轨迹跟踪: {total_revolutions}圈")
        return True
        
    def stop_circle_tracking(self):
        """停止圆形轨迹跟踪"""
        self.is_tracking_circle = False
        self.trajectory_started = False
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"圆形轨迹跟踪已停止，用时: {duration:.1f}秒")
        
    def get_target_position(self):
        """获取当前目标位置"""
        if not self.is_tracking_circle or self.circle_center is None:
            return None
            
        # 计算目标位置
        target_x = self.circle_center[0] + self.circle_radius_px * math.cos(self.current_angle)
        target_y = self.circle_center[1] + self.circle_radius_px * math.sin(self.current_angle)
        
        return (int(target_x), int(target_y))
        
    def update_angle(self):
        """更新角度，检查是否完成轨迹"""
        if not self.is_tracking_circle:
            return
            
        self.current_angle += self.angle_step
        
        # 检查是否完成一圈
        if self.current_angle >= 2 * math.pi:
            self.current_angle -= 2 * math.pi
            self.current_revolution += 1
            
            print(f"完成第 {self.current_revolution}/{self.total_revolutions} 圈")
            
            # 检查是否完成所有圈数
            if self.current_revolution >= self.total_revolutions:
                self.trajectory_completed = True
                print("圆形轨迹跟踪完成！")
                
    def calculate_control_command(self, laser_position, target_position):
        """计算控制指令（PID控制）"""
        if laser_position is None or target_position is None:
            return 0, 0
            
        # 计算误差
        error_x = target_position[0] - laser_position[0]
        error_y = target_position[1] - laser_position[1]
        
        # PID控制计算
        # 比例项
        p_x = self.pid_kp * error_x
        p_y = self.pid_kp * error_y
        
        # 积分项
        self.error_integral_x += error_x
        self.error_integral_y += error_y
        i_x = self.pid_ki * self.error_integral_x
        i_y = self.pid_ki * self.error_integral_y
        
        # 微分项
        d_x = self.pid_kd * (error_x - self.last_error_x)
        d_y = self.pid_kd * (error_y - self.last_error_y)
        
        # 总控制量
        control_x = p_x + i_x + d_x
        control_y = p_y + i_y + d_y
        
        # 更新上次误差
        self.last_error_x = error_x
        self.last_error_y = error_y
        
        # 限制控制量范围
        control_x = max(-100, min(100, control_x))
        control_y = max(-100, min(100, control_y))
        
        return int(control_x), int(control_y)
        
    def update_trajectory_records(self, target_pos, laser_pos):
        """更新轨迹记录"""
        if target_pos:
            self.trajectory_points.append(target_pos)
            if len(self.trajectory_points) > self.max_trail_length:
                self.trajectory_points.pop(0)
                
        if laser_pos:
            self.laser_trail.append(laser_pos)
            if len(self.laser_trail) > self.max_trail_length:
                self.laser_trail.pop(0)
                
    def draw_trajectory_info(self, frame):
        """绘制轨迹信息"""
        if not self.is_tracking_circle:
            return
            
        # 绘制圆形轨迹（目标轨迹）
        if self.circle_center and self.circle_radius_px > 0:
            cv2.circle(frame, self.circle_center, int(self.circle_radius_px), (0, 255, 0), 2)
            
        # 绘制目标位置
        target_pos = self.get_target_position()
        if target_pos:
            cv2.circle(frame, target_pos, 8, (0, 255, 0), -1)  # 绿色目标点
            cv2.putText(frame, "Target", (target_pos[0] + 10, target_pos[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
        # 绘制轨迹历史
        if len(self.trajectory_points) > 1:
            pts = np.array(self.trajectory_points, np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 1)  # 绿色目标轨迹
            
        if len(self.laser_trail) > 1:
            pts = np.array(self.laser_trail, np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 255), 1)  # 紫色激光轨迹
            
        # 显示状态信息
        info_text = f"Circle Tracking: Rev {self.current_revolution}/{self.total_revolutions}"
        cv2.putText(frame, info_text, (10, frame.shape[0] - 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                   
        angle_deg = math.degrees(self.current_angle)
        cv2.putText(frame, f"Angle: {angle_deg:.1f}°", (10, frame.shape[0] - 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                   
        if self.trajectory_completed:
            cv2.putText(frame, "TRAJECTORY COMPLETED!", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

class LaserCircleTrackingSystem:
    """激光圆形轨迹跟踪系统主类"""
    
    def __init__(self):
        # 初始化所有组件
        self.serial_controller = SerialController(config.SERIAL_PORT, config.SERIAL_BAUDRATE)
        self.hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        self.distance_calculator = SimpleDistanceCalculator()
        self.offset_calculator = DistanceOffsetCalculator()
        self.a4_detector = A4PaperDetector()
        self.display_manager = DisplayManager()
        self.laser_detector = LaserSpotDetector()  # 激光光斑检测器
        self.circle_controller = CircleTrajectoryController()  # 新增圆形轨迹控制器

        self.fix_gap = False
        
        # 摄像头
        self.cap = None
        
        # 校准点收集
        self.calibration_points = []  # 存储校准点 [distance_mm, offset_x, offset_y]
        self.max_calibration_points = 3  # 最大收集3个校准点
        
        # 激光追踪相关参数
        self.laser_tracking_enabled = True  # 是否启用激光追踪
        self.target_center = None           # 目标中心点
        self.laser_center = None            # 激光光斑中心点
        
        # 跟踪模式：'center' 或 'circle'
        self.tracking_mode = 'center'       # 默认为中心跟踪模式
        
        # PID控制参数（用于平滑控制云台 - 中心跟踪模式）
        self.pid_kp = 0.5
        self.pid_ki = 0.0
        self.pid_kd = 0.1
        self.error_integral_x = 0
        self.error_integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0

    def read_hmi_command_packet(self, timeout=1.0):
        """读取新格式的HMI指令数据包: AA + 指令内容 + A5 5A"""
        if not self.hmi or not self.hmi.is_connected():
            return None
            
        try:
            # 设置临时超时
            original_timeout = self.hmi.ser.timeout
            self.hmi.ser.timeout = timeout
            
            buffer = b''
            max_length = 1024  # 最大数据包长度
            start_found = False
            
            while len(buffer) < max_length:
                byte = self.hmi.ser.read(1)
                if not byte:
                    break  # 超时或无数据
                
                buffer += byte
                
                # 查找开始符 0xAA
                if not start_found:
                    if byte == b'\xAA':
                        start_found = True
                        buffer = b'\xAA'  # 重置缓冲区，只保留开始符
                    else:
                        buffer = b''  # 清空缓冲区，继续寻找开始符
                    continue
                
                # 已找到开始符，查找结束符 0xA5 0x5A
                if len(buffer) >= 3 and buffer[-2:] == b'\xA5\x5A':
                    # 找到完整的数据包
                    self.hmi.ser.timeout = original_timeout
                    return buffer
            
            # 恢复原始超时设置
            self.hmi.ser.timeout = original_timeout
            return None
            
        except Exception as e:
            print(f"读取HMI指令数据包错误: {e}")
            return None
    
    def parse_hmi_command_packet(self, packet):
        """解析新格式的HMI指令数据包"""
        if not packet or len(packet) < 4:
            return None
        
        try:
            # 验证数据包格式
            if packet[0] != 0xAA:
                return None
            
            if packet[-2:] != b'\xA5\x5A':
                return None
            
            # 提取指令内容（去除开始符和结束符）
            command_bytes = packet[1:-2]
            
            # 显示原始数据包用于调试
            hex_data = ' '.join([f'{b:02X}' for b in packet])
            print(f"解析数据包: {hex_data}")
            
            # 检查是否为二进制坐标格式 (长度为6: x(2字节) + 分隔符(1字节) + y(2字节) + 换行符(1字节))
            if len(command_bytes) == 6 and command_bytes[2] == 0x2C and command_bytes[5] == 0x0A:
                import struct
                try:
                    # 解析二进制坐标: 小端格式的有符号16位整数
                    x = struct.unpack('<h', command_bytes[0:2])[0]  # 有符号16位小端
                    y = struct.unpack('<h', command_bytes[3:5])[0]  # 有符号16位小端
                    print(f"解析二进制坐标: x={x}, y={y}")
                    return (x, y)
                except struct.error as e:
                    print(f"二进制坐标解析失败: {e}")
            
            # 尝试作为文本指令解析
            try:
                command = command_bytes.decode('utf-8', errors='ignore').strip()
                if command:
                    print(f"解析文本指令: '{command}'")
                    return command
            except UnicodeDecodeError:
                pass
            
            print("未知数据包格式")
            return None
                
        except Exception as e:
            print(f"解析HMI指令数据包错误: {e}")
            return None

    def set_fix_gap(self, fix_gap):
        """设置是否修正误差"""
        self.fix_gap = fix_gap
    
    def update_calibration_points(self):
        """更新配置文件中的校准点"""
        try:
            # 更新动态配置
            config.CALIBRATION_POINTS = self.calibration_points.copy()
            
            # 保存到文件
            config.save_to_file()
            
            print("=== 校准点更新完成 ===")
            for i, point in enumerate(self.calibration_points, 1):
                print(f"校准点{i}: 距离={point[0]}mm, 偏移=({point[1]}, {point[2]})")
            
            # 重新初始化偏移计算器以使用新的校准点
            self.offset_calculator = DistanceOffsetCalculator()
            print("偏移计算器已重新初始化")
            
            # 清空校准点列表，准备下一轮收集
            self.calibration_points = []
            print("校准点列表已清空，可以开始新一轮校准")
            
        except Exception as e:
            print(f"更新校准点失败: {e}")
        
    def initialize_camera(self, camera_index=0):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print("错误：无法打开视频流。")
            return False
        
        return True
    
    def calculate_laser_tracking_control(self, target_center, laser_center):
        """计算激光追踪的控制量（使用PID控制）"""
        if target_center is None or laser_center is None:
            return 0, 0
        
        # 计算误差（激光光斑相对目标中心的偏差）
        error_x = target_center[0] - laser_center[0]
        error_y = target_center[1] - laser_center[1]
        
        # PID控制计算
        # 比例项
        p_x = self.pid_kp * error_x
        p_y = self.pid_kp * error_y
        
        # 积分项
        self.error_integral_x += error_x
        self.error_integral_y += error_y
        i_x = self.pid_ki * self.error_integral_x
        i_y = self.pid_ki * self.error_integral_y
        
        # 微分项
        d_x = self.pid_kd * (error_x - self.last_error_x)
        d_y = self.pid_kd * (error_y - self.last_error_y)
        
        # 总控制量
        control_x = p_x + i_x + d_x
        control_y = p_y + i_y + d_y
        
        # 更新上次误差
        self.last_error_x = error_x
        self.last_error_y = error_y
        
        # 限制控制量范围
        control_x = max(-100, min(100, control_x))
        control_y = max(-100, min(100, control_y))
        
        return int(control_x), int(control_y)
    
    def extract_circle_parameters_from_warped(self, warped_image, M):
        """从变换后的图像中提取圆形参数，并转换到原图坐标系"""
        if warped_image is None or M is None:
            return None, None
            
        maxWidth, maxHeight = warped_image.shape[1], warped_image.shape[0]
        
        # 计算像素/厘米比例
        pixels_per_cm_w = maxWidth / config.PHYSICAL_WIDTH_CM
        pixels_per_cm_h = maxHeight / config.PHYSICAL_HEIGHT_CM
        pixels_per_cm = (pixels_per_cm_w + pixels_per_cm_h) / 2.0
        
        # 计算变换图像中的圆形参数
        radius_px = int(config.CIRCLE_RADIUS_CM * pixels_per_cm)
        center_warped = (maxWidth // 2, maxHeight // 2)
        
        # 将圆心转换到原图坐标系
        inv_M = np.linalg.inv(M)
        center_warped_homogeneous = np.array([center_warped[0], center_warped[1], 1], dtype=np.float32)
        original_center_homogeneous = inv_M.dot(center_warped_homogeneous)
        
        # 防止除零错误
        if abs(original_center_homogeneous[2]) < 1e-8:
            print("警告: 透视变换除数接近零")
            return None, None
        
        original_center = (
            int(original_center_homogeneous[0] / original_center_homogeneous[2]),
            int(original_center_homogeneous[1] / original_center_homogeneous[2])
        )
        
        # 计算原图坐标系中的半径（近似）
        # 通过变换一个圆周上的点来估算半径
        test_point_warped = (center_warped[0] + radius_px, center_warped[1])
        test_point_homogeneous = np.array([test_point_warped[0], test_point_warped[1], 1], dtype=np.float32)
        test_point_original_homogeneous = inv_M.dot(test_point_homogeneous)
        
        if abs(test_point_original_homogeneous[2]) < 1e-8:
            # 使用默认半径估算
            original_radius = int(radius_px * 0.8)  # 经验值
        else:
            test_point_original = (
                int(test_point_original_homogeneous[0] / test_point_original_homogeneous[2]),
                int(test_point_original_homogeneous[1] / test_point_original_homogeneous[2])
            )
            original_radius = int(np.sqrt(
                (test_point_original[0] - original_center[0])**2 + 
                (test_point_original[1] - original_center[1])**2
            ))
        
        return original_center, original_radius
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 检测A4纸目标
        detected_rect = self.a4_detector.detect_a4_paper(frame)
        
        # 检测激光光斑
        laser_spot, laser_mask = self.laser_detector.detect_laser_spot(frame)
        
        # 保存当前激光光斑信息用于调试
        self.current_laser_spot = laser_spot
        
        # 更新激光光斑位置
        if laser_spot is not None:
            self.laser_center = (laser_spot[0], laser_spot[1])
        else:
            self.laser_center = None
        
        if detected_rect is None:
            # 如果没有检测到目标，仍然显示激光光斑
            if laser_spot is not None:
                self.laser_detector.draw_laser_spot(frame, laser_spot)
            return frame, None, None, None
        
        # 创建透视变换图像
        warped_image, M, detected_width, warped_size = self.a4_detector.create_warped_image(frame, detected_rect)
        
        if warped_image is None:
            if laser_spot is not None:
                self.laser_detector.draw_laser_spot(frame, laser_spot)
            return frame, None, None, None
        
        # 在变换图像上绘制圆形并获取中心点（目标中心）
        center_x, center_y = self.a4_detector.draw_circle_on_warped(warped_image, frame, M)
        self.target_center = (center_x, center_y)
        
        # 提取圆形参数用于轨迹跟踪
        circle_center, circle_radius = self.extract_circle_parameters_from_warped(warped_image, M)
        if circle_center and circle_radius:
            self.circle_controller.set_circle_parameters(circle_center, circle_radius)
        
        # 距离测量
        distance_mm = self.distance_calculator.calculate_distance_from_width(detected_width, frame.shape[1])
        self.distance_calculator.update_distance_history(distance_mm)
        avg_distance = self.distance_calculator.get_averaged_distance()
        
        # 检查HMI指令（无论是否开启fix_gap都要检查退出指令）
        command = None
        packet = self.read_hmi_command_packet(timeout=0)

        if packet:
            command = self.parse_hmi_command_packet(packet)
            if command:
                # 检查是否为文本指令
                if isinstance(command, str):
                    print(f"收到HMI指令: '{command}'")
                    
                    if command.lower().strip() in ['q', 'quit', 'exit']:
                        print("收到退出命令，程序即将退出...")
                        return frame, None, None, "quit"
                    
                    # 激光追踪开关控制
                    elif command.lower().strip() == 'laser_on':
                        self.laser_tracking_enabled = True
                        print("激光追踪已启用")
                    elif command.lower().strip() == 'laser_off':
                        self.laser_tracking_enabled = False
                        print("激光追踪已禁用")
                    
                    # 新增：圆形轨迹跟踪控制
                    elif command.lower().strip() == 'circle_start':
                        if self.circle_controller.start_circle_tracking(total_revolutions=1):
                            self.tracking_mode = 'circle'
                            print("开始圆形轨迹跟踪模式")
                        else:
                            print("无法开始圆形轨迹跟踪：圆形参数未设置")
                    elif command.lower().strip() == 'circle_stop':
                        self.circle_controller.stop_circle_tracking()
                        self.tracking_mode = 'center'
                        print("停止圆形轨迹跟踪，切换回中心跟踪模式")
                    elif command.lower().strip() == 'mode_center':
                        self.tracking_mode = 'center'
                        print("切换到中心跟踪模式")
                    elif command.lower().strip() == 'mode_circle':
                        self.tracking_mode = 'circle'
                        print("切换到圆形跟踪模式")
                
                # 检查是否为二进制坐标
                elif isinstance(command, tuple):
                    x, y = command
                    print(f"收到二进制坐标: x={x}, y={y}")
                    # 直接更新全局变量
                    global dx_from_uart, dy_from_uart
                    dx_from_uart = x
                    dy_from_uart = y
        
        # 激光追踪控制逻辑
        if self.laser_tracking_enabled and self.target_center is not None:
            if self.laser_center is not None:
                if self.tracking_mode == 'center':
                    # 中心跟踪模式：激光光斑跟踪目标中心
                    control_x, control_y = self.calculate_laser_tracking_control(
                        self.target_center, self.laser_center
                    )
                    
                    # 发送中心跟踪控制指令到串口
                    center_control_cmd = f"CENTER:{control_x},{control_y}\n"
                    print(f"中心跟踪控制: {center_control_cmd.strip()}")
                    self.serial_controller.write(center_control_cmd)
                    
                elif self.tracking_mode == 'circle' and self.circle_controller.is_tracking_circle:
                    # 圆形轨迹跟踪模式
                    target_pos = self.circle_controller.get_target_position()
                    if target_pos:
                        control_x, control_y = self.circle_controller.calculate_control_command(
                            self.laser_center, target_pos
                        )
                        
                        # 发送圆形轨迹控制指令到串口
                        circle_control_cmd = f"CIRCLE:{control_x},{control_y}\n"
                        print(f"圆形轨迹控制: {circle_control_cmd.strip()}")
                        self.serial_controller.write(circle_control_cmd)
                        
                        # 更新角度和轨迹记录
                        self.circle_controller.update_angle()
                        self.circle_controller.update_trajectory_records(target_pos, self.laser_center)
                        
                        # 检查轨迹是否完成
                        if self.circle_controller.trajectory_completed:
                            self.circle_controller.stop_circle_tracking()
                            self.tracking_mode = 'center'
                            print("圆形轨迹完成，自动切换回中心跟踪模式")
            else:
                # 没有检测到激光光斑，发送搜索指令
                search_cmd = "LASER:SEARCH\n"
                self.serial_controller.write(search_cmd)
        
        if self.fix_gap:
            # debug 时，使用串口，手动修正间隙
            
            # 如果已经收到了文本指令，处理其他指令类型
            if command and isinstance(command, str):
                if command.lower().strip() == 'ok':
                    # 收集校准点数据
                    if distance_mm and len(self.calibration_points) < self.max_calibration_points:
                        # 将数据转换为整数并存储
                        calibration_point = [int(distance_mm), int(dx_from_uart), int(dy_from_uart)]
                        self.calibration_points.append(calibration_point)
                        print(f"收集校准点 {len(self.calibration_points)}/{self.max_calibration_points}: {calibration_point}")
                        
                        # 如果收集到3个校准点，更新配置文件
                        if len(self.calibration_points) == self.max_calibration_points:
                            self.update_calibration_points()
                    else:
                        if not distance_mm:
                            print("错误: 距离数据无效，无法收集校准点")
                        else:
                            print(f"已收集足够的校准点({self.max_calibration_points}个)")
                
                else:
                    # 尝试解析为文本坐标数据 "x,y"
                    try:
                        if ',' in command:
                            parts = command.strip().split(',')
                            if len(parts) == 2:
                                x = int(parts[0])
                                y = int(parts[1])
                                dx_from_uart = x
                                dy_from_uart = y
                                print(f"更新偏移量(文本): dx={dx_from_uart}, dy={dy_from_uart}")
                    except ValueError:
                        print(f"无法解析坐标数据: '{command}'")
            
            offset_x, offset_y = dx_from_uart, dy_from_uart
        else:
            # 计算屏幕中心偏移
            if avg_distance:
                offset_x, offset_y = self.offset_calculator.calculate_screen_center_offset(avg_distance)
            else:
                offset_x, offset_y = 0, 0
            
        # 计算动态屏幕中心
        screen_center_x = frame.shape[1] // 2 + offset_x
        screen_center_y = frame.shape[0] // 2 + offset_y
        
        # 计算偏移量并检查对齐
        dx, dy, alignment_status = self.a4_detector.calculate_offset_and_check_alignment(
            center_x, center_y, screen_center_x, screen_center_y, self.serial_controller
        )
        
        # 绘制所有信息
        self.display_manager.draw_detection_info(frame, detected_rect, center_x, center_y)
        self.display_manager.draw_screen_center(frame, screen_center_x, screen_center_y)
        
        # 绘制激光光斑
        if laser_spot is not None:
            self.laser_detector.draw_laser_spot(frame, laser_spot)
            
            # 绘制目标中心与激光光斑的连线（仅在中心跟踪模式）
            if self.target_center is not None and self.tracking_mode == 'center':
                cv2.line(frame, self.target_center, self.laser_center, (0, 255, 255), 2)
                
                # 显示追踪误差
                error_x = self.target_center[0] - self.laser_center[0]
                error_y = self.target_center[1] - self.laser_center[1]
                error_distance = np.sqrt(error_x**2 + error_y**2)
                
                cv2.putText(frame, f"Center Tracking Error: {error_distance:.1f}px", 
                           (10, frame.shape[0] - 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制圆形轨迹跟踪信息
        if self.tracking_mode == 'circle':
            self.circle_controller.draw_trajectory_info(frame)
        
        info_y = self.display_manager.draw_distance_info(frame, distance_mm, avg_distance)
        self.display_manager.draw_offset_info(frame, offset_x, offset_y, dx, dy, alignment_status, info_y)
        self.display_manager.draw_status_info(frame, self.serial_controller.is_connected())
        
        # 显示激光追踪状态和模式
        laser_status = "Laser Tracking: ON" if self.laser_tracking_enabled else "Laser Tracking: OFF"
        laser_color = (0, 255, 0) if self.laser_tracking_enabled else (0, 0, 255)
        cv2.putText(frame, laser_status, (10, frame.shape[0] - 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, laser_color, 2)
        
        # 显示跟踪模式
        mode_text = f"Mode: {self.tracking_mode.upper()}"
        mode_color = (255, 255, 0) if self.tracking_mode == 'circle' else (0, 255, 255)
        cv2.putText(frame, mode_text, (200, frame.shape[0] - 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        
        # 更新变换视图
        self.display_manager.update_transformed_view(warped_image)
        
        return frame, warped_image, distance_mm, avg_distance
    
    def wait_for_start_command(self):
        """等待HMI串口的start指令才开始运行"""
        print("等待HMI串口start指令...")
        print("期待指令: AA 73 74 61 72 74 0A A5 5A (start)")
        
        while True:
            try:
                # 持续读取HMI指令
                packet = self.read_hmi_command_packet(timeout=1.0)
                if packet:
                    command = self.parse_hmi_command_packet(packet)
                    if command and isinstance(command, str):
                        print(f"收到HMI指令: '{command}'")
                        
                        # 检查是否为start指令
                        if command.lower().strip() == 'start':
                            print("收到start指令，开始运行激光圆形跟踪系统...")
                            self.hmi.write(b't0.txt="running"\xff\xff\xff')
                            return True
                        
                        # 检查是否为退出指令
                        if command.lower().strip() in ['q', 'quit', 'exit']:
                            print("收到退出指令，程序退出")
                            return False
                
                # 短暂延时避免CPU占用过高
                import time
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                print("\n用户中断程序")
                return False
            except Exception as e:
                print(f"等待start指令时出错: {e}")
                import time
                time.sleep(0.1)

    def run(self):
        """运行主循环"""
        if not self.initialize_camera():
            return
        
        # 等待HMI串口的start指令
        if not self.wait_for_start_command():
            print("未收到start指令，程序退出")
            return
        
        print("=== 激光圆形轨迹跟踪系统 ===")
        print("距离计算参数:", config.CAMERA_PARAMS)
        print(f"距离范围: {config.MIN_DISTANCE_MM}-{config.MAX_DISTANCE_MM}mm")
        print("系统功能:")
        print("1. A4纸目标检测和跟踪")
        print("2. 蓝紫色激光光斑检测")
        print("3. 激光光斑中心跟踪模式")
        print("4. 激光光斑圆形轨迹跟踪模式（新增）")
        print("5. HMI指令接收和处理")
        
        if self.display_manager.display_enabled:
            print("\n操作:")
            print("- 'q': 退出程序")
            print("- 'h': 清除距离历史记录")
            print("- 'd': 显示当前距离和偏移信息")
            print("- 'l': 切换激光追踪模式")
            print("- 'm': 切换激光检测调试模式")
            print("- 'p': 打印当前激光光斑HSV信息")
            print("- 'c': 开始圆形轨迹跟踪")
            print("- 's': 停止圆形轨迹跟踪")
            print("- '1': 切换到中心跟踪模式")
            print("- '2': 切换到圆形跟踪模式")
        else:
            print("\n运行在无头模式 - 使用 Ctrl+C 退出程序")
        
        print("\nHMI指令:")
        print("- 'start': 开始系统")
        print("- 'laser_on/laser_off': 开启/关闭激光追踪")
        print("- 'circle_start': 开始圆形轨迹跟踪")
        print("- 'circle_stop': 停止圆形轨迹跟踪")
        print("- 'mode_center/mode_circle': 切换跟踪模式")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法接收帧（视频流结束？）。正在退出...")
                break
            
            # 处理帧
            processed_frame, warped_image, distance, avg_distance = self.process_frame(frame)
            
            # 检查是否收到串口退出命令
            if avg_distance == "quit":
                print("收到串口退出命令，程序退出")
                break
            
            # 显示结果
            self.display_manager.show_frames(processed_frame)
            
            # 处理键盘输入
            key_result = self.handle_keyboard_input()
            
            if key_result == 'quit':
                break
        
        self.cleanup()
    
    def handle_keyboard_input(self):
        """处理键盘输入（扩展版本，添加圆形轨迹跟踪控制）"""
        if not self.display_manager.display_enabled:
            # 无显示模式下，添加简单的延时避免CPU占用过高
            import time
            time.sleep(0.01)
            return None
            
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return 'quit'
        elif key == ord('h'):
            self.distance_calculator.clear_history()
            print("距离历史记录已清除")
        elif key == ord('d'):
            avg_distance = self.distance_calculator.get_averaged_distance()
            if avg_distance:
                offset_x, offset_y = self.offset_calculator.calculate_screen_center_offset(avg_distance)
                print(f"当前平均距离: {avg_distance:.1f}mm")
                print(f"计算的屏幕中心偏移: ({offset_x}, {offset_y})")
        elif key == ord('l'):
            # 切换激光追踪模式
            self.laser_tracking_enabled = not self.laser_tracking_enabled
            status = "启用" if self.laser_tracking_enabled else "禁用"
            print(f"激光追踪模式已{status}")
        elif key == ord('m'):
            # 切换激光检测调试模式
            self.laser_detector.set_debug_mode(not self.laser_detector.debug_mode)
            status = "启用" if self.laser_detector.debug_mode else "禁用"
            print(f"激光检测调试模式已{status}")
        elif key == ord('p'):
            # 打印当前检测到的激光光斑HSV信息
            if hasattr(self, 'current_laser_spot') and self.current_laser_spot:
                if len(self.current_laser_spot) == 7:
                    cx, cy, area, circularity, h, s, v = self.current_laser_spot
                    print(f"当前激光光斑HSV: H={h}, S={s}, V={v} at ({cx},{cy})")
                else:
                    print("当前激光光斑信息不包含HSV数据")
        elif key == ord('c'):
            # 开始圆形轨迹跟踪
            if self.circle_controller.start_circle_tracking(total_revolutions=1):
                self.tracking_mode = 'circle'
                print("开始圆形轨迹跟踪模式")
            else:
                print("无法开始圆形轨迹跟踪：请确保A4纸已被检测到")
        elif key == ord('s'):
            # 停止圆形轨迹跟踪
            self.circle_controller.stop_circle_tracking()
            self.tracking_mode = 'center'
            print("停止圆形轨迹跟踪，切换回中心跟踪模式")
        elif key == ord('1'):
            # 切换到中心跟踪模式
            self.tracking_mode = 'center'
            print("切换到中心跟踪模式")
        elif key == ord('2'):
            # 切换到圆形跟踪模式
            self.tracking_mode = 'circle'
            print("切换到圆形跟踪模式")
        
        return None
    
    def cleanup(self):
        """清理资源"""
        # 停止圆形轨迹跟踪
        if self.circle_controller.is_tracking_circle:
            self.circle_controller.stop_circle_tracking()
        
        # 释放摄像头
        if self.cap:
            self.cap.release()
        
        # 清理显示
        self.display_manager.cleanup()
        
        # 关闭串口
        self.serial_controller.close()
        
        # 显示最终统计信息
        self.show_final_statistics()
    
    def show_final_statistics(self):
        """显示最终统计信息"""
        mean_distance, std_distance = self.distance_calculator.get_distance_statistics()
        
        if mean_distance is not None:
            print("\n=== 最终统计信息 ===")
            print(f"平均距离: {mean_distance:.1f} ± {std_distance:.1f} mm")
            
            # 显示最终的屏幕中心偏移
            offset_x, offset_y = self.offset_calculator.calculate_screen_center_offset(mean_distance)
            print(f"最终屏幕中心偏移: ({offset_x}, {offset_y})")
            print(f"激光追踪状态: {'启用' if self.laser_tracking_enabled else '禁用'}")
            print(f"最终跟踪模式: {self.tracking_mode}")
            
            # 圆形轨迹统计
            if hasattr(self.circle_controller, 'current_revolution'):
                print(f"圆形轨迹完成圈数: {self.circle_controller.current_revolution}")

# 参数调整接口类
class SystemParameterManager:
    """系统参数管理器 - 提供运行时参数调整接口"""
    
    @staticmethod
    def update_detection_parameters(mean_inner_val=None, mean_border_val=None):
        """更新检测参数"""
        if mean_inner_val is not None:
            config.MEAN_INNER_VAL = mean_inner_val
        if mean_border_val is not None:
            config.MEAN_BORDER_VAL = mean_border_val
    
    @staticmethod
    def update_camera_parameters(focal_length=None, sensor_width=None, sensor_height=None, calibration_factor=None):
        """更新摄像头参数"""
        if focal_length is not None:
            config.CAMERA_PARAMS["focal_length_mm"] = focal_length
        if sensor_width is not None:
            config.CAMERA_PARAMS["sensor_width_mm"] = sensor_width
        if sensor_height is not None:
            config.CAMERA_PARAMS["sensor_height_mm"] = sensor_height
        if calibration_factor is not None:
            config.CAMERA_PARAMS["calibration_factor"] = calibration_factor
    
    @staticmethod
    def update_tracking_parameters(alignment_threshold=None, track_count_threshold=None):
        """更新跟踪参数"""
        if alignment_threshold is not None:
            config.ALIGNMENT_THRESHOLD = alignment_threshold
        if track_count_threshold is not None:
            config.TRACK_COUNT_THRESHOLD = track_count_threshold
    
    @staticmethod
    def update_distance_range(min_distance=None, max_distance=None):
        """更新距离范围"""
        if min_distance is not None:
            config.MIN_DISTANCE_MM = min_distance
        if max_distance is not None:
            config.MAX_DISTANCE_MM = max_distance
    
    @staticmethod
    def update_circle_tracking_parameters(angle_step=None, pid_kp=None, pid_ki=None, pid_kd=None, total_revolutions=None):
        """更新圆形轨迹跟踪参数"""
        # 这里可以扩展配置文件来存储圆形轨迹跟踪相关参数
        pass
    
    @staticmethod
    def update_laser_parameters(min_area=None, max_area=None, min_circularity=None, 
                               pid_kp=None, pid_ki=None, pid_kd=None):
        """更新激光检测和控制参数"""
        # 这里可以扩展配置文件来存储激光相关参数
        pass
    
    @staticmethod
    def get_current_parameters():
        """获取当前所有参数"""
        return {
            "detection": {
                "mean_inner_val": config.MEAN_INNER_VAL,
                "mean_border_val": config.MEAN_BORDER_VAL
            },
            "camera": config.CAMERA_PARAMS,
            "tracking": {
                "alignment_threshold": config.ALIGNMENT_THRESHOLD,
                "track_count_threshold": config.TRACK_COUNT_THRESHOLD
            },
            "distance_range": {
                "min_distance_mm": config.MIN_DISTANCE_MM,
                "max_distance_mm": config.MAX_DISTANCE_MM
            },
            "calibration_points": config.CALIBRATION_POINTS
        }

def main():
    """主函数"""
    # 创建激光圆形轨迹跟踪系统实例
    laser_circle_tracking_system = LaserCircleTrackingSystem()
    
    # 可选：在运行前调整参数
    # SystemParameterManager.update_detection_parameters(mean_inner_val=105, mean_border_val=75)
    # SystemParameterManager.update_tracking_parameters(alignment_threshold=8)
    
    # 可选：调整激光追踪参数
    # laser_circle_tracking_system.pid_kp = 0.8  # 增加比例控制增益
    # laser_circle_tracking_system.laser_detector.min_area = 5  # 调整最小检测面积
    
    # 可选：调整圆形轨迹跟踪参数
    # laser_circle_tracking_system.circle_controller.angle_step = 0.03  # 调整角度步长（更精细）
    # laser_circle_tracking_system.circle_controller.pid_kp = 1.0  # 调整圆形轨迹PID参数
    
    print("=== 激光圆形轨迹跟踪系统启动 ===")
    print("系统功能:")
    print("1. A4纸目标检测和跟踪")
    print("2. 蓝紫色激光光斑检测")
    print("3. 激光光斑自动追踪目标中心（中心模式）")
    print("4. 激光光斑沿圆形轨迹运动（圆形模式）- 新增功能")
    print("5. 云台控制指令发送")
    print("6. HMI指令接收和处理")
    print("7. 实时轨迹可视化")
    
    # 运行系统
    try:
        laser_circle_tracking_system.run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        laser_circle_tracking_system.cleanup()
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
        laser_circle_tracking_system.cleanup()

if __name__ == '__main__':
    main()