#!/usr/bin/env python3
"""
激光光斑跟踪系统
基于basic2.py，添加蓝紫色激光光斑检测和跟踪功能
主文件 - 包含主要的控制逻辑和激光光斑检测
"""

import cv2
import numpy as np

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
            # # 测试数据1: H(0-6) S(0-178) V(238-255) - 高亮度蓝紫色
            # ([0, 0, 238], [6, 178, 255]),
            # # 测试数据2: H(2-161) S(0-34) V(155-255) - 低饱和度高亮度
            # ([2, 0, 155], [161, 34, 255]),
            # # 测试数据3: H(12-122) S(0-241) V(215-255) - 宽色相范围
            # ([12, 0, 215], [122, 241, 255]),
            # # 测试数据4: H(15-121) S(0-230) V(192-255) - 中等亮度范围
            # ([15, 0, 192], [121, 230, 255]),
            ([0, 0, 255], [0, 0, 255]),
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
        
        # 位置稳定性检查参数
        self.last_position = None           # 上一帧的激光光斑位置 (x, y)
        self.max_position_jump = 80         # 最大允许的位置跳跃像素数
        self.position_history = []          # 位置历史记录
        self.max_history_length = 5        # 最大历史记录长度
        self.stability_threshold = 3        # 稳定性阈值：连续多少帧才认为位置稳定
        self.lost_frames_count = 0          # 连续丢失帧数
        self.max_lost_frames = 10           # 最大允许连续丢失帧数
        
        # 预测和平滑参数
        self.use_prediction = True          # 是否使用位置预测
        self.smoothing_factor = 0.7         # 位置平滑因子 (0-1, 越大越平滑)
        
    def detect_laser_spot(self, frame):
        """检测激光光斑位置（使用多组HSV范围和稳定性检查）"""
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
        
        # 收集所有有效的候选光斑
        candidate_spots = []
        
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
            
            candidate_spots.append((cx, cy, area, circularity, h, s, v, score))
        
        # 按评分排序，获取最佳候选
        candidate_spots.sort(key=lambda x: x[7], reverse=True)
        
        # 使用稳定性检查选择最终位置
        stable_position = self.validate_position_stability(candidate_spots)
        
        if stable_position is None:
            return None, mask_combined
        
        # 查找与稳定位置对应的光斑信息
        best_spot_info = None
        min_distance = float('inf')
        
        for spot in candidate_spots:
            spot_pos = (spot[0], spot[1])
            distance = self.calculate_distance(spot_pos, stable_position)
            if distance < min_distance:
                min_distance = distance
                best_spot_info = spot[:7]  # 排除score
        
        # 如果没找到对应的光斑信息（使用预测位置的情况），创建一个基本信息
        if best_spot_info is None:
            # 使用稳定位置创建基本信息
            h, s, v = hsv[stable_position[1], stable_position[0]]
            best_spot_info = (stable_position[0], stable_position[1], 0, 0, h, s, v)
        
        return best_spot_info, mask_combined
    
    def calculate_distance(self, pos1, pos2):
        """计算两点之间的欧几里得距离"""
        if pos1 is None or pos2 is None:
            return float('inf')
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def predict_next_position(self):
        """基于历史位置预测下一个可能的位置"""
        if len(self.position_history) < 2:
            return self.last_position
        
        # 简单线性预测：基于最近两个位置的速度向量
        if len(self.position_history) >= 2:
            pos1 = self.position_history[-2]
            pos2 = self.position_history[-1]
            
            # 计算速度向量
            vx = pos2[0] - pos1[0]
            vy = pos2[1] - pos1[1]
            
            # 预测下一个位置
            predicted_x = pos2[0] + vx
            predicted_y = pos2[1] + vy
            
            return (int(predicted_x), int(predicted_y))
        
        return self.last_position
    
    def update_position_history(self, position):
        """更新位置历史记录"""
        if position is not None:
            self.position_history.append(position)
            if len(self.position_history) > self.max_history_length:
                self.position_history.pop(0)
    
    def smooth_position(self, new_position, last_position):
        """位置平滑处理"""
        if last_position is None:
            return new_position
        
        # 加权平均平滑
        smooth_x = int(self.smoothing_factor * last_position[0] + 
                      (1 - self.smoothing_factor) * new_position[0])
        smooth_y = int(self.smoothing_factor * last_position[1] + 
                      (1 - self.smoothing_factor) * new_position[1])
        
        return (smooth_x, smooth_y)
    
    def validate_position_stability(self, detected_spots):
        """验证位置稳定性并选择最佳候选位置"""
        if not detected_spots:
            # 没有检测到任何光斑
            self.lost_frames_count += 1
            
            if self.lost_frames_count > self.max_lost_frames:
                # 丢失太多帧，重置跟踪
                self.reset_tracking()
                return None
            
            # 使用预测位置或上一帧位置
            if self.use_prediction and len(self.position_history) >= 2:
                predicted_pos = self.predict_next_position()
                print(f"激光光斑丢失，使用预测位置: {predicted_pos}")
                return predicted_pos
            elif self.last_position:
                print(f"激光光斑丢失，使用上一帧位置: {self.last_position}")
                return self.last_position
            else:
                return None
        
        # 有检测结果，重置丢失计数
        self.lost_frames_count = 0
        
        if self.last_position is None:
            # 第一次检测，直接使用最佳候选
            best_spot = detected_spots[0]
            position = (best_spot[0], best_spot[1])
            self.last_position = position
            self.update_position_history(position)
            return position
        
        # 寻找与上一帧位置最接近的候选
        best_candidate = None
        min_distance = float('inf')
        
        for spot in detected_spots:
            spot_pos = (spot[0], spot[1])
            distance = self.calculate_distance(spot_pos, self.last_position)
            
            if distance < min_distance:
                min_distance = distance
                best_candidate = spot
        
        if best_candidate is None:
            return self.last_position
        
        candidate_pos = (best_candidate[0], best_candidate[1])
        
        # 检查位置跳跃是否过大
        if min_distance > self.max_position_jump:
            print(f"检测到位置跳跃过大: {min_distance:.1f}px > {self.max_position_jump}px")
            
            # 如果有多个候选，尝试其他候选
            if len(detected_spots) > 1:
                for spot in detected_spots[1:]:
                    spot_pos = (spot[0], spot[1])
                    distance = self.calculate_distance(spot_pos, self.last_position)
                    if distance <= self.max_position_jump:
                        candidate_pos = spot_pos
                        min_distance = distance
                        best_candidate = spot
                        print(f"使用备选候选，距离: {distance:.1f}px")
                        break
                else:
                    # 所有候选都跳跃过大，使用预测或保持上一帧位置
                    if self.use_prediction and len(self.position_history) >= 2:
                        predicted_pos = self.predict_next_position()
                        predicted_distance = self.calculate_distance(predicted_pos, self.last_position)
                        if predicted_distance <= self.max_position_jump * 2:  # 放宽预测位置的限制
                            print(f"使用预测位置，距离: {predicted_distance:.1f}px")
                            return predicted_pos
                    
                    print("保持上一帧位置")
                    return self.last_position
        
        # 应用位置平滑
        smoothed_pos = self.smooth_position(candidate_pos, self.last_position)
        
        # 更新状态
        self.last_position = smoothed_pos
        self.update_position_history(smoothed_pos)
        
        return smoothed_pos
    
    def reset_tracking(self):
        """重置跟踪状态"""
        print("重置激光光斑跟踪状态")
        self.last_position = None
        self.position_history.clear()
        self.lost_frames_count = 0
    
    def get_stability_info(self):
        """获取稳定性相关信息"""
        return {
            'last_position': self.last_position,
            'position_history_length': len(self.position_history),
            'lost_frames_count': self.lost_frames_count,
            'max_position_jump': self.max_position_jump,
            'smoothing_factor': self.smoothing_factor
        }
    
    def set_stability_parameters(self, max_jump=None, smoothing_factor=None, 
                                max_lost_frames=None, use_prediction=None):
        """设置稳定性参数"""
        if max_jump is not None:
            self.max_position_jump = max_jump
        if smoothing_factor is not None:
            self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        if max_lost_frames is not None:
            self.max_lost_frames = max_lost_frames
        if use_prediction is not None:
            self.use_prediction = use_prediction
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
            
            # 绘制位置历史轨迹
            if len(self.position_history) > 1:
                points = np.array(self.position_history, dtype=np.int32)
                cv2.polylines(frame, [points], False, (128, 0, 128), 2)
            
            # 添加标签和详细信息
            cv2.putText(frame, f"Laser ({cx},{cy})", 
                       (cx + 15, cy - 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # 显示光斑属性信息
            if area > 0 and circularity > 0:  # 真实检测的光斑
                cv2.putText(frame, f"Area: {area:.0f}", 
                           (cx + 15, cy - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                
                cv2.putText(frame, f"Circ: {circularity:.2f}", 
                           (cx + 15, cy - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            else:  # 预测或保持的位置
                cv2.putText(frame, "Predicted/Hold", 
                           (cx + 15, cy - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # 显示HSV值
            cv2.putText(frame, f"HSV: ({h},{s},{v})", 
                       (cx + 15, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # 显示稳定性信息
            if self.lost_frames_count > 0:
                cv2.putText(frame, f"Lost: {self.lost_frames_count}", 
                           (cx + 15, cy + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 显示位置历史长度
            cv2.putText(frame, f"History: {len(self.position_history)}", 
                       (cx + 15, cy + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
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

class LaserTrackingSystem:
    """激光光斑跟踪系统主类"""
    
    def __init__(self):
        # 初始化所有组件
        self.serial_controller = SerialController(config.SERIAL_PORT, config.SERIAL_BAUDRATE)
        self.hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        self.distance_calculator = SimpleDistanceCalculator()
        self.offset_calculator = DistanceOffsetCalculator()
        self.a4_detector = A4PaperDetector()
        self.display_manager = DisplayManager()
        self.laser_detector = LaserSpotDetector()  # 新增激光光斑检测器

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
        
        # PID控制参数（用于平滑控制云台）
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
                    
                    # 新增：激光追踪开关控制
                    elif command.lower().strip() == 'laser_on':
                        self.laser_tracking_enabled = True
                        print("激光追踪已启用")
                    elif command.lower().strip() == 'laser_off':
                        self.laser_tracking_enabled = False
                        print("激光追踪已禁用")
                
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
                # 计算激光追踪控制量
                control_x, control_y = self.calculate_laser_tracking_control(
                    self.target_center, self.laser_center
                )
                
                # 发送激光追踪控制指令到串口
                laser_control_cmd = f"LASER:{control_x},{control_y}\n"
                print(f"激光追踪控制: {laser_control_cmd.strip()}")
                self.serial_controller.write(laser_control_cmd)
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
            
            # 绘制目标中心与激光光斑的连线
            if self.target_center is not None:
                cv2.line(frame, self.target_center, self.laser_center, (0, 255, 255), 2)
                
                # 显示追踪误差
                error_x = self.target_center[0] - self.laser_center[0]
                error_y = self.target_center[1] - self.laser_center[1]
                error_distance = np.sqrt(error_x**2 + error_y**2)
                
                cv2.putText(frame, f"Tracking Error: {error_distance:.1f}px", 
                           (10, frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        info_y = self.display_manager.draw_distance_info(frame, distance_mm, avg_distance)
        self.display_manager.draw_offset_info(frame, offset_x, offset_y, dx, dy, alignment_status, info_y)
        self.display_manager.draw_status_info(frame, self.serial_controller.is_connected())
        
        # 显示激光追踪状态和稳定性信息
        laser_status = "Laser Tracking: ON" if self.laser_tracking_enabled else "Laser Tracking: OFF"
        laser_color = (0, 255, 0) if self.laser_tracking_enabled else (0, 0, 255)
        cv2.putText(frame, laser_status, (10, frame.shape[0] - 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, laser_color, 2)
        
        # 显示稳定性统计信息
        stability_info = self.laser_detector.get_stability_info()
        cv2.putText(frame, f"Jump Threshold: {stability_info['max_position_jump']}px", 
                   (10, frame.shape[0] - 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(frame, f"Smoothing: {stability_info['smoothing_factor']:.1f}", 
                   (200, frame.shape[0] - 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if stability_info['lost_frames_count'] > 0:
            cv2.putText(frame, f"Lost Frames: {stability_info['lost_frames_count']}", 
                       (320, frame.shape[0] - 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
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
                            print("收到start指令，开始运行激光跟踪系统...")
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
        
        print("=== 激光光斑跟踪系统 ===")
        print("距离计算参数:", config.CAMERA_PARAMS)
        print(f"距离范围: {config.MIN_DISTANCE_MM}-{config.MAX_DISTANCE_MM}mm")
        print("新功能: 蓝紫色激光光斑检测和跟踪")
        
        if self.display_manager.display_enabled:
            print("\n操作:")
            print("- 'q': 退出程序")
            print("- 'h': 清除距离历史记录")
            print("- 'd': 显示当前距离和偏移信息")
            print("- 'l': 切换激光追踪模式")
            print("- 'm': 切换激光检测调试模式")
            print("- 'p': 打印当前激光光斑HSV和稳定性信息")
            print("- 'r': 重置激光光斑跟踪状态")
            print("- 's': 显示稳定性参数")
            print("- '1'/'2': 减少/增加最大位置跳跃阈值")
            print("- '3'/'4': 减少/增加位置平滑因子")
            print("- '5': 切换位置预测模式")
        else:
            print("\n运行在无头模式 - 使用 Ctrl+C 退出程序")
        
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
        """处理键盘输入（扩展版本，添加激光追踪控制）"""
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
            
            # 打印稳定性信息
            stability_info = self.laser_detector.get_stability_info()
            print("稳定性信息:")
            for key, value in stability_info.items():
                print(f"  {key}: {value}")
        elif key == ord('r'):
            # 重置激光光斑跟踪
            self.laser_detector.reset_tracking()
            print("已重置激光光斑跟踪状态")
        elif key == ord('s'):
            # 调整稳定性参数
            print("当前稳定性参数:")
            print(f"  最大位置跳跃: {self.laser_detector.max_position_jump}px")
            print(f"  平滑因子: {self.laser_detector.smoothing_factor}")
            print(f"  最大丢失帧数: {self.laser_detector.max_lost_frames}")
            print(f"  使用预测: {self.laser_detector.use_prediction}")
        elif key == ord('1'):
            # 降低最大位置跳跃阈值
            self.laser_detector.max_position_jump = max(10, self.laser_detector.max_position_jump - 10)
            print(f"最大位置跳跃阈值调整为: {self.laser_detector.max_position_jump}px")
        elif key == ord('2'):
            # 增加最大位置跳跃阈值
            self.laser_detector.max_position_jump = min(200, self.laser_detector.max_position_jump + 10)
            print(f"最大位置跳跃阈值调整为: {self.laser_detector.max_position_jump}px")
        elif key == ord('3'):
            # 降低平滑因子
            self.laser_detector.smoothing_factor = max(0.1, self.laser_detector.smoothing_factor - 0.1)
            print(f"平滑因子调整为: {self.laser_detector.smoothing_factor:.1f}")
        elif key == ord('4'):
            # 增加平滑因子
            self.laser_detector.smoothing_factor = min(0.9, self.laser_detector.smoothing_factor + 0.1)
            print(f"平滑因子调整为: {self.laser_detector.smoothing_factor:.1f}")
        elif key == ord('5'):
            # 切换预测模式
            self.laser_detector.use_prediction = not self.laser_detector.use_prediction
            status = "启用" if self.laser_detector.use_prediction else "禁用"
            print(f"位置预测模式已{status}")
        
        return None
    
    def cleanup(self):
        """清理资源"""
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
    def update_laser_parameters(min_area=None, max_area=None, min_circularity=None, 
                               pid_kp=None, pid_ki=None, pid_kd=None,
                               max_position_jump=None, smoothing_factor=None, 
                               max_lost_frames=None, use_prediction=None):
        """更新激光检测和控制参数"""
        # 这里可以扩展配置文件来存储激光相关参数
        print("激光参数更新功能待实现")
        if max_position_jump is not None:
            print(f"最大位置跳跃: {max_position_jump}")
        if smoothing_factor is not None:
            print(f"平滑因子: {smoothing_factor}")
        if max_lost_frames is not None:
            print(f"最大丢失帧数: {max_lost_frames}")
        if use_prediction is not None:
            print(f"使用预测: {use_prediction}")
    
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
    # 创建激光跟踪系统实例
    laser_tracking_system = LaserTrackingSystem()
    
    # 可选：在运行前调整参数
    # SystemParameterManager.update_detection_parameters(mean_inner_val=105, mean_border_val=75)
    # SystemParameterManager.update_tracking_parameters(alignment_threshold=8)
    
    # 可选：调整激光追踪参数
    # laser_tracking_system.pid_kp = 0.8  # 增加比例控制增益
    # laser_tracking_system.laser_detector.min_area = 5  # 调整最小检测面积
    # laser_tracking_system.laser_detector.set_stability_parameters(
    #     max_jump=30,           # 设置最大位置跳跃为30像素
    #     smoothing_factor=0.8,  # 设置平滑因子为0.8
    #     max_lost_frames=15,    # 设置最大丢失帧数为15
    #     use_prediction=True    # 启用位置预测
    # )
    
    print("=== 激光光斑跟踪系统启动 ===")
    print("系统功能:")
    print("1. A4纸目标检测和跟踪")
    print("2. 蓝紫色激光光斑检测")
    print("3. 激光光斑稳定性跟踪和位置预测")
    print("4. 激光光斑自动追踪目标中心")
    print("5. 云台控制指令发送")
    print("6. HMI指令接收和处理")
    print("7. 实时参数调节和调试功能")
    
    # 运行系统
    try:
        laser_tracking_system.run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        laser_tracking_system.cleanup()
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
        laser_tracking_system.cleanup()

if __name__ == '__main__':
    main()