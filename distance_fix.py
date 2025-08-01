#!/usr/bin/env python3
"""
模块化A4纸跟踪系统 - 集成distance校准逻辑
主文件 - 包含主要的控制逻辑和可调参数接口
"""

import cv2
import numpy as np
import json
import os

# 导入模块化组件
from dynamic_config import config
from serial_controller import SerialController
from distance_offset_calculator import DistanceOffsetCalculator
from distance_calculator import SimpleDistanceCalculator
from a4_detector import A4PaperDetector
from display_manager import DisplayManager

dx_from_uart = 0
dy_from_uart = 0  # 用于从串口接收偏移量

class CalibratedDistanceCalculator:
    """集成distance校准逻辑的距离计算器"""
    
    def __init__(self, max_history=None):
        self.distance_history = []
        self.max_history = max_history or config.MAX_DISTANCE_HISTORY
        
        # 校准参数 - 使用config中的参数作为初始值
        self.camera_params = config.CAMERA_PARAMS.copy()
        
        # 校准相关
        self.calibration_mode = False
        self.calibration_data = []  # 存储校准数据 [(pixel_width, known_distance), ...]
        self.max_calibration_points = 3  # 最大校准点数
        
    def calculate_distance_from_width(self, pixel_width, frame_width):
        """基于宽度计算距离（使用FOV方法）"""
        import math
        
        focal_length = self.camera_params["focal_length_mm"]
        sensor_width = self.camera_params["sensor_width_mm"]
        calibration_factor = self.camera_params["calibration_factor"]
        
        # 计算水平视场角
        fov_horizontal_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        
        # 计算在1米距离处每像素对应的实际距离
        mm_per_pixel_at_1m = (1000 * math.tan(fov_horizontal_rad / 2) * 2) / frame_width
        
        # 计算距离
        if pixel_width > 0 and mm_per_pixel_at_1m > 0:
            distance_mm = (config.A4_WIDTH_MM * 1000) / (pixel_width * mm_per_pixel_at_1m)
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
    
    def add_calibration_point(self, pixel_width, known_distance):
        """添加校准点"""
        if len(self.calibration_data) < self.max_calibration_points:
            self.calibration_data.append((pixel_width, known_distance))
            print(f"添加校准点 {len(self.calibration_data)}/{self.max_calibration_points}: 像素宽度={pixel_width}, 已知距离={known_distance}mm")
            return True
        return False
    
    def perform_calibration(self):
        """执行校准计算"""
        if len(self.calibration_data) < 2:
            print("校准点不足，至少需要2个点")
            return False
        
        print("开始执行校准计算...")
        
        # 提取数据
        pixel_widths = [point[0] for point in self.calibration_data]
        known_distances = [point[1] for point in self.calibration_data]
        
        # 使用最简单的校准方法：计算平均校准因子
        calibration_factors = []
        
        for pixel_width, known_distance in self.calibration_data:
            # 使用当前参数计算理论距离
            theoretical_distance = self._calculate_theoretical_distance(pixel_width, config.cap.get(cv2.CAP_PROP_FRAME_WIDTH) if hasattr(config, 'cap') else 640)
            
            if theoretical_distance and theoretical_distance > 0:
                factor = known_distance / theoretical_distance
                calibration_factors.append(factor)
                print(f"像素宽度{pixel_width}: 理论距离={theoretical_distance:.1f}mm, 实际距离={known_distance}mm, 因子={factor:.4f}")
        
        if calibration_factors:
            # 使用平均校准因子
            new_calibration_factor = np.mean(calibration_factors)
            self.camera_params["calibration_factor"] = new_calibration_factor
            
            print(f"校准完成！新的校准因子: {new_calibration_factor:.4f}")
            
            # 更新config中的参数
            config.CAMERA_PARAMS = self.camera_params.copy()
            
            # 保存到配置文件
            config.save_to_file()
            
            return True
        
        print("校准失败：无法计算有效的校准因子")
        return False
    
    def _calculate_theoretical_distance(self, pixel_width, frame_width):
        """计算理论距离（不使用校准因子）"""
        import math
        
        focal_length = self.camera_params["focal_length_mm"]
        sensor_width = self.camera_params["sensor_width_mm"]
        
        # 计算水平视场角
        fov_horizontal_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        
        # 计算在1米距离处每像素对应的实际距离
        mm_per_pixel_at_1m = (1000 * math.tan(fov_horizontal_rad / 2) * 2) / frame_width
        
        # 计算距离（不使用校准因子）
        if pixel_width > 0 and mm_per_pixel_at_1m > 0:
            distance_mm = (config.A4_WIDTH_MM * 1000) / (pixel_width * mm_per_pixel_at_1m)
            return distance_mm
        return None
    
    def reset_calibration(self):
        """重置校准数据"""
        self.calibration_data.clear()
        print("校准数据已重置")
    
    def enter_calibration_mode(self):
        """进入校准模式"""
        self.calibration_mode = True
        self.reset_calibration()
        print("=== 进入校准模式 ===")
        print("请将A4纸放置在以下距离位置并发送对应的串口数据：")
        print("- 500mm处发送: 50")
        print("- 1000mm处发送: 100") 
        print("- 1500mm处发送: 150")
        print("收到3个校准点后将自动计算校准参数")
    
    def exit_calibration_mode(self):
        """退出校准模式"""
        self.calibration_mode = False
        print("退出校准模式")

class A4TrackingSystem:
    """A4纸跟踪系统主类"""
    
    def __init__(self):
        # 初始化所有组件，使用校准版本的距离计算器
        self.serial_controller = SerialController(config.SERIAL_PORT, config.SERIAL_BAUDRATE)
        self.hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        self.distance_calculator = CalibratedDistanceCalculator()
        self.a4_detector = A4PaperDetector()
        self.display_manager = DisplayManager()
        
        # 摄像头
        self.cap = None
        
        # 自动进入校准模式
        self.distance_calculator.enter_calibration_mode()

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
        """设置是否修正误差（已移除相关功能）"""
        pass
    
    def update_calibration_points(self):
        """更新配置文件中的校准点（已移除相关功能）"""
        pass
        
    def initialize_camera(self, camera_index=0):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print("错误：无法打开视频流。")
            return False
        
        # 保存摄像头引用到config中（用于校准计算）
        config.cap = self.cap
        
        return True
    
    def process_calibration_command(self, command, pixel_width):
        """处理校准相关的命令"""
        if not self.distance_calculator.calibration_mode:
            return
        
        # 校准命令映射：串口数据 -> 实际距离
        calibration_mapping = {
            50: 500,    # 串口数据50 对应 500mm
            100: 1000,  # 串口数据100 对应 1000mm  
            150: 1500   # 串口数据150 对应 1500mm
        }
        
        try:
            # 尝试解析为数字
            uart_value = int(command)
            
            if uart_value in calibration_mapping:
                known_distance = calibration_mapping[uart_value]
                
                if pixel_width > 0:
                    # 添加校准点
                    success = self.distance_calculator.add_calibration_point(pixel_width, known_distance)
                    
                    if success and len(self.distance_calculator.calibration_data) == self.distance_calculator.max_calibration_points:
                        # 收集到足够的校准点，执行校准
                        if self.distance_calculator.perform_calibration():
                            print("=== 距离校准完成 ===")
                            self.distance_calculator.exit_calibration_mode()
                        else:
                            print("校准失败，请重新校准")
                            self.distance_calculator.reset_calibration()
                else:
                    print(f"错误：像素宽度无效({pixel_width})，无法添加校准点")
            else:
                print(f"未知的校准命令: {uart_value}")
                
        except ValueError:
            # 不是数字，可能是其他命令
            pass
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 检测A4纸
        detected_rect = self.a4_detector.detect_a4_paper(frame)
        
        if detected_rect is None:
            return frame, None, None, None, 0
        
        # 创建透视变换图像
        warped_image, M, detected_width, warped_size = self.a4_detector.create_warped_image(frame, detected_rect)
        
        if warped_image is None:
            return frame, None, None, None, 0
        
        # 在变换图像上绘制圆形并获取中心点
        center_x, center_y = self.a4_detector.draw_circle_on_warped(warped_image, frame, M)
        
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
                        return frame, None, None, "quit", detected_width
                    
                    # 处理校准相关命令
                    self.process_calibration_command(command, detected_width)
        
        # 计算屏幕中心（使用固定中心点）
        screen_center_x = frame.shape[1] // 2
        screen_center_y = frame.shape[0] // 2
        
        # 计算偏移量并检查对齐
        dx, dy, alignment_status = self.a4_detector.calculate_offset_and_check_alignment(
            center_x, center_y, screen_center_x, screen_center_y, self.serial_controller
        )
        
        # 绘制所有信息
        self.display_manager.draw_detection_info(frame, detected_rect, center_x, center_y)
        self.display_manager.draw_screen_center(frame, screen_center_x, screen_center_y)
        
        info_y = self.display_manager.draw_distance_info(frame, distance_mm, avg_distance)
        # 移除偏移信息显示
        self.display_manager.draw_status_info(frame, self.serial_controller.is_connected())
        
        # 绘制校准状态
        if self.distance_calculator.calibration_mode:
            cv2.putText(frame, f"CALIBRATION MODE ({len(self.distance_calculator.calibration_data)}/3)", 
                       (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Pixel Width: {detected_width}", 
                       (10, frame.shape[0] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 更新变换视图
        self.display_manager.update_transformed_view(warped_image)
        
        return frame, warped_image, distance_mm, avg_distance, detected_width
    
    def run(self):
        """运行主循环"""
        if not self.initialize_camera():
            return
        
        print("=== 模块化A4纸跟踪系统（集成距离校准） ===")
        print("距离计算参数:", config.CAMERA_PARAMS)
        print(f"距离范围: {config.MIN_DISTANCE_MM}-{config.MAX_DISTANCE_MM}mm")
        
        if self.display_manager.display_enabled:
            print("\n操作:")
            print("- 'q': 退出程序")
            print("- 'h': 清除距离历史记录")
            print("- 'd': 显示当前距离信息")
            print("- 'r': 重置距离校准数据")
            print("- 's': 保存当前配置")
            print("\n距离校准模式（自动开启）:")
            print("- 将A4纸放在500mm处，通过串口发送: 50")
            print("- 将A4纸放在1000mm处，通过串口发送: 100")
            print("- 将A4纸放在1500mm处，通过串口发送: 150")
        else:
            print("\n运行在无头模式 - 使用 Ctrl+C 退出程序")
        
        self.hmi.write(b't0.txt="running"\xff\xff\xff')
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法接收帧（视频流结束？）。正在退出...")
                break
            
            # 处理帧
            processed_frame, warped_image, distance, avg_distance, pixel_width = self.process_frame(frame)
            
            # 检查是否收到串口退出命令
            if avg_distance == "quit":
                print("收到串口退出命令，程序退出")
                break
            
            # 显示结果
            self.display_manager.show_frames(processed_frame)
            
            # 处理键盘输入
            key_result = self.display_manager.handle_keyboard_input(
                self.distance_calculator, None
            )
            
            if key_result == 'quit':
                break
            
            # 处理额外的键盘命令
            if self.display_manager.display_enabled:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):
                    # 重置校准数据
                    self.distance_calculator.reset_calibration()
                elif key == ord('s'):
                    # 保存配置
                    config.save_to_file()
                    print("配置已保存")
        
        self.cleanup()
    
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
            
            # 显示校准参数
            print(f"最终校准参数: {self.distance_calculator.camera_params}")

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
            }
        }

def main():
    """主函数"""
    # 创建系统实例
    tracking_system = A4TrackingSystem()
    
    # 可选：在运行前调整参数
    # SystemParameterManager.update_detection_parameters(mean_inner_val=105, mean_border_val=75)
    # SystemParameterManager.update_tracking_parameters(alignment_threshold=8)
    
    # 运行系统
    try:
        tracking_system.run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        tracking_system.cleanup()
    except Exception as e:
        print(f"程序运行错误: {e}")
        tracking_system.cleanup()

if __name__ == '__main__':
    main()