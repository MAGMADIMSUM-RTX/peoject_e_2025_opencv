#!/usr/bin/env python3
"""
模块化A4纸跟踪系统
主文件 - 包含主要的控制逻辑和可调参数接口
"""

import cv2

# 导入模块化组件
from dynamic_config import config
from serial_controller import SerialController
from distance_offset_calculator import DistanceOffsetCalculator
from distance_calculator import SimpleDistanceCalculator
from a4_detector import A4PaperDetector
from display_manager import DisplayManager

dx_from_uart = 0
dy_from_uart = 0  # 用于从串口接收偏移量

class A4TrackingSystem:
    """A4纸跟踪系统主类"""
    
    def __init__(self):
        # 初始化所有组件
        self.serial_controller = SerialController(config.SERIAL_PORT, config.SERIAL_BAUDRATE)
        self.hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        self.distance_calculator = SimpleDistanceCalculator()
        self.offset_calculator = DistanceOffsetCalculator()
        self.a4_detector = A4PaperDetector()
        self.display_manager = DisplayManager()

        self.fix_gap = False
        
        # 摄像头
        self.cap = None
        
        # 校准点收集
        self.calibration_points = []  # 存储校准点 [distance_mm, offset_x, offset_y]
        self.max_calibration_points = 3  # 最大收集3个校准点

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
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 检测A4纸
        detected_rect = self.a4_detector.detect_a4_paper(frame)
        
        if detected_rect is None:
            return frame, None, None, None
        
        # 创建透视变换图像
        warped_image, M, detected_width, warped_size = self.a4_detector.create_warped_image(frame, detected_rect)
        
        if warped_image is None:
            return frame, None, None, None
        
        # 在变换图像上绘制圆形并获取中心点
        center_x, center_y = self.a4_detector.draw_circle_on_warped(warped_image, frame, M)
        
        # 距离测量
        distance_mm = self.distance_calculator.calculate_distance_from_width(detected_width, frame.shape[1])
        self.distance_calculator.update_distance_history(distance_mm)
        avg_distance = self.distance_calculator.get_averaged_distance()
        
        if self.fix_gap:
            # debug 时，使用串口，手动修正间隙
            global dx_from_uart, dy_from_uart
            
            # 读取以0xA5 0x5A结束的数据包
            packet = self.hmi.read_packet_with_terminator(timeout=0)
            if packet:
                result, message = self.hmi.parse_packet_data(packet)
                print(f"收到数据包: {message}")
                
                if result == 'quit':
                    print("收到退出命令，程序即将退出...")
                    return frame, None, None, "quit"
                
                elif result == 'ok':
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
                
                elif isinstance(result, tuple):
                    # 坐标数据
                    x, y = result
                    dx_from_uart = x
                    dy_from_uart = y
                    print(f"更新偏移量: dx={dx_from_uart}, dy={dy_from_uart}")
            
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
        
        info_y = self.display_manager.draw_distance_info(frame, distance_mm, avg_distance)
        self.display_manager.draw_offset_info(frame, offset_x, offset_y, dx, dy, alignment_status, info_y)
        self.display_manager.draw_status_info(frame, self.serial_controller.is_connected())
        
        # 更新变换视图
        self.display_manager.update_transformed_view(warped_image)
        
        return frame, warped_image, distance_mm, avg_distance
    
    def run(self):
        """运行主循环"""
        if not self.initialize_camera():
            return
        
        print("=== 模块化A4纸跟踪系统 ===")
        print("距离计算参数:", config.CAMERA_PARAMS)
        print(f"距离范围: {config.MIN_DISTANCE_MM}-{config.MAX_DISTANCE_MM}mm")
        
        if self.display_manager.display_enabled:
            print("\n操作:")
            print("- 'q': 退出程序")
            print("- 'h': 清除距离历史记录")
            print("- 'd': 显示当前距离和偏移信息")
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
            key_result = self.display_manager.handle_keyboard_input(
                self.distance_calculator, self.offset_calculator
            )
            
            if key_result == 'quit':
                break
        
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
            
            # 显示最终的屏幕中心偏移
            offset_x, offset_y = self.offset_calculator.calculate_screen_center_offset(mean_distance)
            print(f"最终屏幕中心偏移: ({offset_x}, {offset_y})")

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
            },
            "calibration_points": config.CALIBRATION_POINTS
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
