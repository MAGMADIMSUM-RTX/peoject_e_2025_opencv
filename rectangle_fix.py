#!/usr/bin/env python3
"""
A4纸检测参数调整系统
基于laser_fix修改，专门用于调整MEAN_INNER_VAL和MEAN_BORDER_VAL参数
"""

import cv2

# 导入模块化组件
from dynamic_config import config
from serial_controller import SerialController
from a4_detector import A4PaperDetector
from display_manager import DisplayManager

class A4DetectionTuner:
    """A4纸检测参数调整器"""
    
    def __init__(self):
        # 初始化组件
        self.serial_controller = SerialController(config.SERIAL_PORT, config.SERIAL_BAUDRATE)
        self.hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        self.a4_detector = A4PaperDetector()
        self.display_manager = DisplayManager()
        
        # 摄像头
        self.cap = None
        
        # 当前检测状态
        self.last_detection_status = None
        
        # 参数调整值（从串口接收）
        self.mean_inner_val_from_uart = config.MEAN_INNER_VAL
        self.mean_border_val_from_uart = config.MEAN_BORDER_VAL

    def read_hmi_command_packet(self, timeout=1.0):
        """读取HMI指令数据包: AA + 指令内容 + A5 5A"""
        if not self.hmi or not self.hmi.is_connected():
            return None
            
        try:
            # 设置临时超时
            original_timeout = self.hmi.ser.timeout
            self.hmi.ser.timeout = timeout
            
            buffer = b''
            max_length = 1024
            start_found = False
            
            while len(buffer) < max_length:
                byte = self.hmi.ser.read(1)
                if not byte:
                    break
                
                buffer += byte
                
                # 查找开始符 0xAA
                if not start_found:
                    if byte == b'\xAA':
                        start_found = True
                        buffer = b'\xAA'
                    else:
                        buffer = b''
                    continue
                
                # 查找结束符 0xA5 0x5A
                if len(buffer) >= 3 and buffer[-2:] == b'\xA5\x5A':
                    self.hmi.ser.timeout = original_timeout
                    return buffer
            
            self.hmi.ser.timeout = original_timeout
            return None
            
        except Exception as e:
            print(f"读取HMI指令数据包错误: {e}")
            return None
    
    def parse_hmi_command_packet(self, packet):
        """解析HMI指令数据包（使用laser_fix的逻辑）"""
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

    def process_parameter_command(self, command):
        """处理参数调整命令（按照laser_fix的逻辑）"""
        try:
            # 处理特殊命令
            if isinstance(command, str):
                if command.lower().strip() == 'ok':
                    # 收到ok指令后直接保存当前参数
                    config.save_to_file()
                    print(f"参数已保存: MEAN_INNER_VAL={config.MEAN_INNER_VAL}, MEAN_BORDER_VAL={config.MEAN_BORDER_VAL}")
                    return True
                
                # 尝试解析为文本坐标数据 "inner_val,border_val"
                elif ',' in command:
                    parts = command.strip().split(',')
                    if len(parts) == 2:
                        inner_val = int(parts[0])
                        border_val = int(parts[1])
                        
                        # 更新参数
                        self.mean_inner_val_from_uart = inner_val
                        self.mean_border_val_from_uart = border_val
                        
                        # 更新config中的参数
                        config.MEAN_INNER_VAL = inner_val
                        config.MEAN_BORDER_VAL = border_val
                        
                        print(f"更新检测参数: MEAN_INNER_VAL={inner_val}, MEAN_BORDER_VAL={border_val}")
                        return True
            
            # 处理二进制坐标格式 (x, y) -> (MEAN_INNER_VAL, MEAN_BORDER_VAL)
            elif isinstance(command, tuple):
                inner_val, border_val = command
                
                # 更新参数
                self.mean_inner_val_from_uart = inner_val
                self.mean_border_val_from_uart = border_val
                
                # 更新config中的参数
                config.MEAN_INNER_VAL = inner_val
                config.MEAN_BORDER_VAL = border_val
                
                print(f"更新检测参数(二进制): MEAN_INNER_VAL={inner_val}, MEAN_BORDER_VAL={border_val}")
                return True
                
        except ValueError as e:
            print(f"无法解析参数数据: {e}")
        
        return False
    
    def send_initial_parameters(self):
        """程序启动时发送当前参数到HMI"""
        inner_val = config.MEAN_INNER_VAL
        border_val = config.MEAN_BORDER_VAL
        
        # 发送MEAN_INNER_VAL到n0
        command1 = f'n0.val={inner_val}'
        self.hmi.write(command1.encode() + b'\xff\xff\xff')
        
        # 发送MEAN_BORDER_VAL到n1
        command2 = f'n1.val={border_val}'
        self.hmi.write(command2.encode() + b'\xff\xff\xff')
        
        print(f"已发送初始参数到HMI: n0.val={inner_val}, n1.val={border_val}")
    
    def send_detection_status(self, detected):
        """发送检测状态到HMI"""
        if detected != self.last_detection_status:
            if detected:
                self.hmi.write(b't0.txt="Detected"\xff\xff\xff')
                print("发送状态: 检测到目标")
            else:
                self.hmi.write(b't0.txt="Undetected"\xff\xff\xff')
                print("发送状态: 未检测到")
            
            self.last_detection_status = detected
    
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
        
        # 发送检测状态
        self.send_detection_status(detected_rect is not None)
        
        # 检查HMI指令
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
                        return frame, "quit"
                
                # 检查是否为二进制坐标
                elif isinstance(command, tuple):
                    x, y = command
                    print(f"收到二进制坐标: x={x}, y={y}")
                
                # 处理参数调整命令
                self.process_parameter_command(command)
        
        # 绘制检测结果
        if detected_rect is not None:
            # 绘制检测到的轮廓
            cv2.drawContours(frame, [detected_rect], -1, (0, 255, 0), 3)
            cv2.putText(frame, "A4 Paper Detected", 
                       (detected_rect.ravel()[0], detected_rect.ravel()[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 计算中心点
            M = cv2.moments(detected_rect)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Center", (center_x - 30, center_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 绘制当前参数信息
        info_y = 30
        cv2.putText(frame, f"MEAN_INNER_VAL: {config.MEAN_INNER_VAL}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        cv2.putText(frame, f"MEAN_BORDER_VAL: {config.MEAN_BORDER_VAL}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        # 显示检测状态
        status_text = "检测到A4纸" if detected_rect is not None else "未检测到A4纸"
        status_color = (0, 255, 0) if detected_rect is not None else (0, 0, 255)
        cv2.putText(frame, status_text, (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        info_y += 30
        
        # 显示串口状态
        serial_status = "Serial: ON" if self.serial_controller.is_connected() else "Serial: OFF"
        serial_color = (0, 255, 0) if self.serial_controller.is_connected() else (0, 0, 255)
        cv2.putText(frame, serial_status, (10, frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, serial_color, 2)
        
        hmi_status = "HMI: ON" if self.hmi.is_connected() else "HMI: OFF"
        hmi_color = (0, 255, 0) if self.hmi.is_connected() else (0, 0, 255)
        cv2.putText(frame, hmi_status, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, hmi_color, 2)
        
        # 发送参数到串口（用于调试）
        print(f"MEAN_INNER_VAL: {config.MEAN_INNER_VAL}, MEAN_BORDER_VAL: {config.MEAN_BORDER_VAL}")
        self.serial_controller.write(f"{config.MEAN_INNER_VAL},{config.MEAN_BORDER_VAL}\n")
        
        return frame, None
    
    def run(self):
        """运行主循环"""
        if not self.initialize_camera():
            return
        
        print("=== A4纸检测参数调整系统 ===")
        print(f"初始参数: MEAN_INNER_VAL={config.MEAN_INNER_VAL}, MEAN_BORDER_VAL={config.MEAN_BORDER_VAL}")
        
        if self.display_manager.display_enabled:
            print("\n操作:")
            print("- 'q': 退出程序")
            print("\n通过HMI串口发送参数:")
            print("- 二进制坐标格式: (inner_val, border_val)")
            print("- 文本格式: 'inner_val,border_val' (如: '105,75')")
            print("- 'ok': 保存当前参数到配置文件")
            print("- 'q'/'quit'/'exit': 退出程序")
        else:
            print("\n运行在无头模式 - 使用 Ctrl+C 退出程序")
        
        # 初始化HMI显示
        self.hmi.write(b't0.txt="running"\xff\xff\xff')
        
        # 发送当前参数到HMI
        self.send_initial_parameters()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法接收帧（视频流结束？）。正在退出...")
                break
            
            # 处理帧
            processed_frame, exit_signal = self.process_frame(frame)
            
            # 检查退出信号
            if exit_signal == "quit":
                print("收到退出命令，程序退出")
                break
            
            # 显示结果
            self.display_manager.show_frames(processed_frame)
            
            # 处理键盘输入
            if self.display_manager.display_enabled:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
            else:
                # 无头模式下添加延时
                import time
                time.sleep(0.01)
        
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
        self.hmi.close()
        
        # 显示最终参数
        print("\n=== 最终参数 ===")
        print(f"MEAN_INNER_VAL: {config.MEAN_INNER_VAL}")
        print(f"MEAN_BORDER_VAL: {config.MEAN_BORDER_VAL}")

def main():
    """主函数"""
    # 创建参数调整器实例
    tuner = A4DetectionTuner()
    
    # 运行系统
    try:
        tuner.run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
        tuner.cleanup()
    except Exception as e:
        print(f"程序运行错误: {e}")
        tuner.cleanup()

if __name__ == '__main__':
    main()