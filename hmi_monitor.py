#!/usr/bin/env python3
"""
简化的串口数据监控脚本
实时显示从HMI串口接收到的数据
"""

import time
import signal
import sys
from serial_controller import SerialController
from dynamic_config import config

class SerialMonitor:
    def __init__(self):
        self.running = True
        self.serial_ctrl = None
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n\n收到中断信号，正在退出...")
        self.running = False
    
    def start_monitoring(self):
        """开始监控串口数据"""
        print("=== HMI串口数据监控 ===")
        print(f"端口: {config.HMI_PORT}")
        print(f"波特率: {config.HMI_BAUDRATE}")
        print("按 Ctrl+C 停止监控\n")
        
        # 连接串口
        self.serial_ctrl = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        
        if not self.serial_ctrl.is_connected():
            print("错误: 无法连接HMI串口")
            return
        
        print("HMI串口连接成功，开始监控...\n")
        
        packet_count = 0
        
        try:
            while self.running:
                # 读取以0xA5 0x5A结束的数据包
                packet = self.serial_ctrl.read_packet_with_terminator(timeout=0)
                
                if packet:
                    packet_count += 1
                    timestamp = time.strftime("%H:%M:%S")
                    
                    # 显示原始数据
                    hex_data = ' '.join([f'0x{b:02X}' for b in packet])
                    print(f"[{timestamp}] #{packet_count:04d} 原始数据: {hex_data}")
                    
                    # 解析数据包
                    result, message = self.serial_ctrl.parse_packet_data(packet)
                    print(f"                解析结果: {message}")
                    
                    if result == 'quit':
                        print("                -> 收到退出命令!")
                        break
                    elif result == 'ok':
                        print("                -> 收到确认命令!")
                    elif isinstance(result, tuple):
                        x, y = result
                        print(f"                -> 坐标数据: ({x}, {y})")
                    
                    print()  # 空行分隔
                
                # 短暂延时
                time.sleep(0.01)
                
        except Exception as e:
            print(f"\n监控过程中出错: {e}")
        
        finally:
            if self.serial_ctrl:
                self.serial_ctrl.close()
            print(f"\n监控结束，总共接收 {packet_count} 个数据包")

def main():
    """主函数"""
    monitor = SerialMonitor()
    monitor.start_monitoring()

if __name__ == '__main__':
    main()
