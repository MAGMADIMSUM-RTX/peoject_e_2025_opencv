#!/usr/bin/env python3
"""
串口数据发送测试脚本
用于测试串口数据的解析功能
"""

import struct
import time
from serial_controller import SerialController
from dynamic_config import config

def create_binary_packet(x, y):
    """创建二进制数据包"""
    # int32 (x) + ',' + int32 (y) + '\n'
    packet = struct.pack('<i', x)  # x (小端)
    packet += b','                 # 逗号
    packet += struct.pack('<i', y)  # y (小端) 
    packet += b'\n'                # 换行符
    return packet

def send_test_data():
    """发送测试数据"""
    print("=== 串口数据发送测试 ===")
    
    # 连接到主串口（假设用于发送测试数据）
    serial_ctrl = SerialController(config.SERIAL_PORT, config.SERIAL_BAUDRATE)
    
    if not serial_ctrl.is_connected():
        print("错误: 无法连接串口")
        return
    
    print("串口连接成功，开始发送测试数据...\n")
    
    # 测试数据
    test_data = [
        (5, -2),      # 正常坐标
        (0, 0),       # 原点
        (-10, 15),    # 负数坐标
        (100, -50),   # 大数值
        (113, 0),     # 退出命令
        (111, 107),   # 确认命令
    ]
    
    try:
        for i, (x, y) in enumerate(test_data, 1):
            print(f"发送数据 {i}: ({x}, {y})")
            
            # 发送二进制格式
            binary_packet = create_binary_packet(x, y)
            serial_ctrl.write(binary_packet)
            print(f"  二进制: {[f'0x{b:02X}' for b in binary_packet]}")
            
            time.sleep(1)  # 延时1秒
            
            # 发送文本格式（可选）
            text_packet = f"{x},{y}\n".encode('utf-8')
            serial_ctrl.write(text_packet)
            print(f"  文本: '{x},{y}\\n'")
            
            time.sleep(1)  # 延时1秒
            print()
    
    except KeyboardInterrupt:
        print("\n发送被中断")
    
    finally:
        serial_ctrl.close()
        print("测试完成，串口已关闭")

def send_specific_command(command):
    """发送特定命令"""
    serial_ctrl = SerialController(config.SERIAL_PORT, config.SERIAL_BAUDRATE)
    
    if not serial_ctrl.is_connected():
        print("错误: 无法连接串口")
        return
    
    if command == 'quit':
        # 发送退出命令
        packet = create_binary_packet(113, 0)
        serial_ctrl.write(packet)
        print("已发送退出命令")
    
    elif command == 'ok':
        # 发送确认命令
        packet = create_binary_packet(111, 107)
        serial_ctrl.write(packet)
        print("已发送确认命令")
    
    else:
        print(f"未知命令: {command}")
    
    serial_ctrl.close()

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python send_test_data.py test      # 发送测试数据序列")
        print("  python send_test_data.py quit      # 发送退出命令")
        print("  python send_test_data.py ok        # 发送确认命令")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'test':
        send_test_data()
    elif command in ['quit', 'ok']:
        send_specific_command(command)
    else:
        print(f"未知命令: {command}")

if __name__ == '__main__':
    main()
