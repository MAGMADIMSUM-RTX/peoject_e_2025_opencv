#!/usr/bin/env python3
"""
测试start指令功能
"""

import time
from serial_controller import SerialController
from dynamic_config import config

def send_start_command():
    """发送start指令到HMI串口"""
    print("测试发送start指令...")
    
    try:
        hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        if not hmi.is_connected():
            print("无法连接到HMI串口")
            return
        
        # 构造start指令数据包: AA + "start\n" + A5 5A
        # "start\n" = 73 74 61 72 74 0A
        start_packet = b'\xAA\x73\x74\x61\x72\x74\x0A\xA5\x5A'
        
        print("发送数据包:")
        hex_data = ' '.join([f'{b:02X}' for b in start_packet])
        print(f"  {hex_data}")
        print("  对应: AA + 'start\\n' + A5 5A")
        
        hmi.write(start_packet)
        print("start指令已发送!")
        
        hmi.close()
        
    except Exception as e:
        print(f"发送start指令失败: {e}")

def send_quit_command():
    """发送quit指令到HMI串口"""
    print("测试发送quit指令...")
    
    try:
        hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
        if not hmi.is_connected():
            print("无法连接到HMI串口")
            return
        
        # 构造quit指令数据包: AA + "q\n" + A5 5A
        # "q\n" = 71 0A
        quit_packet = b'\xAA\x71\x0A\xA5\x5A'
        
        print("发送数据包:")
        hex_data = ' '.join([f'{b:02X}' for b in quit_packet])
        print(f"  {hex_data}")
        print("  对应: AA + 'q\\n' + A5 5A")
        
        hmi.write(quit_packet)
        print("quit指令已发送!")
        
        hmi.close()
        
    except Exception as e:
        print(f"发送quit指令失败: {e}")

def main():
    """主函数"""
    print("=== start指令测试工具 ===")
    print("1. start - 发送start指令")
    print("2. quit  - 发送quit指令")
    print("3. exit  - 退出测试工具")
    
    while True:
        command = input("\n请选择操作 (start/quit/exit): ").strip().lower()
        
        if command == 'start':
            send_start_command()
        elif command == 'quit':
            send_quit_command()
        elif command == 'exit':
            print("退出测试工具")
            break
        else:
            print("无效指令，请输入 start、quit 或 exit")

if __name__ == '__main__':
    main()
