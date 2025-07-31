#!/usr/bin/env python3
"""
串口数据读取和解析脚本
支持二进制和文本两种数据格式
"""

import sys
import time
import struct
from serial_controller import SerialController
from dynamic_config import config

def parse_binary_packet(packet):
    """解析二进制数据包"""
    if len(packet) != 10:
        return None, f"数据包长度错误: 期望10字节，得到{len(packet)}字节"
    
    try:
        # 前4字节: 第一个int32 (小端)
        x = struct.unpack('<i', packet[0:4])[0]
        
        # 第5字节: 逗号
        if packet[4] != 0x2C:
            return None, f"分隔符错误: 期望逗号(0x2C)，得到0x{packet[4]:02X}"
        
        # 第6-9字节: 第二个int32 (小端)
        y = struct.unpack('<i', packet[5:9])[0]
        
        # 第10字节: 换行符
        if packet[9] != 0x0A:
            return None, f"结束符错误: 期望换行符(0x0A)，得到0x{packet[9]:02X}"
        
        return (x, y), "解析成功"
        
    except Exception as e:
        return None, f"解析异常: {e}"

def parse_text_data(text):
    """解析文本数据"""
    try:
        text = text.strip()
        
        # 检查特殊命令
        if text.lower() == 'q':
            return 'quit', "退出命令"
        elif text.lower() == 'ok':
            return 'ok', "确认命令"
        
        # 解析坐标数据
        parts = text.split(',')
        if len(parts) == 2:
            x = int(parts[0])
            y = int(parts[1])
            return (x, y), "文本解析成功"
        else:
            return None, f"格式错误: 期望2个值，得到{len(parts)}个"
            
    except ValueError as e:
        return None, f"数值转换错误: {e}"
    except Exception as e:
        return None, f"文本解析异常: {e}"

def monitor_serial_data(port=None, baudrate=None, duration=None):
    """监控串口数据"""
    
    # 使用配置文件中的HMI串口设置
    if port is None:
        port = config.HMI_PORT
    if baudrate is None:
        baudrate = config.HMI_BAUDRATE
    
    print(f"=== 串口数据监控器 ===")
    print(f"端口: {port}")
    print(f"波特率: {baudrate}")
    if duration:
        print(f"监控时长: {duration}秒")
    print("按 Ctrl+C 停止监控\n")
    
    # 创建串口控制器
    serial_ctrl = SerialController(port, baudrate)
    
    if not serial_ctrl.is_connected():
        print("错误: 无法连接串口")
        return
    
    print("串口连接成功，开始监控数据...\n")
    
    start_time = time.time()
    packet_count = 0
    
    try:
        while True:
            # 检查是否超时
            if duration and (time.time() - start_time) > duration:
                print(f"\n监控时间({duration}秒)已到，停止监控")
                break
            
            # 尝试读取二进制数据包
            packet = serial_ctrl.ser.read(10) if serial_ctrl.ser else None
            
            if packet and len(packet) > 0:
                packet_count += 1
                timestamp = time.strftime("%H:%M:%S")
                
                print(f"[{timestamp}] 数据包 #{packet_count}")
                print(f"原始数据: {[f'0x{b:02X}' for b in packet]}")
                
                if len(packet) == 10:
                    # 尝试解析为二进制数据包
                    result, message = parse_binary_packet(packet)
                    print(f"二进制解析: {message}")
                    if result:
                        x, y = result
                        print(f"坐标: ({x}, {y})")
                        
                        # 检查特殊命令
                        if x == 113 and y == 0:
                            print("-> 检测到退出命令!")
                        elif x == 111 and y == 107:
                            print("-> 检测到确认命令!")
                else:
                    # 尝试解析为文本数据
                    try:
                        text = packet.decode('utf-8', errors='ignore')
                        print(f"文本数据: '{text}'")
                        result, message = parse_text_data(text)
                        print(f"文本解析: {message}")
                        if result and isinstance(result, tuple):
                            x, y = result
                            print(f"坐标: ({x}, {y})")
                        elif result == 'quit':
                            print("-> 检测到退出命令!")
                        elif result == 'ok':
                            print("-> 检测到确认命令!")
                    except Exception as e:
                        print(f"文本解析失败: {e}")
                
                print("-" * 50)
            
            # 短暂延时，避免CPU占用过高
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\n\n用户中断监控")
    finally:
        serial_ctrl.close()
        print(f"总共接收到 {packet_count} 个数据包")
        print("串口已关闭")

def test_specific_packet():
    """测试特定数据包解析"""
    print("=== 测试数据包解析 ===\n")
    
    # 测试用例
    test_cases = [
        {
            "name": "正常坐标 (5, -2)",
            "data": bytes([0x05, 0x00, 0x00, 0x00, 0x2C, 0xFE, 0xFF, 0xFF, 0xFF, 0x0A])
        },
        {
            "name": "退出命令 (113, 0)",
            "data": bytes([0x71, 0x00, 0x00, 0x00, 0x2C, 0x00, 0x00, 0x00, 0x00, 0x0A])
        },
        {
            "name": "确认命令 (111, 107)",
            "data": bytes([0x6F, 0x00, 0x00, 0x00, 0x2C, 0x6B, 0x00, 0x00, 0x00, 0x0A])
        },
        {
            "name": "大正数 (1000, 500)",
            "data": bytes([0xE8, 0x03, 0x00, 0x00, 0x2C, 0xF4, 0x01, 0x00, 0x00, 0x0A])
        },
        {
            "name": "负数对 (-10, -20)",
            "data": bytes([0xF6, 0xFF, 0xFF, 0xFF, 0x2C, 0xEC, 0xFF, 0xFF, 0xFF, 0x0A])
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_case['name']}")
        print(f"原始数据: {[f'0x{b:02X}' for b in test_case['data']]}")
        
        result, message = parse_binary_packet(test_case['data'])
        print(f"解析结果: {message}")
        
        if result:
            x, y = result
            print(f"坐标: ({x}, {y})")
            
            # 特殊命令检查
            if x == 113 and y == 0:
                print("-> 这是退出命令")
            elif x == 111 and y == 107:
                print("-> 这是确认命令")
        
        print("-" * 40)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python serial_monitor.py monitor [端口] [波特率] [时长]")
        print("  python serial_monitor.py test")
        print("\n示例:")
        print("  python serial_monitor.py monitor")
        print("  python serial_monitor.py monitor /dev/ttyUSB0 115200")
        print("  python serial_monitor.py monitor /dev/ttyUSB0 115200 30")
        print("  python serial_monitor.py test")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'monitor':
        port = sys.argv[2] if len(sys.argv) > 2 else None
        baudrate = int(sys.argv[3]) if len(sys.argv) > 3 else None
        duration = int(sys.argv[4]) if len(sys.argv) > 4 else None
        
        monitor_serial_data(port, baudrate, duration)
        
    elif command == 'test':
        test_specific_packet()
        
    else:
        print(f"未知命令: {command}")
        print("支持的命令: monitor, test")

if __name__ == '__main__':
    main()
