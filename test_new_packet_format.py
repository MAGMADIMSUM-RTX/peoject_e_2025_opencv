#!/usr/bin/env python3
"""
测试新的数据包格式 (以0xA5 0x5A结束)
"""

from serial_controller import SerialController
from dynamic_config import config

def test_new_packet_format():
    """测试新的数据包格式解析"""
    print("=== 测试新数据包格式 ===\n")
    
    # 创建模拟的串口控制器
    serial_ctrl = SerialController(None, None)  # 不实际连接
    
    # 测试用例
    test_cases = [
        {
            "name": "退出命令 'q'",
            "data": b'q\x0A\xA5\x5A',
            "expected": "quit"
        },
        {
            "name": "确认命令 'ok'",
            "data": b'ok\x0A\xA5\x5A',
            "expected": "ok"
        },
        {
            "name": "文本坐标 '5,-2'",
            "data": b'5,-2\x0A\xA5\x5A',
            "expected": "(5, -2)"
        },
        {
            "name": "二进制坐标 (5, -2)",
            "data": b'\x05\x00\x00\x00,\xFE\xFF\xFF\xFF\x0A\xA5\x5A',
            "expected": "(5, -2)"
        },
        {
            "name": "大数值坐标 (1000, -500)",
            "data": b'1000,-500\x0A\xA5\x5A',
            "expected": "(1000, -500)"
        },
        {
            "name": "未知数据",
            "data": b'hello\xA5\x5A',
            "expected": "unknown"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_case['name']}")
        print(f"原始数据: {' '.join([f'0x{b:02X}' for b in test_case['data']])}")
        
        # 提取数据包部分（去除结束符）
        if test_case['data'].endswith(b'\xA5\x5A'):
            packet = test_case['data'][:-2]
        else:
            packet = test_case['data']
        
        # 解析数据包
        result, message = serial_ctrl.parse_packet_data(packet)
        print(f"解析结果: {message}")
        
        if result == 'quit':
            print("-> 解析为退出命令")
        elif result == 'ok':
            print("-> 解析为确认命令")
        elif isinstance(result, tuple):
            x, y = result
            print(f"-> 解析为坐标: ({x}, {y})")
        else:
            print("-> 解析失败或未知格式")
        
        print("-" * 50)

def test_packet_reading_simulation():
    """模拟数据包读取过程"""
    print("\n=== 模拟数据包读取过程 ===\n")
    
    # 模拟接收到的字节流
    byte_stream = (
        b'5,-2\x0A\xA5\x5A'  # 第一个数据包
        b'ok\x0A\xA5\x5A'   # 第二个数据包
        b'q\x0A\xA5\x5A'    # 第三个数据包
    )
    
    print(f"模拟字节流: {' '.join([f'0x{b:02X}' for b in byte_stream])}")
    print()
    
    # 手动模拟数据包分割过程
    buffer = b''
    packet_count = 0
    
    for byte in byte_stream:
        buffer += bytes([byte])
        
        # 检查是否找到结束符
        if len(buffer) >= 2 and buffer[-2:] == b'\xA5\x5A':
            packet_count += 1
            packet = buffer[:-2]  # 去除结束符
            
            print(f"数据包 {packet_count}:")
            print(f"  内容: {' '.join([f'0x{b:02X}' for b in packet])}")
            
            # 创建串口控制器实例来解析
            serial_ctrl = SerialController(None, None)
            result, message = serial_ctrl.parse_packet_data(packet)
            print(f"  解析: {message}")
            
            if isinstance(result, tuple):
                print(f"  坐标: {result}")
            elif result:
                print(f"  命令: {result}")
            
            print()
            buffer = b''  # 清空缓冲区

if __name__ == '__main__':
    test_new_packet_format()
    test_packet_reading_simulation()
