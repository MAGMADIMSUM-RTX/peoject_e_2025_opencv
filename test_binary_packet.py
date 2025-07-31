#!/usr/bin/env python3
"""
测试二进制数据包解析
"""

import struct

def test_binary_packet():
    """测试二进制数据包格式解析"""
    
    # 模拟接收到的数据包: "5,-2\n"
    # 0x05, 0x00, 0x00, 0x00, 0x2C, 0xFE, 0xFF, 0xFF, 0xFF, 0x0A
    test_packet = bytes([0x05, 0x00, 0x00, 0x00, 0x2C, 0xFE, 0xFF, 0xFF, 0xFF, 0x0A])
    
    print("原始数据包:", [f"0x{b:02X}" for b in test_packet])
    
    if len(test_packet) == 10:
        # 前4字节: 第一个int32 (小端)
        x = struct.unpack('<i', test_packet[0:4])[0]
        print(f"第一个数字 (x): {x}")
        
        # 第5字节: 逗号
        comma = test_packet[4]
        print(f"分隔符: 0x{comma:02X} ({',' if comma == 0x2C else '不是逗号'})")
        
        # 第6-9字节: 第二个int32 (小端)  
        y = struct.unpack('<i', test_packet[5:9])[0]
        print(f"第二个数字 (y): {y}")
        
        # 第10字节: 换行符
        newline = test_packet[9]
        newline_char = '\\n' if newline == 0x0A else '不是换行符'
        print(f"结束符: 0x{newline:02X} ({newline_char})")
        
        print(f"解析结果: ({x}, {y})")
        print(f"期望结果: (5, -2)")
        
        # 验证负数编码
        print("\n=== 负数编码验证 ===")
        negative_two_bytes = struct.pack('<i', -2)
        print(f"-2 的小端编码: {[f'0x{b:02X}' for b in negative_two_bytes]}")
        print(f"与数据包匹配: {negative_two_bytes == test_packet[5:9]}")

if __name__ == '__main__':
    test_binary_packet()
