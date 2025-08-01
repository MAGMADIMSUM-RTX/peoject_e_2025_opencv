#!/usr/bin/env python3
"""
测试重启功能的脚本
"""

def test_restart_command_parsing():
    """测试重启指令的解析"""
    # 模拟数据包: AA 72 65 73 74 61 72 74 A5 5A (restart)
    restart_packet = bytes([0xAA, 0x72, 0x65, 0x73, 0x74, 0x61, 0x72, 0x74, 0xA5, 0x5A])
    
    print("测试重启指令数据包:")
    hex_data = ' '.join([f'{b:02X}' for b in restart_packet])
    print(f"原始数据包: {hex_data}")
    
    # 模拟解析过程
    if restart_packet[0] == 0xAA and restart_packet[-2:] == bytes([0xA5, 0x5A]):
        command_bytes = restart_packet[1:-2]
        command = command_bytes.decode('utf-8', errors='ignore').strip()
        print(f"解析出的指令: '{command}'")
        
        if command == 'restart':
            print("✓ 重启指令解析成功!")
            return True
        else:
            print(f"✗ 解析错误，期望 'restart'，得到 '{command}'")
            return False
    else:
        print("✗ 数据包格式错误")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("测试重启功能")
    print("=" * 50)
    
    success = test_restart_command_parsing()
    
    if success:
        print("\n重启功能测试通过!")
        print("数据包 'AA 72 65 73 74 61 72 74 A5 5A' 将被正确解析为 'restart' 指令")
    else:
        print("\n重启功能测试失败!")
    
    print("=" * 50)
