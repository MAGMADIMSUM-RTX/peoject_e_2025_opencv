#!/usr/bin/env python3
"""
测试 poweroff 功能的脚本
"""

def test_poweroff_command_parsing():
    """测试 poweroff 指令的解析"""
    # 模拟数据包: AA + "poweroff" + A5 5A
    # "poweroff" 的ASCII码：70 6F 77 65 72 6F 66 66
    poweroff_packet = bytes([0xAA, 0x70, 0x6F, 0x77, 0x65, 0x72, 0x6F, 0x66, 0x66, 0xA5, 0x5A])
    
    print("测试 poweroff 指令数据包:")
    hex_data = ' '.join([f'{b:02X}' for b in poweroff_packet])
    print(f"原始数据包: {hex_data}")
    
    # 模拟解析过程
    if poweroff_packet[0] == 0xAA and poweroff_packet[-2:] == bytes([0xA5, 0x5A]):
        command_bytes = poweroff_packet[1:-2]
        command = command_bytes.decode('utf-8', errors='ignore').strip()
        print(f"解析出的指令: '{command}'")
        
        if command == 'poweroff':
            print("✓ poweroff 指令解析成功!")
            return True
        else:
            print(f"✗ 解析错误，期望 'poweroff'，得到 '{command}'")
            return False
    else:
        print("✗ 数据包格式错误")
        return False

def test_poweroff_hex_values():
    """测试 poweroff 的十六进制值"""
    poweroff_str = "poweroff"
    hex_values = [f'{ord(c):02X}' for c in poweroff_str]
    print(f"\n'poweroff' 的十六进制表示:")
    print(f"字符串: {poweroff_str}")
    print(f"十六进制: {' '.join(hex_values)}")
    print(f"完整数据包: AA {' '.join(hex_values)} A5 5A")

if __name__ == '__main__':
    print("=" * 50)
    print("测试 poweroff 功能")
    print("=" * 50)
    
    test_poweroff_hex_values()
    print()
    success = test_poweroff_command_parsing()
    
    if success:
        print("\npoweroff 功能测试通过!")
        print("当收到对应的数据包时，系统将执行 'sudo poweroff' 命令")
    else:
        print("\npoweroff 功能测试失败!")
    
    print("=" * 50)
