#!/usr/bin/env python3
"""
测试HMI数据包解析，特别是退出指令 'q'
"""

def parse_hmi_command_packet(packet):
    """解析新格式的HMI指令数据包"""
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
        
        # 解码指令内容
        command = command_bytes.decode('utf-8', errors='ignore').strip()
        
        return command if command else None
            
    except Exception as e:
        print(f"解析HMI指令数据包错误: {e}")
        return None

def test_quit_command():
    """测试退出指令解析"""
    print("测试HMI退出指令解析")
    print("=" * 40)
    
    # 测试数据包: AA 71 0A A5 5A
    test_packet = bytes([0xAA, 0x71, 0x0A, 0xA5, 0x5A])
    
    print(f"测试数据包: {' '.join([f'{b:02X}' for b in test_packet])}")
    
    # 分析每个字节
    print("\n字节分析:")
    print(f"0xAA = 开始符")
    print(f"0x71 = 字符 '{chr(0x71)}' (ASCII)")
    print(f"0x0A = 换行符 '\\n'")
    print(f"0xA5 0x5A = 结束符")
    
    # 解析指令
    command = parse_hmi_command_packet(test_packet)
    print(f"\n解析结果: '{command}'")
    print(f"长度: {len(command) if command else 0}")
    
    # 测试退出条件
    if command:
        is_quit = command.lower().strip() in ['q', 'quit', 'exit']
        print(f"是否为退出指令: {is_quit}")
        
        # 详细检查
        print(f"\n详细检查:")
        print(f"原始: '{command}'")
        print(f"小写: '{command.lower()}'")
        print(f"去空格: '{command.lower().strip()}'")
        print(f"字符编码: {[ord(c) for c in command]}")
    
    # 测试其他退出指令
    print("\n" + "=" * 40)
    print("测试其他退出指令:")
    
    test_cases = [
        ("quit", bytes([0xAA]) + b"quit\n" + bytes([0xA5, 0x5A])),
        ("exit", bytes([0xAA]) + b"exit\n" + bytes([0xA5, 0x5A])),
        ("Q", bytes([0xAA]) + b"Q\n" + bytes([0xA5, 0x5A])),
    ]
    
    for name, packet in test_cases:
        hex_str = ' '.join([f'{b:02X}' for b in packet])
        print(f"\n{name}: {hex_str}")
        cmd = parse_hmi_command_packet(packet)
        if cmd:
            is_quit = cmd.lower().strip() in ['q', 'quit', 'exit']
            print(f"  解析: '{cmd}' -> 退出指令: {is_quit}")

if __name__ == '__main__':
    test_quit_command()
