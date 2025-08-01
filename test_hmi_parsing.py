#!/usr/bin/env python3
"""
测试HMI数据包解析功能
验证 'q' 指令的解析是否正确
"""

def test_hmi_packet_parsing():
    """测试HMI数据包解析"""
    
    def parse_hmi_command_packet(packet):
        """解析HMI指令数据包（复制自basic2.py的逻辑）"""
        if not packet or len(packet) < 4:
            return None
        
        try:
            # 验证数据包格式
            if packet[0] != 0xAA:
                print(f"错误: 开始符不正确 (0x{packet[0]:02X})")
                return None
            
            if packet[-2:] != b'\xA5\x5A':
                print("错误: 结束符不正确")
                return None
            
            # 提取指令内容（去除开始符和结束符）
            command_bytes = packet[1:-2]
            
            # 显示原始数据
            hex_data = ' '.join([f'{b:02X}' for b in packet])
            print(f"原始数据包: {hex_data}")
            print(f"指令字节: {' '.join([f'{b:02X}' for b in command_bytes])}")
            
            # 解码指令内容
            command = command_bytes.decode('utf-8', errors='ignore').strip()
            
            print(f"解码结果: '{command}'")
            print(f"指令长度: {len(command)}")
            
            return command if command else None
                
        except Exception as e:
            print(f"解析错误: {e}")
            return None
    
    # 测试数据包: AA 71 0A A5 5A
    print("=" * 50)
    print("测试HMI数据包解析")
    print("=" * 50)
    
    test_packet = bytes([0xAA, 0x71, 0x0A, 0xA5, 0x5A])
    print("测试数据包: AA 71 0A A5 5A")
    print()
    
    result = parse_hmi_command_packet(test_packet)
    
    print(f"\n解析结果: '{result}'")
    
    if result:
        print(f"小写处理: '{result.lower().strip()}'")
        
        # 测试匹配条件
        if result.lower().strip() in ['q', 'quit', 'exit']:
            print("✓ 匹配退出条件")
        else:
            print("✗ 不匹配退出条件")
            
        # 显示ASCII值
        for i, char in enumerate(result):
            print(f"字符{i}: '{char}' (ASCII: {ord(char)})")
    else:
        print("解析失败")
    
    print("\n" + "=" * 50)

if __name__ == '__main__':
    test_hmi_packet_parsing()
