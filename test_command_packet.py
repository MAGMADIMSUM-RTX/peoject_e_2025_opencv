#!/usr/bin/env python3
"""
测试新的指令数据包格式解析
格式: AA + 指令内容 + A5 5A
"""

def test_command_packet_parsing():
    """测试指令数据包解析功能"""
    
    # 模拟 MainController 的解析方法
    def parse_command_packet(packet):
        """解析新格式的指令数据包"""
        if not packet or len(packet) < 4:
            return None
        
        try:
            # 验证数据包格式
            if packet[0] != 0xAA:
                print(f"错误: 数据包开始符不正确 (0x{packet[0]:02X})")
                return None
            
            if packet[-2:] != b'\xA5\x5A':
                print(f"错误: 数据包结束符不正确")
                return None
            
            # 提取指令内容（去除开始符和结束符）
            command_bytes = packet[1:-2]
            
            # 显示原始数据用于调试
            hex_data = ' '.join([f'{b:02X}' for b in packet])
            print(f"原始数据包: {hex_data}")
            
            # 解码指令内容
            command = command_bytes.decode('utf-8', errors='ignore').strip()
            
            if command:
                return command
            else:
                print("警告: 解析出的指令为空")
                return None
                
        except Exception as e:
            print(f"解析指令数据包错误: {e}")
            hex_data = ' '.join([f'0x{b:02X}' for b in packet])
            print(f"错误数据包: {hex_data}")
            return None
    
    # 测试用例
    test_cases = [
        {
            'name': '示例指令: basic2.py',
            'hex': 'AA 62 61 73 69 63 32 2e 70 79 0A A5 5A',
            'expected': 'basic2.py'
        },
        {
            'name': '指令: basic3',
            'hex': 'AA 62 61 73 69 63 33 0A A5 5A',
            'expected': 'basic3'
        },
        {
            'name': '指令: help',
            'hex': 'AA 68 65 6C 70 0A A5 5A',
            'expected': 'help'
        },
        {
            'name': '指令: quit',
            'hex': 'AA 71 75 69 74 0A A5 5A',
            'expected': 'quit'
        },
        {
            'name': '错误: 缺少开始符',
            'hex': '62 61 73 69 63 32 0A A5 5A',
            'expected': None
        },
        {
            'name': '错误: 缺少结束符',
            'hex': 'AA 62 61 73 69 63 32 0A',
            'expected': None
        }
    ]
    
    print("=" * 60)
    print("测试新指令数据包格式解析")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['name']}")
        print(f"输入: {test_case['hex']}")
        
        # 将十六进制字符串转换为字节
        hex_bytes = test_case['hex'].replace(' ', '')
        try:
            packet = bytes.fromhex(hex_bytes)
        except ValueError as e:
            print(f"错误: 无效的十六进制格式 - {e}")
            continue
        
        # 解析数据包
        result = parse_command_packet(packet)
        
        # 检查结果
        expected = test_case['expected']
        if result == expected:
            print(f"✓ 测试通过: '{result}'")
        else:
            print(f"✗ 测试失败: 期望 '{expected}', 得到 '{result}'")
        
        print("-" * 40)
    
    print("\n测试完成!")

if __name__ == '__main__':
    test_command_packet_parsing()
