#!/usr/bin/env python3
"""
模拟HMI指令发送工具
用于测试主控制器的指令接收功能
"""

import time
import sys

def create_command_packet(command):
    """创建新格式的指令数据包
    
    Args:
        command (str): 要发送的指令
        
    Returns:
        bytes: 格式化的数据包 (AA + 指令内容 + A5 5A)
    """
    # 确保指令以换行符结尾
    if not command.endswith('\n'):
        command += '\n'
    
    # 编码指令
    command_bytes = command.encode('utf-8')
    
    # 构造数据包: AA + 指令内容 + A5 5A
    packet = b'\xAA' + command_bytes + b'\xA5\x5A'
    
    return packet

def print_packet_hex(packet):
    """打印数据包的十六进制格式"""
    hex_data = ' '.join([f'{b:02X}' for b in packet])
    return hex_data

def main():
    """主函数"""
    print("=" * 60)
    print("HMI指令数据包生成工具")
    print("=" * 60)
    print("格式: AA + 指令内容 + A5 5A")
    print("支持的指令示例:")
    print("  basic2, basic3, help, quit, list, status")
    print("  或任何Python文件名（如 test_script）")
    print("=" * 60)
    
    # 预定义的测试指令
    test_commands = [
        'help',
        'list', 
        'status',
        'basic2',
        'basic3',
        'test_script',
        'quit'
    ]
    
    print("\n预定义测试指令:")
    for i, cmd in enumerate(test_commands, 1):
        packet = create_command_packet(cmd)
        hex_data = print_packet_hex(packet)
        print(f"{i}. {cmd:<12} -> {hex_data}")
    
    print("\n" + "=" * 60)
    print("交互模式 - 输入指令生成数据包")
    print("输入 'exit' 退出")
    print("=" * 60)
    
    while True:
        try:
            command = input("\n请输入指令: ").strip()
            
            if not command:
                continue
                
            if command.lower() in ['exit', 'quit']:
                print("退出程序")
                break
            
            # 生成数据包
            packet = create_command_packet(command)
            hex_data = print_packet_hex(packet)
            
            print(f"指令: '{command}'")
            print(f"数据包: {hex_data}")
            print(f"长度: {len(packet)} 字节")
            
            # 解析验证
            if len(packet) >= 4 and packet[0] == 0xAA and packet[-2:] == b'\xA5\x5A':
                content = packet[1:-2].decode('utf-8', errors='ignore').strip()
                print(f"验证: 解析出的指令为 '{content}'")
            else:
                print("错误: 数据包格式不正确")
        
        except KeyboardInterrupt:
            print("\n\n用户中断，退出程序")
            break
        except Exception as e:
            print(f"错误: {e}")

if __name__ == '__main__':
    main()
