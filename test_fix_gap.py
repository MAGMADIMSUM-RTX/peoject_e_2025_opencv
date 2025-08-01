#!/usr/bin/env python3
"""
测试basic2.py的fix_gap模式设置
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fix_gap_setting():
    """测试fix_gap模式设置"""
    try:
        from basic2 import A4TrackingSystem
        
        print("创建A4TrackingSystem实例...")
        system = A4TrackingSystem()
        
        print(f"初始fix_gap状态: {system.fix_gap}")
        
        # 设置fix_gap模式
        system.set_fix_gap(True)
        print(f"设置后fix_gap状态: {system.fix_gap}")
        
        # 检查HMI连接状态
        if system.hmi:
            print(f"HMI串口连接状态: {system.hmi.is_connected()}")
        else:
            print("HMI串口未初始化")
        
        # 测试数据包解析
        test_packet = bytes([0xAA, 0x71, 0x0A, 0xA5, 0x5A])
        print(f"\n测试数据包: {' '.join([f'{b:02X}' for b in test_packet])}")
        
        command = system.parse_hmi_command_packet(test_packet)
        if command:
            is_quit = command.lower().strip() in ['q', 'quit', 'exit']
            print(f"解析结果: '{command}' -> 是否退出: {is_quit}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_fix_gap_setting()
