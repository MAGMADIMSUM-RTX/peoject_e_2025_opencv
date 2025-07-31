#!/usr/bin/env python3
"""
测试basic3.py的未检测信号发送功能
"""

import time
import sys
sys.path.append('/home/lc/code')

from basic3 import A4TrackingSystem
from dynamic_config import config

def test_no_detection_signal():
    """测试未检测到矩形时的信号发送功能"""
    print("=== 测试未检测信号发送功能 ===")
    
    # 创建系统实例
    system = A4TrackingSystem()
    
    print(f"未检测信号发送间隔: {system.no_detection_send_interval}秒")
    print(f"串口状态: {'已连接' if system.serial_controller.is_connected() else '未连接'}")
    
    # 测试发送未检测信号
    print("\n测试发送未检测信号...")
    try:
        system.send_no_detection_signal()
        print("✓ 未检测信号发送成功")
    except Exception as e:
        print(f"✗ 未检测信号发送失败: {e}")
    
    # 测试定时发送逻辑
    print(f"\n测试定时发送逻辑（间隔{system.no_detection_send_interval}秒）...")
    
    # 模拟连续调用process_frame但没有检测到矩形
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    test_duration = 2.0  # 测试2秒
    frame_count = 0
    
    while time.time() - start_time < test_duration:
        # 模拟处理帧（没有A4纸）
        result = system.process_frame(dummy_frame)
        frame_count += 1
        
        time.sleep(0.05)  # 模拟帧处理间隔
    
    print(f"测试完成: 处理了{frame_count}帧，耗时{test_duration}秒")
    
    # 清理资源
    system.cleanup()

def test_detection_to_no_detection_transition():
    """测试从检测到未检测的转换"""
    print("\n=== 测试检测状态转换 ===")
    
    system = A4TrackingSystem()
    
    # 模拟有检测和无检测的情况
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("模拟场景: A4纸进入视野 -> 离开视野")
    
    # 这里只是演示逻辑，实际检测需要真实的A4纸图像
    print("1. 模拟A4纸在视野中（实际需要真实图像）")
    print("2. 模拟A4纸离开视野...")
    
    for i in range(10):
        result = system.process_frame(dummy_frame)
        print(f"   帧 {i+1}: 未检测到矩形，应该发送(1000,0)信号")
        time.sleep(0.1)
    
    system.cleanup()

if __name__ == '__main__':
    try:
        test_no_detection_signal()
        test_detection_to_no_detection_transition()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
    
    print("\n测试完成")
