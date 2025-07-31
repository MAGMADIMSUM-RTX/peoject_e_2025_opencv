#!/usr/bin/env python3
"""
测试基于距离递减的永久检测逻辑
模拟：检测到矩形一次后，即使后续检测失败也使用计算的距离
"""

import cv2
import numpy as np
import time
from basic3 import A4TrackingSystem

def create_test_frame_with_rect():
    """创建带有矩形的测试帧"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(50)  # 深灰色背景
    
    # 绘制一个白色矩形模拟A4纸
    cv2.rectangle(frame, (200, 150), (440, 330), (255, 255, 255), -1)
    cv2.rectangle(frame, (200, 150), (440, 330), (0, 0, 0), 2)
    
    return frame

def create_test_frame_without_rect():
    """创建不带矩形的测试帧"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame.fill(50)  # 深灰色背景，没有明显的矩形
    
    # 添加一些噪声
    noise = np.random.randint(0, 30, (480, 640, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    
    return frame

def test_distance_calculation_logic():
    """测试基于距离递减的逻辑"""
    print("=== 测试基于距离递减的永久检测逻辑 ===")
    
    # 创建跟踪系统实例
    system = A4TrackingSystem()
    
    print(f"初始检测状态: {system.rect_detected_once}")
    print(f"初始有效距离: {system.last_valid_distance}")
    print(f"初始失败计数: {system.detection_failure_count}")
    
    # 第一阶段：模拟没有检测到矩形的帧
    print("\n第一阶段：处理无矩形帧（应该发送未检测信号）")
    no_rect_frame = create_test_frame_without_rect()
    
    for i in range(2):
        print(f"  处理帧 {i+1}:")
        result = system.process_frame(no_rect_frame)
        print(f"    检测状态: {system.rect_detected_once}")
        print(f"    返回结果: {'有数据' if result[1] is not None else '无数据'}")
        time.sleep(0.1)
    
    # 第二阶段：模拟检测到矩形的帧
    print("\n第二阶段：处理有矩形帧（应该首次检测成功）")
    rect_frame = create_test_frame_with_rect()
    
    print("  处理有矩形的帧:")
    result = system.process_frame(rect_frame)
    print(f"    检测状态: {system.rect_detected_once}")
    print(f"    有效距离: {system.last_valid_distance}")
    print(f"    失败计数: {system.detection_failure_count}")
    print(f"    返回结果: {'有数据' if result[1] is not None else '无数据'}")
    
    # 第三阶段：再次处理无矩形帧，测试距离递减逻辑
    print("\n第三阶段：再次处理无矩形帧（应该使用距离递减逻辑）")
    
    for i in range(5):
        print(f"  处理帧 {i+1}:")
        result = system.process_frame(no_rect_frame)
        print(f"    检测状态: {system.rect_detected_once}")
        print(f"    有效距离: {system.last_valid_distance}")
        print(f"    失败计数: {system.detection_failure_count}")
        if system.last_valid_distance:
            expected_distance = system.last_valid_distance / (1 + system.detection_failure_count)
            print(f"    期望距离: {expected_distance:.1f}mm")
        print(f"    实际距离: {result[2]:.1f}mm" if result[2] else "无距离")
        print(f"    返回结果: {'有数据' if result[1] is not None else '无数据'}")
        time.sleep(0.1)
    
    # 第四阶段：再次处理有矩形帧，测试重置逻辑
    print("\n第四阶段：再次处理有矩形帧（应该重置失败计数）")
    
    print("  处理有矩形的帧:")
    result = system.process_frame(rect_frame)
    print(f"    检测状态: {system.rect_detected_once}")
    print(f"    有效距离: {system.last_valid_distance}")
    print(f"    失败计数: {system.detection_failure_count}")
    print(f"    返回结果: {'有数据（真实矩形）' if result[1] is not None else '无数据'}")
    
    # 清理
    system.cleanup()
    print("\n=== 测试完成 ===")

def test_distance_progression():
    """测试距离递减的数学逻辑"""
    print("\n=== 测试距离递减数学逻辑 ===")
    
    initial_distance = 1000.0  # 初始距离1000mm
    
    print(f"初始距离: {initial_distance}mm")
    
    for failure_count in range(1, 11):
        calculated_distance = initial_distance / (1 + failure_count)
        print(f"失败次数 {failure_count}: {calculated_distance:.1f}mm")

if __name__ == "__main__":
    test_distance_calculation_logic()
    test_distance_progression()
