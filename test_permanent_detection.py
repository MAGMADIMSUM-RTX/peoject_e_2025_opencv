#!/usr/bin/env python3
"""
测试永久检测逻辑
模拟：检测到矩形一次后，即使后续检测失败也不再发送未检测信号
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

def test_permanent_detection_logic():
    """测试永久检测逻辑"""
    print("=== 测试永久检测逻辑 ===")
    
    # 创建跟踪系统实例
    system = A4TrackingSystem()
    
    print(f"初始检测状态: {system.rect_detected_once}")
    
    # 第一阶段：模拟没有检测到矩形的帧
    print("\n第一阶段：处理无矩形帧（应该发送未检测信号）")
    no_rect_frame = create_test_frame_without_rect()
    
    for i in range(3):
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
    print(f"    返回结果: {'有数据' if result[1] is not None else '无数据'}")
    
    # 第三阶段：再次处理无矩形帧，但此时应该不再发送未检测信号
    print("\n第三阶段：再次处理无矩形帧（应该使用虚拟矩形，不发送未检测信号）")
    
    for i in range(5):
        print(f"  处理帧 {i+1}:")
        result = system.process_frame(no_rect_frame)
        print(f"    检测状态: {system.rect_detected_once}")
        print(f"    返回结果: {'有数据（虚拟矩形）' if result[1] is not None else '无数据'}")
        time.sleep(0.1)
    
    # 第四阶段：再次处理有矩形帧
    print("\n第四阶段：再次处理有矩形帧（应该正常检测）")
    
    print("  处理有矩形的帧:")
    result = system.process_frame(rect_frame)
    print(f"    检测状态: {system.rect_detected_once}")
    print(f"    返回结果: {'有数据（真实矩形）' if result[1] is not None else '无数据'}")
    
    # 清理
    system.cleanup()
    print("\n=== 测试完成 ===")

def test_virtual_rect_creation():
    """测试虚拟矩形创建功能"""
    print("\n=== 测试虚拟矩形创建 ===")
    
    system = A4TrackingSystem()
    test_frame = create_test_frame_without_rect()
    
    virtual_rect = system.create_virtual_rect(test_frame)
    
    print("虚拟矩形四个角点:")
    for i, point in enumerate(virtual_rect):
        print(f"  点{i+1}: ({point[0]}, {point[1]})")
    
    # 计算矩形的宽度和高度
    width = virtual_rect[1][0] - virtual_rect[0][0]
    height = virtual_rect[2][1] - virtual_rect[1][1]
    aspect_ratio = width / height
    
    print(f"虚拟矩形尺寸: {width} x {height}")
    print(f"长宽比: {aspect_ratio:.3f} (A4标准比例: {210/297:.3f})")
    
    system.cleanup()

if __name__ == "__main__":
    test_permanent_detection_logic()
    test_virtual_rect_creation()
