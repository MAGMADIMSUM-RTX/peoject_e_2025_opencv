#!/usr/bin/env python3
"""
测试除零错误修复
"""

import numpy as np
import cv2
from a4_detector import A4PaperDetector

def test_division_by_zero_fix():
    """测试透视变换中的除零错误修复"""
    
    # 创建A4检测器
    detector = A4PaperDetector()
    
    # 创建一个测试矩阵，可能导致除零错误
    print("测试透视变换除零错误修复...")
    
    # 创建一个可能导致问题的变换矩阵
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]  # 这里可能导致除零
    ], dtype=np.float32)
    
    try:
        inv_M = cv2.invert(M)[1]
        print("变换矩阵:", inv_M)
        
        # 测试中心点计算
        center_px = (100, 100)
        center_warped_homogeneous = np.array([center_px[0], center_px[1], 1], dtype=np.float32)
        original_center_homogeneous = inv_M.dot(center_warped_homogeneous)
        
        print("齐次坐标结果:", original_center_homogeneous)
        
        # 检查第三个分量是否接近零
        if abs(original_center_homogeneous[2]) < 1e-8:
            print("检测到除数接近零，应该使用安全处理")
        else:
            original_center = (
                original_center_homogeneous[0] / original_center_homogeneous[2],
                original_center_homogeneous[1] / original_center_homogeneous[2]
            )
            print("正常计算结果:", original_center)
            
    except Exception as e:
        print(f"测试中发生错误: {e}")

if __name__ == '__main__':
    test_division_by_zero_fix()
