#!/usr/bin/env python3
"""
模块化A4纸跟踪系统使用示例
展示如何使用不同的接口和参数调整功能
"""

from main_modular import A4TrackingSystem, SystemParameterManager

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    system = A4TrackingSystem()
    system.run()

def example_with_parameter_adjustment():
    """带参数调整的使用示例"""
    print("=== 带参数调整的使用示例 ===")
    
    # 在运行前调整参数
    SystemParameterManager.update_detection_parameters(
        mean_inner_val=105,  # 稍微提高内部亮度阈值
        mean_border_val=75   # 稍微降低边框暗度阈值
    )
    
    SystemParameterManager.update_tracking_parameters(
        alignment_threshold=8,      # 稍微放宽对齐阈值
        track_count_threshold=5     # 增加跟踪确认次数
    )
    
    SystemParameterManager.update_distance_range(
        min_distance=400,   # 允许更近的距离
        max_distance=2000   # 允许更远的距离
    )
    
    # 显示当前参数
    params = SystemParameterManager.get_current_parameters()
    print("当前参数设置:")
    for category, values in params.items():
        print(f"  {category}: {values}")
    
    # 运行系统
    system = A4TrackingSystem()
    system.run()

def example_custom_calibration():
    """自定义校准点示例"""
    print("=== 自定义校准点示例 ===")
    
    # 创建系统
    system = A4TrackingSystem()
    
    # 更新校准点（如果你有新的校准数据）
    new_calibration_points = [
        (500, -5, 2),    # 500mm: offset_x=-5, offset_y=2
        (800, 0, 0),     # 800mm: offset_x=0, offset_y=0
        (1200, 15, -4),  # 1200mm: offset_x=15, offset_y=-4
        (1600, 25, -8)   # 1600mm: offset_x=25, offset_y=-8
    ]
    
    system.offset_calculator.update_calibration_points(new_calibration_points)
    
    # 运行系统
    system.run()

def main():
    """主函数 - 选择运行哪个示例"""
    print("请选择运行示例:")
    print("1. 基本使用")
    print("2. 带参数调整")
    print("3. 自定义校准点")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_with_parameter_adjustment()
    elif choice == "3":
        example_custom_calibration()
    else:
        print("无效选择，运行基本示例")
        example_basic_usage()

if __name__ == '__main__':
    main()
