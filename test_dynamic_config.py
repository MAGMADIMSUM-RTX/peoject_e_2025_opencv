#!/usr/bin/env python3
"""
测试dynamic_config的功能
"""

from dynamic_config import config

def test_config():
    """测试配置功能"""
    print("=== Dynamic Config 测试 ===")
    
    # 测试读取配置
    print(f"当前检测参数:")
    print(f"  MEAN_INNER_VAL: {config.MEAN_INNER_VAL}")
    print(f"  MEAN_BORDER_VAL: {config.MEAN_BORDER_VAL}")
    
    print(f"\n串口配置:")
    print(f"  ENABLE_SERIAL: {config.ENABLE_SERIAL}")
    print(f"  SERIAL_PORT: {config.SERIAL_PORT}")
    print(f"  SERIAL_BAUDRATE: {config.SERIAL_BAUDRATE}")
    
    print(f"\nHMI配置:")
    print(f"  ENABLE_HMI: {config.ENABLE_HMI}")
    print(f"  HMI_PORT: {config.HMI_PORT}")
    print(f"  HMI_BAUDRATE: {config.HMI_BAUDRATE}")
    
    print(f"\n显示配置:")
    print(f"  ENABLE_DISPLAY: {config.ENABLE_DISPLAY}")
    
    # 测试修改配置
    print("\n=== 测试配置修改 ===")
    original_inner_val = config.MEAN_INNER_VAL
    config.MEAN_INNER_VAL = 120
    print(f"修改后 MEAN_INNER_VAL: {config.MEAN_INNER_VAL}")
    
    # 测试保存配置
    print("\n=== 测试配置保存 ===")
    config.save_to_file()
    
    # 恢复原值
    config.MEAN_INNER_VAL = original_inner_val
    print(f"恢复 MEAN_INNER_VAL: {config.MEAN_INNER_VAL}")
    
    # 测试获取所有配置
    print("\n=== 所有配置 ===")
    all_config = config.get_all()
    for key, value in all_config.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

if __name__ == '__main__':
    test_config()
