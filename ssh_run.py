#!/usr/bin/env python3
"""
SSH模式运行脚本 - 最小修改版本
只添加必要的错误处理，让程序在SSH下不报错
"""

import os
import sys

# 设置环境变量以避免GUI相关错误
os.environ['DISPLAY'] = ''

# 导入主程序
from main_modular import A4TrackingSystem

def main():
    """主函数"""
    print("=== A4纸跟踪系统 (SSH模式) ===")
    print("自动检测SSH环境并禁用显示窗口")
    
    try:
        # 创建系统实例
        tracking_system = A4TrackingSystem()
        
        # 运行系统
        tracking_system.run()
        
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序结束")

if __name__ == '__main__':
    main()
