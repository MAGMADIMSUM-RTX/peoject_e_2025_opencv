#!/usr/bin/env python3
"""
动态修改配置的工具脚本
"""

from dynamic_config import config

def enable_display():
    """启用显示功能"""
    config.ENABLE_DISPLAY = True
    config.save_to_file()
    print(f"显示功能已启用: {config.ENABLE_DISPLAY}")

def disable_display():
    """禁用显示功能"""
    config.ENABLE_DISPLAY = False
    config.save_to_file()
    print(f"显示功能已禁用: {config.ENABLE_DISPLAY}")

def show_display_status():
    """显示当前显示状态"""
    print(f"当前显示状态: {config.ENABLE_DISPLAY}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python config_tool.py enable   # 启用显示")
        print("  python config_tool.py disable  # 禁用显示")
        print("  python config_tool.py status   # 查看状态")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == 'enable':
        enable_display()
    elif action == 'disable':
        disable_display()
    elif action == 'status':
        show_display_status()
    else:
        print(f"未知操作: {action}")
        sys.exit(1)
