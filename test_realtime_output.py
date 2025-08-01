#!/usr/bin/env python3
"""
实时输出测试脚本
用于测试main.py的实时输出显示功能
"""

import time
import sys

def main():
    print("=== 实时输出测试开始 ===", flush=True)
    
    for i in range(10):
        print(f"[{time.strftime('%H:%M:%S')}] 测试输出 {i+1}/10", flush=True)
        time.sleep(1)
    
    print("模拟一些错误输出...", file=sys.stderr, flush=True)
    print("错误: 这是一个测试错误", file=sys.stderr, flush=True)
    
    print("=== 实时输出测试完成 ===", flush=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
