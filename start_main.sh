#!/bin/bash
# 主控制器启动脚本

echo "==========================================="
echo "主控制器启动脚本"
echo "==========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查必要文件
if [ ! -f "main.py" ]; then
    echo "错误: 未找到 main.py"
    exit 1
fi

if [ ! -f "dynamic_config.py" ]; then
    echo "错误: 未找到 dynamic_config.py"
    exit 1
fi

if [ ! -f "serial_controller.py" ]; then
    echo "错误: 未找到 serial_controller.py"
    exit 1
fi

echo "文件检查完成，启动主控制器..."
echo ""

# 启动主控制器
python3 main.py

echo ""
echo "主控制器已退出"
