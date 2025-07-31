#!/bin/bash

# A4纸跟踪系统启动脚本
# 用于开机自启动或SSH远程启动

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置环境变量
export DISPLAY=""  # 禁用X11转发
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 日志文件
LOG_FILE="/tmp/a4_tracker_$(date +%Y%m%d_%H%M%S).log"

echo "A4纸跟踪系统启动中..."
echo "脚本目录: $SCRIPT_DIR"
echo "日志文件: $LOG_FILE"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 检查必要的Python模块
python3 -c "import cv2, numpy, serial" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python模块 (cv2, numpy, serial)"
    echo "请运行: pip3 install opencv-python numpy pyserial"
    exit 1
fi

# 检查摄像头
if [ ! -e /dev/video0 ]; then
    echo "警告: 未检测到摄像头设备 /dev/video0"
fi

# 检查串口设备 (可选)
SERIAL_DEVICE="/dev/serial/by-id/usb-ATK_ATK-HSWL-CMSIS-DAP_ATK_20190528-if00"
if [ ! -e "$SERIAL_DEVICE" ]; then
    echo "警告: 未检测到串口设备 $SERIAL_DEVICE"
fi

# 启动系统
echo "启动A4纸跟踪系统 (无头模式)..."
python3 headless_tracker.py --log-file "$LOG_FILE" "$@"

EXIT_CODE=$?
echo "系统退出，退出码: $EXIT_CODE"
exit $EXIT_CODE
