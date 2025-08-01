# 主控制器使用说明

## 概述

`main.py` 是一个主控制器，通过 HMI 串口读取指令，然后阻塞执行相应的 Python 脚本。当脚本执行完成后，主控制器会继续等待下一个指令。

## 功能特点

- 🔌 **HMI串口通信**: 通过HMI串口接收指令
- 🚀 **脚本执行管理**: 阻塞执行其他Python脚本
- 🛑 **进程控制**: 支持中断和终止正在运行的脚本
- 📋 **指令映射**: 预定义的脚本快捷指令
- 📊 **实时输出**: 显示被执行脚本的实时输出
- 🔄 **循环运行**: 脚本执行完成后自动返回等待状态

## 启动方式

### 方法1: 直接运行
```bash
python3 main.py
```

### 方法2: 使用启动脚本
```bash
./start_main.sh
```

## 支持的指令

### 系统控制指令
- `help` / `h` / `?` - 显示帮助信息
- `quit` / `exit` / `q` - 退出程序
- `list` / `ls` - 列出所有可用脚本
- `status` / `stat` - 显示系统状态

### 预定义脚本快捷指令
- `basic2` - 执行 basic2.py
- `basic3` - 执行 basic3.py
- `basic2222` - 执行 basic2222.py
- `basic2_old` - 执行 basic2_old.py
- `basic3_old` - 执行 basic3_old.py
- `main_modular` - 执行 main_modular.py
- `distance` - 执行 distance.py
- `bw_tool` - 执行 BW_tool.py
- `rectangle_detect` - 执行 retangle_detect.py
- `circle_draw` - 执行 circle_draw.py
- `hmi_monitor` - 执行 hmi_monitor.py
- `serial_monitor` - 执行 serial_monitor.py
- `config_tool` - 执行 config_tool.py

### 直接文件名
也可以直接输入任何Python文件名（带或不带 .py 扩展名）：
- `test_script` - 执行 test_script.py
- `example_usage.py` - 执行 example_usage.py

## 串口指令格式

指令需要通过HMI串口发送，支持以下格式：

1. **文本指令** (推荐)
   ```
   basic2\n
   help\n
   quit\n
   ```

2. **数据包格式**
   ```
   指令内容 + 0xA5 0x5A (结束符)
   ```

## 配置要求

确保 `runtime_config.json` 或 `dynamic_config.py` 中的HMI配置正确：

```json
{
  "ENABLE_HMI": true,
  "HMI_PORT": "/dev/serial/by-id/usb-Arm_DAPLink_CMSIS-DAP_52D9DC36C41C91B341B13A63F8731D01-if01",
  "HMI_BAUDRATE": 115200
}
```

## 使用流程

1. **启动主控制器**
   ```bash
   python3 main.py
   ```

2. **等待串口连接**
   - 程序会自动连接HMI串口
   - 显示连接状态和配置信息

3. **发送指令**
   - 通过HMI串口发送指令
   - 指令必须以换行符结尾

4. **脚本执行**
   - 主控制器接收指令后开始执行对应脚本
   - 实时显示脚本输出
   - 主控制器进入阻塞状态，直到脚本完成

5. **返回等待状态**
   - 脚本执行完成后，主控制器继续等待下一个指令

## 示例会话

```
============================================================
主控制器启动
============================================================
初始化HMI串口连接...
HMI端口: /dev/serial/by-id/usb-Arm_DAPLink_CMSIS-DAP_52D9DC36C41C91B341B13A63F8731D01-if01
波特率: 115200
HMI串口连接成功!

等待HMI串口指令... (端口: /dev/serial/by-id/...)
注意: 指令需要以换行符结尾

收到指令: 'basic2'

准备执行: basic2
开始执行脚本: basic2.py
==================================================
[basic2.py的输出内容...]
==================================================
脚本 'basic2.py' 执行完成，返回码: 0

脚本 'basic2' 执行完成，继续等待指令...

等待下一个指令...
```

## 错误处理

- **串口连接失败**: 检查HMI串口配置和硬件连接
- **脚本不存在**: 确认脚本文件存在于当前目录
- **脚本执行失败**: 查看脚本的错误输出
- **进程中断**: 支持 Ctrl+C 中断当前执行的脚本

## 信号处理

- **SIGINT (Ctrl+C)**: 优雅退出，终止当前执行的脚本
- **SIGTERM**: 系统终止信号处理

## 测试

使用提供的测试脚本验证功能：

1. 启动主控制器
2. 发送指令: `test_script`
3. 观察测试脚本的执行过程
4. 脚本完成后返回等待状态

## 故障排除

### 常见问题

1. **HMI串口连接失败**
   - 检查串口设备是否存在
   - 确认串口权限
   - 验证波特率设置

2. **指令无响应**
   - 确认指令格式正确
   - 检查是否包含换行符
   - 验证串口数据传输

3. **脚本执行异常**
   - 查看脚本依赖是否满足
   - 检查Python环境
   - 确认文件权限

### 调试模式

可以使用 `hmi_monitor.py` 监控HMI串口数据：
```bash
python3 hmi_monitor.py
```

这将显示所有接收到的原始数据，帮助调试指令格式问题。
