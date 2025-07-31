# 串口数据监控工具使用说明

## 概述

本工具包提供了多个脚本来监控和测试串口数据，支持二进制和文本两种数据格式。

## 数据格式

### 二进制格式 (10字节)
```
[int32_x][,][int32_y][\n]
 4字节   1字节 4字节   1字节
```
- 整数采用小端序编码
- 分隔符为ASCII逗号 (0x2C)
- 结束符为换行符 (0x0A)

**示例**: `5,-2` 对应
```
0x05 0x00 0x00 0x00 0x2C 0xFE 0xFF 0xFF 0xFF 0x0A
```

### 文本格式
```
"x,y\n"
```

**示例**: `"5,-2\n"`

## 特殊命令

| 命令 | 二进制格式 | 文本格式 | 说明 |
|------|------------|----------|------|
| 退出 | (113, 0) | "q" | 退出程序 |
| 确认 | (111, 107) | "ok" | 收集校准点 |

## 脚本说明

### 1. serial_monitor.py - 完整串口监控器

**功能**: 
- 监控指定串口的数据
- 自动识别二进制和文本格式
- 提供详细的解析信息

**用法**:
```bash
# 监控默认HMI串口
python3 serial_monitor.py monitor

# 监控指定串口
python3 serial_monitor.py monitor /dev/ttyUSB0 115200

# 监控30秒后自动停止
python3 serial_monitor.py monitor /dev/ttyUSB0 115200 30

# 测试数据包解析
python3 serial_monitor.py test
```

### 2. hmi_monitor.py - 简化HMI监控器

**功能**:
- 专门监控HMI串口
- 实时显示接收到的数据
- 自动识别特殊命令

**用法**:
```bash
python3 hmi_monitor.py
```

**输出示例**:
```
[14:30:15] #0001 二进制数据: (   5,   -2) -> 坐标数据
[14:30:16] #0002 二进制数据: ( 113,    0) -> 退出命令
[14:30:17] #0003 文本数据: 'ok' -> 确认命令
```

### 3. send_test_data.py - 测试数据发送器

**功能**:
- 发送测试数据到串口
- 支持发送特定命令
- 用于测试接收端

**用法**:
```bash
# 发送测试数据序列
python3 send_test_data.py test

# 发送退出命令
python3 send_test_data.py quit

# 发送确认命令  
python3 send_test_data.py ok
```

## 配置

所有脚本使用 `runtime_config.json` 中的串口配置:

```json
{
  "HMI_PORT": "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
  "HMI_BAUDRATE": 115200,
  "SERIAL_PORT": "/dev/serial/by-id/usb-ATK_ATK-HSWL-CMSIS-DAP_ATK_20190528-if00",
  "SERIAL_BAUDRATE": 115200
}
```

## 使用流程

1. **启动监控器**:
   ```bash
   python3 hmi_monitor.py
   ```

2. **在另一个终端发送测试数据** (可选):
   ```bash
   python3 send_test_data.py test
   ```

3. **观察输出**:
   - 监控器会显示接收到的所有数据
   - 自动识别坐标数据和特殊命令
   - 显示时间戳和数据包编号

4. **停止监控**:
   - 按 `Ctrl+C` 停止监控
   - 或发送退出命令: `python3 send_test_data.py quit`

## 故障排除

### 串口连接失败
- 检查串口设备是否存在: `ls -l /dev/serial/by-id/`
- 检查权限: `sudo chmod 666 /dev/ttyUSB*`
- 确认串口未被其他程序占用

### 数据解析失败
- 使用 `python3 serial_monitor.py test` 测试解析功能
- 检查数据包长度是否为10字节
- 验证分隔符和结束符是否正确

### 配置问题
- 检查 `runtime_config.json` 文件
- 确认串口路径和波特率设置正确

## 集成到主程序

在 `basic2.py` 中已经集成了二进制数据包读取功能:

```python
# 读取二进制数据包
binary_data = self.hmi.read_binary_packet(timeout=0)
if binary_data:
    x, y = binary_data
    # 处理坐标数据
```

这确保了主程序能够正确处理新的二进制数据格式。
