# 模块化A4纸跟踪系统

这是一个模块化重构的A4纸检测和跟踪系统，保留了原有的所有功能，同时提供了更好的代码组织和参数调整接口。

## 文件结构

```
├── config.py                     # 静态配置文件 - 所有可调参数
├── dynamic_config.py             # 动态配置系统 - 支持运行时修改
├── serial_controller.py          # 串口控制模块
├── distance_offset_calculator.py # 距离偏移计算模块
├── distance_calculator.py        # 距离计算模块
├── a4_detector.py                # A4纸检测模块
├── display_manager.py            # 显示管理模块
├── main_modular.py               # 主程序文件
├── example_usage.py              # 使用示例
├── runtime_config_example.py     # 运行时配置示例
└── basic2.py                     # 原始文件（保留）
```

## 模块说明

### 1. config.py
包含所有可调参数：
- A4纸尺寸参数
- 检测参数（阈值、形态学操作等）
- 摄像头校准参数
- 串口配置
- 跟踪参数
- 距离范围设置
- 校准点数据

### 2. serial_controller.py
串口通信控制：
- 自动初始化串口连接
- 安全的数据发送和接收
- 支持读取完整行数据（以\n结尾）
- 连接状态管理
- 错误处理

### 3. distance_offset_calculator.py
屏幕中心偏移计算：
- 基于距离的多项式拟合
- 动态偏移计算
- 校准点验证
- 支持运行时更新校准数据

### 4. distance_calculator.py
距离测量：
- 基于A4纸宽度的距离计算
- 距离历史记录和平均
- 统计信息计算

### 5. a4_detector.py
A4纸检测核心：
- 图像预处理和轮廓检测
- A4纸验证（白纸黑边）
- 透视变换
- 圆形绘制和投影
- 对齐状态检测

### 6. display_manager.py
显示管理：
- 所有信息的绘制和显示
- 键盘输入处理
- 窗口管理

### 7. main_modular.py
主程序：
- 系统集成和协调
- 主循环控制
- 参数管理接口

## 使用方法

### 基本使用
```python
from main_modular import A4TrackingSystem

system = A4TrackingSystem()
system.run()
```

### 带参数调整
```python
from basic2 import A4TrackingSystem, SystemParameterManager

# 调整检测参数
SystemParameterManager.update_detection_parameters(
    mean_inner_val=105,
    mean_border_val=75
)

# 调整跟踪参数
SystemParameterManager.update_tracking_parameters(
    alignment_threshold=8,
    track_count_threshold=5
)

system = A4TrackingSystem()
system.run()
```

### 自定义校准
```python
system = A4TrackingSystem()

# 更新校准点
new_points = [
    (600, -3, 0),
    (1000, 12, -3),
    (1300, 19, -5)
]
system.offset_calculator.update_calibration_points(new_points)

system.run()
```

## 参数调整接口

### SystemParameterManager类提供以下方法：

1. `update_detection_parameters()` - 更新检测阈值
2. `update_camera_parameters()` - 更新摄像头参数
3. `update_tracking_parameters()` - 更新跟踪参数
4. `update_distance_range()` - 更新距离范围
5. `get_current_parameters()` - 获取当前所有参数

## 主要保留的功能

1. **A4纸检测** - 完整保留原有的检测算法
2. **距离测量** - 基于宽度的距离计算
3. **动态偏移** - 距离相关的屏幕中心偏移
4. **串口通信** - 偏移数据发送和对齐信号
5. **实时显示** - 所有可视化信息
6. **参数调整** - 运行时参数修改能力

## 优势

1. **模块化** - 每个功能独立模块，便于维护
2. **可配置** - 所有参数集中管理，易于调整
3. **可扩展** - 模块间松耦合，便于添加新功能
4. **接口丰富** - 提供多种使用方式和参数调整接口
5. **向后兼容** - 保留原有文件，功能完全一致

## 运行示例

```bash
# 本地运行（带界面）
python main_modular.py

# SSH远程运行（自动无头模式）
python ssh_run.py

# 运行示例程序（包含参数调整示例）
python example_usage.py

# 完整无头模式（可选）
python runtime_config_example.py
```

## 串口通信功能

### 发送数据
系统会自动发送偏移数据：
```
-dx,dy\n  # 负的x偏移，y偏移
```

### 读取数据
支持多种读取方式：
```python
# 读取完整行（推荐）
line = serial_controller.read_line(timeout=1.0)

# 逐字节读取直到\n
data = serial_controller.read_until_newline()

# 检查缓冲区
if serial_controller.in_waiting() > 0:
    data = serial_controller.read_line()
```

### 测试串口通信
```bash
python serial_test.py
```

## SSH/无头模式运行

系统会自动检测SSH环境并禁用图像显示窗口：
- 检测SSH_CLIENT、SSH_TTY环境变量
- 检测DISPLAY环境变量为空
- 自动切换到无头模式，避免显示相关错误

### 快速SSH运行
```bash
python ssh_run.py
```

### 手动禁用显示
在config.py中设置：
```python
ENABLE_DISPLAY = False
```

## 运行时配置修改

### 方法1: 通过SystemParameterManager（推荐）
```python
from main_modular import SystemParameterManager

# 在程序运行前修改
SystemParameterManager.update_detection_parameters(mean_inner_val=105)
SystemParameterManager.update_tracking_parameters(alignment_threshold=8)
```

### 方法2: 使用动态配置系统
```python
from dynamic_config import config

# 运行时直接修改
config.MEAN_INNER_VAL = 105
config.ALIGNMENT_THRESHOLD = 8
config.save_to_file()  # 保存到文件
```

### 方法3: GUI界面实时调整
```bash
python runtime_config_example.py
# 选择GUI模式，可以通过滑块实时调整参数
```

### 方法4: 键盘快捷键调整
```bash
python runtime_config_example.py
# 选择键盘模式，使用数字键1-8调整参数
```

## 键盘操作

### 基本操作
- `q`: 退出程序
- `h`: 清除距离历史记录
- `d`: 显示当前距离和偏移信息

### 运行时参数调整（在runtime_config_example.py中）
- `1/2`: 调整内部亮度阈值 (+/-)
- `3/4`: 调整边框暗度阈值 (+/-)
- `5/6`: 调整对齐阈值 (+/-)
- `7/8`: 调整跟踪计数阈值 (+/-)
- `s`: 保存配置到文件
- `r`: 重置为默认配置
