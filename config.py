# 配置文件 - 包含所有可调参数

# A4纸的实际尺寸
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# 检测参数
MEAN_INNER_VAL = 100
MEAN_BORDER_VAL = 80

# 固定的已校准距离测量参数
CAMERA_PARAMS = {
    "focal_length_mm": 2.8,
    "sensor_width_mm": 5.37,
    "sensor_height_mm": 4.04,
    "calibration_factor": 2.0776785714285713
}

# 串口控制宏定义
ENABLE_SERIAL = True  # 设置为 False 可以禁用串口功能
SERIAL_PORT = '/dev/serial/by-id/usb-ATK_ATK-HSWL-CMSIS-DAP_ATK_20190528-if00'
SERIAL_BAUDRATE = 115200

# 跟踪参数
ALIGNMENT_THRESHOLD = 6  # 对齐阈值
TRACK_COUNT_THRESHOLD = 4  # 跟踪计数阈值

# 距离计算参数
MAX_DISTANCE_HISTORY = 5  # 距离历史记录最大长度
MIN_DISTANCE_MM = 500     # 最小距离限制
MAX_DISTANCE_MM = 1500    # 最大距离限制

# 圆形绘制参数
CIRCLE_RADIUS_CM = 6.0    # 圆形半径（厘米）
PHYSICAL_WIDTH_CM = 26.0  # A4纸物理宽度（厘米）
PHYSICAL_HEIGHT_CM = 18.0 # A4纸物理高度（厘米）

# 图像处理参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_C = 2
CONTOUR_AREA_THRESHOLD = 1000
APPROX_EPSILON_FACTOR = 0.02

# 形态学操作参数
MORPHOLOGY_KERNEL_SIZE = (10, 10)

# 屏幕中心偏移校准点
CALIBRATION_POINTS = [
    (600, -3, 0),    # 600mm: offset_x=-3, offset_y=0
    (1000, 12, -3),  # 1000mm: offset_x=12, offset_y=-3
    (1300, 19, -5)   # 1300mm: offset_x=19, offset_y=-5
]
