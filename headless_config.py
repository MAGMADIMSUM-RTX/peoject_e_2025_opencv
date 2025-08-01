# A4纸跟踪系统 - 无头模式配置文件
# 该文件用于无头模式运行的参数配置

# 显示控制 (无头模式配置)
ENABLE_DISPLAY = False          # 禁用所有显示功能
ENABLE_GUI_WINDOWS = False      # 禁用OpenCV窗口
ENABLE_CONSOLE_OUTPUT = True    # 保留控制台输出

# 串口配置
ENABLE_SERIAL = True
SERIAL_PORT = '/dev/serial/by-id/usb-ATK_ATK-HSWL-CMSIS-DAP_ATK_20190528-if00'
SERIAL_BAUDRATE = 115200

# 检测参数 (可根据实际环境调整)
MEAN_INNER_VAL = 100
MEAN_BORDER_VAL = 80

# 跟踪参数
ALIGNMENT_THRESHOLD = 6
TRACK_COUNT_THRESHOLD = 4

# 距离范围
MIN_DISTANCE_MM = 500
MAX_DISTANCE_MM = 1500

# 摄像头参数
CAMERA_PARAMS = {
    "focal_length_mm": 2.8,
    "sensor_width_mm": 5.37,
    "sensor_height_mm": 4.04,
    "calibration_factor": 2.0776785714285713
}

# 校准点数据
CALIBRATION_POINTS = [
    [600, -3, 0],
    [1000, 12, -3],
    [1300, 19, -5]
]

# 系统配置
MAX_DISTANCE_HISTORY = 5
CIRCLE_RADIUS_CM = 6.0
PHYSICAL_WIDTH_CM = 26.0
PHYSICAL_HEIGHT_CM = 18.0

# 图像处理参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = [8, 8]
GAUSSIAN_BLUR_KERNEL = [5, 5]
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_C = 2
CONTOUR_AREA_THRESHOLD = 1000
APPROX_EPSILON_FACTOR = 0.02
MORPHOLOGY_KERNEL_SIZE = [10, 10]

# A4纸尺寸
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
