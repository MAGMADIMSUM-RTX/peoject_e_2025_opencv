# 动态配置系统 - 支持运行时修改参数

import json
import os
from typing import Dict, Any, Optional

class DynamicConfig:
    """动态配置管理器 - 支持运行时修改和持久化"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "runtime_config.json"
        self._config = {}
        self._load_default_config()
        self._load_from_file()
    
    def _load_default_config(self):
        """加载默认配置"""
        self._config = {
            # A4纸的实际尺寸
            "A4_WIDTH_MM": 210.0,
            "A4_HEIGHT_MM": 297.0,
            
            # 检测参数
            "MEAN_INNER_VAL": 100,
            "MEAN_BORDER_VAL": 80,
            
            # 摄像头校准参数
            "CAMERA_PARAMS": {
                "focal_length_mm": 2.8,
                "sensor_width_mm": 5.37,
                "sensor_height_mm": 4.04,
                "calibration_factor": 2.0776785714285713
            },
            
            # 串口配置
            "ENABLE_SERIAL": True,
            "SERIAL_PORT": '/dev/serial/by-id/usb-ATK_ATK-HSWL-CMSIS-DAP_ATK_20190528-if00',
            "SERIAL_BAUDRATE": 115200,
            
            # HMI串口配置
            "ENABLE_HMI": True,
            "HMI_PORT": '/dev/serial0',
            "HMI_BAUDRATE": 115200,
            
            # 显示控制配置
            "ENABLE_DISPLAY": False,
            
            # 跟踪参数
            "ALIGNMENT_THRESHOLD": 6,
            "TRACK_COUNT_THRESHOLD": 4,
            
            # 距离计算参数
            "MAX_DISTANCE_HISTORY": 5,
            "MIN_DISTANCE_MM": 500,
            "MAX_DISTANCE_MM": 1500,
            
            # 圆形绘制参数
            "CIRCLE_RADIUS_CM": 6.0,
            "PHYSICAL_WIDTH_CM": 26.0,
            "PHYSICAL_HEIGHT_CM": 18.0,
            
            # 图像处理参数
            "CLAHE_CLIP_LIMIT": 2.0,
            "CLAHE_TILE_GRID_SIZE": [8, 8],
            "GAUSSIAN_BLUR_KERNEL": [5, 5],
            "ADAPTIVE_THRESH_BLOCK_SIZE": 11,
            "ADAPTIVE_THRESH_C": 2,
            "CONTOUR_AREA_THRESHOLD": 1000,
            "APPROX_EPSILON_FACTOR": 0.02,
            
            # 形态学操作参数
            "MORPHOLOGY_KERNEL_SIZE": [10, 10],
            
            # 屏幕中心偏移校准点
            "CALIBRATION_POINTS": [
                [600, -3, 0],
                [1000, 12, -3],
                [1300, 19, -5]
            ]
        }
    
    def _load_from_file(self):
        """从文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
                print(f"已从 {self.config_file} 加载配置")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
    
    def save_to_file(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get(self, key: str, default=None):
        """获取配置值"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """批量更新配置"""
        self._config.update(config_dict)
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()
    
    def reset_to_default(self):
        """重置为默认配置"""
        self._load_default_config()
    
    # 便捷属性访问
    @property
    def MEAN_INNER_VAL(self):
        return self.get("MEAN_INNER_VAL")
    
    @MEAN_INNER_VAL.setter
    def MEAN_INNER_VAL(self, value):
        self.set("MEAN_INNER_VAL", value)
    
    @property
    def MEAN_BORDER_VAL(self):
        return self.get("MEAN_BORDER_VAL")
    
    @MEAN_BORDER_VAL.setter
    def MEAN_BORDER_VAL(self, value):
        self.set("MEAN_BORDER_VAL", value)
    
    @property
    def CAMERA_PARAMS(self):
        return self.get("CAMERA_PARAMS")
    
    @CAMERA_PARAMS.setter
    def CAMERA_PARAMS(self, value):
        self.set("CAMERA_PARAMS", value)
    
    @property
    def ALIGNMENT_THRESHOLD(self):
        return self.get("ALIGNMENT_THRESHOLD")
    
    @ALIGNMENT_THRESHOLD.setter
    def ALIGNMENT_THRESHOLD(self, value):
        self.set("ALIGNMENT_THRESHOLD", value)
    
    @property
    def TRACK_COUNT_THRESHOLD(self):
        return self.get("TRACK_COUNT_THRESHOLD")
    
    @TRACK_COUNT_THRESHOLD.setter
    def TRACK_COUNT_THRESHOLD(self, value):
        self.set("TRACK_COUNT_THRESHOLD", value)
    
    @property
    def CALIBRATION_POINTS(self):
        return self.get("CALIBRATION_POINTS")
    
    @CALIBRATION_POINTS.setter
    def CALIBRATION_POINTS(self, value):
        self.set("CALIBRATION_POINTS", value)
    
    # 添加更多属性...
    def __getattr__(self, name):
        """动态属性访问"""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """动态属性设置"""
        if name.startswith('_') or name in ['config_file']:
            super().__setattr__(name, value)
        else:
            if hasattr(self, '_config'):
                self._config[name] = value
            else:
                super().__setattr__(name, value)

# 全局配置实例
config = DynamicConfig()

# 为了向后兼容，导出所有配置变量
def _update_globals():
    """更新全局变量以保持向后兼容性"""
    globals().update(config.get_all())

_update_globals()

# 提供更新全局变量的函数
def refresh_globals():
    """刷新全局变量（在配置更改后调用）"""
    _update_globals()
