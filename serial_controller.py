import serial
import time
from config import ENABLE_SERIAL, SERIAL_PORT, SERIAL_BAUDRATE

class SerialController:
    """串口控制器"""
    
    def __init__(self):
        self.ser = None
        self.is_enabled = ENABLE_SERIAL
        self.initialize_serial()
    
    def initialize_serial(self):
        """初始化串口连接"""
        if not self.is_enabled:
            print("串口功能已禁用")
            return
            
        try:
            self.ser = serial.Serial(
                port=SERIAL_PORT,
                baudrate=SERIAL_BAUDRATE,
            )
            print("串口已成功连接")
        except Exception as e:
            print(f"串口连接失败: {e}")
            print("程序将在无串口模式下运行")
            self.is_enabled = False
            self.ser = None
    
    def write(self, data):
        """安全的串口写入函数"""
        if self.is_enabled and self.ser is not None:
            try:
                if isinstance(data, str):
                    data = data.encode()
                self.ser.write(data)
            except Exception as e:
                print(f"串口写入错误: {e}")
    
    def close(self):
        """关闭串口连接"""
        if self.is_enabled and self.ser is not None:
            self.ser.close()
            print("串口已关闭")
    
    def is_connected(self):
        """检查串口是否连接"""
        return self.is_enabled and self.ser is not None
