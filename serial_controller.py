import serial
    def initialize_serial(self):
        """初始化串口连接"""
        if not self.is_enabled:
            print("串口功能已禁用")
            return
            
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
            )
            print("串口已成功连接")
        except Exception as e:
            print(f"串口连接失败: {e}")
            print("程序将在无串口模式下运行")
            self.is_enabled = False
            self.ser = Nonedynamic_config import config

class SerialController:
    """串口控制器"""
    
    def __init__(self, port=None, baudrate=None):
        self.ser = None
        self.port = port or config.SERIAL_PORT
        self.baudrate = baudrate or config.SERIAL_BAUDRATE
        self.is_enabled = config.ENABLE_SERIAL
        self.initialize_serial()
    
    def initialize_serial(self, port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
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

    # 读取串口数据
    def read(self, size=1):
        """从串口读取指定字节数的数据"""
        if self.is_enabled and self.ser is not None:
            try:
                return self.ser.read(size)
            except Exception as e:
                print(f"串口读取错误: {e}")
                return None
        return None
    
    def read_line(self, timeout=0):
        """读取以\\n结尾的一整行数据
        
        Args:
            timeout (float): 超时时间，单位：秒。默认1.0秒
            
        Returns:
            str: 读取到的行数据（已去除换行符），失败返回None
        """
        if not (self.is_enabled and self.ser is not None):
            return None
            
        try:
            # 设置临时超时
            original_timeout = self.ser.timeout
            self.ser.timeout = timeout
            
            # 读取一行数据
            line = self.ser.readline()
            
            # 恢复原始超时设置
            self.ser.timeout = original_timeout
            
            if line:
                # 解码并去除换行符
                return line.decode('utf-8', errors='ignore').strip()
            return None
            
        except Exception as e:
            print(f"串口读行错误: {e}")
            return None
    
    def read_until_newline(self, max_length=1024):
        """逐字节读取直到遇到\\n"""
        if not (self.is_enabled and self.ser is not None):
            return None
            
        try:
            buffer = b''
            while len(buffer) < max_length:
                byte = self.ser.read(1)
                if not byte:
                    break
                buffer += byte
                if byte == b'\n':
                    break
            
            if buffer:
                return buffer.decode('utf-8', errors='ignore').strip()
            return None
            
        except Exception as e:
            print(f"串口读取错误: {e}")
            return None
    
    def in_waiting(self):
        """返回输入缓冲区中等待的字节数"""
        if self.is_enabled and self.ser is not None:
            try:
                return self.ser.in_waiting
            except Exception as e:
                print(f"检查串口缓冲区错误: {e}")
                return 0
        return 0
    
    def set_timeout(self, timeout):
        """设置串口超时时间
        
        Args:
            timeout (float): 超时时间，单位：秒。None表示无限等待，0表示非阻塞
        """
        if self.is_enabled and self.ser is not None:
            try:
                self.ser.timeout = timeout
            except Exception as e:
                print(f"设置串口超时错误: {e}")
    
    def flush_input(self):
        """清空输入缓冲区"""
        if self.is_enabled and self.ser is not None:
            try:
                self.ser.reset_input_buffer()
            except Exception as e:
                print(f"清空输入缓冲区错误: {e}")
    
    def flush_output(self):
        """清空输出缓冲区"""
        if self.is_enabled and self.ser is not None:
            try:
                self.ser.reset_output_buffer()
            except Exception as e:
                print(f"清空输出缓冲区错误: {e}")
