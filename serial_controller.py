import serial
from dynamic_config import config

class SerialController:
    """串口控制器"""
    
    def __init__(self, port=None, baudrate=None):
        self.ser = None
        self.port = port or config.SERIAL_PORT
        self.baudrate = baudrate or config.SERIAL_BAUDRATE
        self.is_enabled = config.ENABLE_SERIAL
        self.initialize_serial()
    
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
    
    def read_packet_with_terminator(self, timeout=1.0):
        """读取以0xA5 0x5A结束的数据包
        
        Args:
            timeout (float): 超时时间，单位：秒
            
        Returns:
            bytes: 读取到的数据包（不包含结束符），失败返回None
        """
        if not (self.is_enabled and self.ser is not None):
            return None
            
        try:
            # 设置临时超时
            original_timeout = self.ser.timeout
            self.ser.timeout = timeout
            
            buffer = b''
            max_length = 1024  # 最大数据包长度
            
            while len(buffer) < max_length:
                byte = self.ser.read(1)
                if not byte:
                    break  # 超时或无数据
                
                buffer += byte
                
                # 检查是否找到结束符 0xA5 0x5A
                if len(buffer) >= 2 and buffer[-2:] == b'\xA5\x5A':
                    # 找到结束符，返回数据（去除结束符）
                    self.ser.timeout = original_timeout
                    return buffer[:-2]
            
            # 恢复原始超时设置
            self.ser.timeout = original_timeout
            
            # 如果到这里说明没找到完整的数据包
            if buffer:
                print(f"数据包不完整，缓冲区长度: {len(buffer)}")
            return None
            
        except Exception as e:
            print(f"读取数据包错误: {e}")
            return None
    
    def parse_packet_data(self, packet):
        """解析数据包内容
        
        Args:
            packet (bytes): 原始数据包
            
        Returns:
            tuple: (result, message) 
                   result可能是: (x, y)坐标, 'quit', 'ok', 或None
        """
        if not packet:
            return None, "空数据包"
        
        try:
            # 检查特殊命令
            if packet == b'q\x0A':  # "q\n"
                return 'quit', "退出命令"
            elif packet == b'ok\x0A':  # "ok\n"
                return 'ok', "确认命令"
            
            # 尝试解析为坐标数据
            # 检查是否包含换行符
            if b'\x0A' in packet:
                # 去除换行符
                data = packet.replace(b'\x0A', b'')
                
                # 先尝试解析为文本坐标（优先级更高）
                try:
                    text = data.decode('utf-8', errors='ignore')
                    if ',' in text:
                        parts = text.split(',')
                        if len(parts) == 2:
                            x = int(parts[0])
                            y = int(parts[1])
                            return (x, y), "文本坐标解析成功"
                except:
                    pass
                
                # 如果文本解析失败，尝试解析为二进制坐标 (int32, int32)
                if len(data) == 9 and data[4] == 0x2C:  # 包含逗号分隔符
                    import struct
                    try:
                        x = struct.unpack('<i', data[0:4])[0]
                        y = struct.unpack('<i', data[5:9])[0]
                        return (x, y), "二进制坐标解析成功"
                    except:
                        pass
            
            # 如果都不匹配，返回原始数据信息
            hex_data = ' '.join([f'0x{b:02X}' for b in packet])
            return None, f"未知格式数据: {hex_data}"
            
        except Exception as e:
            return None, f"解析异常: {e}"
