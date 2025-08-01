#!/usr/bin/env python3
"""
主控制器 - 通过HMI串口读取指令并执行相应的脚本
"""

import subprocess
import sys
import time
import signal
from pathlib import Path
from serial_controller import SerialController
from dynamic_config import config

class MainController:
    """主控制器类"""
    
    def __init__(self):
        self.running = True
        self.hmi = None
        self.current_process = None
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # 支持的脚本映射
        self.script_map = {
            'basic2': 'basic2.py',
            'basic3': 'basic3.py',
            'basic2222': 'basic2222.py',
            'basic2_old': 'basic2_old.py',
            'basic3_old': 'basic3_old.py',
            'main_modular': 'main_modular.py',
            'distance': 'distance.py',
            'bw_tool': 'BW_tool.py',
            'rectangle_detect': 'retangle_detect.py',
            'circle_draw': 'circle_draw.py',
            'hmi_monitor': 'hmi_monitor.py',
            'serial_monitor': 'serial_monitor.py',
            'config_tool': 'config_tool.py',
        }
    
    def signal_handler(self, signum, frame):
        """处理信号"""
        print(f"\n收到信号 {signum}，正在退出...")
        self.running = False
    
    def initialize_hmi(self):
        """初始化HMI串口连接"""
        print("初始化HMI串口连接...")
        print(f"HMI端口: {config.HMI_PORT}")
        print(f"波特率: {config.HMI_BAUDRATE}")
        
        try:
            self.hmi = SerialController(config.HMI_PORT, config.HMI_BAUDRATE)
            if self.hmi.is_connected():
                print("HMI串口连接成功!")
                return True
            else:
                print("HMI串口连接失败!")
                return False
        except Exception as e:
            print(f"初始化HMI串口时出错: {e}")
            return False
    
    def read_command(self, timeout=1.0):
        """从HMI串口读取指令
        
        Args:
            timeout: 读取超时时间（秒）
            
        Returns:
            str: 读取到的指令，如果没有则返回None
        """
        if not self.hmi or not self.hmi.is_connected():
            return None
        
        try:
            # 读取新格式的数据包: AA + 指令内容 + A5 5A
            packet = self.read_command_packet(timeout=timeout)
            if packet:
                # 解析数据包内容
                command = self.parse_command_packet(packet)
                if command:
                    print(f"收到指令: '{command}'")
                    return command
            
            # 兼容旧格式：尝试读取以换行符结尾的指令
            command = self.hmi.read_line(timeout=0.1)
            if command:
                command = command.strip()
                if command:
                    print(f"收到指令(旧格式): '{command}'")
                    return command
            
            # 兼容旧格式：尝试读取数据包
            packet = self.hmi.read_packet_with_terminator(timeout=0.1)
            if packet:
                result, message = self.hmi.parse_packet_data(packet)
                print(f"收到数据包(旧格式): {message}")
                
                if result == 'quit':
                    return 'quit'
                elif result == 'ok':
                    return 'ok'
                elif isinstance(result, str):
                    return result
                    
        except Exception as e:
            print(f"读取指令时出错: {e}")
        
        return None
    
    def read_command_packet(self, timeout=1.0):
        """读取新格式的指令数据包: AA + 指令内容 + A5 5A
        
        Args:
            timeout: 读取超时时间（秒）
            
        Returns:
            bytes: 读取到的完整数据包，失败返回None
        """
        if not self.hmi or not self.hmi.is_connected():
            return None
            
        try:
            # 设置临时超时
            original_timeout = self.hmi.ser.timeout
            self.hmi.ser.timeout = timeout
            
            buffer = b''
            max_length = 1024  # 最大数据包长度
            start_found = False
            
            while len(buffer) < max_length:
                byte = self.hmi.ser.read(1)
                if not byte:
                    break  # 超时或无数据
                
                buffer += byte
                
                # 查找开始符 0xAA
                if not start_found:
                    if byte == b'\xAA':
                        start_found = True
                        buffer = b'\xAA'  # 重置缓冲区，只保留开始符
                    else:
                        buffer = b''  # 清空缓冲区，继续寻找开始符
                    continue
                
                # 已找到开始符，查找结束符 0xA5 0x5A
                if len(buffer) >= 3 and buffer[-2:] == b'\xA5\x5A':
                    # 找到完整的数据包
                    self.hmi.ser.timeout = original_timeout
                    return buffer
            
            # 恢复原始超时设置
            self.hmi.ser.timeout = original_timeout
            
            # 如果到这里说明没找到完整的数据包
            if buffer:
                print(f"数据包不完整，缓冲区长度: {len(buffer)}")
                hex_data = ' '.join([f'0x{b:02X}' for b in buffer])
                print(f"缓冲区内容: {hex_data}")
            return None
            
        except Exception as e:
            print(f"读取指令数据包错误: {e}")
            return None
    
    def parse_command_packet(self, packet):
        """解析新格式的指令数据包
        
        Args:
            packet (bytes): 完整的数据包 (AA + 指令内容 + A5 5A)
            
        Returns:
            str: 解析出的指令，失败返回None
        """
        if not packet or len(packet) < 4:
            return None
        
        try:
            # 验证数据包格式
            if packet[0] != 0xAA:
                print(f"错误: 数据包开始符不正确 (0x{packet[0]:02X})")
                return None
            
            if packet[-2:] != b'\xA5\x5A':
                print("错误: 数据包结束符不正确")
                return None
            
            # 提取指令内容（去除开始符和结束符）
            command_bytes = packet[1:-2]
            
            # 显示原始数据用于调试
            hex_data = ' '.join([f'{b:02X}' for b in packet])
            print(f"原始数据包: {hex_data}")
            
            # 解码指令内容
            command = command_bytes.decode('utf-8', errors='ignore').strip()
            
            if command:
                return command
            else:
                print("警告: 解析出的指令为空")
                return None
                
        except Exception as e:
            print(f"解析指令数据包错误: {e}")
            hex_data = ' '.join([f'0x{b:02X}' for b in packet])
            print(f"错误数据包: {hex_data}")
            return None
    
    def execute_script(self, script_name):
        """执行指定的脚本
        
        Args:
            script_name: 脚本名称或文件名
            
        Returns:
            bool: 执行是否成功
        """
        # 检查脚本是否存在于映射中
        if script_name in self.script_map:
            script_file = self.script_map[script_name]
        else:
            # 直接使用提供的文件名
            script_file = script_name
            if not script_file.endswith('.py'):
                script_file += '.py'
        
        # 检查文件是否存在
        script_path = Path(script_file)
        if not script_path.exists():
            print(f"错误: 脚本文件 '{script_file}' 不存在")
            return False
        
        print(f"开始执行脚本: {script_file}")
        print("=" * 50)
        
        try:
            # 方法1: 使用subprocess.run直接继承输出流
            result = subprocess.run(
                [sys.executable, '-u', script_file],
                text=True,
                check=False  # 不在非零退出时抛出异常
            )
            
            return_code = result.returncode
            
            print("=" * 50)
            if return_code == 0:
                print(f"脚本 '{script_file}' 执行完成，返回码: {return_code}")
                return True
            else:
                print(f"脚本 '{script_file}' 执行失败，返回码: {return_code}")
                return False
                
        except KeyboardInterrupt:
            print("\n用户中断了脚本执行")
            return False
            
        except Exception as e:
            print(f"执行脚本时出错: {e}")
            return False
    
    def show_help(self):
        """显示帮助信息"""
        print("\n" + "=" * 60)
        print("主控制器 - 指令帮助")
        print("=" * 60)
        print("支持的指令:")
        print("  help          - 显示此帮助信息")
        print("  quit/exit     - 退出程序")
        print("  list          - 列出所有可用脚本")
        print("  status        - 显示系统状态")
        print("")
        print("支持的脚本:")
        for key, script in self.script_map.items():
            print(f"  {key:<15} - 执行 {script}")
        print("")
        print("也可以直接输入Python文件名（带或不带.py扩展名）")
        print("例如: basic2, basic2.py, test_basic3.py")
        print("=" * 60)
    
    def list_scripts(self):
        """列出所有可用的脚本"""
        print("\n可用脚本列表:")
        print("-" * 40)
        
        # 显示映射中的脚本
        print("快捷指令:")
        for key, script in self.script_map.items():
            script_path = Path(script)
            status = "✓" if script_path.exists() else "✗"
            print(f"  {status} {key:<15} -> {script}")
        
        # 查找目录中的其他Python文件
        print("\n其他Python文件:")
        py_files = list(Path('.').glob('*.py'))
        mapped_files = set(self.script_map.values())
        
        for py_file in sorted(py_files):
            if py_file.name not in mapped_files and py_file.name != 'main.py':
                print(f"  ✓ {py_file.name}")
        print("-" * 40)
    
    def show_status(self):
        """显示系统状态"""
        print("\n系统状态:")
        print("-" * 30)
        print(f"HMI串口状态: {'已连接' if self.hmi and self.hmi.is_connected() else '未连接'}")
        print(f"HMI端口: {config.HMI_PORT}")
        print(f"HMI波特率: {config.HMI_BAUDRATE}")
        print(f"当前进程: {'无' if not self.current_process else '有脚本正在运行'}")
        print("-" * 30)
    
    def run(self):
        """运行主控制循环"""
        print("=" * 60)
        print("主控制器启动")
        print("=" * 60)
        
        # 初始化HMI串口
        if not self.initialize_hmi():
            print("无法初始化HMI串口，程序退出")
            return 1
        
        # 显示帮助信息
        self.show_help()
        
        print(f"\n等待HMI串口指令... (端口: {config.HMI_PORT})")
        print("注意: 指令需要以换行符结尾")
        self.hmi.write(b't0.txt="ready"\xff\xff\xff')
        time.sleep(0.01)  # 等待HMI准备好
        
        try:
            while self.running:
                # 读取指令
                command = self.read_command(timeout=0.5)
                
                if not command:
                    # 没有指令，继续等待
                    time.sleep(0.1)
                    continue
                
                # 处理指令
                command = command.lower().strip()
                
                # if command in ['quit', 'exit', 'q']:
                #     print("收到退出指令，程序结束")
                #     break
                # elif command in ['help', 'h', '?']:
                if command in ['help', 'h', '?']:
                    self.show_help()
                elif command in ['list', 'ls']:
                    self.list_scripts()
                elif command in ['status', 'stat']:
                    self.show_status()
                elif command:
                    # 尝试执行脚本
                    print(f"\n准备执行: {command}")
                    time.sleep(0.01)
                    self.hmi.write(b't0.txt="waiting to start"\xff\xff\xff')
                    time.sleep(0.01)
                    success = self.execute_script(command)
                    
                    if success:
                        print(f"\n脚本 '{command}' 执行完成，继续等待指令...")
                    else:
                        print(f"\n脚本 '{command}' 执行失败，继续等待指令...")
                    
                    print("\n等待下一个指令...")
                    self.hmi.write(b't0.txt="ready"\xff\xff\xff')
                
        except KeyboardInterrupt:
            print("\n用户中断程序")
        except Exception as e:
            print(f"主循环出错: {e}")
        finally:
            # 清理资源
            if self.hmi:
                self.hmi.close()
                print("HMI串口已关闭")
        
        print("主控制器退出")
        return 0

def main():
    """主函数"""
    controller = MainController()
    return controller.run()

if __name__ == '__main__':
    sys.exit(main())
