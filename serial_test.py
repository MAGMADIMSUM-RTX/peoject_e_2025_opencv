#!/usr/bin/env python3
"""
串口通信示例 - 展示如何读取以\n结尾的完整数据
"""

import time
from serial_controller import SerialController

def test_serial_read():
    """测试串口读取功能"""
    print("=== 串口读取测试 ===")
    
    # 创建串口控制器
    serial_ctrl = SerialController()
    
    if not serial_ctrl.is_connected():
        print("串口未连接，无法进行测试")
        return
    
    print("串口已连接，开始测试...")
    print("支持的读取方法:")
    print("1. read_line() - 使用pyserial的readline()方法")
    print("2. read_until_newline() - 逐字节读取直到\\n")
    print("3. 检查缓冲区和读取完整数据")
    
    # 设置超时
    serial_ctrl.set_timeout(0.5)
    
    try:
        while True:
            # 方法1: 使用readline()
            print("\n--- 使用 read_line() 方法 ---")
            line = serial_ctrl.read_line(timeout=1.0)
            if line:
                print(f"读取到行数据: '{line}'")
            else:
                print("未读取到数据")
            
            # 方法2: 检查缓冲区
            print("\n--- 检查缓冲区 ---")
            waiting = serial_ctrl.in_waiting()
            print(f"缓冲区中有 {waiting} 字节数据")
            
            if waiting > 0:
                # 方法3: 逐字节读取
                print("使用 read_until_newline() 方法")
                data = serial_ctrl.read_until_newline()
                if data:
                    print(f"读取到数据: '{data}'")
            
            # 发送测试数据
            test_message = f"Test message {int(time.time())}\n"
            print(f"发送测试数据: '{test_message.strip()}'")
            serial_ctrl.write(test_message)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n测试中断")
    
    finally:
        serial_ctrl.close()
        print("串口已关闭")

def interactive_serial_communication():
    """交互式串口通信"""
    print("=== 交互式串口通信 ===")
    
    serial_ctrl = SerialController()
    
    if not serial_ctrl.is_connected():
        print("串口未连接")
        return
    
    print("串口已连接")
    print("输入消息并按回车发送，输入'quit'退出")
    print("程序会显示接收到的完整行数据")
    
    # 设置较短的超时以便快速响应
    serial_ctrl.set_timeout(0.1)
    
    try:
        while True:
            # 检查是否有输入数据
            if serial_ctrl.in_waiting() > 0:
                received_line = serial_ctrl.read_line(timeout=0.5)
                if received_line:
                    print(f"接收: {received_line}")
            
            # 检查用户输入（非阻塞）
            try:
                import select
                import sys
                
                if select.select([sys.stdin], [], [], 0)[0]:
                    user_input = input()
                    if user_input.lower() == 'quit':
                        break
                    
                    # 发送数据，确保以\n结尾
                    message = user_input + '\n'
                    serial_ctrl.write(message)
                    print(f"发送: {user_input}")
            
            except ImportError:
                # Windows系统不支持select，使用简单的input
                try:
                    user_input = input("输入消息 (或'quit'退出): ")
                    if user_input.lower() == 'quit':
                        break
                    
                    message = user_input + '\n'
                    serial_ctrl.write(message)
                    print(f"发送: {user_input}")
                    
                    # 等待响应
                    time.sleep(0.5)
                    if serial_ctrl.in_waiting() > 0:
                        received_line = serial_ctrl.read_line(timeout=1.0)
                        if received_line:
                            print(f"接收: {received_line}")
                
                except EOFError:
                    break
            
            time.sleep(0.01)  # 避免CPU占用过高
    
    except KeyboardInterrupt:
        print("\n通信中断")
    
    finally:
        serial_ctrl.close()
        print("串口已关闭")

def main():
    """主函数"""
    print("串口通信测试程序")
    print("1. 基本读取测试")
    print("2. 交互式通信")
    
    choice = input("请选择测试模式 (1/2): ").strip()
    
    if choice == "1":
        test_serial_read()
    elif choice == "2":
        interactive_serial_communication()
    else:
        print("无效选择")

if __name__ == '__main__':
    main()
