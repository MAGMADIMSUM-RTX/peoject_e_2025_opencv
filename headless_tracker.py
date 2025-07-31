#!/usr/bin/env python3
"""
A4纸跟踪系统 - 无头模式运行脚本
适用于SSH连接、开机自启动等无显示环境
"""

import sys
import time
import signal
import logging
import os
from datetime import datetime

# 导入模块化组件
from dynamic_config import config
from main_modular import A4TrackingSystem, SystemParameterManager

class HeadlessA4TrackingSystem:
    """无头模式A4纸跟踪系统"""
    
    def __init__(self, log_file=None):
        # 设置无头模式
        config.ENABLE_DISPLAY = False
        
        self.tracking_system = None
        self.running = True
        self.log_file = log_file
        
        # 设置日志
        self.setup_logging()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """设置日志记录"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        if self.log_file:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[
                    logging.FileHandler(self.log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format=log_format,
                handlers=[logging.StreamHandler(sys.stdout)]
            )
        
        self.logger = logging.getLogger(__name__)
    
    def signal_handler(self, signum, frame):
        """处理退出信号"""
        self.logger.info(f"收到信号 {signum}, 正在优雅退出...")
        self.running = False
    
    def run(self):
        """运行系统"""
        self.logger.info("=== A4纸跟踪系统启动 (无头模式) ===")
        self.logger.info(f"PID: {os.getpid()}")
        self.logger.info(f"工作目录: {os.getcwd()}")
        self.logger.info(f"显示模式: 禁用")
        self.logger.info(f"串口模式: {'启用' if config.ENABLE_SERIAL else '禁用'}")
        
        try:
            # 创建跟踪系统
            self.tracking_system = A4TrackingSystem()
            
            # 初始化摄像头
            if not self.tracking_system.initialize_camera():
                self.logger.error("摄像头初始化失败")
                return False
            
            self.logger.info("摄像头初始化成功")
            self.logger.info("系统开始运行...")
            
            frame_count = 0
            last_status_time = time.time()
            
            while self.running:
                ret, frame = self.tracking_system.cap.read()
                if not ret:
                    self.logger.warning("无法读取摄像头帧，尝试重新连接...")
                    time.sleep(1)
                    continue
                
                # 处理帧
                try:
                    processed_frame, warped_image, distance, avg_distance = \
                        self.tracking_system.process_frame(frame)
                    
                    frame_count += 1
                    
                    # 定期输出状态
                    current_time = time.time()
                    if current_time - last_status_time >= 10:  # 每10秒输出一次
                        self.log_status(frame_count, avg_distance)
                        last_status_time = current_time
                        frame_count = 0
                    
                except Exception as e:
                    self.logger.error(f"处理帧时出错: {e}")
                    time.sleep(0.1)
                    continue
                
                # 防止CPU占用过高
                time.sleep(0.01)
            
            self.logger.info("系统正常退出")
            return True
            
        except Exception as e:
            self.logger.error(f"系统运行错误: {e}")
            return False
        
        finally:
            if self.tracking_system:
                self.tracking_system.cleanup()
    
    def log_status(self, frame_count, avg_distance):
        """记录系统状态"""
        serial_status = "已连接" if self.tracking_system.serial_controller.is_connected() else "未连接"
        distance_str = f"{avg_distance:.1f}mm" if avg_distance else "未知"
        
        self.logger.info(f"状态: 处理了{frame_count}帧, 平均距离: {distance_str}, 串口: {serial_status}")

def create_systemd_service():
    """创建systemd服务文件"""
    service_content = f"""[Unit]
Description=A4 Paper Tracking System (Headless)
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'pi')}
WorkingDirectory={os.getcwd()}
ExecStart=/usr/bin/python3 {os.path.join(os.getcwd(), 'headless_tracker.py')}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_file = "/tmp/a4-tracker.service"
    with open(service_file, 'w') as f:
        f.write(service_content)
    
    print(f"服务文件已创建: {service_file}")
    print("\n要安装为系统服务，请运行:")
    print(f"sudo cp {service_file} /etc/systemd/system/")
    print("sudo systemctl enable a4-tracker.service")
    print("sudo systemctl start a4-tracker.service")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='A4纸跟踪系统 - 无头模式')
    parser.add_argument('--log-file', '-l', help='日志文件路径')
    parser.add_argument('--create-service', action='store_true', help='创建systemd服务文件')
    parser.add_argument('--daemon', '-d', action='store_true', help='以守护进程模式运行')
    
    # 参数调整选项
    parser.add_argument('--inner-threshold', type=int, help='内部亮度阈值')
    parser.add_argument('--border-threshold', type=int, help='边框暗度阈值')
    parser.add_argument('--align-threshold', type=int, help='对齐阈值')
    parser.add_argument('--track-count', type=int, help='跟踪计数阈值')
    
    args = parser.parse_args()
    
    if args.create_service:
        create_systemd_service()
        return
    
    # 调整参数
    if args.inner_threshold:
        SystemParameterManager.update_detection_parameters(mean_inner_val=args.inner_threshold)
    if args.border_threshold:
        SystemParameterManager.update_detection_parameters(mean_border_val=args.border_threshold)
    if args.align_threshold:
        SystemParameterManager.update_tracking_parameters(alignment_threshold=args.align_threshold)
    if args.track_count:
        SystemParameterManager.update_tracking_parameters(track_count_threshold=args.track_count)
    
    # 守护进程模式
    if args.daemon:
        import daemon
        import daemon.pidfile
        
        pid_file = '/tmp/a4-tracker.pid'
        
        with daemon.DaemonContext(
            pidfile=daemon.pidfile.PIDLockFile(pid_file),
            working_directory=os.getcwd(),
        ):
            system = HeadlessA4TrackingSystem(args.log_file)
            system.run()
    else:
        # 普通模式
        system = HeadlessA4TrackingSystem(args.log_file)
        success = system.run()
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
