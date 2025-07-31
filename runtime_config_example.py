#!/usr/bin/env python3
"""
运行时配置管理示例
展示如何在程序运行过程中动态修改配置参数
"""

import cv2
import numpy as np
import time
from threading import Thread
import tkinter as tk
from tkinter import ttk

from dynamic_config import config
from main_modular import A4TrackingSystem

class RuntimeConfigGUI:
    """运行时配置GUI界面"""
    
    def __init__(self, tracking_system):
        self.tracking_system = tracking_system
        self.root = tk.Tk()
        self.root.title("A4跟踪系统 - 运行时配置")
        self.root.geometry("400x600")
        
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 检测参数
        detection_frame = ttk.LabelFrame(self.root, text="检测参数")
        detection_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(detection_frame, text="内部亮度阈值:").grid(row=0, column=0, sticky="w")
        self.inner_val_var = tk.DoubleVar(value=config.MEAN_INNER_VAL)
        inner_scale = ttk.Scale(detection_frame, from_=50, to=200, 
                               variable=self.inner_val_var, orient="horizontal")
        inner_scale.grid(row=0, column=1, sticky="ew")
        inner_scale.bind("<Motion>", self.on_inner_val_change)
        
        ttk.Label(detection_frame, text="边框暗度阈值:").grid(row=1, column=0, sticky="w")
        self.border_val_var = tk.DoubleVar(value=config.MEAN_BORDER_VAL)
        border_scale = ttk.Scale(detection_frame, from_=30, to=150, 
                                variable=self.border_val_var, orient="horizontal")
        border_scale.grid(row=1, column=1, sticky="ew")
        border_scale.bind("<Motion>", self.on_border_val_change)
        
        # 跟踪参数
        tracking_frame = ttk.LabelFrame(self.root, text="跟踪参数")
        tracking_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(tracking_frame, text="对齐阈值:").grid(row=0, column=0, sticky="w")
        self.align_threshold_var = tk.DoubleVar(value=config.ALIGNMENT_THRESHOLD)
        align_scale = ttk.Scale(tracking_frame, from_=2, to=20, 
                               variable=self.align_threshold_var, orient="horizontal")
        align_scale.grid(row=0, column=1, sticky="ew")
        align_scale.bind("<Motion>", self.on_align_threshold_change)
        
        ttk.Label(tracking_frame, text="跟踪计数阈值:").grid(row=1, column=0, sticky="w")
        self.track_count_var = tk.DoubleVar(value=config.TRACK_COUNT_THRESHOLD)
        track_count_scale = ttk.Scale(tracking_frame, from_=1, to=10, 
                                     variable=self.track_count_var, orient="horizontal")
        track_count_scale.grid(row=1, column=1, sticky="ew")
        track_count_scale.bind("<Motion>", self.on_track_count_change)
        
        # 距离参数
        distance_frame = ttk.LabelFrame(self.root, text="距离参数")
        distance_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(distance_frame, text="最小距离(mm):").grid(row=0, column=0, sticky="w")
        self.min_distance_var = tk.DoubleVar(value=config.MIN_DISTANCE_MM)
        min_dist_scale = ttk.Scale(distance_frame, from_=200, to=800, 
                                  variable=self.min_distance_var, orient="horizontal")
        min_dist_scale.grid(row=0, column=1, sticky="ew")
        min_dist_scale.bind("<Motion>", self.on_min_distance_change)
        
        ttk.Label(distance_frame, text="最大距离(mm):").grid(row=1, column=0, sticky="w")
        self.max_distance_var = tk.DoubleVar(value=config.MAX_DISTANCE_MM)
        max_dist_scale = ttk.Scale(distance_frame, from_=1000, to=3000, 
                                  variable=self.max_distance_var, orient="horizontal")
        max_dist_scale.grid(row=1, column=1, sticky="ew")
        max_dist_scale.bind("<Motion>", self.on_max_distance_change)
        
        # 控制按钮
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Button(button_frame, text="保存配置", command=self.save_config).pack(side="left", padx=5)
        ttk.Button(button_frame, text="重置默认", command=self.reset_config).pack(side="left", padx=5)
        ttk.Button(button_frame, text="显示当前配置", command=self.show_config).pack(side="left", padx=5)
        
        # 配置列权重
        for frame in [detection_frame, tracking_frame, distance_frame]:
            frame.columnconfigure(1, weight=1)
    
    def on_inner_val_change(self, event=None):
        """内部亮度阈值改变"""
        config.MEAN_INNER_VAL = int(self.inner_val_var.get())
        print(f"内部亮度阈值更新为: {config.MEAN_INNER_VAL}")
    
    def on_border_val_change(self, event=None):
        """边框暗度阈值改变"""
        config.MEAN_BORDER_VAL = int(self.border_val_var.get())
        print(f"边框暗度阈值更新为: {config.MEAN_BORDER_VAL}")
    
    def on_align_threshold_change(self, event=None):
        """对齐阈值改变"""
        config.ALIGNMENT_THRESHOLD = int(self.align_threshold_var.get())
        print(f"对齐阈值更新为: {config.ALIGNMENT_THRESHOLD}")
    
    def on_track_count_change(self, event=None):
        """跟踪计数阈值改变"""
        config.TRACK_COUNT_THRESHOLD = int(self.track_count_var.get())
        print(f"跟踪计数阈值更新为: {config.TRACK_COUNT_THRESHOLD}")
    
    def on_min_distance_change(self, event=None):
        """最小距离改变"""
        config.MIN_DISTANCE_MM = int(self.min_distance_var.get())
        print(f"最小距离更新为: {config.MIN_DISTANCE_MM}mm")
    
    def on_max_distance_change(self, event=None):
        """最大距离改变"""
        config.MAX_DISTANCE_MM = int(self.max_distance_var.get())
        print(f"最大距离更新为: {config.MAX_DISTANCE_MM}mm")
    
    def save_config(self):
        """保存配置到文件"""
        config.save_to_file()
        print("配置已保存到文件")
    
    def reset_config(self):
        """重置为默认配置"""
        config.reset_to_default()
        # 更新GUI显示
        self.inner_val_var.set(config.MEAN_INNER_VAL)
        self.border_val_var.set(config.MEAN_BORDER_VAL)
        self.align_threshold_var.set(config.ALIGNMENT_THRESHOLD)
        self.track_count_var.set(config.TRACK_COUNT_THRESHOLD)
        self.min_distance_var.set(config.MIN_DISTANCE_MM)
        self.max_distance_var.set(config.MAX_DISTANCE_MM)
        print("配置已重置为默认值")
    
    def show_config(self):
        """显示当前配置"""
        print("\n=== 当前配置 ===")
        print(f"检测参数: 内部亮度={config.MEAN_INNER_VAL}, 边框暗度={config.MEAN_BORDER_VAL}")
        print(f"跟踪参数: 对齐阈值={config.ALIGNMENT_THRESHOLD}, 跟踪计数={config.TRACK_COUNT_THRESHOLD}")
        print(f"距离范围: {config.MIN_DISTANCE_MM}-{config.MAX_DISTANCE_MM}mm")
        print("================\n")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()

class RuntimeConfigManager:
    """运行时配置管理器"""
    
    def __init__(self):
        self.tracking_system = None
        self.gui = None
    
    def start_with_gui(self):
        """带GUI界面启动"""
        print("=== 带GUI的运行时配置管理 ===")
        print("启动跟踪系统...")
        
        # 在单独线程中启动跟踪系统
        self.tracking_system = A4TrackingSystem()
        tracking_thread = Thread(target=self.tracking_system.run, daemon=True)
        tracking_thread.start()
        
        # 稍等一下让跟踪系统初始化
        time.sleep(1)
        
        # 启动GUI
        self.gui = RuntimeConfigGUI(self.tracking_system)
        self.gui.run()
    
    def start_with_keyboard_control(self):
        """键盘控制启动"""
        print("=== 键盘控制的运行时配置管理 ===")
        print("键盘命令:")
        print("- 1/2: 调整内部亮度阈值 (+/-)")
        print("- 3/4: 调整边框暗度阈值 (+/-)")
        print("- 5/6: 调整对齐阈值 (+/-)")
        print("- 7/8: 调整跟踪计数阈值 (+/-)")
        print("- s: 保存配置")
        print("- r: 重置配置")
        print("- d: 显示当前配置")
        print("- q: 退出")
        
        self.tracking_system = A4TrackingSystem()
        
        # 重写键盘处理
        original_handle_keyboard = self.tracking_system.display_manager.handle_keyboard_input
        
        def enhanced_keyboard_handler(distance_calc, offset_calc):
            key = cv2.waitKey(1) & 0xFF
            
            # 原有功能
            if key == ord('q'):
                return 'quit'
            elif key == ord('h'):
                distance_calc.clear_history()
                print("距离历史记录已清除")
            
            # 新的配置调整功能
            elif key == ord('1'):
                config.MEAN_INNER_VAL = min(200, config.MEAN_INNER_VAL + 5)
                print(f"内部亮度阈值: {config.MEAN_INNER_VAL}")
            elif key == ord('2'):
                config.MEAN_INNER_VAL = max(50, config.MEAN_INNER_VAL - 5)
                print(f"内部亮度阈值: {config.MEAN_INNER_VAL}")
            elif key == ord('3'):
                config.MEAN_BORDER_VAL = min(150, config.MEAN_BORDER_VAL + 5)
                print(f"边框暗度阈值: {config.MEAN_BORDER_VAL}")
            elif key == ord('4'):
                config.MEAN_BORDER_VAL = max(30, config.MEAN_BORDER_VAL - 5)
                print(f"边框暗度阈值: {config.MEAN_BORDER_VAL}")
            elif key == ord('5'):
                config.ALIGNMENT_THRESHOLD = min(20, config.ALIGNMENT_THRESHOLD + 1)
                print(f"对齐阈值: {config.ALIGNMENT_THRESHOLD}")
            elif key == ord('6'):
                config.ALIGNMENT_THRESHOLD = max(2, config.ALIGNMENT_THRESHOLD - 1)
                print(f"对齐阈值: {config.ALIGNMENT_THRESHOLD}")
            elif key == ord('7'):
                config.TRACK_COUNT_THRESHOLD = min(10, config.TRACK_COUNT_THRESHOLD + 1)
                print(f"跟踪计数阈值: {config.TRACK_COUNT_THRESHOLD}")
            elif key == ord('8'):
                config.TRACK_COUNT_THRESHOLD = max(1, config.TRACK_COUNT_THRESHOLD - 1)
                print(f"跟踪计数阈值: {config.TRACK_COUNT_THRESHOLD}")
            elif key == ord('s'):
                config.save_to_file()
                print("配置已保存")
            elif key == ord('r'):
                config.reset_to_default()
                print("配置已重置")
            elif key == ord('d'):
                print(f"\n当前配置:")
                print(f"  内部亮度阈值: {config.MEAN_INNER_VAL}")
                print(f"  边框暗度阈值: {config.MEAN_BORDER_VAL}")
                print(f"  对齐阈值: {config.ALIGNMENT_THRESHOLD}")
                print(f"  跟踪计数阈值: {config.TRACK_COUNT_THRESHOLD}")
                print()
            
            return None
        
        # 替换键盘处理函数
        self.tracking_system.display_manager.handle_keyboard_input = enhanced_keyboard_handler
        
        # 运行系统
        self.tracking_system.run()

def main():
    """主函数"""
    print("请选择运行模式:")
    print("1. GUI界面模式")
    print("2. 键盘控制模式")
    
    choice = input("请输入选择 (1-2): ").strip()
    
    manager = RuntimeConfigManager()
    
    if choice == "1":
        try:
            manager.start_with_gui()
        except ImportError:
            print("GUI模式需要tkinter库，切换到键盘控制模式")
            manager.start_with_keyboard_control()
    elif choice == "2":
        manager.start_with_keyboard_control()
    else:
        print("无效选择，使用键盘控制模式")
        manager.start_with_keyboard_control()

if __name__ == '__main__':
    main()
