#!/usr/bin/env python3
"""
距离校准工具
用于校准A4纸检测系统的距离测量精度
"""

import cv2
import json
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# 添加当前目录到路径以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from a4_detector import A4PaperDetector
    from distance_calculator import SimpleDistanceCalculator
    from dynamic_config import config
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保所有必需的模块都在当前目录中")
    sys.exit(1)

class DistanceCalibrationTool:
    """距离校准工具"""
    
    def __init__(self, config_file="runtime_config.json"):
        self.config_file = config_file
        self.detector = A4PaperDetector()
        self.distance_calc = SimpleDistanceCalculator()
        self.calibration_data = []
        
        # 加载当前配置
        self.load_config()
        
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {self.config_file} 不存在")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            sys.exit(1)
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            print(f"配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def capture_calibration_point(self, actual_distance_mm):
        """捕获一个校准点"""
        print(f"\n=== 校准距离: {actual_distance_mm}mm ===")
        print("请将A4纸放置在指定距离处，按空格键捕获数据，ESC退出")
        print("尝试不同角度的倾斜来测试透视校正效果")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return None
        
        measurements = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_height, frame_width = frame.shape[:2]
            
            # 检测A4纸
            detected_rect = self.detector.detect_a4_paper(frame)
            
            if detected_rect is not None:
                # 计算透视校正后的尺寸
                corrected_width, corrected_height, center = self.detector.calculate_perspective_corrected_dimensions(
                    detected_rect, frame_width, frame_height)
                
                if corrected_width is not None:
                    # 计算各种距离
                    distance_original = self.distance_calc.calculate_distance_from_width(corrected_width, frame_width)
                    distance_corrected = self.distance_calc.calculate_distance_from_corrected_width(corrected_width, frame_width)
                    distance_3d = self.distance_calc.calculate_3d_distance_with_homography(detected_rect, frame_width, frame_height)
                    distance_advanced = self.distance_calc.calculate_advanced_perspective_distance(detected_rect, frame_width, frame_height)
                    distance_hybrid = self.distance_calc.calculate_distance_hybrid(corrected_width, detected_rect, frame_width, frame_height)
                    
                    # 在图像上显示信息
                    cv2.drawContours(frame, [detected_rect], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                    
                    # 计算倾斜程度指标
                    from a4_detector import order_points
                    rect_pts = order_points(detected_rect.reshape(4, 2))
                    (tl, tr, br, bl) = rect_pts
                    
                    width_top = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
                    width_bottom = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
                    height_left = np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)
                    height_right = np.sqrt((br[0] - tr[0])**2 + (br[1] - tr[1])**2)
                    
                    width_ratio = min(width_top, width_bottom) / max(width_top, width_bottom)
                    height_ratio = min(height_left, height_right) / max(height_left, height_right)
                    tilt_score = (width_ratio + height_ratio) / 2
                    
                    info_text = [
                        f"Actual: {actual_distance_mm}mm",
                        f"Original: {distance_original:.0f}mm" if distance_original else "Original: None",
                        f"Corrected: {distance_corrected:.0f}mm" if distance_corrected else "Corrected: None",
                        f"3D: {distance_3d:.0f}mm" if distance_3d else "3D: None",
                        f"Advanced: {distance_advanced:.0f}mm" if distance_advanced else "Advanced: None",
                        f"Hybrid: {distance_hybrid:.0f}mm" if distance_hybrid else "Hybrid: None",
                        f"Tilt Score: {tilt_score:.3f} (1.0=straight)",
                        "Space: Capture, ESC: Exit"
                    ]
                    
                    for i, text in enumerate(info_text):
                        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                else:
                    cv2.putText(frame, "A4 detected but size calculation failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No A4 paper detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Distance Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and detected_rect is not None and corrected_width is not None:
                # 捕获当前测量
                measurement = {
                    'actual_distance': actual_distance_mm,
                    'pixel_width': corrected_width,
                    'distance_original': distance_original,
                    'distance_corrected': distance_corrected,
                    'distance_3d': distance_3d,
                    'distance_advanced': distance_advanced,
                    'distance_hybrid': distance_hybrid,
                    'tilt_score': tilt_score,
                    'detected_rect': detected_rect.tolist()
                }
                measurements.append(measurement)
                print(f"已捕获测量点 #{len(measurements)}: 实际={actual_distance_mm}mm, 计算={distance_hybrid:.0f}mm, 倾斜={tilt_score:.3f}")
                
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return measurements
    
    def collect_calibration_data(self, distances):
        """收集校准数据"""
        print("开始收集校准数据...")
        print("每个距离建议采集3-5个测量点以提高准确性")
        print("建议在每个距离下测试不同的倾斜角度")
        
        for distance in distances:
            measurements = self.capture_calibration_point(distance)
            if measurements:
                self.calibration_data.extend(measurements)
                print(f"距离 {distance}mm: 采集了 {len(measurements)} 个测量点")
            else:
                print(f"距离 {distance}mm: 未采集到有效数据")
        
        print(f"\n总共采集了 {len(self.calibration_data)} 个校准点")
    
    def analyze_calibration_data(self):
        """分析校准数据"""
        if not self.calibration_data:
            print("没有校准数据可分析")
            return
        
        print("\n=== 校准数据分析 ===")
        
        # 提取数据
        actual_distances = [d['actual_distance'] for d in self.calibration_data]
        original_distances = [d['distance_original'] for d in self.calibration_data if d['distance_original']]
        corrected_distances = [d['distance_corrected'] for d in self.calibration_data if d['distance_corrected']]
        d3d_distances = [d['distance_3d'] for d in self.calibration_data if d['distance_3d']]
        advanced_distances = [d['distance_advanced'] for d in self.calibration_data if d['distance_advanced']]
        hybrid_distances = [d['distance_hybrid'] for d in self.calibration_data if d['distance_hybrid']]
        
        # 分析倾斜度对准确性的影响
        straight_data = [d for d in self.calibration_data if d.get('tilt_score', 0) > 0.9]
        tilted_data = [d for d in self.calibration_data if d.get('tilt_score', 0) <= 0.9]
        
        print(f"直立样本: {len(straight_data)} 个")
        print(f"倾斜样本: {len(tilted_data)} 个")
        
        # 计算误差统计
        def calculate_error_stats(measured, actual_subset):
            if not measured:
                return None, None, None
            errors = [abs(m - a) for m, a in zip(measured, actual_subset)]
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            return mean_error, std_error, max_error
        
        methods = [
            ("Original", original_distances),
            ("Corrected", corrected_distances),
            ("3D", d3d_distances),
            ("Advanced", advanced_distances),
            ("Hybrid", hybrid_distances)
        ]
        
        print("\n误差统计 (mm):")
        print(f"{'Method':<12} {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12} {'Count':<8}")
        print("-" * 60)
        
        best_method = None
        best_error = float('inf')
        
        for method_name, distances in methods:
            if distances:
                mean_err, std_err, max_err = calculate_error_stats(distances, actual_distances[:len(distances)])
                print(f"{method_name:<12} {mean_err:<12.1f} {std_err:<12.1f} {max_err:<12.1f} {len(distances):<8}")
                
                if mean_err < best_error:
                    best_error = mean_err
                    best_method = method_name
        
        print(f"\n最佳方法: {best_method} (平均误差: {best_error:.1f}mm)")
        
        # 分析倾斜对各方法的影响
        if straight_data and tilted_data:
            print("\n倾斜度影响分析:")
            print("直立A4纸 vs 倾斜A4纸的误差对比:")
            
            for method_name, _ in methods:
                straight_errors = []
                tilted_errors = []
                
                for d in straight_data:
                    if d.get(f'distance_{method_name.lower()}'):
                        error = abs(d[f'distance_{method_name.lower()}'] - d['actual_distance'])
                        straight_errors.append(error)
                
                for d in tilted_data:
                    if d.get(f'distance_{method_name.lower()}'):
                        error = abs(d[f'distance_{method_name.lower()}'] - d['actual_distance'])
                        tilted_errors.append(error)
                
                if straight_errors and tilted_errors:
                    straight_mean = np.mean(straight_errors)
                    tilted_mean = np.mean(tilted_errors)
                    improvement = ((tilted_mean - straight_mean) / straight_mean) * 100
                    print(f"{method_name}: 直立={straight_mean:.1f}mm, 倾斜={tilted_mean:.1f}mm, 变化={improvement:+.1f}%")
        
        return best_method
    
    def simple_optimize_calibration_factors(self):
        """简单的校准因子优化（不使用scipy）"""
        if not self.calibration_data:
            print("没有校准数据进行优化")
            return
        
        print("\n=== 简单优化校准因子 ===")
        
        # 提取实际距离和测量距离
        actual_distances = np.array([d['actual_distance'] for d in self.calibration_data])
        
        # 选择最佳方法的距离数据
        best_distances = []
        for d in self.calibration_data:
            if d.get('distance_advanced'):
                best_distances.append(d['distance_advanced'])
            elif d.get('distance_hybrid'):
                best_distances.append(d['distance_hybrid'])
            elif d.get('distance_corrected'):
                best_distances.append(d['distance_corrected'])
            else:
                best_distances.append(d.get('distance_original', 0))
        
        best_distances = np.array(best_distances)
        
        if len(best_distances) == 0:
            print("没有有效的距离数据进行优化")
            return False
        
        # 计算当前误差
        current_errors = np.abs(best_distances - actual_distances)
        current_mean_error = np.mean(current_errors)
        
        print(f"当前平均误差: {current_mean_error:.2f}mm")
        
        # 简单的线性校正：寻找最佳的乘法因子
        best_factor = None
        best_mean_error = current_mean_error
        
        # 在0.5到2.0之间搜索最佳校正因子
        for factor in np.arange(0.5, 2.0, 0.01):
            corrected_distances = best_distances * factor
            errors = np.abs(corrected_distances - actual_distances)
            mean_error = np.mean(errors)
            
            if mean_error < best_mean_error:
                best_mean_error = mean_error
                best_factor = factor
        
        if best_factor is not None:
            print(f"找到最佳校正因子: {best_factor:.4f}")
            print(f"优化后平均误差: {best_mean_error:.2f}mm")
            print(f"误差改善: {((current_mean_error - best_mean_error) / current_mean_error * 100):.1f}%")
            
            # 更新配置中的高级透视因子
            current_factor = self.config_data["CAMERA_PARAMS"].get("advanced_perspective_factor", 1.0)
            new_factor = current_factor * best_factor
            
            self.config_data["CAMERA_PARAMS"]["advanced_perspective_factor"] = float(new_factor)
            print(f"更新 advanced_perspective_factor: {current_factor:.4f} -> {new_factor:.4f}")
            
            return True
        else:
            print("未找到更好的校正因子")
            return False
    
    def save_calibration_results(self):
        """保存校准结果"""
        # 保存校准数据
        calibration_file = self.config_file.replace('.json', '_calibration_data.json')
        try:
            with open(calibration_file, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_data, f, indent=2, ensure_ascii=False)
            print(f"校准数据已保存到: {calibration_file}")
        except Exception as e:
            print(f"保存校准数据失败: {e}")
        
        # 保存更新的配置
        self.save_config()
    
    def interactive_calibration(self):
        """交互式校准过程"""
        print("=== A4纸距离校准工具 ===")
        print("这个工具将帮助您校准A4纸检测系统的距离测量精度")
        
        # 获取校准距离
        print("\n请输入要校准的距离 (单位: mm，用空格分隔):")
        print("建议使用: 600 800 1000 1200 1400")
        distance_input = input("距离: ").strip()
        
        try:
            distances = [int(d) for d in distance_input.split()]
            if not distances:
                distances = [600, 800, 1000, 1200, 1400]  # 默认距离
        except ValueError:
            print("输入格式错误，使用默认距离")
            distances = [600, 800, 1000, 1200, 1400]
        
        print(f"将校准以下距离: {distances}")
        
        # 收集校准数据
        self.collect_calibration_data(distances)
        
        if not self.calibration_data:
            print("未收集到校准数据，退出")
            return
        
        # 分析数据
        best_method = self.analyze_calibration_data()
        
        # 优化参数
        print("\n是否进行参数优化? (y/n): ", end='')
        if input().lower().startswith('y'):
            if self.simple_optimize_calibration_factors():
                print("参数优化完成")
            else:
                print("参数优化失败")
        
        # 保存结果
        print("\n是否保存校准结果? (y/n): ", end='')
        if input().lower().startswith('y'):
            self.save_calibration_results()
            print("校准完成！")
        else:
            print("校准结果未保存")

def main():
    parser = argparse.ArgumentParser(description="A4纸距离校准工具")
    parser.add_argument("--config", default="runtime_config.json", help="配置文件路径")
    parser.add_argument("--distances", nargs='+', type=int, help="校准距离列表 (mm)")
    parser.add_argument("--auto", action='store_true', help="自动模式，使用默认参数")
    
    args = parser.parse_args()
    
    tool = DistanceCalibrationTool(args.config)
    
    if args.auto:
        distances = args.distances or [600, 800, 1000, 1200, 1400]
        tool.collect_calibration_data(distances)
        tool.analyze_calibration_data()
        tool.simple_optimize_calibration_factors()
        tool.save_calibration_results()
    else:
        tool.interactive_calibration()

if __name__ == "__main__":
    main() {'Mean Error':<12} {'Std Error':<12} {'Max Error':<12} {'Count':<8}")
        print("-" * 60)
        
        best_method = None
        best_error = float('inf')
        
        for method_name, distances in methods:
            if distances:
                mean_err, std_err, max_err = calculate_error_stats(distances, actual_distances)
                print(f"{method_name:<12} {mean_err:<12.1f} {std_err:<12.1f} {max_err:<12.1f} {len(distances):<8}")
                
                if mean_err < best_error:
                    best_error = mean_err
                    best_method = method_name
        
        print(f"\n最佳方法: {best_method} (平均误差: {best_error:.1f}mm)")
        
        return best_method
    
    def optimize_calibration_factors(self):
        """优化校准因子"""
        if not self.calibration_data:
            print("没有校准数据进行优化")
            return
        
        print("\n=== 优化校准因子 ===")
        
        # 提取实际距离和像素宽度
        actual_distances = np.array([d['actual_distance'] for d in self.calibration_data])
        pixel_widths = np.array([d['pixel_width'] for d in self.calibration_data])
        
        # 当前相机参数
        focal_length = self.config_data["CAMERA_PARAMS"]["focal_length_mm"]
        sensor_width = self.config_data["CAMERA_PARAMS"]["sensor_width_mm"]
        frame_width = 640  # 假设的帧宽度，实际使用时应该从实际帧获取
        
        def distance_function(pixel_width, calibration_factor, perspective_weight):
            """距离计算函数"""
            fov_horizontal_rad = 2 * np.arctan(sensor_width / (2 * focal_length))
            mm_per_pixel_at_1m = (1000 * np.tan(fov_horizontal_rad / 2) * 2) / frame_width
            distance_mm = (self.config_data["A4_WIDTH_MM"] * 1000) / (pixel_width * mm_per_pixel_at_1m)
            return distance_mm * calibration_factor * perspective_weight
        
        def objective_function(params):
            """优化目标函数"""
            calibration_factor, perspective_weight = params
            predicted_distances = [distance_function(pw, calibration_factor, perspective_weight) 
                                 for pw in pixel_widths]
            mse = np.mean((predicted_distances - actual_distances) ** 2)
            return mse
        
        # 初始参数
        initial_params = [
            self.config_data["CAMERA_PARAMS"]["calibration_factor"],
            self.config_data["CAMERA_PARAMS"].get("perspective_correction_weight", 1.0)
        ]
        
        # 参数边界
        bounds = [(0.1, 10.0), (0.1, 3.0)]  # calibration_factor, perspective_weight
        
        # 优化
        result = minimize(objective_function, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimized_calibration_factor, optimized_perspective_weight = result.x
            
            print(f"优化前:")
            print(f"  calibration_factor: {initial_params[0]:.6f}")
            print(f"  perspective_correction_weight: {initial_params[1]:.6f}")
            print(f"  MSE: {objective_function(initial_params):.2f}")
            
            print(f"\n优化后:")
            print(f"  calibration_factor: {optimized_calibration_factor:.6f}")
            print(f"  perspective_correction_weight: {optimized_perspective_weight:.6f}")
            print(f"  MSE: {result.fun:.2f}")
            
            # 更新配置
            self.config_data["CAMERA_PARAMS"]["calibration_factor"] = float(optimized_calibration_factor)
            self.config_data["CAMERA_PARAMS"]["perspective_correction_weight"] = float(optimized_perspective_weight)
            
            return True
        else:
            print("优化失败:", result.message)
            return False
    
    def save_calibration_results(self):
        """保存校准结果"""
        # 保存校准数据
        calibration_file = self.config_file.replace('.json', '_calibration_data.json')
        try:
            with open(calibration_file, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_data, f, indent=2, ensure_ascii=False)
            print(f"校准数据已保存到: {calibration_file}")
        except Exception as e:
            print(f"保存校准数据失败: {e}")
        
        # 保存更新的配置
        self.save_config()
    
    def interactive_calibration(self):
        """交互式校准过程"""
        print("=== A4纸距离校准工具 ===")
        print("这个工具将帮助您校准A4纸检测系统的距离测量精度")
        
        # 获取校准距离
        print("\n请输入要校准的距离 (单位: mm，用空格分隔):")
        print("建议使用: 600 800 1000 1200 1400")
        distance_input = input("距离: ").strip()
        
        try:
            distances = [int(d) for d in distance_input.split()]
            if not distances:
                distances = [600, 800, 1000, 1200, 1400]  # 默认距离
        except ValueError:
            print("输入格式错误，使用默认距离")
            distances = [600, 800, 1000, 1200, 1400]
        
        print(f"将校准以下距离: {distances}")
        
        # 收集校准数据
        self.collect_calibration_data(distances)
        
        if not self.calibration_data:
            print("未收集到校准数据，退出")
            return
        
        # 分析数据
        best_method = self.analyze_calibration_data()
        
        # 优化参数
        print("\n是否进行参数优化? (y/n): ", end='')
        if input().lower().startswith('y'):
            if self.optimize_calibration_factors():
                print("参数优化完成")
            else:
                print("参数优化失败")
        
        # 保存结果
        print("\n是否保存校准结果? (y/n): ", end='')
        if input().lower().startswith('y'):
            self.save_calibration_results()
            print("校准完成！")
        else:
            print("校准结果未保存")

def main():
    parser = argparse.ArgumentParser(description="A4纸距离校准工具")
    parser.add_argument("--config", default="runtime_config.json", help="配置文件路径")
    parser.add_argument("--distances", nargs='+', type=int, help="校准距离列表 (mm)")
    parser.add_argument("--auto", action='store_true', help="自动模式，使用默认参数")
    
    args = parser.parse_args()
    
    tool = DistanceCalibrationTool(args.config)
    
    if args.auto:
        distances = args.distances or [600, 800, 1000, 1200, 1400]
        tool.collect_calibration_data(distances)
        tool.analyze_calibration_data()
        tool.optimize_calibration_factors()
        tool.save_calibration_results()
    else:
        tool.interactive_calibration()

if __name__ == "__main__":
    main()