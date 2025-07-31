import cv2
import numpy as np
import json
import os

# A4纸的实际尺寸
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# 阈值参数文件
THRESHOLD_CONFIG_FILE = "threshold_config.json"

class ThresholdAdjustmentTool:
    """A4纸检测阈值实时调节工具"""
    
    def __init__(self):
        # 默认阈值参数
        self.params = {
            "mean_inner_val": 100,      # 纸张内部亮度阈值
            "mean_border_val": 80,      # 黑边暗度阈值
            "clahe_clip_limit": 2.0,    # CLAHE对比度限制
            "clahe_tile_size": 8,       # CLAHE网格大小
            "gaussian_blur_size": 5,    # 高斯模糊核大小
            "adaptive_thresh_block": 11, # 自适应阈值块大小
            "adaptive_thresh_c": 2,     # 自适应阈值常数
            "contour_area_min": 1000,   # 最小轮廓面积
            "approx_epsilon": 0.02,     # 轮廓近似精度
            "erosion_kernel_size": 10,  # 腐蚀核大小
            "erosion_iterations": 1,    # 腐蚀迭代次数
            "dilation_iterations": 1    # 膨胀迭代次数
        }
        
        # 加载保存的参数
        self.load_parameters()
        
        # 创建滑动条窗口
        self.create_trackbars()
        
        # 调试信息显示标志
        self.show_debug = True
        self.show_threshold = False
        self.show_contours = False
        
    def load_parameters(self):
        """加载保存的参数"""
        if os.path.exists(THRESHOLD_CONFIG_FILE):
            try:
                with open(THRESHOLD_CONFIG_FILE, 'r') as f:
                    saved_params = json.load(f)
                    self.params.update(saved_params)
                print(f"已加载阈值参数: {THRESHOLD_CONFIG_FILE}")
            except Exception as e:
                print(f"加载阈值参数失败: {e}")
    
    def save_parameters(self):
        """保存当前参数"""
        try:
            with open(THRESHOLD_CONFIG_FILE, 'w') as f:
                json.dump(self.params, f, indent=2)
            print(f"阈值参数已保存到: {THRESHOLD_CONFIG_FILE}")
        except Exception as e:
            print(f"保存阈值参数失败: {e}")
    
    def create_trackbars(self):
        """创建调节滑动条"""
        cv2.namedWindow('Threshold Controls', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Threshold Controls', 400, 800)
        
        # 主要阈值参数
        cv2.createTrackbar('Inner Val', 'Threshold Controls', 
                          self.params["mean_inner_val"], 255, self.on_trackbar_change)
        cv2.createTrackbar('Border Val', 'Threshold Controls', 
                          self.params["mean_border_val"], 255, self.on_trackbar_change)
        
        # CLAHE参数
        cv2.createTrackbar('CLAHE Clip', 'Threshold Controls', 
                          int(self.params["clahe_clip_limit"] * 10), 100, self.on_trackbar_change)
        cv2.createTrackbar('CLAHE Tile', 'Threshold Controls', 
                          self.params["clahe_tile_size"], 16, self.on_trackbar_change)
        
        # 预处理参数
        cv2.createTrackbar('Blur Size', 'Threshold Controls', 
                          self.params["gaussian_blur_size"], 15, self.on_trackbar_change)
        cv2.createTrackbar('Adaptive Block', 'Threshold Controls', 
                          self.params["adaptive_thresh_block"], 31, self.on_trackbar_change)
        cv2.createTrackbar('Adaptive C', 'Threshold Controls', 
                          self.params["adaptive_thresh_c"], 20, self.on_trackbar_change)
        
        # 轮廓参数
        cv2.createTrackbar('Min Area', 'Threshold Controls', 
                          self.params["contour_area_min"] // 100, 100, self.on_trackbar_change)
        cv2.createTrackbar('Approx Epsilon', 'Threshold Controls', 
                          int(self.params["approx_epsilon"] * 100), 10, self.on_trackbar_change)
        
        # 形态学操作参数
        cv2.createTrackbar('Erosion Size', 'Threshold Controls', 
                          self.params["erosion_kernel_size"], 20, self.on_trackbar_change)
        cv2.createTrackbar('Erosion Iter', 'Threshold Controls', 
                          self.params["erosion_iterations"], 5, self.on_trackbar_change)
        cv2.createTrackbar('Dilation Iter', 'Threshold Controls', 
                          self.params["dilation_iterations"], 5, self.on_trackbar_change)
    
    def on_trackbar_change(self, val):
        """滑动条变化回调函数"""
        # 更新参数
        self.params["mean_inner_val"] = cv2.getTrackbarPos('Inner Val', 'Threshold Controls')
        self.params["mean_border_val"] = cv2.getTrackbarPos('Border Val', 'Threshold Controls')
        self.params["clahe_clip_limit"] = cv2.getTrackbarPos('CLAHE Clip', 'Threshold Controls') / 10.0
        self.params["clahe_tile_size"] = max(1, cv2.getTrackbarPos('CLAHE Tile', 'Threshold Controls'))
        self.params["gaussian_blur_size"] = max(1, cv2.getTrackbarPos('Blur Size', 'Threshold Controls'))
        if self.params["gaussian_blur_size"] % 2 == 0:
            self.params["gaussian_blur_size"] += 1  # 确保为奇数
        self.params["adaptive_thresh_block"] = max(3, cv2.getTrackbarPos('Adaptive Block', 'Threshold Controls'))
        if self.params["adaptive_thresh_block"] % 2 == 0:
            self.params["adaptive_thresh_block"] += 1  # 确保为奇数
        self.params["adaptive_thresh_c"] = cv2.getTrackbarPos('Adaptive C', 'Threshold Controls')
        self.params["contour_area_min"] = cv2.getTrackbarPos('Min Area', 'Threshold Controls') * 100
        self.params["approx_epsilon"] = cv2.getTrackbarPos('Approx Epsilon', 'Threshold Controls') / 100.0
        self.params["erosion_kernel_size"] = max(1, cv2.getTrackbarPos('Erosion Size', 'Threshold Controls'))
        self.params["erosion_iterations"] = cv2.getTrackbarPos('Erosion Iter', 'Threshold Controls')
        self.params["dilation_iterations"] = cv2.getTrackbarPos('Dilation Iter', 'Threshold Controls')
    
    def order_points(self, pts):
        """整理四个顶点的顺序"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def detect_a4_paper_with_debug(self, frame):
        """
        带调试信息的A4纸检测
        """
        original_frame = frame.copy()
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用CLAHE改善对比度
        if self.params["clahe_clip_limit"] > 0:
            clahe = cv2.createCLAHE(
                clipLimit=self.params["clahe_clip_limit"], 
                tileGridSize=(self.params["clahe_tile_size"], self.params["clahe_tile_size"])
            )
            gray = clahe.apply(gray)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, 
                                  (self.params["gaussian_blur_size"], self.params["gaussian_blur_size"]), 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            self.params["adaptive_thresh_block"], 
            self.params["adaptive_thresh_c"]
        )
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        detected_rect = None
        debug_info = {
            "total_contours": len(contours),
            "valid_4_sided": 0,
            "area_filtered": 0,
            "convex_filtered": 0,
            "color_validated": 0,
            "mean_inner_vals": [],
            "mean_border_vals": []
        }
        
        debug_frame = frame.copy()
        contour_debug_frame = frame.copy()
        
        for i, contour in enumerate(contours):
            # 轮廓近似
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.params["approx_epsilon"] * peri, True)
            
            # 绘制所有轮廓（用于调试）
            if self.show_contours:
                cv2.drawContours(contour_debug_frame, [contour], -1, (0, 255, 0), 2)
                cv2.putText(contour_debug_frame, f"{i}", tuple(contour[0][0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 检查是否为四边形
            if len(approx) == 4:
                debug_info["valid_4_sided"] += 1
                
                # 检查面积
                area = cv2.contourArea(approx)
                if area < self.params["contour_area_min"]:
                    continue
                debug_info["area_filtered"] += 1
                
                # 检查是否为凸形
                if not cv2.isContourConvex(approx):
                    continue
                debug_info["convex_filtered"] += 1
                
                # 颜色验证
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [approx], -1, (255), -1)
                
                # 创建腐蚀和膨胀核
                if self.params["erosion_kernel_size"] > 0:
                    kernel = np.ones((self.params["erosion_kernel_size"], self.params["erosion_kernel_size"]), np.uint8)
                    inner_mask = cv2.erode(mask, kernel, iterations=self.params["erosion_iterations"])
                    outer_mask = cv2.dilate(mask, kernel, iterations=self.params["dilation_iterations"])
                else:
                    inner_mask = mask
                    outer_mask = mask
                
                border_mask = cv2.subtract(outer_mask, mask)
                
                # 计算平均颜色值
                mean_inner_val = cv2.mean(gray, mask=inner_mask)[0] if cv2.countNonZero(inner_mask) > 0 else 0
                mean_border_val = cv2.mean(gray, mask=border_mask)[0] if cv2.countNonZero(border_mask) > 0 else 0
                
                debug_info["mean_inner_vals"].append(mean_inner_val)
                debug_info["mean_border_vals"].append(mean_border_val)
                
                # 绘制候选轮廓（用于调试）
                cv2.drawContours(debug_frame, [approx], -1, (255, 0, 0), 2)
                cv2.putText(debug_frame, f"I:{mean_inner_val:.1f} B:{mean_border_val:.1f}", 
                           tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # 颜色阈值检查
                if (mean_inner_val > self.params["mean_inner_val"] and 
                    mean_border_val < self.params["mean_border_val"]):
                    debug_info["color_validated"] += 1
                    detected_rect = approx
                    cv2.drawContours(debug_frame, [approx], -1, (0, 255, 0), 3)
                    break
        
        # 显示调试信息
        if self.show_debug:
            info_y = 30
            cv2.putText(debug_frame, f"Total Contours: {debug_info['total_contours']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(debug_frame, f"4-Sided: {debug_info['valid_4_sided']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(debug_frame, f"Area OK: {debug_info['area_filtered']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(debug_frame, f"Convex: {debug_info['convex_filtered']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(debug_frame, f"Color OK: {debug_info['color_validated']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 显示当前阈值
            info_y += 35
            cv2.putText(debug_frame, f"Inner Thresh: {self.params['mean_inner_val']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            info_y += 20
            cv2.putText(debug_frame, f"Border Thresh: {self.params['mean_border_val']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 显示检测到的平均值范围
            if debug_info["mean_inner_vals"]:
                info_y += 25
                inner_range = f"{min(debug_info['mean_inner_vals']):.1f}-{max(debug_info['mean_inner_vals']):.1f}"
                cv2.putText(debug_frame, f"Inner Range: {inner_range}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            if debug_info["mean_border_vals"]:
                info_y += 20
                border_range = f"{min(debug_info['mean_border_vals']):.1f}-{max(debug_info['mean_border_vals']):.1f}"
                cv2.putText(debug_frame, f"Border Range: {border_range}", 
                           (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # 如果检测到A4纸，添加成功标识
        if detected_rect is not None:
            cv2.putText(debug_frame, "A4 DETECTED!", (10, debug_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(debug_frame, "NO A4 DETECTED", (10, debug_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return debug_frame, thresh, contour_debug_frame, detected_rect is not None
    
    def run(self):
        """运行阈值调节工具"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        print("=== A4纸检测阈值实时调节工具 ===")
        print("操作说明:")
        print("- 'q': 退出程序")
        print("- 's': 保存当前参数")
        print("- 'r': 重置为默认参数")
        print("- 'd': 切换调试信息显示")
        print("- 't': 显示阈值处理结果")
        print("- 'c': 显示轮廓检测过程")
        print("- 'h': 显示帮助信息")
        print("\n调节滑动条实时调整检测参数...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误：无法读取帧")
                break
            
            # 处理帧
            debug_frame, thresh_frame, contour_frame, detected = self.detect_a4_paper_with_debug(frame)
            
            # 显示结果
            cv2.imshow('A4 Detection Debug', debug_frame)
            
            if self.show_threshold:
                cv2.imshow('Threshold Result', thresh_frame)
            
            if self.show_contours:
                cv2.imshow('Contour Detection', contour_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_parameters()
            elif key == ord('r'):
                # 重置为默认参数
                self.params = {
                    "mean_inner_val": 100,
                    "mean_border_val": 80,
                    "clahe_clip_limit": 2.0,
                    "clahe_tile_size": 8,
                    "gaussian_blur_size": 5,
                    "adaptive_thresh_block": 11,
                    "adaptive_thresh_c": 2,
                    "contour_area_min": 1000,
                    "approx_epsilon": 0.02,
                    "erosion_kernel_size": 10,
                    "erosion_iterations": 1,
                    "dilation_iterations": 1
                }
                # 更新滑动条
                cv2.setTrackbarPos('Inner Val', 'Threshold Controls', self.params["mean_inner_val"])
                cv2.setTrackbarPos('Border Val', 'Threshold Controls', self.params["mean_border_val"])
                # ... 更新其他滑动条
                print("参数已重置为默认值")
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"调试信息显示: {'开' if self.show_debug else '关'}")
            elif key == ord('t'):
                self.show_threshold = not self.show_threshold
                if not self.show_threshold:
                    cv2.destroyWindow('Threshold Result')
                print(f"阈值结果显示: {'开' if self.show_threshold else '关'}")
            elif key == ord('c'):
                self.show_contours = not self.show_contours
                if not self.show_contours:
                    cv2.destroyWindow('Contour Detection')
                print(f"轮廓检测显示: {'开' if self.show_contours else '关'}")
            elif key == ord('h'):
                print("\n=== 操作说明 ===")
                print("滑动条参数说明:")
                print("- Inner Val: 纸张内部亮度阈值 (越高越严格)")
                print("- Border Val: 黑边暗度阈值 (越低越严格)")
                print("- CLAHE Clip: 对比度增强限制 (0-10)")
                print("- CLAHE Tile: 自适应网格大小 (1-16)")
                print("- Blur Size: 高斯模糊核大小 (奇数)")
                print("- Adaptive Block: 自适应阈值块大小 (奇数)")
                print("- Adaptive C: 自适应阈值常数")
                print("- Min Area: 最小轮廓面积 (×100)")
                print("- Approx Epsilon: 轮廓近似精度 (0.01-0.10)")
                print("- Erosion Size: 腐蚀核大小")
                print("- Erosion/Dilation Iter: 形态学操作迭代次数")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    tool = ThresholdAdjustmentTool()
    tool.run()

if __name__ == '__main__':
    main()