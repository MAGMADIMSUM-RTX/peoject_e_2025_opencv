import cv2
import numpy as np
import time

# 帧率显示标志
SHOW_FPS = True  # 改为 True 显示帧率，False 不显示


# 打开摄像头
cap = cv2.VideoCapture(0)
# 设置分辨率为1080p
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
if not cap.isOpened():
    raise RuntimeError('无法打开摄像头')

# 帧率相关变量
if SHOW_FPS:
    prev_time = time.time()
    frame_count = 0
    fps = 0

# 各步骤总耗时和计数
total_time_capture = 0
total_time_gray = 0
total_time_blur = 0
total_time_thresh = 0
total_time_contour = 0
total_time_find = 0
total_time_draw = 0
total_time_total = 0
step_count = 0


while True:
    t0 = time.time()
    ret, frame = cap.read()
    t1 = time.time()
    if not ret:
        print("无法读取摄像头帧")
        break

    # 帧率统计
    if SHOW_FPS:
        frame_count += 1
        curr_time = time.time()
        if curr_time - prev_time >= 1.0:
            fps = frame_count / (curr_time - prev_time)
            prev_time = curr_time
            frame_count = 0

    t2 = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t3 = time.time()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    t4 = time.time()
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    t5 = time.time()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    t6 = time.time()

    max_area = 0
    outer_idx = -1
    inner_idx = -1

    # 寻找最大中空多边形（有父子关系的轮廓）
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, h in enumerate(hierarchy):
            # h[2]是子轮廓索引，h[3]是父轮廓索引
            if h[2] != -1 and h[3] == -1:  # 外轮廓有子轮廓且无父轮廓
                area = cv2.contourArea(contours[i])
                if area > max_area:
                    max_area = area
                    outer_idx = i
                    inner_idx = h[2]
    t7 = time.time()

    display = frame.copy()
    if outer_idx != -1 and inner_idx != -1:
        # 画外线为绿色
        cv2.drawContours(display, contours, outer_idx, (0, 255, 0), 2)
        # 画内线为红色
        cv2.drawContours(display, contours, inner_idx, (0, 0, 255), 2)
    t8 = time.time()

    # 累加各步骤耗时
    total_time_capture += (t1 - t0)
    total_time_gray += (t3 - t2)
    total_time_blur += (t4 - t3)
    total_time_thresh += (t5 - t4)
    total_time_contour += (t6 - t5)
    total_time_find += (t7 - t6)
    total_time_draw += (t8 - t7)
    total_time_total += (t8 - t0)
    step_count += 1
    # 显示分辨率和帧率
    h, w = display.shape[:2]
    if SHOW_FPS:
        cv2.putText(display, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(display, f"Res: {w}x{h}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 打印各步骤耗时
    print(f"采集:{(t1-t0)*1000:.1f}ms 灰度:{(t3-t2)*1000:.1f}ms 模糊:{(t4-t3)*1000:.1f}ms 阈值:{(t5-t4)*1000:.1f}ms 轮廓:{(t6-t5)*1000:.1f}ms 查找:{(t7-t6)*1000:.1f}ms 绘制:{(t8-t7)*1000:.1f}ms 总:{(t8-t0)*1000:.1f}ms")

    # 每10帧打印一次平均耗时
    if step_count % 10 == 0:
        print("\n===== 10帧平均耗时统计 =====")
        print(f"采集: {total_time_capture * 1000 / step_count:.1f}ms")
        print(f"灰度: {total_time_gray * 1000 / step_count:.1f}ms")
        print(f"模糊: {total_time_blur * 1000 / step_count:.1f}ms")
        print(f"阈值: {total_time_thresh * 1000 / step_count:.1f}ms")
        print(f"轮廓: {total_time_contour * 1000 / step_count:.1f}ms")
        print(f"查找: {total_time_find * 1000 / step_count:.1f}ms")
        print(f"绘制: {total_time_draw * 1000 / step_count:.1f}ms")
        print(f"总: {total_time_total * 1000 / step_count:.1f}ms\n")

    cv2.imshow('result', display)
    key = cv2.waitKey(1)
    if key == 27:  # 按ESC退出
        break

cap.release()
cv2.destroyAllWindows()

# 输出平均耗时
# 程序结束时仍保留最终平均耗时输出
if step_count > 0:
    print("\n===== 最终平均耗时统计 =====")
    print(f"采集: {total_time_capture * 1000 / step_count:.1f}ms")
    print(f"灰度: {total_time_gray * 1000 / step_count:.1f}ms")
    print(f"模糊: {total_time_blur * 1000 / step_count:.1f}ms")
    print(f"阈值: {total_time_thresh * 1000 / step_count:.1f}ms")
    print(f"轮廓: {total_time_contour * 1000 / step_count:.1f}ms")
    print(f"查找: {total_time_find * 1000 / step_count:.1f}ms")
    print(f"绘制: {total_time_draw * 1000 / step_count:.1f}ms")
    print(f"总: {total_time_total * 1000 / step_count:.1f}ms")
