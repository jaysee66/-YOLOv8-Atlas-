import cv2
import torch
from ultralytics import YOLO
import time
import numpy as np
from collections import deque

# 加载训练好的最佳模型
model = YOLO('../switch/best.pt')  # 替换为你的模型路径

# 视频文件路径
video = '../data/YN090013.MP4'  # 替换为你的4K视频路径
output = '../data/output_video.mp4'

# 打开视频文件
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# 获取视频属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建视频写入对象（如果需要保存结果）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

# 类别名称
class_names = ['holothurian', 'echinus', 'scallop', 'starfish']

# 性能统计
frame_count = 0
total_time = 0
detection_counts = {name: 0 for name in class_names}
confidence_threshold = 0.5  # 置信度阈值

# 帧缓存，用于单帧后退功能
frame_buffer = deque(maxlen=5)  # 缓存最近的5帧

# 控制变量
paused = False
step_mode = False  # 单帧模式

# 字体和颜色设置
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # 红色
box_colors = {
    'holothurian': (0, 255, 0),  # 绿色
    'echinus': (0, 0, 255),  # 红色
    'scallop': (0, 255, 255),  # 黄色
    'starfish': (255, 0, 0)  # 蓝色
}

# 创建性能监控窗口
cv2.namedWindow('Underwater Object Detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Performance Metrics', cv2.WINDOW_NORMAL)

# 性能数据记录
timing_data = {
    'frame_times': [],
    'fps_values': [],
    'detections_per_frame': []
}

# 循环处理每一帧
while cap.isOpened():
    if not paused or step_mode:
        ret, frame = cap.read()
        if not ret:
            break

        # 将当前帧加入缓存
        frame_buffer.append(frame.copy())

        frame_count += 1
        start_time = time.time()

        # 使用模型进行预测
        results = model(frame, conf=confidence_threshold, verbose=False)

        # 计算处理时间
        processing_time = time.time() - start_time
        total_time += processing_time
        timing_data['frame_times'].append(processing_time)

        # 获取当前帧率
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        timing_data['fps_values'].append(current_fps)

        # 获取检测结果
        boxes = []
        confs = []
        cls_ids = []
        detections_this_frame = 0

        for result in results:
            result_boxes = result.boxes.xyxy.cpu().numpy()
            result_confs = result.boxes.conf.cpu().numpy()
            result_cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            boxes.extend(result_boxes)
            confs.extend(result_confs)
            cls_ids.extend(result_cls_ids)

            detections_this_frame += len(result_boxes)

        timing_data['detections_per_frame'].append(detections_this_frame)

        # 绘制检测结果
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[cls_id]

            # 更新检测计数
            detection_counts[class_name] += 1

            # 绘制边界框
            color = box_colors.get(class_name, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制类别标签和置信度
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        font, font_scale, color, font_thickness)

        # 重置单帧模式
        step_mode = False

    else:  # 暂停状态且不是单帧模式
        # 使用最后一帧
        if frame_buffer:
            frame = frame_buffer[-1].copy()
        else:
            # 如果没有缓存帧，尝试读取一帧
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame.copy())

    # 绘制性能信息
    progress = f"Frame: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)"
    fps_info = f"FPS: {current_fps:.1f} (Processing: {processing_time * 1000:.1f}ms)"
    detection_info = ", ".join([f"{k}: {v}" for k, v in detection_counts.items()])
    status = "PAUSED" if paused else "PLAYING"

    # 绘制信息面板
    cv2.putText(frame, progress, (10, 30), font, font_scale, text_color, font_thickness)
    cv2.putText(frame, fps_info, (10, 70), font, font_scale, text_color, font_thickness)
    cv2.putText(frame, detection_info, (10, 110), font, font_scale, text_color, font_thickness)
    cv2.putText(frame, f"Status: {status}", (10, 150), font, font_scale,
                (0, 0, 255) if paused else (0, 255, 0), font_thickness)
    cv2.putText(frame, "Controls: [SPACE] Pause/Resume, [→] Step, [←] Back", (10, frame_height - 30),
                font, 0.7, (255, 255, 255), 1)

    # 创建性能图表
    metrics_img = np.zeros((300, 600, 3), dtype=np.uint8)

    # 绘制FPS图表
    if len(timing_data['fps_values']) > 1:
        max_fps = max(timing_data['fps_values'][-100:])
        min_fps = min(timing_data['fps_values'][-100:]) if len(timing_data['fps_values']) > 1 else 0
        avg_fps = sum(timing_data['fps_values'][-100:]) / len(timing_data['fps_values'][-100:])

        # 绘制FPS曲线
        for i in range(1, min(100, len(timing_data['fps_values']))):
            idx1 = len(timing_data['fps_values']) - i
            idx2 = len(timing_data['fps_values']) - i - 1
            if idx2 < 0:
                break

            y1 = int(250 - (timing_data['fps_values'][idx1] / max_fps * 200) if max_fps > 0 else 0)
            y2 = int(250 - (timing_data['fps_values'][idx2] / max_fps * 200) if max_fps > 0 else 0)
            cv2.line(metrics_img, (600 - i * 6, y1), (600 - (i + 1) * 6, y2), ((0, 0, 255)), 2)

        # 添加标签
        cv2.putText(metrics_img, "FPS Performance", (10, 20), font, 0.7, (0, 255, 0), 1)
        cv2.putText(metrics_img, f"Current: {current_fps:.1f}", (10, 50), font, 0.6, (0, 255, 0), 1)
        cv2.putText(metrics_img, f"Avg (100f): {avg_fps:.1f}", (10, 80), font, 0.6, (0, 255, 0), 1)
        cv2.putText(metrics_img, f"Min: {min_fps:.1f}, Max: {max_fps:.1f}", (10, 110), font, 0.6, (0, 255, 0), 1)

    # 绘制检测数量图表
    if len(timing_data['detections_per_frame']) > 1:
        max_detections = max(timing_data['detections_per_frame'][-100:]) if timing_data['detections_per_frame'] else 1

        # 绘制检测数量柱状图
        for i in range(min(100, len(timing_data['detections_per_frame']))):
            idx = len(timing_data['detections_per_frame']) - i - 1
            if idx < 0:
                break

            detections = timing_data['detections_per_frame'][idx]
            height = int(detections / max_detections * 100) if max_detections > 0 else 0
            cv2.rectangle(metrics_img, (i * 6, 299), (i * 6 + 4, 299 - height), (255, 0, 0), -1)

        # 添加标签
        cv2.putText(metrics_img, f"Detections: {detections_this_frame}", (300, 50), font, 0.6, (255, 0, 0), 1)
        cv2.putText(metrics_img, f"Total: {sum(detection_counts.values())}", (300, 80), font, 0.6, (255, 0, 0), 1)

    # 显示处理结果
    cv2.imshow('Underwater Object Detection', frame)
    cv2.imshow('Performance Metrics', metrics_img)

    # 保存结果帧
    out.write(frame)

    # 处理键盘输入
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # 空格键暂停/继续
        paused = not paused
        step_mode = False
    elif key == ord('q'):  # q键退出
        break
    elif key == 83 or key == 3:  # 右箭头键 (Windows/Linux: 83, Mac: 3)
        if paused:
            step_mode = True
    elif key == 81 or key == 2:  # 左箭头键 (Windows/Linux: 81, Mac: 2)
        if paused and frame_count > 1 and len(frame_buffer) > 1:
            # 后退一帧
            frame_count -= 1
            # 移除当前帧
            frame_buffer.pop()
            # 获取前一帧
            frame = frame_buffer[-1].copy()
            # 重新处理前一帧（但不计入统计）
            step_mode = True

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

# 打印最终统计信息
print("\n===== Video Processing Summary =====")
print(f"Total frames processed: {frame_count}")
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average FPS: {frame_count / total_time:.2f}")
print("Detection counts:")
for class_name, count in detection_counts.items():
    print(f"  {class_name}: {count} detections")

# 保存性能数据到文件
with open('performance_metrics.csv', 'w') as f:
    f.write("Frame,ProcessingTime(s),FPS,Detections\n")
    for i in range(len(timing_data['frame_times'])):
        frame_time = timing_data['frame_times'][i]
        fps = timing_data['fps_values'][i] if i < len(timing_data['fps_values']) else 0
        detections = timing_data['detections_per_frame'][i] if i < len(timing_data['detections_per_frame']) else 0
        f.write(f"{i + 1},{frame_time:.4f},{fps:.2f},{detections}\n")

print("Performance metrics saved to performance_metrics.csv")