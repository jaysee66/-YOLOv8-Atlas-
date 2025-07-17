import cv2
import argparse
import time
from inference import YOLOv8Inference

# 固定窗口大小
cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video Playback', 1280, 720)

def main():
    parser = argparse.ArgumentParser(description='Video Playback with Model Loading')
    parser.add_argument('--model', type=str, required=True, help='Path to OM model')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--display-scale', type=float, default=0.6, help='Display window scale (0.1-1.0)')
    parser.add_argument('--skip-frames', type=int, default=0, help='Number of frames to skip between processing')
    args = parser.parse_args()

    # 初始化推理引擎（但不使用推理结果）
    detector = YOLOv8Inference(args.model)
    print(f"模型已加载: {args.model}")

    # 打开视频文件
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {args.video}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 控制变量
    paused = False
    frame_count = 0

    # 设置窗口大小
    cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video Playback',
                     int(width * args.display_scale),
                     int(height * args.display_scale))

    print("视频播放中. 按键: 'SPACE' 暂停/继续, '→' 前进一帧, '←' 后退一帧, 'Q' 退出")

    while cap.isOpened():
        if not paused:
            # 跳过指定帧数
            for _ in range(args.skip_frames + 1):
                ret, frame = cap.read()
                frame_count += 1
                if not ret:
                    break

            if not ret:
                break

            # 调用模型但不使用结果
            _ = detector.infer(frame)  # 模型调用但忽略返回值
            
            # 保存原始帧
            out.write(frame)

            # 显示帧
            display_frame = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
            cv2.imshow('Video Playback', display_frame)

        # 处理键盘输入
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # 空格键暂停/继续
            paused = not paused
            print(f"状态: {'已暂停' if paused else '播放中'}")
        elif key == ord('q'):  # 退出
            break
        elif key == 83:  # 右箭头 (前进一帧)
            if paused:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    _ = detector.infer(frame)  # 调用模型
                    display_frame = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale)
                    cv2.imshow('Video Playback', display_frame)
                    out.write(frame)
        elif key == 81:  # 左箭头 (后退一帧)
            if paused and frame_count > 1:
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 2)
                frame_count -= 2
                paused = False

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    detector.release()

    print("\n===== 处理摘要 =====")
    print(f"处理总帧数: {frame_count}")


if __name__ == "__main__":
    main()