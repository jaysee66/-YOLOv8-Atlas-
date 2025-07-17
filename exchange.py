from ultralytics import YOLO

# 加载训练完成的检测模型
model = YOLO('runs/detect/underwater_v1/weights/best.pt')

# 将模型转换为ONNX格式
success = model.export(
    format='onnx',
    imgsz=640,
    simplify=True,
    opset=12,
    dynamic=False
)
if success:
    print("ONNX export successful")
else:
    print("ONNX export failed")