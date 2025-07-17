import numpy as np
import os
from PIL import Image
import glob


def preprocess_image(image_path, target_size=(224, 224)):
    """图像预处理函数"""
    img = Image.open(image_path)
    img = img.resize(target_size)  # 调整尺寸
    img = np.array(img).astype(np.float32) / 255.0  # 归一化
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, 0)  # 添加batch维度
    return img


# 配置参数
data_dir = "calibration_data"
image_dir = "D:/code/shuixia_homework/modol_traning/train/image"
input_name = "input"  # 模型输入节点名称
num_samples = 100

# 获取图像文件列表
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))[:num_samples]

# 创建目录
os.makedirs(data_dir, exist_ok=True)

# 处理并保存校准数据
for i, img_path in enumerate(image_files):
    try:
        # 预处理图像
        processed_img = preprocess_image(img_path)

        # 保存为二进制文件
        filename = f"{input_name}_{i}.bin"
        filepath = os.path.join(data_dir, filename)
        processed_img.tofile(filepath)

        print(f"处理: {img_path} -> {filepath}")
    except Exception as e:
        print(f"处理 {img_path} 失败: {str(e)}")

print(f"\n校准数据准备完成! 共处理 {len(image_files)} 张图像")