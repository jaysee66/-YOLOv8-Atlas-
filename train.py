import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


# 1. 数据准备 - 将VOC格式转换为YOLO格式
def change_voc_to_yolo(xml_path, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)

    annotations = []
    for obj in root.findall('object'):
        # 过滤水草类别
        if obj.find('name').text == 'waterweeds':
            continue

        class_name = obj.find('name').text

        if class_name not in classes:
            continue

        class_id = classes.index(class_name)
        # 边界框坐标转换
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # YOLO格式转换（归一化中心坐标）
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        # 处理越界标注
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.01, min(1.0, width))  # 防止零宽
        height = max(0.01, min(1.0, height))

        annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )

    return annotations


# 2. 创建数据集目录结构
def new_dataset(base_path):
    # 创建目录
    os.makedirs(os.path.join(base_path, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels', 'val'), exist_ok=True)

    # 定义目标类别（过滤水草）
    classes = ['holothurian', 'echinus', 'scallop', 'starfish']

    # 获取所有XML标注文件
    xml_dir = os.path.join(base_path, 'box')
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    # 分层抽样划分训练/验证集（8:2）
    train_files, val_files = train_test_split(
        xml_files,
        test_size=0.2,
        random_state=42 # 确保可复现
    )

    # 处理训练集
    for xml_file in train_files:
        base_name = os.path.splitext(xml_file)[0]
        img_src = os.path.join(base_path, 'image', f"{base_name}.jpg")
        img_dst = os.path.join(base_path, 'images', 'train', f"{base_name}.jpg")

        # 复制图像
        shutil.copy(img_src, img_dst)

        # 转换并保存标签
        annotations = change_voc_to_yolo(os.path.join(xml_dir, xml_file), classes)
        with open(os.path.join(base_path, 'labels', 'train', f"{base_name}.txt"), 'w') as f:
            f.write("\n".join(annotations))

    # 处理验证集
    for xml_file in val_files:
        base_name = os.path.splitext(xml_file)[0]
        img_src = os.path.join(base_path, 'image', f"{base_name}.jpg")
        img_dst = os.path.join(base_path, 'images', 'val', f"{base_name}.jpg")

        # 复制图像
        shutil.copy(img_src, img_dst)

        # 转换并保存标签
        annotations = change_voc_to_yolo(os.path.join(xml_dir, xml_file), classes)
        with open(os.path.join(base_path, 'labels', 'val', f"{base_name}.txt"), 'w') as f:
            f.write("\n".join(annotations))

    # 创建数据集配置文件
    dataset_config = f"""
    path: {base_path}
    train: images/train
    val: images/val

    names:
      0: holothurian
      1: echinus
      2: scallop
      3: starfish
    """

    with open(os.path.join(base_path, 'underwater.yaml'), 'w') as f:
        f.write(dataset_config)

    return os.path.join(base_path, 'underwater.yaml')


# 3. 训练YOLOv8模型
def train_v8(dataset_yaml):
    # 加载预训练模型 (选择合适尺寸)
    model = YOLO('yolov8m.pt')  # 中等尺寸模型

    # 训练配置
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,  # 匹配部署尺寸
        batch=16,  # 根据GPU显存调整
        device=0,  # 使用第一块GPU
        name='underwater_v1',  # 实验名称
        optimizer='AdamW',  # 优化器选择
        lr0=0.001,  # 初始学习率
        lrf=0.01,  # 最终学习率 = 0.001 * 0.01=0.00001
        weight_decay=0.0005,
        dropout=0.1,  # 防止过拟合
        hsv_h=0.015,  # 色相扰动±1.5%
        hsv_s=0.7,  # 饱和度增强
        hsv_v=0.4,  # 明度增强
        translate=0.1,  # 平移增强
        scale=0.5,  # 缩放增强
        flipud=0.3,  # 上下翻转概率
        fliplr=0.5,  # 左右翻转概率
        mosaic=1.0,  # Mosaic增强概率
        mixup=0.1,  # Mixup增强概率
        copy_paste=0.1,  # 复制粘贴增强
        auto_augment='randaugment',  # 自动增强策略
        erasing=0.4,  # 随机擦除概率
        close_mosaic=10,  # 最后10个epoch关闭mosaic
        patience=20,  # 早停耐心值
        save_period=5,  # 每5个epoch保存一次
        amp=True  # 自动混合精度
    )

    return results


# 4. 模型验证
def verify_model(model_path, dataset_yaml):
    model = YOLO(model_path)
    metrics = model.val(
        data=dataset_yaml,
        batch=16,
        imgsz=640,
        conf=0.001,  # 低置信度阈值以评估召回率
        iou=0.6,  # NMS的IoU阈值
        save_json=True,  # 保存JSON格式结果
        save_hybrid=True,  # 保存混合结果
        plots=True  # 生成评估图表
    )

    # 打印关键指标
    print(f"mAP@0.5: {metrics.box.map:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map50_95:.4f}")
    print(f"Precision: {metrics.box.precision.mean():.4f}")
    print(f"Recall: {metrics.box.recall.mean():.4f}")

    return metrics


# 主执行流程
if __name__ == "__main__":
    # 设置路径
    BASE_PATH = './train'  # 数据集根目录

    # 步骤1: 准备数据集
    dataset_yaml = new_dataset(BASE_PATH)
    print(f"Dataset prepared at: {dataset_yaml}")
    print("数据集准备完成")
    # 步骤2: 训练模型
    train_results = train_v8(dataset_yaml)

    best_model_path = 'D:/code/shuixia_homework/modol_traning/runs/detect/underwater_v1/weights/best.pt'

    best_model = YOLO(best_model_path)

    print("模型训练完成")

    # 步骤3: 验证模型
    metrics = verify_model(best_model, dataset_yaml)

    print("验证模型完成")