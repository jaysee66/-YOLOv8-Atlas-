# inference.py
import cv2
import numpy as np
import acl
import time
import ctypes
from ctypes import *

# 定义ACL常量
ACL_SUCCESS = 0
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2

class AclResource:
    def __init__(self):
        self.device_id = 0
        self.context = None
        self.stream = None

    def init(self):
        # 初始化ACL
        ret = acl.init()
        if ret != ACL_SUCCESS:
            raise Exception(f"ACL init failed, ret={ret}")

        # 设置设备
        ret = acl.rt.set_device(self.device_id)
        if ret != ACL_SUCCESS:
            acl.finalize()
            raise Exception(f"ACL set device failed, ret={ret}")

        # 创建Context
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != ACL_SUCCESS:
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            raise Exception(f"ACL create context failed, ret={ret}")

        # 创建Stream
        self.stream, ret = acl.rt.create_stream()
        if ret != ACL_SUCCESS:
            acl.rt.destroy_context(self.context)
            acl.rt.reset_device(self.device_id)
            acl.finalize()
            raise Exception(f"ACL create stream failed, ret={ret}")

        print("ACL resource initialized successfully")

    def release(self):
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("ACL resources released")

class YOLOv8Inference:
    def __init__(self, model_path):
        # 初始化ACL资源
        self.acl_resource = AclResource()
        self.acl_resource.init()
        
        # 加载模型
        self.model_path = model_path
        self.model_id = None
        self.input_data = None
        self.output_data = None
        self.model_id, self.input_data, self.output_data = self._load_model()
        
        # 模型参数
        self.input_width = 640
        self.input_height = 640
        self.class_names = ['holothurian', 'echinus', 'scallop', 'starfish']
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        self.num_classes = len(self.class_names)
        
        # 性能统计
        self.frame_count = 0
        self.total_time = 0
        self.detection_counts = {name: 0 for name in self.class_names}
        
        # 预分配内存
        self.input_buffer = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)

    def _load_model(self):
        # 加载OM模型
        result = acl.mdl.load_from_file(self.model_path)
        
        # 检查返回类型
        if isinstance(result, tuple):
            model_id = result[0]
        else:
            model_id = result
        
        # 检查模型ID是否有效
        if model_id == 0:
            error_msg = acl.util.get_error() if hasattr(acl.util, 'get_error') else "Unknown error"
            raise ValueError(f"Model load failed: {error_msg}")
        
        # 获取模型描述信息
        model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(model_desc, model_id)
        if ret != ACL_SUCCESS:
            acl.mdl.unload(model_id)
            raise ValueError(f"Get model description failed, ret={ret}")
        
        # 获取输入尺寸
        input_size = acl.mdl.get_input_size_by_index(model_desc, 0)
        output_size = acl.mdl.get_output_size_by_index(model_desc, 0)
        
        # 分配输入输出内存
        input_result = None
        output_result = None
        
        try:
            # 分配输入内存
            input_result = acl.rt.malloc(input_size, ACL_MEM_MALLOC_HUGE_FIRST)
            # 分配输出内存（fp16）
            output_result = acl.rt.malloc(output_size, ACL_MEM_MALLOC_HUGE_FIRST)
            
            # 检查内存分配是否成功
            if input_result is None or output_result is None:
                raise RuntimeError("Memory allocation failed")
        except Exception as e:
            # 清理资源
            if model_desc:
                acl.mdl.destroy_desc(model_desc)
            if model_id:
                acl.mdl.unload(model_id)
            raise RuntimeError(f"Memory allocation error: {str(e)}")
        
        # 处理返回值
        if isinstance(input_result, tuple):
            input_data = input_result[0]
        else:
            input_data = input_result
        
        if isinstance(output_result, tuple):
            output_data = output_result[0]
        else:
            output_data = output_result
        
        # 打印模型信息
        print(f"Model loaded: input_size={input_size}, output_size={output_size}")
        
        return model_id, input_data, output_data

    def preprocess(self, frame):
        # 硬件加速缩放
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        # 使用单精度浮点
        normalized = resized.astype(np.float32) / 255.0
        
        # 转换为NCHW格式
        return normalized.transpose(2, 0, 1)[np.newaxis, :, :, :]

    def infer(self, frame):
        try:
            start_time = time.time()
            
            # 预处理并填充预分配缓冲区
            resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            self.input_buffer[0] = normalized.transpose(2, 0, 1)
            
            # 获取内存地址和大小
            input_ptr = self.input_buffer.ctypes.data_as(POINTER(c_float))
            input_size = self.input_buffer.nbytes
            
            # 确保地址是整数类型
            input_ptr_int = ctypes.addressof(input_ptr.contents) if hasattr(input_ptr, 'contents') else ctypes.cast(input_ptr, c_void_p).value
            
            # 执行内存拷贝
            ret = acl.rt.memcpy(self.input_data, input_size, 
                                input_ptr_int, input_size, 
                                ACL_MEMCPY_HOST_TO_DEVICE)
            if ret != ACL_SUCCESS:
                print(f"Memory copy failed, ret={ret}")
                return []
        
            # 创建输入输出数据集
            input_dataset = acl.mdl.create_dataset()
            input_buffer = acl.create_data_buffer(self.input_data, input_size)
            acl.mdl.add_dataset_buffer(input_dataset, input_buffer)
            
            # 获取输出大小
            model_desc = acl.mdl.create_desc()
            acl.mdl.get_desc(model_desc, self.model_id)
            output_size_val = acl.mdl.get_output_size_by_index(model_desc, 0)
            
            # 创建输出数据集
            output_dataset = acl.mdl.create_dataset()
            output_buffer = acl.create_data_buffer(self.output_data, output_size_val)
            acl.mdl.add_dataset_buffer(output_dataset, output_buffer)
            
            # 执行推理
            ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
            if ret != ACL_SUCCESS:
                print(f"Model execute failed, ret={ret}")
                return []
            
            # 将输出数据复制回主机（fp16）
            output_data_ptr = acl.get_data_buffer_addr(output_buffer)
            output_data_size = acl.get_data_buffer_size(output_buffer)
            
            # 创建fp16数组接收数据
            num_output_elements = output_data_size // 2  # fp16每个元素2字节
            host_output = np.zeros(num_output_elements, dtype=np.float16)
            
            # 获取主机内存指针
            host_ptr = host_output.ctypes.data_as(POINTER(c_ushort))
            host_ptr_int = ctypes.addressof(host_ptr.contents)
            
            # 执行内存拷贝
            ret = acl.rt.memcpy(host_ptr_int, output_data_size,
                                output_data_ptr, output_data_size,
                                ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != ACL_SUCCESS:
                print(f"Output memory copy failed, ret={ret}")
                return []
            
            # 后处理
            detections = self.postprocess(host_output, frame.shape)
            
            # 更新性能统计
            self.frame_count += 1
            self.total_time += time.time() - start_time
            
            # 销毁数据集和缓冲区
            acl.destroy_data_buffer(input_buffer)
            acl.destroy_data_buffer(output_buffer)
            acl.mdl.destroy_dataset(input_dataset)
            acl.mdl.destroy_dataset(output_dataset)
            
            return detections
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def postprocess(self, output, orig_shape):
        try:
            # 将fp16输出转换为float32进行计算
            output_data = output.astype(np.float32)
            
            # 计算元素总数
            total_elements = output_data.size
            print(f"Total elements in output: {total_elements}")
            
            # 计算每个检测框的元素数 (4个坐标 + 4个类别分数)
            elements_per_box = 4 + self.num_classes
            
            # 确保总元素数可以被每个检测框的元素数整除
            if total_elements % elements_per_box != 0:
                print(f"Unexpected output size: {total_elements} elements, not divisible by {elements_per_box}")
                return []
            
            # 计算检测框数量
            num_boxes = total_elements // elements_per_box
            print(f"Number of boxes: {num_boxes}")
            
            # 重塑为(num_boxes, elements_per_box)
            output_data = output_data.reshape(num_boxes, elements_per_box)
            
            # 分离边界框坐标和类别分数
            boxes = output_data[:, :4]  # cx, cy, w, h (归一化坐标)
            scores = output_data[:, 4:4 + self.num_classes]  # 类别分数
            
            # 找到每个框的最佳类别和置信度
            class_ids = np.argmax(scores, axis=1)
            max_scores = np.max(scores, axis=1)
            
            # 应用置信度阈值
            valid_mask = max_scores > 0.5
            boxes = boxes[valid_mask]
            max_scores = max_scores[valid_mask]
            class_ids = class_ids[valid_mask]
            
            # 如果没有检测结果，返回空列表
            if len(boxes) == 0:
                return []
            
            # 获取原始图像尺寸
            orig_h, orig_w = orig_shape[:2]
            
            # 计算缩放比例 (保持宽高比)
            scale = min(self.input_width / orig_w, self.input_height / orig_h)
            pad_w = (self.input_width - orig_w * scale) / 2
            pad_h = (self.input_height - orig_h * scale) / 2
            
            # 坐标转换（向量化）
            cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            
            # 将归一化坐标转换为绝对坐标 (考虑缩放和填充)
            # 首先移除填充并反转缩放
            x1 = (cx - w/2 - pad_w) / scale
            y1 = (cy - h/2 - pad_h) / scale
            x2 = (cx + w/2 - pad_w) / scale
            y2 = (cy + h/2 - pad_h) / scale
            
            # 边界裁剪
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            # 构建结果
            detections = []
            for i in range(len(x1)):
                class_id = int(class_ids[i])
                
                # 计算边界框坐标
                x_min, y_min = int(x1[i]), int(y1[i])
                x_max, y_max = int(x2[i]), int(y2[i])
                
                # 确保边界框有效
                if x_max > x_min and y_max > y_min:
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id],
                        'confidence': float(max_scores[i]),
                        'bbox': [x_min, y_min, x_max, y_max]
                    })
                    self.detection_counts[self.class_names[class_id]] += 1
                else:
                    print(f"Invalid bbox: ({x_min}, {y_min}, {x_max}, {y_max})")
            
            return detections
        except Exception as e:
            print(f"Postprocess error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def draw_results(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            color = self.colors[det['class_id']]
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制性能信息
        if self.frame_count > 0:
            avg_time = self.total_time / self.frame_count
            current_fps = 1.0 / avg_time
        else:
            current_fps = 0
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制检测统计
        stats_text = ", ".join([f"{k}: {v}" for k, v in self.detection_counts.items()])
        cv2.putText(frame, stats_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return frame

    def release(self):
        # 释放模型相关资源
        if self.model_id:
            acl.mdl.unload(self.model_id)
        if self.input_data:
            acl.rt.free(self.input_data)
        if self.output_data:
            acl.rt.free(self.output_data)
        # 释放ACL资源
        self.acl_resource.release()