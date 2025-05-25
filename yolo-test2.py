import cv2
import time
import numpy as np
import onnxruntime as ort

# 配置参数
conf_threshold = 0.5  # 置信度阈值
iou_threshold = 0.4  # NMS的IOU阈值
class_names = ['class0', 'class1', 'class2', 'class3']  # 类别名称

# 加载ONNX模型
model_path = "./weights/cs2警0胸1头匪2胸3头.onnx"  # 替换为您的ONNX文件路径
session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # 获取输入形状 (1,3,640,640)

# UDP流地址
udp_url = "udp://@192.168.1.2:9999"
cap = cv2.VideoCapture(udp_url)


def preprocess(frame):
    # 调整大小并保持宽高比填充为正方形
    h, w = frame.shape[:2]
    scale = min(640 / w, 640 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # 创建填充后的画布
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # 预处理 (BGR转RGB、归一化、调整维度)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True)
    return blob, (scale, (w, h))


def nms(boxes, scores, iou_threshold):
    # 简单实现NMS
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess(outputs, meta):
    # 解析模型输出
    outputs = outputs[0][0]  # 假设输出形状为(1, num_boxes, 6)
    boxes = outputs[:, :4]
    scores = outputs[:, 4]
    class_ids = outputs[:, 5].astype(int)

    # 过滤低置信度检测
    valid = scores > conf_threshold
    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    # 转换坐标到原始图像尺寸
    scale, (orig_w, orig_h) = meta
    boxes /= scale  # 缩放回原始尺寸
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # 应用NMS
    if len(boxes) > 0:
        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

    return boxes.astype(int), class_ids, scores


# 初始化帧率计算
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理
    blob, meta = preprocess(frame)

    # 推理
    outputs = session.run(None, {input_name: blob})

    # 后处理
    boxes, class_ids, scores = postprocess(outputs, meta)

    # 绘制检测结果
    for box, cls_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = box
        color = (0, 255, 0)  # 绿色框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]}: {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 计算并显示FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('UDP Stream with Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()