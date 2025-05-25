import cv2
import time
import torch
import onnxruntime
import numpy as np
import threading
import kmNet

# UDP 流的地址
udp_url = "udp://@192.168.1.2:9999"

# 创建 VideoCapture 对象
cap = cv2.VideoCapture(udp_url)

# 设置缓冲区大小，确保视频流始终是最新的
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 检查是否成功打开视频流
if not cap.isOpened():
    print("Error: Could not open UDP stream.")
    exit()

# 初始化帧率和延迟变量
frame_count = 0
start_time = time.time()

# 加载 ONNX 模型
model_path = "./weights/cs2警0胸1头匪2胸3头.onnx"  # 替换为你的 ONNX 文件路径

# 指定使用 GPU 执行提供程序
providers = ['CUDAExecutionProvider']
session = onnxruntime.InferenceSession(model_path, providers=providers)

# 获取模型输入的期望形状
input_shape = session.get_inputs()[0].shape
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 模型输入的宽度和高度
model_width = input_shape[2]
model_height = input_shape[3]

# 定义类别颜色映射
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]  # 分别对应类别 0, 1, 2, 3

kmNet.init("192.168.2.188", "5410", "15327019")
last_click_time = time.time()  # 上次点击的时间
click_cooldown = 0.5  # 点击冷却时间，单位为秒


def click():
    kmNet.left(1)  # 鼠标左键按下
    time.sleep(0.05)  # 按下后等待一段时间
    kmNet.left(0)  # 鼠标左键松开
    # kmNet.move(100, 20)


# 创建一个锁对象
click_lock = threading.Lock()


def click_non_blocking():
    # 尝试获取锁，如果获取成功，执行 click 函数，否则直接返回
    if click_lock.acquire(blocking=False):
        try:
            thread = threading.Thread(target=click)
            thread.start()
        finally:
            click_lock.release()  # 立即释放锁，允许其他线程尝试获取
    else:
        print("Another click is already in progress. Discarding this click request.")


def preprocess(frame):
    # 调整大小以匹配模型输入
    resized = cv2.resize(frame, (model_width, model_height))

    # 归一化
    input_data = resized.astype(np.float32) / 255.0

    # 转换为 CHW 格式
    input_data = np.transpose(input_data, (2, 0, 1))

    # 添加批量维度
    input_data = np.expand_dims(input_data, axis=0)

    return input_data


def box_in_center(x1, y1, x2, y2):
    predict_length = 5
    if x1 - predict_length < 320 / 2 < x2 + predict_length and y1 < 320 / 2 < y2:
        return True
    return False


def postprocess(outputs, frame, threshold=0.5):
    global last_click_time
    detections = outputs[0][0]  # 提取 batch 中的第一个元素
    height, width = frame.shape[:2]
    result_frame = frame.copy()  # 创建一个副本以避免修改原始帧

    for detection in detections:
        # 边界框坐标
        x1 = int(detection[0])
        y1 = int(detection[1])
        box_width = int(detection[2])
        box_height = int(detection[3])

        # 置信度
        confidence = float(detection[4])

        # 类别概率
        class_probabilities = detection[5:]
        class_id = np.argmax(class_probabilities)
        class_confidence = class_probabilities[class_id]

        # 合并置信度和类别置信度
        final_confidence = confidence * class_confidence

        if final_confidence < threshold:
            continue

        # 绘制边界框和标签
        color = colors[class_id]
        # cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(result_frame, (int(x1 - box_width / 2), int(y1 - box_height / 2)),
                      (int(x1 + box_width / 2), int(y1 + box_height / 2)), color, 2)
        label = f"{class_id}: {final_confidence:.2f}"
        cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if box_in_center(int(x1 - box_width*1.4 / 2), int(y1 - box_height*1.4 / 2),
                         int(x1 + box_width*1.4 / 2), int(y1 + box_height*1.4 / 2)):
            current_time = time.time()
            if current_time - last_click_time > click_cooldown:
                click_non_blocking()
                last_click_time = current_time

    # 在画面中心绘制一个十字线
    center_x, center_y = width // 2, height // 2
    line_length = 5
    cv2.line(result_frame, (center_x - line_length, center_y), (center_x + line_length, center_y), (255, 0, 0), 2)
    cv2.line(result_frame, (center_x, center_y - line_length), (center_x, center_y + line_length), (255, 0, 0), 2)


    return result_frame


while True:
    ret, frame = cap.read()

    if ret:
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        input_data = preprocess(frame)

        # 运行推理
        outputs = session.run([output_name], {input_name: input_data})

        result_frame = postprocess(outputs, frame)
        # result_frame = frame

        cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('UDP Stream with Detection', result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Error: Could not read frame.")
        continue

cap.release()
cv2.destroyAllWindows()
