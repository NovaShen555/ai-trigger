import cv2
import time

# UDP 流的地址
udp_url = "udp://@192.168.1.2:9999"

# 创建 VideoCapture 对象
cap = cv2.VideoCapture(udp_url)

# 检查是否成功打开视频流
if not cap.isOpened():
    print("Error: Could not open UDP stream.")
    exit()

# 初始化帧率和延迟变量
frame_count = 0
start_time = time.time()

while True:
    # 获取当前时间用于计算延迟
    current_time = time.time()

    # 读取一帧
    ret, frame = cap.read()

    # 如果读取成功，显示帧
    if ret:

        # 计算帧率
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # 在左上角显示帧率和延迟
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('UDP Stream with FPS and Delay', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Error: Could not read frame.")
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()