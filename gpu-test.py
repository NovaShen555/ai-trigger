import torch
import onnxruntime

print(onnxruntime.__version__)
print(onnxruntime.get_device())  # 如果得到的输出结果是GPU，所以按理说是找到了GPU的
ort_session = onnxruntime.InferenceSession("./weights/cs2警0胸1头匪2胸3头.onnx",
                                           providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())
