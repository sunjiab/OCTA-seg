import torch
print("PyTorch 版本:", torch.__version__)
print("CUDA 版本:", torch.version.cuda)
print("GPU 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))

# 测试基本张量运算
x = torch.tensor([1.0, 2.0, 3.0])
print("张量运算测试:", x * 2)
