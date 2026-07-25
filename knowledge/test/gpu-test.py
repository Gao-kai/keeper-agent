import torch

print(f"是否支持GPU:{torch.cuda.is_available()}")
print(f"设备名:{torch.cuda.get_device_name()}")