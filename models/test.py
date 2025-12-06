import torch

# Which GPU is being used
print(torch.cuda.get_device_name(0))

# How many GPUs are available
print(torch.cuda.device_count())

# Current GPU index
print(torch.cuda.current_device())

# CUDA version
print(torch.version.cuda)