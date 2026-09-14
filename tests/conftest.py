# Initialize PyTorch DLLs before other native libraries to prevent Windows OpenMP runtime collision
try:
    import torch
except Exception:
    pass
