import torch 


# Disable TF32 precision
# Reduces precision for improved speed
# With small values we want full precision 
torch.backends.cudnn.allow_tf32 = False

# Disable reduced precision 
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# Set default tensors to 64-bit float 
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# Enable full precision
torch.backends.cudnn.enabled = True

# Stops the backend from choosing the fastest algorithms 
torch.backends.cudnn.benchmark = False

# Use MKL for linalg backend for accuracy 
if torch.backends.mkl.is_available():
    torch.backends.mkl.enabled = True

def set_torch_config():
    # Set backend flags and settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float64)
    torch.set_default_tensor_type(torch.DoubleTensor)
    if torch.backends.mkl.is_available():
        torch.backends.mkl.enabled = True
