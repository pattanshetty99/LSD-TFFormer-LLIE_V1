import torch

def psnr(pred, target):
    mse = torch.mean((pred-target)**2)
    return 20*torch.log10(1.0/torch.sqrt(mse))
