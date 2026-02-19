import torch.nn.functional as F

def ssim_loss(x, y):
    return 1 - F.mse_loss(x, y)
