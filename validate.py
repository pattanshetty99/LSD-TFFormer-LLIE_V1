import torch
from utils.metrics import psnr

def validate(model, loader, device):
    model.eval()
    total_psnr = 0

    with torch.no_grad():
        for low, high in loader:
            low, high = low.to(device), high.to(device)
            out = model(low)
            total_psnr += psnr(out, high).item()

    return total_psnr / len(loader)
