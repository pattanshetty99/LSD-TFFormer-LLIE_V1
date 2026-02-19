import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import *
from models.lsd_tf_former import LSD_TFFormer
from datasets.llie_dataset import LLIE_Dataset
from losses.ssim import ssim_loss
from losses.perceptual import PerceptualLoss
from utils.scheduler import get_scheduler
from utils.checkpoint import save_checkpoint
from validate import validate

device = torch.device(DEVICE)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

train_dataset = LLIE_Dataset(TRAIN_LOW, TRAIN_HIGH)
val_dataset = LLIE_Dataset(VAL_LOW, VAL_HIGH)

train_loader = DataLoader(train_dataset,
                          BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(val_dataset,
                        BATCH_SIZE)

model = LSD_TFFormer().to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LR)

scheduler = get_scheduler(optimizer,
                          EPOCHS)

scaler = GradScaler()

l1 = nn.L1Loss()
perceptual = PerceptualLoss()

best_psnr = 0

for epoch in range(EPOCHS):
    model.train()

    for low, high in train_loader:
        low, high = low.to(device), high.to(device)

        optimizer.zero_grad()

        with autocast():
            out = model(low)
            loss = l1(out, high) \
                   + 0.1*ssim_loss(out, high) \
                   + 0.1*perceptual(out, high)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

    val_psnr = validate(model,
                        val_loader,
                        device)

    print(f"Epoch {epoch} | PSNR: {val_psnr:.3f}")

    if val_psnr > best_psnr:
        best_psnr = val_psnr
        save_checkpoint(model,
                        optimizer,
                        epoch,
                        f"{SAVE_DIR}/best.pth")
