import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import *
from models.lsd_tf_former import LSD_TFFormer

# ===============================
# SETTINGS
# ===============================

EVAL_LOW = "data/test/low"
EVAL_HIGH = "data/test/high"

device = torch.device(DEVICE)

# ===============================
# Load Model
# ===============================

model = LSD_TFFormer().to(device)
checkpoint = torch.load(f"{SAVE_DIR}/best.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("Model Loaded for Evaluation")

psnr_scores = []
ssim_scores = []

image_list = sorted(os.listdir(EVAL_LOW))

with torch.no_grad():
    for img_name in tqdm(image_list):

        low_path = os.path.join(EVAL_LOW, img_name)
        high_path = os.path.join(EVAL_HIGH, img_name)

        low = cv2.imread(low_path)
        high = cv2.imread(high_path)

        low_rgb = cv2.cvtColor(low, cv2.COLOR_BGR2RGB) / 255.0
        high_rgb = cv2.cvtColor(high, cv2.COLOR_BGR2RGB) / 255.0

        tensor = torch.tensor(low_rgb).permute(2,0,1).unsqueeze(0).float().to(device)

        output = model(tensor)
        output = output.squeeze(0).permute(1,2,0).cpu().numpy()
        output = np.clip(output, 0, 1)

        # PSNR
        psnr_val = peak_signal_noise_ratio(high_rgb, output, data_range=1.0)

        # SSIM
        ssim_val = structural_similarity(high_rgb,
                                          output,
                                          multichannel=True,
                                          data_range=1.0)

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

# ===============================
# Final Results
# ===============================

avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)

print("\n==============================")
print(f"Average PSNR: {avg_psnr:.3f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")
print("==============================")
