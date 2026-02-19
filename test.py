import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

from config import *
from models.lsd_tf_former import LSD_TFFormer

# ===============================
# SETTINGS
# ===============================

TEST_LOW = "data/test/low"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device(DEVICE)

# ===============================
# Load Model
# ===============================

model = LSD_TFFormer().to(device)
checkpoint = torch.load(f"{SAVE_DIR}/best.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

print("Model Loaded Successfully")
print("Using device:", device)

# ===============================
# Inference
# ===============================

image_list = sorted(os.listdir(TEST_LOW))

with torch.no_grad():
    for img_name in tqdm(image_list):

        img_path = os.path.join(TEST_LOW, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().to(device)

        output = model(tensor)

        output = output.squeeze(0).permute(1,2,0).cpu().numpy()
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), output)

print("Testing Completed. Results saved in:", OUTPUT_DIR)
