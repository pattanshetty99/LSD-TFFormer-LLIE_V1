import os

TRAIN_LOW = "data/train/low"
TRAIN_HIGH = "data/train/high"
VAL_LOW = "data/val/low"
VAL_HIGH = "data/val/high"

BATCH_SIZE = 2
EPOCHS = 250
LR = 2e-4
PATCH_SIZE = None  # full 512
NUM_WORKERS = 4

SAVE_DIR = "checkpoints"
DEVICE = "cuda"

os.makedirs(SAVE_DIR, exist_ok=True)
