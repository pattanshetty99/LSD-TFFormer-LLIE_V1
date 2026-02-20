import os
import cv2
import torch
import glob
from torch.utils.data import Dataset


class LLIE_Dataset(Dataset):
    def __init__(self, low_dir, high_dir, extensions=("png", "jpg", "jpeg")):
        """
        Args:
            low_dir (str): Path to low-light images
            high_dir (str): Path to ground-truth images
            extensions (tuple): Supported image extensions
        """

        self.low_dir = low_dir
        self.high_dir = high_dir

        # Collect all supported image files
        self.low_paths = []
        self.high_paths = []

        for ext in extensions:
            self.low_paths += glob.glob(os.path.join(low_dir, f"*.{ext}"))
            self.high_paths += glob.glob(os.path.join(high_dir, f"*.{ext}"))

        # Sort to ensure pairing consistency
        self.low_paths = sorted(self.low_paths)
        self.high_paths = sorted(self.high_paths)

        # Safety checks
        if len(self.low_paths) == 0:
            raise ValueError(f"No images found in {low_dir}")

        if len(self.high_paths) == 0:
            raise ValueError(f"No images found in {high_dir}")

        if len(self.low_paths) != len(self.high_paths):
            raise ValueError(
                f"Mismatch: {len(self.low_paths)} low images "
                f"and {len(self.high_paths)} high images"
            )

        print(f"Loaded {len(self.low_paths)} image pairs.")

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):

        low_path = self.low_paths[idx]
        high_path = self.high_paths[idx]

        # Read images
        low = cv2.imread(low_path)
        high = cv2.imread(high_path)

        if low is None:
            raise ValueError(f"Failed to load image: {low_path}")
        if high is None:
            raise ValueError(f"Failed to load image: {high_path}")

        # Convert BGR → RGB
        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB) / 255.0
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB) / 255.0

        # Convert to tensor (C, H, W)
        low = torch.from_numpy(low).permute(2, 0, 1).float()
        high = torch.from_numpy(high).permute(2, 0, 1).float()

        return low, high
