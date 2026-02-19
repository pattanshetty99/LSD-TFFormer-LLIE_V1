import os
import cv2
import torch
from torch.utils.data import Dataset

class LLIE_Dataset(Dataset):
    def __init__(self, low_dir, high_dir):
        self.low_images = sorted(os.listdir(low_dir))
        self.high_images = sorted(os.listdir(high_dir))
        self.low_dir = low_dir
        self.high_dir = high_dir

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low = cv2.imread(os.path.join(self.low_dir,
                                      self.low_images[idx]))
        high = cv2.imread(os.path.join(self.high_dir,
                                       self.high_images[idx]))

        low = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)/255.
        high = cv2.cvtColor(high, cv2.COLOR_BGR2RGB)/255.

        low = torch.tensor(low).permute(2,0,1).float()
        high = torch.tensor(high).permute(2,0,1).float()

        return low, high
