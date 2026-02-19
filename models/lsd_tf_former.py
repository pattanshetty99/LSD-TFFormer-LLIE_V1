import torch
import torch.nn as nn
from models.swin_blocks import SwinBlock

class LSD_TFFormer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 base_dim=64,
                 depth=6,
                 heads=8,
                 window_size=8):

        super().__init__()

        # ===== Encoder =====
        self.stem = nn.Conv2d(in_channels, base_dim, 3,1,1)

        self.down1 = nn.Conv2d(base_dim,
                               base_dim*2,
                               4,2,1)

        self.down2 = nn.Conv2d(base_dim*2,
                               base_dim*4,
                               4,2,1)

        # ===== Swin Transformer Bottleneck =====
        self.blocks = nn.ModuleList([
            SwinBlock(base_dim*4,
                      heads,
                      window_size,
                      shift_size=0 if i%2==0 else window_size//2)
            for i in range(depth)
        ])

        # ===== Decoder =====
        self.up1 = nn.ConvTranspose2d(
            base_dim*4,
            base_dim*2,
            4,2,1
        )

        self.up2 = nn.ConvTranspose2d(
            base_dim*2,
            base_dim,
            4,2,1
        )

        self.output = nn.Conv2d(base_dim,
                                in_channels,
                                3,1,1)

    def forward(self, x):

        x1 = self.stem(x)      # 512
        x2 = self.down1(x1)    # 256
        x3 = self.down2(x2)    # 128

        B,C,H,W = x3.shape
        x3 = x3.permute(0,2,3,1)

        for blk in self.blocks:
            x3 = blk(x3)

        x3 = x3.permute(0,3,1,2)

        x = self.up1(x3) + x2
        x = self.up2(x) + x1

        return self.output(x)
