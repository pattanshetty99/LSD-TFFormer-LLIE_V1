import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import DropPath, LayerScale

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    windows = x.permute(0,1,3,2,4,5).contiguous()
    return windows.view(-1, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B,
                     H // window_size,
                     W // window_size,
                     window_size,
                     window_size,
                     -1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    return x.view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3,
                                  self.num_heads,
                                  C // self.num_heads)
        q,k,v = qkv.permute(2,0,3,1,4)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1,2).reshape(B_,N,C)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window_size=8,
                 shift_size=0, drop_path=0.1):

        super().__init__()

        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window_size)

        self.ls1 = LayerScale(dim)
        self.drop_path = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )
        self.ls2 = LayerScale(dim)

    import torch.nn.functional as F

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x

        # LayerNorm
        x = self.norm1(x)

        # 🔥 PAD TO MULTIPLE OF WINDOW SIZE
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = x.shape[1], x.shape[2]

        # Shift
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2)
            )

        # Window partition
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)

        # Reverse windows
        x = window_reverse(
            attn_windows,
            self.window_size,
            Hp, Wp
        )

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )

        # 🔥 REMOVE PAD
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :]

        # Residual 1
        x = shortcut + self.drop_path(self.ls1(x))

        # Residual 2
        x = x + self.drop_path(
            self.ls2(
                self.mlp(self.norm2(x))
            )
        )

        return x
