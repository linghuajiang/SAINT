# some of following codes are borrowed from https://github.com/lucidrains/enformer-pytorch

import math
import torch
from torch import nn, einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class ATACStem(nn.Module):
    """Learnable log-scale + InstanceNorm for RPGC-normalized ATAC signal."""
    def __init__(self, out_channels):
        super().__init__()
        self.log_scale = nn.Parameter(torch.ones(1))
        self.conv = nn.Conv1d(1, out_channels, kernel_size=15, padding="same")
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)

    def forward(self, atac):
        atac = torch.log1p(atac * self.log_scale.abs())
        return self.norm(self.conv(atac))


class ConvBlock(nn.Module):
    def __init__(self, in_channels,out_channels=None, kernel_size=1,
                 conv_type="standard", gn_groups=16):
        super(ConvBlock, self).__init__()
        if conv_type == "separable":
            self.norm = nn.Identity()
            depthwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=in_channels, padding = 'same', bias = False)
            pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.conv_layer = nn.Sequential(depthwise_conv, pointwise_conv)
            self.activation = nn.Identity()
        else:
            g = min(gn_groups, in_channels)
            while in_channels % g != 0 and g > 1:
                g -= 1
            self.norm = nn.GroupNorm(num_groups=g, num_channels=in_channels, eps=0.001)
            # self.norm = nn.BatchNorm1d(in_channels, eps = 0.001)
            self.activation =nn.GELU(approximate='tanh')
            self.conv_layer = nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=kernel_size,
                padding='same')    
            
    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv_layer(x)
        return x

class DualStem(nn.Module):
    def __init__(self, dna_channels=4, atac_channels=1, out_channels=192):
        super().__init__()
        # DNA path
        self.conv_layer = nn.Conv1d(
            dna_channels, out_channels, kernel_size=15, padding="same")

        # ATAC path — independent feature extraction
        self.atac_stem = nn.Sequential(
            nn.Conv1d(atac_channels, out_channels, kernel_size=15, padding="same"),
            nn.GELU(approximate="tanh"),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
        )

        # Cross-modal gate: takes both DNA and ATAC features as input
        # learns per-channel how much ATAC should modulate DNA
        self.cross_gate = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Each latent channel can independently adjust its dependence on ATAC
        self.channel_alpha = nn.Parameter(torch.randn(out_channels) * 0.01)

        self.pool = nn.MaxPool1d(kernel_size=2)


    def forward(self, x):
        dna = x[:, :4, :]
        atac = x[:, 4:5, :]

        # DNA features — identical to pretrained behavior
        dna_feat = self.conv_layer(dna)          # (B, 192, L)

        # ATAC features
        atac_feat = self.atac_stem(atac)         # (B, 192, L)

        # Cross-modal gate conditioned on both signals
        combined = torch.cat([dna_feat, atac_feat], dim=1)  # (B, 384, L)
        gate = self.cross_gate(combined)                     # (B, 192, L)

        # Per-channel residual addition
        # channel_alpha shape: (192,) → (1, 192, 1) for broadcasting
        alpha = torch.sigmoid(self.channel_alpha).view(1, -1, 1)  # (1, 192, 1)
        x = dna_feat + alpha * gate * atac_feat

        return self.pool(x), self.pool(atac_feat)

class ATACDownBlock(nn.Module):
    """
    Downsamples ATAC features to match each encoder stage.
    This branch is lightweight but gives multi-scale ATAC context.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel, gn_groups=16):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=kernel_size, gn_groups=gn_groups),
            nn.MaxPool1d(kernel_size=pool_kernel, padding=0),
        )

    def forward(self, x):
        return self.block(x)


class FiLM1d(nn.Module):
    """
    Feature-wise linear modulation:
        y = x * (1 + gamma) + beta
    gamma/beta are produced from ATAC features at the same scale.
    """
    def __init__(self, x_channels, cond_channels, hidden_channels=None, gn_groups=16):
        super().__init__()
        hidden_channels = hidden_channels or cond_channels

        g = min(gn_groups, cond_channels)
        while cond_channels % g != 0 and g > 1:
            g -= 1

        self.cond_net = nn.Sequential(
            nn.GroupNorm(num_groups=g, num_channels=cond_channels, eps=1e-3),
            nn.GELU(approximate="tanh"),
            nn.Conv1d(cond_channels, hidden_channels, kernel_size=1),
            nn.GELU(approximate="tanh"),
            nn.Conv1d(hidden_channels, 2 * x_channels, kernel_size=1),
        )

    def forward(self, x, cond):
        film = self.cond_net(cond)           # (B, 2C, L)
        gamma, beta = torch.chunk(film, 2, dim=1)
        return x * (1.0 + gamma) + beta


class CNN_atac(nn.Module):
    def __init__(self, in_ch: int = 5):
        super().__init__()

        # Main encoder
        self.stem = DualStem(dna_channels=4, atac_channels=1, out_channels=192)

        self.down1 = nn.Sequential(
            ConvBlock(in_channels=192, out_channels=256, kernel_size=5),
            nn.MaxPool1d(kernel_size=5, padding=0),
        )
        self.down2 = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=320, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
        )
        self.down3 = nn.Sequential(
            ConvBlock(in_channels=320, out_channels=384, kernel_size=5),
            nn.MaxPool1d(kernel_size=5, padding=0),
        )
        self.down4 = nn.Sequential(
            ConvBlock(in_channels=384, out_channels=448, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, padding=0),
        )
        self.down5 = nn.Sequential(
            ConvBlock(in_channels=448, out_channels=512, kernel_size=5),
            nn.MaxPool1d(kernel_size=5, padding=0),
        )

        # Multi-scale ATAC encoder aligned to each stage
        # stem already outputs atac at L/2 with 192 ch
        self.atac_down1 = ATACDownBlock(192, 256, kernel_size=5, pool_kernel=5)  # -> L/10
        self.atac_down2 = ATACDownBlock(256, 320, kernel_size=5, pool_kernel=2)  # -> L/20
        self.atac_down3 = ATACDownBlock(320, 384, kernel_size=5, pool_kernel=5)  # -> L/100

        # FiLM after each block
        self.film1 = FiLM1d(x_channels=256, cond_channels=256)
        self.film2 = FiLM1d(x_channels=320, cond_channels=320)
        self.film3 = FiLM1d(x_channels=384, cond_channels=384)

        self.num_channels = 512

    def forward(self, x: torch.Tensor):
        """
        x: (B, 5, L_bp), channel order = [DNA(4), ATAC(1)]
        returns:
            main output: (B, 512, L/1000)
            skips: dict of intermediate main features
        """
        # Stem
        x, atac_s0 = self.stem(x)        # both at L/2, channels=192

        # Main path + ATAC path + FiLM
        x = self.down1(x)                # (B, 256, L/10)
        atac1 = self.atac_down1(atac_s0) # (B, 256, L/10)
        x = self.film1(x, atac1)
        x1 = x

        x = self.down2(x)                # (B, 320, L/20)
        atac2 = self.atac_down2(atac1)   # (B, 320, L/20)
        x = self.film2(x, atac2)
        x2 = x

        x = self.down3(x)                # (B, 384, L/100)
        atac3 = self.atac_down3(atac2)   # (B, 384, L/100)
        x = self.film3(x, atac3)
        x3 = x

        x = self.down4(x)                # (B, 448, L/200)
        x4 = x

        x = self.down5(x)                # (B, 512, L/1000)

        return x, {"6k": x3, "3k": x4, "30k": x2, "60k": x1}