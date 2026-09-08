# Some of the codes are borrowed from Borzoi
import os,sys
import torch,math
import torch.nn as nn
import torch.nn.functional as F
from backbone import build_backbone
from layers import ConvBlock

class MultiDilationSepConv(nn.Module):
    def __init__(self, c, k=3, dilations=[1, 2, 4], gn_groups=16):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(c, c, k, padding=d, dilation=d, groups=c, bias=False)
            for d in dilations
        ])
        self.pw = nn.Conv1d(c * len(dilations), c, 1)
        g = min(gn_groups, c)
        while c % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, c, eps=1e-3)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        x = self.pw(torch.cat(outs, dim=1))
        return self.act(self.gn(x))

class SepConv(nn.Module):
    def __init__(self, c, k=3, gn_groups=16):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding="same", groups=c, bias=False)
        self.pw = nn.Conv1d(c, c, 1, bias=True)
        g = min(gn_groups, c)
        while c % g != 0 and g > 1:
            g -= 1
        self.gn = nn.GroupNorm(g, c, eps=1e-3)
        self.act = nn.GELU(approximate="tanh")
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(self.gn(x))
        return x

class rep_upres(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        # Progressive upsampling for 100bp pathway
        self.upsampling_unet1 = nn.Sequential(
            ConvBlock(in_channels = embed_dim, out_channels = embed_dim,  kernel_size = 1),
            nn.Upsample(scale_factor = 5,mode="linear", align_corners=False),
        )

        self.separable1 = MultiDilationSepConv(embed_dim, k=3)
        self.upsampling_unet0 = nn.Sequential(
            ConvBlock(in_channels = embed_dim,out_channels = embed_dim,kernel_size = 1),
            nn.Upsample(scale_factor = 2,mode="linear", align_corners=False),
        )

        self.separable0 = MultiDilationSepConv(embed_dim, k=3)
        self.horizontal_conv0 = ConvBlock(in_channels = 384, out_channels = embed_dim, kernel_size = 1)
        self.horizontal_conv1 = ConvBlock(in_channels = 448, out_channels = embed_dim,kernel_size = 1)
    def forward(self, x, local_embed,atac_raw):
        # print(x.shape,local_embed["3k"].shape,local_embed["6k"].shape)
        x = x.permute(0,2,1)
        x = self.upsampling_unet1(x)
        x += self.horizontal_conv1(local_embed["3k"])
        x = self.separable1(x)
        x = self.upsampling_unet0(x)
        x += self.horizontal_conv0(local_embed["6k"])
        x = self.separable0(x)
        return x.permute(0,2,1)

class Downstream_groseq_model(nn.Module):
    def __init__(self,pretrain_model,upres_unet,embed_dim,crop):
        super().__init__()
        self.head_tt = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
        self.head_bru = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
        # self.head_netcage = nn.Sequential(
        #     nn.Linear(embed_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128,2)
        # ) 
        self.pretrain_model=pretrain_model
        self.upres_unet=upres_unet
        self.crop=crop

    def forward(self,x):
        atac_raw = x[:, 4:5, :]
        x,local_embed =self.pretrain_model(x)
        x=self.upres_unet(x,local_embed,atac_raw)
        tt_pred=self.head_tt(x[:,self.crop*10:-self.crop*10,:] if self.crop > 0 else x)
        bru_pred=self.head_bru(x[:,self.crop*10:-self.crop*10,:] if self.crop > 0 else x)
        # netcage_pred=self.head_netcage(x[:,self.crop*10:-self.crop*10,:] if self.crop > 0 else x)
        return [tt_pred,bru_pred]

def build_model(args):
    pretrain_model=build_backbone(args)
    upres_unet = rep_upres(args.embed_dim)
    model=Downstream_groseq_model(
        pretrain_model=pretrain_model,
        upres_unet=upres_unet,
        embed_dim=args.embed_dim,
        crop=args.crop
    )
    return model
