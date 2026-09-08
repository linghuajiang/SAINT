import torch,math
from torch import nn, Tensor
from transformers import TransformerEncoderRel
from einops import rearrange,repeat
from layers import CNN_atac
from einops.layers.torch import Rearrange
import copy
from typing import Optional

class OneKbModel(nn.Module):
    def __init__(self,
                 hidden_dim: int = 512,
                 embed_dim: int = 512,
                 num_class: int = 247,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 2,
                 nheads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 bins_1kb: int = 600,
                 crop: int = 50,
                 return_embed: bool = False,
                 return_local: bool = False):
        super().__init__()
        self.backbone = CNN_atac(in_ch=5)
        self.bins_1kb = bins_1kb
        self.crop = crop
        self.return_embed = return_embed
        self.return_local = return_local

        self.input_proj = nn.Conv1d(self.backbone.num_channels, embed_dim, kernel_size=1)

        self.transformer = TransformerEncoderRel(
            d_model=embed_dim,
            nhead=nheads,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,

            key_size=64,
            num_position_features=32,
            attention_dropout=dropout,
            position_dropout=0.01,
            seq_len_train=bins_1kb,
        )

        self.head = nn.Sequential(
            nn.GroupNorm(16, embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1,padding='same'),
            nn.Dropout(0.1),
        )

        self.cls_prediction_head=nn.Linear(embed_dim,num_class)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, 5, 600000)
        return:
            if return_embed: (B, L_1kb, embed_dim)
            else: (B, L_1kb - 2*crop, num_targets)
        """
        # 1) CNN encoder
        x, local_embed = self.backbone(x)

        # 2) project to Transformer hidden_dim
        x = self.input_proj(x)

        # 3) Transformer over 1kb tokens
        x = x.transpose(1, 2)
        x = self.transformer(x)

        # 4) head over 1kb tokens
        x = x.transpose(1, 2)
        x = self.head(x)           
        x_embed = x.transpose(1, 2)

        if self.return_embed:
            return x_embed
        if self.return_local:
            return x_embed, local_embed

        out = self.cls_prediction_head(x_embed[:, self.crop:-self.crop, :])

        return out

def build_backbone(args):
    model = OneKbModel(
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        bins_1kb=args.bins,
        crop=args.crop,
        return_embed=args.return_embed,
        return_local=args.return_local
    )
    return model
