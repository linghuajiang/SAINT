import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import copy
import math
import torch.nn as nn

def borzoi_central_mask(positions, features, seq_len_train):
    """
    positions: (2L - 1,)
    return:    (2L - 1, features)
    """
    pow_rate = math.exp(
        math.log(seq_len_train + 1) / features
    )

    center_widths = torch.pow(
        torch.tensor(
            pow_rate,
            device=positions.device,
            dtype=torch.float32
        ),
        torch.arange(
            1,
            features + 1,
            device=positions.device,
            dtype=torch.float32
        )
    ) - 1.0

    return (center_widths[None, :] > positions.abs()[:, None]).float()


def borzoi_positional_embed(seq_len, feature_size, device, seq_len_train=None):
    if feature_size % 2 != 0:
        raise ValueError("num_position_features must be divisible by 2")

    if seq_len_train is None:
        seq_len_train = seq_len

    distances = torch.arange(-seq_len + 1, seq_len, device=device, dtype=torch.float32)

    base = borzoi_central_mask(distances, feature_size // 2, seq_len_train)

    directional = (
        torch.sign(distances)[:, None] * base
    )

    return torch.cat([base, directional], dim=-1)

def relative_shift(x):
    """
    x:
        (B, H, L, 2L-1)

    return:
        (B, H, L, L)
    """
    B, H, L, _ = x.shape

    x = torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1)                           # (B,H,L,2L)

    t2 = x.size(-1)

    x = x.reshape(B, H, t2, L)
    x = x[:, :, 1:, :]
    x = x.reshape(B, H, L, t2 - 1)

    return x[:, :, :, : (t2 + 1) // 2]


# ============================================================
# Borzoi-style MHA
# ============================================================

class MHSA_RelBias(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        key_size=None,
        num_position_features=32,
        attention_dropout=0.1,
        position_dropout=0.01,
        seq_len_train=600,
        zero_init_output=True,
    ):
        super().__init__()

        assert d_model % nhead == 0
        if key_size is None:
            key_size = d_model // nhead

        self.nhead = nhead
        self.key_size = key_size
        self.value_size = d_model // nhead

        self.scale = key_size ** -0.5

        self.num_position_features = (num_position_features)
        self.seq_len_train = seq_len_train

        # Q / K use Borzoi-style fixed key dimension
        self.to_q = nn.Linear(d_model, key_size * nhead, bias=False)
        self.to_k = nn.Linear(d_model, key_size * nhead, bias=False)

        # V preserves d_model across all heads
        self.to_v = nn.Linear(d_model, self.value_size * nhead, bias=False)
        self.to_rel_k = nn.Linear(num_position_features, key_size * nhead, bias=False)

        self.rel_content_bias = nn.Parameter(
            torch.empty(
                1, nhead, 1, key_size
            )
        )

        self.rel_pos_bias = nn.Parameter(
            torch.empty(
                1, nhead, 1, key_size
            )
        )

        nn.init.normal_(self.rel_content_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=0.02)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.pos_dropout = nn.Dropout(position_dropout)

        self.to_out = nn.Linear(self.value_size * nhead, d_model)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
            nn.init.zeros_(self.to_out.bias)


    def _split_heads(self, x, head_dim):
        B, L, _ = x.shape

        return (x.view(B, L, self.nhead, head_dim).transpose(1, 2))


    def forward(self, x, key_padding_mask=None):
        B, L, _ = x.shape

        # ----------------------------------------------------
        # Q / K / V
        # ----------------------------------------------------

        q = self._split_heads(
            self.to_q(x),
            self.key_size
        )

        k = self._split_heads(
            self.to_k(x),
            self.key_size
        )

        v = self._split_heads(
            self.to_v(x),
            self.value_size
        )

        q = q * self.scale

        rw = self.rel_content_bias.to(q.dtype)
        rr = self.rel_pos_bias.to(q.dtype)

        content_logits = torch.einsum(
            "bhid,bhjd->bhij",
            q + rw,
            k
        )

        # ----------------------------------------------------
        # relative positional attention
        # ----------------------------------------------------

        pos = borzoi_positional_embed(
            seq_len=L,
            feature_size=self.num_position_features,
            device=x.device,
            seq_len_train=self.seq_len_train
        ).to(x.dtype)

        pos = self.pos_dropout(pos)

        rel_k = self.to_rel_k(pos)

        rel_k = (
            rel_k
            .view(
                2 * L - 1,
                self.nhead,
                self.key_size
            )
            .permute(1, 0, 2)
        )

        rel_logits = torch.einsum("bhid,hjd->bhij", q + rr, rel_k)

        rel_logits = relative_shift(rel_logits)

        logits = (content_logits + rel_logits)

        if key_padding_mask is not None:

            mask = (
                key_padding_mask[
                    :, None, None, :
                ]
                .to(torch.bool)
            )

            logits = logits.masked_fill(
                mask,
                torch.finfo(
                    logits.dtype
                ).min
            )

        attn = torch.softmax(logits, dim=-1)

        attn = self.attn_dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)

        out = (
            out.transpose(1, 2)
               .contiguous()
               .view(
                   B,
                   L,
                   self.nhead
                   * self.value_size
               )
        )

        return self.to_out(out)

class TransformerEncoderLayerRel(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        key_size=64,
        num_position_features=32,
        attention_dropout=0.1,
        position_dropout=0.01,
        seq_len_train=600,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attn = MHSA_RelBias(
            d_model=d_model,
            nhead=nhead,
            key_size=None,
            num_position_features=num_position_features,
            attention_dropout=attention_dropout,
            position_dropout=position_dropout,
            seq_len_train=seq_len_train,
        )

        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.drop2 = nn.Dropout(dropout)


    def forward(self, x, key_padding_mask=None):
        x = x + self.drop1(
            self.attn(
                self.norm1(x),
                key_padding_mask=key_padding_mask
            )
        )

        x = x + self.drop2(
            self.ff(
                self.norm2(x)
            )
        )

        return x


class TransformerEncoderRel(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        key_size=64,
        num_position_features=32,
        attention_dropout=0.1,
        position_dropout=0.01,
        seq_len_train=600,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayerRel(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                key_size=key_size,
                num_position_features=num_position_features,
                attention_dropout=attention_dropout,
                position_dropout=position_dropout,
                seq_len_train=seq_len_train,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)


    def forward(self, x, key_padding_mask=None):

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return self.norm(x)