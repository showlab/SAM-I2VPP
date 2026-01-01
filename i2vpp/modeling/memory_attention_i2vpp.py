# -*- coding: utf-8 -*-
# @FileName: memory_attention_i2vpp.py
# @Time    : 25/5/25 17:27
# @Author  : Haiyang Mei
# @E-mail  : haiyang.mei@outlook.com

import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from i2vpp.modeling.sam.transformer import RoPEAttention
from i2vpp.modeling.i2vpp_utils import get_activation_fn, get_clones, LayerNorm2d


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        INF = -torch.diag(torch.tensor(float("inf")).cuda().repeat(height), 0).unsqueeze(0).repeat(m_batchsize * width, 1, 1)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + INF).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x


class ScaleAdapter(nn.Module):
    def __init__(self, in_dim: int, num_heads: int = 4, expand_ratio: int = 2):
        super().__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.in_dim    = in_dim
        self.num_heads = num_heads
        self.head_dim  = in_dim // num_heads

        # ——— Learnable Scaling ———
        self.norm   = nn.LayerNorm(in_dim)
        self.gamma  = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

        # ——— Per-Head Scale-Specific DW-Conv ———
        for i in range(num_heads):
            # 3579
            ks  = 3 + 2*i
            pad = 1 + i

            conv = nn.Conv2d(
                self.head_dim, self.head_dim,
                kernel_size=ks, padding=pad,
                groups=self.head_dim,
                bias=False
            )
            setattr(self, f"local_conv_{i+1}", conv)

        # ——— Channel Expansion & Projection Back to in_dim ———
        self.proj0 = nn.Conv2d(
            in_dim,
            in_dim * expand_ratio,
            kernel_size=1,
            groups=self.head_dim,
            bias=False
        )
        self.ln0   = LayerNorm2d(in_dim * expand_ratio)
        self.proj1 = nn.Conv2d(
            in_dim * expand_ratio,
            in_dim,
            kernel_size=1,
            bias=True
        )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, hw_shapes: tuple[int,int]) -> torch.Tensor:
        """
        x: [B, N, C=in_dim]
        hw_shapes: (H, W) and N == H*W
        """
        # Outer skip connection directly uses the original x
        identity = x

        B, N, C = x.shape
        H, W    = hw_shapes

        # 1) Learnable Scaling
        x_scaled = self.norm(x) * self.gamma + x * self.gammax  # [B,N,C]

        # 2) reshape → feature map
        feat = x_scaled.permute(0,2,1).contiguous().view(B, C, H, W)  # [B,C,H,W]
        heads = feat.chunk(self.num_heads, dim=1)       # list of [B,head_dim,H,W]

        # 3) Scale-Concatenated Convolution across Multiple Heads within Each Group
        for i, head in enumerate(heads):
            conv = getattr(self, f"local_conv_{i+1}")
            out = conv(head)                # [B,head_dim,H,W]
            out = out.view(B, self.head_dim, 1, H, W)
            if i == 0:
                s_out = out                # [B,head_dim,1,H,W]
            else:
                s_out = torch.cat([s_out, out], dim=2)  # → [B,head_dim,heads,H,W]

        # 4) Reassemble Back to the Full Channel Dimension
        s_out = s_out.view(B, C, H, W)  # [B,C,H,W], C=head_dim*heads

        # 5) Channel Expansion → LN → GELU → Projection Back
        s_out = self.proj1(self.act(self.ln0(self.proj0(s_out))))  # [B,C,H,W]

        # 6) Flatten Back to Sequence
        s_out = s_out.view(B, C, -1).permute(0,2,1).contiguous()   # [B,N,C]

        # 7) Outer Residual Connection
        return identity + s_out


class MemoryAttentionLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.cross_attn_image = cross_attention

        # feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropouts
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # activation
        self.activation = get_activation_fn(activation)

        # positional flags
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

        # criss-cross
        inter_channels = d_model // 4
        self.conva = nn.Sequential(
            nn.Conv2d(d_model, inter_channels, 3, padding=1, bias=False),
            LayerNorm2d(inter_channels),
            nn.GELU(),
        )
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            LayerNorm2d(inter_channels),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(d_model + inter_channels, d_model, 1, 1, 0, bias=False),
            LayerNorm2d(d_model),
            nn.GELU(),
        )

        # ScaleAdapter after each sub-layer
        self.scaleadapter_sa = ScaleAdapter(d_model)
        self.scaleadapter_ca = ScaleAdapter(d_model)
        self.scaleadapter_mlp = ScaleAdapter(d_model)

        # Mask prediction head for deep supervision
        self.mask_conv = nn.Conv2d(d_model, 1, 5, 1, 2)

    def _forward_sa(self, tgt: Tensor, query_pos: Optional[Tensor], recurrence: int = 2) -> Tensor:
        tgt_norm = self.norm1(tgt)
        x = tgt_norm + query_pos if self.pos_enc_at_attn else tgt_norm

        # reshape to feature map
        B, N, C = x.size()
        H = W = int(math.sqrt(N))
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # criss-cross attention block
        output = self.conva(x)
        for _ in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], dim=1))
        output = x + self.dropout1(output)

        # back to token
        output = output.flatten(2).permute(0, 2, 1).contiguous()

        return output

    def _forward_ca(
        self,
        tgt: Tensor,
        memory: Tensor,
        query_pos: Optional[Tensor],
        pos: Optional[Tensor],
        num_k_exclude_rope: int = 0,
    ) -> Tensor:
        tgt_norm = self.norm2(tgt)
        q = tgt_norm + query_pos if self.pos_enc_at_cross_attn_queries else tgt_norm
        k = memory + pos if self.pos_enc_at_cross_attn_keys else memory

        # cross-attention
        if isinstance(self.cross_attn_image, RoPEAttention) and num_k_exclude_rope > 0:
            tgt2 = self.cross_attn_image(q=q, k=k, v=memory, num_k_exclude_rope=num_k_exclude_rope)
        else:
            tgt2 = self.cross_attn_image(q=q, k=k, v=memory)

        output = tgt + self.dropout2(tgt2)

        return output

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            output: Tensor, features after each layer [B, N, C]
            mask_pred: Tensor, mask prediction at each layer [B, N, 1]
        Mask prediction is performed after each layer for deep supervision
        """
        # 1. scaleadapter + self-attn
        tgt = self.scaleadapter_sa(tgt, hw_shapes=(64, 64))
        tgt2 = self._forward_sa(tgt, query_pos)

        # 2. scaleadapter + cross-attn
        tgt2 = self.scaleadapter_ca(tgt2, hw_shapes=(64, 64))
        tgt3 = self._forward_ca(tgt2, memory, query_pos, pos, num_k_exclude_rope)

        # 3. mlp
        tgt3 = self.scaleadapter_mlp(tgt3, hw_shapes=(64, 64))
        mlp_input = self.norm3(tgt3)
        mlp_out = self.linear2(self.dropout(self.activation(self.linear1(mlp_input))))
        output = tgt3 + self.dropout3(mlp_out)

        # 4. Deep Supervision Mask Prediction Head
        B, N, C = output.size()
        H = W = int(math.sqrt(N))
        feat_map = output.permute(0, 2, 1).contiguous().view(B, C, H, W)
        mask = self.mask_conv(feat_map)  # [B,1,H,W]
        # Upsample by 16× to [B, 1, 16*H, 16*W]
        mask_up = F.interpolate(
            mask,
            scale_factor=16,
            mode='bilinear',
            align_corners=False
        )

        return output, mask_up


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs: list of [N B C]
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            normed: Tensor, final normalized features [B, N, C]
            masks: List[Tensor], per-layer mask predictions [B, 1, H, W]
        """
        if isinstance(curr, list):
            curr, curr_pos = curr[0], curr_pos[0]

        # align batch-first
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            output = output.transpose(0, 1).contiguous()
            curr_pos = curr_pos.transpose(0, 1).contiguous()
            memory = memory.transpose(0, 1).contiguous()
            memory_pos = memory_pos.transpose(0, 1).contiguous()

        masks: List[torch.Tensor] = []
        for layer in self.layers:
            # Per-layer output features and mask predictions
            output, mask_pred = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                num_k_exclude_rope=(num_obj_ptr_tokens if isinstance(layer.cross_attn_image, RoPEAttention) else 0),
            )
            masks.append(mask_pred)

        normed = self.norm(output)
        if self.batch_first:
            normed = normed.transpose(0, 1).contiguous()

        masks_tensor = torch.cat(masks, dim=1)

        return normed, masks_tensor
