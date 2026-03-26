"""3D Swin Transformer bottleneck for the hybrid model.

A lightweight 3D Swin block used as the bottleneck between the
residual CNN encoder and transposed-conv decoder.

Reference: AutoLungDx Res-U-Net + ViT on LUNA16 (arXiv, 2025);
hybrid U-Net–Transformer (Nature Sci. Reports, Jan 2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition (B, D, H, W, C) into non-overlapping windows of size W³."""
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size,
        window_size,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, window_size**3, C)


def window_reverse(windows: torch.Tensor, window_size: int, D: int, H: int, W: int) -> torch.Tensor:
    """Reverse window_partition."""
    B = int(windows.shape[0] / (D * H * W / window_size**3))
    x = windows.view(
        B,
        D // window_size,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


class WindowAttention3D(nn.Module):
    """3D window-based multi-head self-attention."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        # Relative position bias table: (2W-1)^3 entries
        ws = window_size
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) ** 3, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid([torch.arange(ws)] * 3, indexing="ij")
        ).flatten(1)  # (3, ws³)
        rel = coords[:, :, None] - coords[:, None, :]  # (3, ws³, ws³)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += ws - 1
        rel[..., 1] += ws - 1
        rel[..., 2] += ws - 1
        rel[..., 0] *= (2 * ws - 1) ** 2
        rel[..., 1] *= 2 * ws - 1
        self.register_buffer("rel_pos_index", rel.sum(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_,
            N,
            3,
            self.num_heads,
            C // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)].view(N, N, self.num_heads)
        attn = attn + rel_bias.permute(2, 0, 1).unsqueeze(0)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock3D(nn.Module):
    """Single 3D Swin Transformer block (W-MSA + MLP, no shift for simplicity)."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W) — will be converted to (B, D, H, W, C) for attention."""
        B, C, D, H, W = x.shape
        ws = self.window_size

        # Pad to multiple of window_size
        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        _, _, Dp, Hp, Wp = x.shape

        x_seq = x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
        windows = window_partition(x_seq, ws)  # (nW*B, ws³, C)

        attn_out = self.attn(self.norm1(windows))
        attn_out = windows + attn_out
        attn_out = attn_out + self.mlp(self.norm2(attn_out))

        x_seq = window_reverse(attn_out, ws, Dp, Hp, Wp)
        x_out = x_seq.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)

        # Remove padding
        if pad_d or pad_h or pad_w:
            x_out = x_out[:, :, :D, :H, :W]
        return x_out


class SwinBottleneck(nn.Module):
    """Stack of Swin blocks used as the transformer bottleneck."""

    def __init__(self, dim: int, depth: int = 2, num_heads: int = 8, window_size: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SwinBlock3D(dim, num_heads, window_size) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
