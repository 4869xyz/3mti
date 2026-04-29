import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CrossModalSelfAttentionFusion(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        #创建了多个transformer块，这里两个块
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )
        #融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, z_h: torch.Tensor, z_l: torch.Tensor):
        # z_h, z_l: [B, D, H, W]
        if z_h.shape != z_l.shape:
            raise ValueError(f"Shape mismatch between modalities: {z_h.shape} vs {z_l.shape}")

        bsz, dim, h, w = z_h.shape
        num_tokens = h * w

        t_h = z_h.flatten(2).transpose(1, 2)  # [B, N, D]把hsi的特征从cnn格式变成transformer格式[B,H*W,D]
        t_l = z_l.flatten(2).transpose(1, 2)  # [B, N, D]
        tokens = torch.cat([t_h, t_l], dim=1)  # [B, 2N, D]在token维度进拼接，前面121个token是hsi，后面121个维度是lidar

        for block in self.blocks:
            tokens = block(tokens)#自注意力让两个token相互交换信息

        t_h_refined, t_l_refined = tokens.split(num_tokens, dim=1)#融合后的token再拆回去两个模态
        z_h_refined = t_h_refined.transpose(1, 2).reshape(bsz, dim, h, w)#变成cnn特征图格式
        z_l_refined = t_l_refined.transpose(1, 2).reshape(bsz, dim, h, w)
        #TODO 这个fuse直接cat的啊，换一下高级一点的呢
        z_fused = self.fuse(torch.cat([z_h_refined, z_l_refined], dim=1))
        aux = {
            "z_h_refined": z_h_refined,
            "z_l_refined": z_l_refined,
        }
        return z_fused, aux
