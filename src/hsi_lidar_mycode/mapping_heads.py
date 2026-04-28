from typing import Optional

import torch
import torch.nn as nn

#自动选择groupnorm的num-groups参数
def _pick_group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _pick_group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),#分组归一化，比batchnorm更加稳定
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

#方向特征映射器
class DirectionalFeatureMapper(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,#潜在特征
        num_blocks: int = 4,#要堆多少个残差块
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        #hid决定中间特征的通道数
        hid = int(hidden_dim) if hidden_dim is not None else latent_dim
        groups = _pick_group_count(hid)#为计算中间维度计算合适的groupnorm组数
        self.in_proj = nn.Sequential(
            nn.Conv2d(latent_dim, hid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hid),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hid, dropout=dropout) for _ in range(num_blocks)])
        self.out_proj = nn.Conv2d(hid, latent_dim, kernel_size=3, padding=1)
        #out-proj投影回原来的latent-dim

    def forward(self, z_cond: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(z_cond)
        x = self.blocks(x)
        return self.out_proj(x)
