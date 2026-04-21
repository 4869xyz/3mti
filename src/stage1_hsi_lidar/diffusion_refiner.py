import torch
import torch.nn as nn


class ResidualBlock(nn.Module):#一个标准的resnet-style残差块
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class FeatureDiffusionRefiner(nn.Module):
    """One-step latent denoiser used as diffusion-style feature refiner.
    一步潜变量去噪器，用作扩散风格的特征精炼器
    
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        noise_min: float = 0.05,#加噪声的范围强度
        noise_max: float = 0.20,
    ) -> None:
        super().__init__()
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.refiner = nn.Sequential(
            #输入是latent_dim+1，1是多出来的噪声强度图
            nn.Conv2d(latent_dim + 1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_dim, hidden_dim),#三个深度残差结构
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            #最后一层把通道数压回去latent_dim，得到残差预测
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, z_fused: torch.Tensor):
        # z_fused: [B, D, H, W]
        if z_fused.ndim != 4:#输入为4D张量，否则报错
            raise ValueError(f"FeatureDiffusionRefiner expects 4D tensor, got {tuple(z_fused.shape)}")

        bsz, _, h, w = z_fused.shape
        if self.training:
            sigma = torch.empty(bsz, 1, 1, 1, device=z_fused.device, dtype=z_fused.dtype).uniform_(
                self.noise_min, self.noise_max
            )
            noise = torch.randn_like(z_fused)
            z_noisy = z_fused + sigma * noise#原始特征加噪得到
        else:
            #评估模式不加，直接把原始图当作噪声强度图
            sigma = torch.zeros(bsz, 1, 1, 1, device=z_fused.device, dtype=z_fused.dtype)
            noise = torch.zeros_like(z_fused)
            z_noisy = z_fused

        sigma_map = sigma.expand(bsz, 1, h, w)
        denoise_input = torch.cat([z_noisy, sigma_map], dim=1)
        residual = self.refiner(denoise_input)
        z_refined = z_noisy + residual#得到精炼后的特征

        aux = {
            "z_noisy": z_noisy,
            "sigma": sigma.squeeze(-1).squeeze(-1).squeeze(-1),
            "noise": noise,
        }
        return z_refined, aux
