import torch
import torch.nn as nn


class HSI3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder3d = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.project2d = nn.Sequential(
            nn.Conv2d(32, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, C, P, P]
        if x.ndim != 5:
            raise ValueError(f"HSI3DEncoder expects 5D tensor, got {tuple(x.shape)}")

        feat3d = self.encoder3d(x)
        feat2d = feat3d.mean(dim=2)
        return self.project2d(feat2d)


class LiDAR2DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, P, P]
        if x.ndim != 4:
            raise ValueError(f"LiDAR2DEncoder expects 4D tensor, got {tuple(x.shape)}")

        return self.encoder(x)
