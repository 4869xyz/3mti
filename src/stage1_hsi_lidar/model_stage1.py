from typing import Dict

import torch
import torch.nn as nn

try:
    from .diffusion_refiner import FeatureDiffusionRefiner
    from .encoders import HSI3DEncoder, LiDAR2DEncoder
    from .fusion_csm import CrossModalSelfAttentionFusion
except ImportError:
    from diffusion_refiner import FeatureDiffusionRefiner
    from encoders import HSI3DEncoder, LiDAR2DEncoder
    from fusion_csm import CrossModalSelfAttentionFusion


class Stage1HSILiDARDiffusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 128,
        csm_heads: int = 4,
        csm_depth: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hsi_encoder = HSI3DEncoder(in_channels=1, latent_dim=latent_dim)
        self.lidar_encoder = LiDAR2DEncoder(in_channels=1, latent_dim=latent_dim)
        self.csm = CrossModalSelfAttentionFusion(
            dim=latent_dim,
            num_heads=csm_heads,
            depth=csm_depth,
            dropout=dropout,
        )
        self.diff_refiner = FeatureDiffusionRefiner(latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_h = self.hsi_encoder(hsi)
        z_l = self.lidar_encoder(lidar)

        z_fused, aux = self.csm(z_h, z_l)
        z_refined, diff_aux = self.diff_refiner(z_fused)
        logits = self.classifier(z_refined)

        return {
            "logits": logits,
            "z_h": z_h,
            "z_l": z_l,
            "z_fused": z_fused,
            "z_refined": z_refined,
            "aux": aux,
            "diff_aux": diff_aux,
        }
