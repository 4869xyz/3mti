from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from .diffusion_compensator import FeatureConditionalDiffusionCompensator
except ImportError:
    from diffusion_compensator import FeatureConditionalDiffusionCompensator


class LiDARPriorHead(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_classes: int = 15,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_l: torch.Tensor) -> torch.Tensor:
        return self.net(z_l)


class ClassConditionalLatentCompensator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        diffusion_steps: int = 60,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.proto_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.compensator = FeatureConditionalDiffusionCompensator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            diffusion_steps=diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

    def _enhance_condition(self, condition: torch.Tensor, class_proto: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 4:
            raise ValueError(f"condition must be [B,C,H,W], got {tuple(condition.shape)}")
        if class_proto.ndim != 2 or class_proto.shape[0] != condition.shape[0] or class_proto.shape[1] != self.latent_dim:
            raise ValueError(
                f"class_proto must be [B,{self.latent_dim}], got {tuple(class_proto.shape)} "
                f"for condition {tuple(condition.shape)}"
            )
        proto_map = self.proto_proj(class_proto).to(dtype=condition.dtype).view(condition.shape[0], self.latent_dim, 1, 1)
        return condition + proto_map

    def training_forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        init_latent: torch.Tensor,
        class_proto: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        condition_enhanced = self._enhance_condition(condition=condition, class_proto=class_proto)
        out = self.compensator.training_forward(
            target=target,
            condition=condition_enhanced,
            init_latent=init_latent,
        )
        out["condition_enhanced"] = condition_enhanced
        return out

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        init_latent: torch.Tensor,
        class_proto: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        condition_enhanced = self._enhance_condition(condition=condition, class_proto=class_proto)
        return self.compensator.sample(
            condition=condition_enhanced,
            num_steps=num_steps,
            init_latent=init_latent,
        )
