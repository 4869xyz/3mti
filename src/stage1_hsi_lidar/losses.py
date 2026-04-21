from typing import Dict

import torch
import torch.nn.functional as F


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def alignment_loss(z_h: torch.Tensor, z_l: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    g_h = F.adaptive_avg_pool2d(z_h, output_size=1).flatten(1)
    g_l = F.adaptive_avg_pool2d(z_l, output_size=1).flatten(1)
    cos = F.cosine_similarity(g_h, g_l, dim=1, eps=eps)
    return (1.0 - cos).mean()


def diffusion_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z_pred, z_target.detach())


def total_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_align: float,
    lambda_diff: float,
) -> Dict[str, torch.Tensor]:
    loss_cls = classification_loss(outputs["logits"], labels)
    loss_align = alignment_loss(outputs["z_h"], outputs["z_l"])
    loss_diff = diffusion_loss(outputs["z_refined"], outputs["z_fused"])
    loss = loss_cls + lambda_align * loss_align + lambda_diff * loss_diff

    return {
        "loss": loss,
        "loss_cls": loss_cls,
        "loss_align": loss_align,
        "loss_diff": loss_diff,
    }
