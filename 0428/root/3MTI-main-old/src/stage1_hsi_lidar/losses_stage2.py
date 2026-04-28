from typing import Dict

import torch
import torch.nn.functional as F


def alignment_loss(z_h: torch.Tensor, z_l: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    g_h = F.adaptive_avg_pool2d(z_h, output_size=1).flatten(1)
    g_l = F.adaptive_avg_pool2d(z_l, output_size=1).flatten(1)
    cos = F.cosine_similarity(g_h, g_l, dim=1, eps=eps)
    return (1.0 - cos).mean()


def total_stage2_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_cls_missing: float,
    lambda_noise: float,
    lambda_recon: float,
    lambda_align: float,
    lambda_refine: float,
) -> Dict[str, torch.Tensor]:
    loss_cls_full = F.cross_entropy(outputs["logits_full"], labels)
    loss_cls_hsi_missing = F.cross_entropy(outputs["logits_hsi_missing"], labels)
    loss_cls_lidar_missing = F.cross_entropy(outputs["logits_lidar_missing"], labels)
    loss_cls_missing = 0.5 * (loss_cls_hsi_missing + loss_cls_lidar_missing)

    loss_noise = 0.5 * (outputs["loss_noise_h"] + outputs["loss_noise_l"])
    loss_recon_h = F.mse_loss(outputs["z_h_pred"], outputs["z_h"].detach())
    loss_recon_l = F.mse_loss(outputs["z_l_pred"], outputs["z_l"].detach())
    loss_recon = 0.5 * (loss_recon_h + loss_recon_l)

    loss_align = alignment_loss(outputs["z_h"], outputs["z_l"])
    loss_refine = F.mse_loss(outputs["z_refined_full"], outputs["z_fused_full"].detach())

    loss = (
        loss_cls_full
        + lambda_cls_missing * loss_cls_missing
        + lambda_noise * loss_noise
        + lambda_recon * loss_recon
        + lambda_align * loss_align
        + lambda_refine * loss_refine
    )

    return {
        "loss": loss,
        "loss_cls_full": loss_cls_full,
        "loss_cls_hsi_missing": loss_cls_hsi_missing,
        "loss_cls_lidar_missing": loss_cls_lidar_missing,
        "loss_cls_missing": loss_cls_missing,
        "loss_noise": loss_noise,
        "loss_recon": loss_recon,
        "loss_align": loss_align,
        "loss_refine": loss_refine,
    }


def total_stage2_single_missing_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_noise: float,
    lambda_recon: float,
    lambda_align: float,
    lambda_refine: float,
) -> Dict[str, torch.Tensor]:
    loss_cls_missing = F.cross_entropy(outputs["logits_missing"], labels)
    loss_noise = outputs["loss_noise"]
    loss_recon = F.mse_loss(outputs["z_pred"], outputs["z_target"].detach())

    loss_align = alignment_loss(outputs["z_h"], outputs["z_l"])
    loss_refine = F.mse_loss(outputs["z_refined_missing"], outputs["z_fused_missing"].detach())

    loss = (
        loss_cls_missing
        + lambda_noise * loss_noise
        + lambda_recon * loss_recon
        + lambda_align * loss_align
        + lambda_refine * loss_refine
    )

    return {
        "loss": loss,
        "loss_cls_missing": loss_cls_missing,
        "loss_noise": loss_noise,
        "loss_recon": loss_recon,
        "loss_align": loss_align,
        "loss_refine": loss_refine,
    }
