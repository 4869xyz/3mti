from typing import Dict

import torch
import torch.nn.functional as F


def alignment_loss(z_h: torch.Tensor, z_l: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    g_h = F.adaptive_avg_pool2d(z_h, output_size=1).flatten(1)
    g_l = F.adaptive_avg_pool2d(z_l, output_size=1).flatten(1)
    cos = F.cosine_similarity(g_h, g_l, dim=1, eps=eps)
    return (1.0 - cos).mean()


def cosine_gap_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_gap = F.adaptive_avg_pool2d(pred, output_size=1).flatten(1)
    target_gap = F.adaptive_avg_pool2d(target, output_size=1).flatten(1)
    cos = F.cosine_similarity(pred_gap, target_gap, dim=1, eps=eps)
    return (1.0 - cos).mean()


def total_stage2_mapper_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_cls_mapper: float = 1.0,
    lambda_map_recon: float = 1.0,
    lambda_map_cos: float = 0.5,
) -> Dict[str, torch.Tensor]:
    cls_losses = {}
    for key in ("logits_hsi_mapped", "logits_lidar_mapped", "logits_enhanced"):
        if key in outputs:
            cls_losses[f"loss_cls_{key.replace('logits_', '')}"] = F.cross_entropy(outputs[key], labels)
    if cls_losses:
        loss_cls_mapper = sum(cls_losses.values()) / len(cls_losses)
    else:
        loss_cls_mapper = F.cross_entropy(outputs["logits"], labels)

    loss_map_recon_h = F.l1_loss(outputs["z_h_map"], outputs["z_h"].detach())
    loss_map_recon_l = F.l1_loss(outputs["z_l_map"], outputs["z_l"].detach())
    loss_map_recon = 0.5 * (loss_map_recon_h + loss_map_recon_l)

    loss_map_cos_h = cosine_gap_loss(outputs["z_h_map"], outputs["z_h"].detach())
    loss_map_cos_l = cosine_gap_loss(outputs["z_l_map"], outputs["z_l"].detach())
    loss_map_cos = 0.5 * (loss_map_cos_h + loss_map_cos_l)

    loss = (
        lambda_cls_mapper * loss_cls_mapper
        + lambda_map_recon * loss_map_recon
        + lambda_map_cos * loss_map_cos
    )

    loss_dict = {
        "loss": loss,
        "loss_cls_mapper": loss_cls_mapper,
        "loss_map_recon": loss_map_recon,
        "loss_map_cos": loss_map_cos,
        "loss_map_recon_h": loss_map_recon_h,
        "loss_map_recon_l": loss_map_recon_l,
        "loss_map_cos_h": loss_map_cos_h,
        "loss_map_cos_l": loss_map_cos_l,
    }
    loss_dict.update(cls_losses)
    return loss_dict


def total_stage3_single_missing_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_cls_missing: float = 1.0,
    lambda_noise: float = 0.5,
    lambda_recon: float = 0.5,
    lambda_align: float = 0.05,
    lambda_refine: float = 0.2,
) -> Dict[str, torch.Tensor]:
    loss_cls_missing = F.cross_entropy(outputs["logits_missing"], labels)
    loss_noise = outputs["loss_noise"]
    loss_recon = F.mse_loss(outputs["z_pred"], outputs["z_target"].detach())
    if outputs.get("mode") == "hsi_missing":
        loss_align = alignment_loss(outputs["z_pred"], outputs["z_l"].detach())
    elif outputs.get("mode") == "lidar_missing":
        loss_align = alignment_loss(outputs["z_h"].detach(), outputs["z_pred"])
    else:
        loss_align = alignment_loss(outputs["z_h"], outputs["z_l"])
    loss_refine = F.mse_loss(outputs["z_refined_missing"], outputs["z_fused_missing"].detach())

    loss = (
        lambda_cls_missing * loss_cls_missing
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


def total_stage2_single_missing_loss(*args, **kwargs) -> Dict[str, torch.Tensor]:
    return total_stage3_single_missing_loss(*args, **kwargs)
