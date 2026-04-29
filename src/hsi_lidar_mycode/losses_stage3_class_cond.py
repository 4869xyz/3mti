from typing import Dict

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)


def prototype_loss(
    z_pred: torch.Tensor,
    class_proto_target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    feat = F.adaptive_avg_pool2d(z_pred, output_size=1).flatten(1)
    feat = F.normalize(feat, dim=1, eps=eps)
    proto = F.normalize(class_proto_target, dim=1, eps=eps)
    return F.mse_loss(feat, proto)


def cosine_gap_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_gap = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
    y_gap = F.adaptive_avg_pool2d(y, output_size=1).flatten(1)
    x_gap = F.normalize(x_gap, dim=1, eps=eps)
    y_gap = F.normalize(y_gap, dim=1, eps=eps)
    return 1.0 - (x_gap * y_gap).sum(dim=1).mean()


def total_stage3_hsi_class_cond_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_cls: float = 1.0,
    lambda_kd: float = 0.7,
    lambda_proto: float = 0.3,
    lambda_noise: float = 0.4,
    lambda_recon: float = 0.2,
    lambda_align: float = 0.03,
    lambda_prior: float = 0.3,
    kd_temperature: float = 2.0,
) -> Dict[str, torch.Tensor]:
    loss_cls = F.cross_entropy(outputs["logits_missing"], labels)
    loss_kd = kd_loss(
        student_logits=outputs["logits_missing"],
        teacher_logits=outputs["logits_teacher"].detach(),
        temperature=kd_temperature,
    )
    loss_proto = prototype_loss(
        z_pred=outputs["z_h_pred"],
        class_proto_target=outputs["class_proto_target"].detach(),
    )
    loss_noise = outputs["loss_noise"]
    loss_recon = F.smooth_l1_loss(outputs["z_h_pred"], outputs["z_h"].detach())
    loss_align = cosine_gap_loss(outputs["z_h_pred"], outputs["z_h"].detach())
    loss_prior = F.cross_entropy(outputs["prior_logits"], labels)

    loss = (
        lambda_cls * loss_cls
        + lambda_kd * loss_kd
        + lambda_proto * loss_proto
        + lambda_noise * loss_noise
        + lambda_recon * loss_recon
        + lambda_align * loss_align
        + lambda_prior * loss_prior
    )

    return {
        "loss": loss,
        "loss_cls": loss_cls,
        "loss_kd": loss_kd,
        "loss_proto": loss_proto,
        "loss_noise": loss_noise,
        "loss_recon": loss_recon,
        "loss_align": loss_align,
        "loss_prior": loss_prior,
    }
