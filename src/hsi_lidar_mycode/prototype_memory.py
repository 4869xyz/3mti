from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def gap_2d(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,C,H,W], got {tuple(z.shape)}")
    return F.adaptive_avg_pool2d(z, output_size=1).flatten(1)


class PrototypeMemoryBank(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dim: int = 128,
        momentum: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")

        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("prototypes", torch.zeros(num_classes, dim, dtype=torch.float32))
        self.register_buffer("counts", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def reset(self) -> None:
        self.prototypes.zero_()
        self.counts.zero_()

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        if feats.ndim != 2 or feats.shape[1] != self.dim:
            raise ValueError(f"feats must be [B,{self.dim}], got {tuple(feats.shape)}")
        if labels.ndim != 1 or labels.shape[0] != feats.shape[0]:
            raise ValueError(f"labels must be [B], got {tuple(labels.shape)} for feats {tuple(feats.shape)}")

        feats = F.normalize(feats.detach(), dim=1, eps=self.eps)
        labels = labels.detach().long()
        for cls_idx in labels.unique().tolist():
            cls_idx = int(cls_idx)
            if cls_idx < 0 or cls_idx >= self.num_classes:
                raise ValueError(f"Label index {cls_idx} outside [0, {self.num_classes - 1}]")
            mask = labels == cls_idx
            cls_feat = F.normalize(feats[mask].mean(dim=0), dim=0, eps=self.eps)
            if int(self.counts[cls_idx].item()) == 0:
                self.prototypes[cls_idx].copy_(cls_feat)
            else:
                old = self.prototypes[cls_idx]
                mixed = self.momentum * old + (1.0 - self.momentum) * cls_feat
                self.prototypes[cls_idx].copy_(F.normalize(mixed, dim=0, eps=self.eps))
            self.counts[cls_idx] += int(mask.sum().item())

    @torch.no_grad()
    def build_from_loader(
        self,
        model,
        loader,
        device,
        label_key: str = "label",
        hsi_key: str = "hsi",
        lidar_key: str = "lidar",
    ) -> None:
        self.reset()
        sums = torch.zeros_like(self.prototypes, device=device)
        counts = torch.zeros_like(self.counts, device=device)

        was_training = model.training
        model.eval()
        for batch in loader:
            hsi = batch[hsi_key].to(device, non_blocking=True)
            lidar = batch[lidar_key].to(device, non_blocking=True)
            labels = batch[label_key].to(device, non_blocking=True).long()
            encoded = model.encode_modalities(hsi, lidar)
            feats = F.normalize(gap_2d(encoded["z_h"]), dim=1, eps=self.eps)

            for cls_idx in labels.unique().tolist():
                cls_idx = int(cls_idx)
                if cls_idx < 0 or cls_idx >= self.num_classes:
                    raise ValueError(f"Label index {cls_idx} outside [0, {self.num_classes - 1}]")
                mask = labels == cls_idx
                sums[cls_idx] += feats[mask].sum(dim=0)
                counts[cls_idx] += int(mask.sum().item())

        valid = counts > 0
        if valid.any():
            proto = sums[valid] / counts[valid].float().unsqueeze(1)
            self.prototypes[valid].copy_(F.normalize(proto, dim=1, eps=self.eps).to(self.prototypes.device))
            self.counts[valid].copy_(counts[valid].to(self.counts.device))

        if was_training:
            model.train()

    def has_valid_prototypes(self, labels: Optional[torch.Tensor] = None) -> bool:
        if labels is None:
            return bool((self.counts > 0).all().item())
        labels = labels.detach().long()
        return bool((self.counts[labels] > 0).all().item())

    def lookup(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        if labels.ndim != 1:
            raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")
        if labels.numel() > 0:
            min_label = int(labels.min().item())
            max_label = int(labels.max().item())
            if min_label < 0 or max_label >= self.num_classes:
                raise ValueError(f"Label range [{min_label}, {max_label}] outside [0, {self.num_classes - 1}]")
        return self.prototypes.index_select(0, labels)

    def soft_lookup(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim != 2 or probs.shape[1] != self.num_classes:
            raise ValueError(f"probs must be [B,{self.num_classes}], got {tuple(probs.shape)}")
        return probs.to(dtype=self.prototypes.dtype) @ self.prototypes
