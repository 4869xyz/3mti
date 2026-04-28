import random
from typing import Tuple

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class MisalignmentAugment:
    """Apply light geometric perturbation to one modality for robustness."""

    def __init__(
        self,
        prob: float = 0.6,
        max_shift: int = 1,
        max_rotate: float = 5.0,
        min_scale: float = 0.95,
        max_scale: float = 1.05,
        hflip_p: float = 0.1,
        vflip_p: float = 0.1,
    ) -> None:
        self.prob = prob
        self.max_shift = max_shift
        self.max_rotate = max_rotate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C, H, W]
        tx = random.randint(-self.max_shift, self.max_shift)
        ty = random.randint(-self.max_shift, self.max_shift)
        angle = random.uniform(-self.max_rotate, self.max_rotate)
        scale = random.uniform(self.min_scale, self.max_scale)

        out = TF.affine(
            x,
            angle=angle,
            translate=[tx, ty],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        if random.random() < self.hflip_p:
            out = TF.hflip(out)
        if random.random() < self.vflip_p:
            out = TF.vflip(out)
        return out

    def __call__(
        self, hsi_patch: torch.Tensor, lidar_patch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hsi_patch: [1, C, P, P], lidar_patch: [1, P, P]
        if random.random() > self.prob:
            return hsi_patch, lidar_patch

        if random.random() < 0.5:
            hsi_aug = self._transform(hsi_patch.squeeze(0))
            hsi_patch = hsi_aug.unsqueeze(0)
        else:
            lidar_patch = self._transform(lidar_patch)

        return hsi_patch, lidar_patch
