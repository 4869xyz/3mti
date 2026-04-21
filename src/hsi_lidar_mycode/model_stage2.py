from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from .diffusion_compensator import FeatureConditionalDiffusionCompensator
    from .diffusion_refiner import FeatureDiffusionRefiner
    from .encoders import HSI3DEncoder, LiDAR2DEncoder
    from .fusion_csm import CrossModalSelfAttentionFusion
    from .mapping_heads import DirectionalFeatureMapper
except ImportError:
    from diffusion_compensator import FeatureConditionalDiffusionCompensator
    from diffusion_refiner import FeatureDiffusionRefiner
    from encoders import HSI3DEncoder, LiDAR2DEncoder
    from fusion_csm import CrossModalSelfAttentionFusion
    from mapping_heads import DirectionalFeatureMapper


class Stage2HSILiDARMissingModalityClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        latent_dim: int = 128,
        csm_heads: int = 4,
        csm_depth: int = 2,
        dropout: float = 0.3,
        diffusion_hidden_dim: int = 128,
        diffusion_steps: int = 100,
        diffusion_beta_start: float = 1e-4,
        diffusion_beta_end: float = 2e-2,
        mapper_num_blocks: int = 4,
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
        self.map_h_from_l = DirectionalFeatureMapper(
            latent_dim=latent_dim,
            num_blocks=mapper_num_blocks,
            dropout=dropout,
        )
        self.map_l_from_h = DirectionalFeatureMapper(
            latent_dim=latent_dim,
            num_blocks=mapper_num_blocks,
            dropout=dropout,
        )

        self.lidar_to_hsi = FeatureConditionalDiffusionCompensator(
            latent_dim=latent_dim,
            hidden_dim=diffusion_hidden_dim,
            diffusion_steps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
        )
        self.hsi_to_lidar = FeatureConditionalDiffusionCompensator(
            latent_dim=latent_dim,
            hidden_dim=diffusion_hidden_dim,
            diffusion_steps=diffusion_steps,
            beta_start=diffusion_beta_start,
            beta_end=diffusion_beta_end,
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        backbone_modules = [
            self.hsi_encoder,
            self.lidar_encoder,
            self.csm,
            self.diff_refiner,
            self.classifier,
        ]
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = trainable

    def load_stage1_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = False,
        map_location: str = "cpu",
    ) -> Dict[str, object]:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt)
        load_info = self.load_state_dict(state_dict, strict=False)

        ignored_prefixes = ("lidar_to_hsi.", "hsi_to_lidar.", "map_h_from_l.", "map_l_from_h.")
        ignored_missing = [k for k in load_info.missing_keys if k.startswith(ignored_prefixes)]
        effective_missing = [k for k in load_info.missing_keys if k not in ignored_missing]
        unexpected = list(load_info.unexpected_keys)

        if strict and (effective_missing or unexpected):
            raise RuntimeError(
                "Strict stage1 loading failed. "
                f"missing_keys={effective_missing}, unexpected_keys={unexpected}"
            )

        return {
            "missing_keys": list(load_info.missing_keys),
            "ignored_missing_keys": ignored_missing,
            "unexpected_keys": unexpected,
        }

    def _classify_from_latents(self, z_h: torch.Tensor, z_l: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_fused, aux = self.csm(z_h, z_l)
        z_refined, diff_aux = self.diff_refiner(z_fused)
        logits = self.classifier(z_refined)
        return {
            "logits": logits,
            "z_fused": z_fused,
            "z_refined": z_refined,
            "aux": aux,
            "diff_aux": diff_aux,
        }

    def encode_modalities(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_h = self.hsi_encoder(hsi)
        z_l = self.lidar_encoder(lidar)
        return {"z_h": z_h, "z_l": z_l}

    def build_enhanced_latents(self, z_h: torch.Tensor, z_l: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_h_map = self.map_h_from_l(z_l)
        z_l_map = self.map_l_from_h(z_h)
        z_h_enh = z_h + z_h_map
        z_l_enh = z_l + z_l_map
        return {
            "z_h_map": z_h_map,
            "z_l_map": z_l_map,
            "z_h_enh": z_h_enh,
            "z_l_enh": z_l_enh,
        }

    def forward_mapper_only(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(hsi, lidar)
        z_h = encoded["z_h"]
        z_l = encoded["z_l"]
        maps = self.build_enhanced_latents(z_h, z_l)
        out = self._classify_from_latents(maps["z_h_enh"], maps["z_l_enh"])
        return {
            "z_h": z_h,
            "z_l": z_l,
            "z_h_map": maps["z_h_map"],
            "z_l_map": maps["z_l_map"],
            "z_h_enh": maps["z_h_enh"],
            "z_l_enh": maps["z_l_enh"],
            "logits": out["logits"],
            "z_fused": out["z_fused"],
            "z_refined": out["z_refined"],
            "aux": out["aux"],
            "diff_aux": out["diff_aux"],
        }

    def forward_train_missing_hsi(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(hsi, lidar)
        z_h = encoded["z_h"]
        z_l = encoded["z_l"]
        z_h_map = self.map_h_from_l(z_l)

        comp_h = self.lidar_to_hsi.training_forward(target=z_h, condition=z_l, init_latent=z_h_map)
        out_missing = self._classify_from_latents(comp_h["x0_pred"], z_l)

        return {
            "mode": "hsi_missing",
            "z_h": z_h,
            "z_l": z_l,
            "z_map": z_h_map,
            "z_target": z_h,
            "z_pred": comp_h["x0_pred"],
            "res_target": comp_h["residual_target"],
            "res_pred": comp_h["residual_pred"],
            "loss_noise": comp_h["loss_noise"],
            "logits_missing": out_missing["logits"],
            "z_fused_missing": out_missing["z_fused"],
            "z_refined_missing": out_missing["z_refined"],
            "aux_missing": out_missing["aux"],
            "diff_aux_missing": out_missing["diff_aux"],
        }

    def forward_train_missing_lidar(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(hsi, lidar)
        z_h = encoded["z_h"]
        z_l = encoded["z_l"]
        z_l_map = self.map_l_from_h(z_h)

        comp_l = self.hsi_to_lidar.training_forward(target=z_l, condition=z_h, init_latent=z_l_map)
        out_missing = self._classify_from_latents(z_h, comp_l["x0_pred"])

        return {
            "mode": "lidar_missing",
            "z_h": z_h,
            "z_l": z_l,
            "z_map": z_l_map,
            "z_target": z_l,
            "z_pred": comp_l["x0_pred"],
            "res_target": comp_l["residual_target"],
            "res_pred": comp_l["residual_pred"],
            "loss_noise": comp_l["loss_noise"],
            "logits_missing": out_missing["logits"],
            "z_fused_missing": out_missing["z_fused"],
            "z_refined_missing": out_missing["z_refined"],
            "aux_missing": out_missing["aux"],
            "diff_aux_missing": out_missing["diff_aux"],
        }

    def forward_train(self, hsi: torch.Tensor, lidar: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode_modalities(hsi, lidar)
        z_h = encoded["z_h"]
        z_l = encoded["z_l"]
        maps = self.build_enhanced_latents(z_h, z_l)
        z_h_map = maps["z_h_map"]
        z_l_map = maps["z_l_map"]

        out_full = self._classify_from_latents(z_h, z_l)
        out_full_enh = self._classify_from_latents(maps["z_h_enh"], maps["z_l_enh"])

        comp_h = self.lidar_to_hsi.training_forward(target=z_h, condition=z_l, init_latent=z_h_map)
        comp_l = self.hsi_to_lidar.training_forward(target=z_l, condition=z_h, init_latent=z_l_map)

        out_hsi_missing = self._classify_from_latents(comp_h["x0_pred"], z_l)
        out_lidar_missing = self._classify_from_latents(z_h, comp_l["x0_pred"])

        return {
            "z_h": z_h,
            "z_l": z_l,
            "z_h_map": z_h_map,
            "z_l_map": z_l_map,
            "z_h_enh": maps["z_h_enh"],
            "z_l_enh": maps["z_l_enh"],
            "z_h_pred": comp_h["x0_pred"],
            "z_l_pred": comp_l["x0_pred"],
            "loss_noise_h": comp_h["loss_noise"],
            "loss_noise_l": comp_l["loss_noise"],
            "logits_full": out_full["logits"],
            "logits_full_enh": out_full_enh["logits"],
            "logits_hsi_missing": out_hsi_missing["logits"],
            "logits_lidar_missing": out_lidar_missing["logits"],
            "z_fused_full": out_full["z_fused"],
            "z_refined_full": out_full["z_refined"],
            "aux_full": out_full["aux"],
            "diff_aux_full": out_full["diff_aux"],
            "z_fused_full_enh": out_full_enh["z_fused"],
            "z_refined_full_enh": out_full_enh["z_refined"],
            "aux_full_enh": out_full_enh["aux"],
            "diff_aux_full_enh": out_full_enh["diff_aux"],
        }

    @torch.no_grad()
    def forward_mode(
        self,
        hsi: Optional[torch.Tensor],
        lidar: Optional[torch.Tensor],
        mode: str = "full",
        sampling_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        mode = mode.lower().strip()
        if mode in {"full", "full_mapper"}:
            if hsi is None or lidar is None:
                raise ValueError("Mode 'full/full_mapper' requires both hsi and lidar tensors.")
            encoded = self.encode_modalities(hsi, lidar)
            if mode == "full_mapper":
                maps = self.build_enhanced_latents(encoded["z_h"], encoded["z_l"])
                z_h = maps["z_h_enh"]
                z_l = maps["z_l_enh"]
            else:
                z_h = encoded["z_h"]
                z_l = encoded["z_l"]
        elif mode == "hsi_missing":
            if lidar is None:
                raise ValueError("Mode 'hsi_missing' requires lidar tensor.")
            z_l = self.lidar_encoder(lidar)
            z_h_map = self.map_h_from_l(z_l)
            z_h = self.lidar_to_hsi.sample(z_l, num_steps=sampling_steps, init_latent=z_h_map)
        elif mode == "lidar_missing":
            if hsi is None:
                raise ValueError("Mode 'lidar_missing' requires hsi tensor.")
            z_h = self.hsi_encoder(hsi)
            z_l_map = self.map_l_from_h(z_h)
            z_l = self.hsi_to_lidar.sample(z_h, num_steps=sampling_steps, init_latent=z_l_map)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        out = self._classify_from_latents(z_h, z_l)
        return {
            "logits": out["logits"],
            "z_h": z_h,
            "z_l": z_l,
            "z_fused": out["z_fused"],
            "z_refined": out["z_refined"],
        }

    def forward(
        self,
        hsi: Optional[torch.Tensor],
        lidar: Optional[torch.Tensor],
        mode: str = "train",
        sampling_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        mode = mode.lower().strip()
        if mode == "train":
            if hsi is None or lidar is None:
                raise ValueError("Mode 'train' requires both hsi and lidar tensors.")
            return self.forward_train(hsi, lidar)
        return self.forward_mode(hsi=hsi, lidar=lidar, mode=mode, sampling_steps=sampling_steps)
