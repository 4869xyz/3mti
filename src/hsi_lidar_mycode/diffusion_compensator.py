import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

#分组数，从大往小尝试，每个分组更细的话，归一化更加灵活
def _pick_group_count(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1

#时间步编码模块，就是把比如t=37，编码为一个更加适合神经网络处理的向量
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        if emb_dim <= 0:
            raise ValueError(f"emb_dim must be > 0, got {emb_dim}")
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:#要求必须输入是一维的[B]
            raise ValueError(f"Expected 1D timesteps, got shape {tuple(t.shape)}")

        half_dim = self.emb_dim // 2
        if half_dim == 0:
            return t.float().unsqueeze(1)

        exponent = torch.arange(half_dim, device=t.device, dtype=torch.float32)
        exponent = exponent / max(half_dim - 1, 1)
        frequencies = torch.exp(-math.log(10000.0) * exponent)
        args = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

#残差块
class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _pick_group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

#条件噪声预测器
class ConditionalNoisePredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        timestep_embed_dim: int,#时间步嵌入向量的维度
        max_timestep: int,#最大时间步
    ) -> None:
        super().__init__()
        self.max_timestep = max(max_timestep, 1)

        in_channels = latent_dim * 2 + 1#计算 stem 输入层的通道数。因为会将 x_t（噪声图）、condition（条件图）和时间步映射 t_map 在通道维度拼接
        groups = _pick_group_count(hidden_dim)
        #定义输入处理模块
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.time_embed = SinusoidalTimeEmbedding(timestep_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(timestep_embed_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.Sequential(
            ResidualConvBlock(hidden_dim),
            ResidualConvBlock(hidden_dim),
            ResidualConvBlock(hidden_dim),
        )
        #输出层，降通道映射回去latent-dim通道，输出x_t形状相同的噪声预测图
        self.head = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x_t.shape != condition.shape:
            raise ValueError(f"Shape mismatch: x_t={tuple(x_t.shape)} condition={tuple(condition.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x_t.shape[0]:
            raise ValueError(
                f"timesteps must be [B], got {tuple(timesteps.shape)} with batch {x_t.shape[0]}"
            )

        bsz, _, h, w = x_t.shape
        t_norm = (timesteps.float() / float(self.max_timestep)).view(bsz, 1, 1, 1)
        t_map = t_norm.expand(bsz, 1, h, w).to(dtype=x_t.dtype, device=x_t.device)

        h_feat = torch.cat([x_t, condition, t_map], dim=1)
        h_feat = self.stem(h_feat)

        t_embed = self.time_mlp(self.time_embed(timesteps))
        h_feat = h_feat + t_embed.to(dtype=h_feat.dtype).view(bsz, -1, 1, 1)

        h_feat = self.blocks(h_feat)
        return self.head(h_feat)


class FeatureConditionalDiffusionCompensator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        diffusion_steps: int = 100,#时间步
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        timestep_embed_dim: int = 128,
        clip_x0: Optional[float] = None,
    ) -> None:
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError(f"diffusion_steps must be >= 2, got {diffusion_steps}")
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError(f"beta_start/beta_end must be > 0, got {beta_start}, {beta_end}")
        if beta_end <= beta_start:
            raise ValueError(f"beta_end must be > beta_start, got {beta_start}, {beta_end}")

        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps
        self.clip_x0 = clip_x0
        #噪声预测器
        self.noise_predictor = ConditionalNoisePredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            timestep_embed_dim=timestep_embed_dim,
            max_timestep=diffusion_steps - 1,
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = values.index_select(0, timesteps).view(-1, 1, 1, 1)
        return out.to(dtype=x.dtype, device=x.device)

    #前向扩散过程，给定干净的残差x0，噪声noise，时间步t，计算带噪声的残差
    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, timesteps, x0)
        sqrt_omb = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, x0)
        return sqrt_ab * x0 + sqrt_omb * noise

    #从预测的噪声和当前带噪声的残差反推预测的干净残差
    def predict_x0_from_eps(self, x_t: torch.Tensor, timesteps: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, timesteps, x_t)
        sqrt_omb = self._extract(self.sqrt_one_minus_alpha_bars, timesteps, x_t)
        x0_pred = (x_t - sqrt_omb * eps_pred) / torch.clamp(sqrt_ab, min=1e-8)
        return x0_pred

    def training_forward(
        self,
        target: torch.Tensor,#真实目标图像
        condition: torch.Tensor,#条件图像
        init_latent: Optional[torch.Tensor] = None,#初始估计
    ) -> Dict[str, torch.Tensor]:
        if target.shape != condition.shape:
            raise ValueError(f"target/condition shape mismatch: {tuple(target.shape)} vs {tuple(condition.shape)}")
        if target.ndim != 4:
            raise ValueError(f"Expected 4D tensors [B,C,H,W], got {tuple(target.shape)}")
        #初始化init-latent张量或者检查形状匹配
        if init_latent is None:
            init_latent = torch.zeros_like(target)
        elif init_latent.shape != target.shape:
            raise ValueError(
                f"init_latent shape mismatch: {tuple(init_latent.shape)} vs target={tuple(target.shape)}"
            )

        # Residual diffusion: r0 = target - coarse mapper output.这个是要建模的残差
        target_residual = target - init_latent

        bsz = target_residual.shape[0]
        timesteps = torch.randint(0, self.diffusion_steps, (bsz,), device=target.device, dtype=torch.long)
        noise = torch.randn_like(target_residual)
        x_t = self.q_sample(target_residual, timesteps, noise)#随机采样时间步和噪声，加噪得到x-t


        eps_pred = self.noise_predictor(x_t, condition, timesteps)#噪声预测器预测噪声
        loss_noise = F.mse_loss(eps_pred, noise)#计算mse损失

        residual_pred = self.predict_x0_from_eps(x_t, timesteps, eps_pred)
        if self.clip_x0 is not None:
            residual_pred = residual_pred.clamp(-self.clip_x0, self.clip_x0)
        x0_pred = init_latent + residual_pred

        return {
            "x_t": x_t,
            "timesteps": timesteps,
            "noise": noise,
            "eps_pred": eps_pred,
            "residual_target": target_residual,
            "residual_pred": residual_pred,
            "init_latent": init_latent,
            "x0_pred": x0_pred,
            "loss_noise": loss_noise,
        }

    def _build_sampling_schedule(self, num_steps: Optional[int]) -> List[int]:
        if num_steps is None or num_steps >= self.diffusion_steps:
            return list(range(self.diffusion_steps - 1, -1, -1))

        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")

        raw = torch.linspace(self.diffusion_steps - 1, 0, steps=num_steps).round().long().tolist()
        schedule: List[int] = []
        seen = set()
        for t in raw:
            t_int = int(t)
            if t_int not in seen:
                schedule.append(t_int)
                seen.add(t_int)
        if schedule[-1] != 0:
            schedule.append(0)
        return schedule

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: Optional[int] = None,
        init_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if condition.ndim != 4:
            raise ValueError(f"condition must be 4D [B,C,H,W], got {tuple(condition.shape)}")
        if init_latent is None:
            init_latent = torch.zeros_like(condition)
        elif init_latent.shape != condition.shape:
            raise ValueError(
                f"init_latent shape mismatch: {tuple(init_latent.shape)} vs condition={tuple(condition.shape)}"
            )

        # Diffusion state is residual r_t.
        x_t = torch.randn_like(condition)
        schedule = self._build_sampling_schedule(num_steps=num_steps)

        for idx, t_cur in enumerate(schedule):
            t_next = schedule[idx + 1] if idx + 1 < len(schedule) else -1
            timesteps = torch.full(
                (condition.shape[0],),
                int(t_cur),
                device=condition.device,
                dtype=torch.long,
            )
            eps_pred = self.noise_predictor(x_t, condition, timesteps)
            residual_pred = self.predict_x0_from_eps(x_t, timesteps, eps_pred)

            if self.clip_x0 is not None:
                residual_pred = residual_pred.clamp(-self.clip_x0, self.clip_x0)

            if t_next < 0:
                x_t = residual_pred
                continue

            alpha_bar_next = self.alpha_bars[t_next].to(device=x_t.device, dtype=x_t.dtype)
            x_t = torch.sqrt(alpha_bar_next) * residual_pred + torch.sqrt(1.0 - alpha_bar_next) * eps_pred

        return init_latent + x_t
