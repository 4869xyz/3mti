from dataclasses import dataclass


@dataclass
class Stage2Config:
    dataset_name: str = "houston"
    data_root: str = "data"
    patch_size: int = 11
    hsi_pca_dim: int = 30
    use_pca: bool = True
    train_per_class: int = 40
    use_misalign_aug: bool = True

    latent_dim: int = 128
    csm_heads: int = 4
    csm_depth: int = 2
    dropout: float = 0.3
    mapper_num_blocks: int = 4

    diffusion_hidden_dim: int = 128
    diffusion_steps: int = 60
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 2e-2
    sampling_steps: int = 30

    batch_size: int = 64
    epochs: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-4

    lambda_cls_missing: float = 1.0
    lambda_noise: float = 0.5
    lambda_recon: float = 0.5
    lambda_align: float = 0.05
    lambda_refine: float = 0.2
    lambda_map_recon: float = 1.0
    lambda_map_cos: float = 0.5

    eval_start_epoch: int = 1
    eval_interval: int = 1
    eval_modes: str = "full,hsi_missing,lidar_missing"

    stage1_ckpt: str = ""
    strict_stage1_load: bool = False
    freeze_backbone_epochs: int = 0

    seed: int = 42
    num_workers: int = 0
    log_batch_interval: int = 50
    output_dir: str = "outputs/stage2_houston_missing_modality"
