from dataclasses import dataclass


@dataclass
class Stage2Config:
    dataset_name: str = "houston"
    data_root: str = "/root/autodl-tmp"
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

    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4

    lambda_cls_mapper: float = 1.0
    lambda_map_recon: float = 1.0
    lambda_map_cos: float = 0.5

    eval_start_epoch: int = 1
    eval_interval: int = 1

    stage1_ckpt: str = "/root/3MTI-main-old/outputs/stage1_houston/best.pt"
    strict_stage1_load: bool = False

    seed: int = 42
    num_workers: int = 0
    log_batch_interval: int = 50
    output_dir: str = "outputs/stage2_houston_mapper"
