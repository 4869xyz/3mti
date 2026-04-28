from dataclasses import dataclass


@dataclass
class Stage1Config:
    dataset_name: str = "houston"
    data_root: str = "/root/autodl-tmp"
    patch_size: int = 11
    hsi_pca_dim: int = 30
    use_pca: bool = True
    latent_dim: int = 128
    batch_size: int = 64
    epochs: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_align: float = 0.1
    lambda_diff: float = 0.2
    use_misalign_aug: bool = True
    eval_start_epoch: int = 1
    eval_interval: int = 1
    csm_heads: int = 4
    csm_depth: int = 2
    dropout: float = 0.3
    seed: int = 42
    num_workers: int = 0
    log_batch_interval: int = 50
    train_per_class: int = 40
    output_dir: str = "outputs/stage1_houston"
