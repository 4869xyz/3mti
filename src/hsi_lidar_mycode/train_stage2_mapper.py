import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .config_stage2 import Stage2Config
    from .dataset_remote import build_houston_loaders
    from .log_utils import build_logger, format_confusion_matrix, save_confusion_matrix
    from .losses_stage2 import total_stage2_mapper_loss
    from .metrics import compute_classification_metrics
    from .model_stage2 import Stage2HSILiDARMissingModalityClassifier
except ImportError:
    from config_stage2 import Stage2Config
    from dataset_remote import build_houston_loaders
    from log_utils import build_logger, format_confusion_matrix, save_confusion_matrix
    from losses_stage2 import total_stage2_mapper_loss
    from metrics import compute_classification_metrics
    from model_stage2 import Stage2HSILiDARMissingModalityClassifier


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def should_evaluate(epoch: int, eval_start_epoch: int, eval_interval: int) -> bool:
    return epoch >= eval_start_epoch and (epoch - eval_start_epoch) % eval_interval == 0


def _to_scalar_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().item()) for k, v in loss_dict.items()}


def _mapper_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k.startswith(("map_h_from_l.", "map_l_from_h."))
    }


def _build_optimizer(model, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found when building optimizer.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_cls_mapper: float,
    lambda_map_recon: float,
    lambda_map_cos: float,
    logger,
    epoch: int,
    log_batch_interval: int,
):
    model.train()
    model.use_mapper_pretrain_mode()
    total_samples = 0
    meter = {
        "loss": 0.0,
        "loss_cls_mapper": 0.0,
        "loss_cls_hsi_mapped": 0.0,
        "loss_cls_lidar_mapped": 0.0,
        "loss_cls_enhanced": 0.0,
        "loss_map_recon": 0.0,
        "loss_map_cos": 0.0,
        "loss_map_recon_h": 0.0,
        "loss_map_recon_l": 0.0,
        "loss_map_cos_h": 0.0,
        "loss_map_cos_l": 0.0,
    }

    pbar = tqdm(loader, desc="Train-S2-Mapper", leave=False)
    for step, batch in enumerate(pbar, start=1):
        hsi = batch["hsi"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model.forward_mapper_only(hsi, lidar)
        losses = total_stage2_mapper_loss(
            outputs=outputs,
            labels=labels,
            lambda_cls_mapper=lambda_cls_mapper,
            lambda_map_recon=lambda_map_recon,
            lambda_map_cos=lambda_map_cos,
        )

        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()

        bsz = labels.size(0)
        total_samples += bsz
        scalar_losses = _to_scalar_dict(losses)
        for key in meter:
            meter[key] += scalar_losses[key] * bsz

        pbar.set_postfix(
            loss=f"{scalar_losses['loss']:.4f}",
            cls=f"{scalar_losses['loss_cls_mapper']:.4f}",
            recon=f"{scalar_losses['loss_map_recon']:.4f}",
            cos=f"{scalar_losses['loss_map_cos']:.4f}",
        )

        if log_batch_interval > 0 and step % log_batch_interval == 0:
            logger.info(
                "Epoch %03d Batch %04d | loss=%.6f cls=%.6f cls_h=%.6f cls_l=%.6f cls_enh=%.6f "
                "map_recon=%.6f map_cos=%.6f "
                "recon_h=%.6f recon_l=%.6f cos_h=%.6f cos_l=%.6f",
                epoch,
                step,
                scalar_losses["loss"],
                scalar_losses["loss_cls_mapper"],
                scalar_losses["loss_cls_hsi_mapped"],
                scalar_losses["loss_cls_lidar_mapped"],
                scalar_losses["loss_cls_enhanced"],
                scalar_losses["loss_map_recon"],
                scalar_losses["loss_map_cos"],
                scalar_losses["loss_map_recon_h"],
                scalar_losses["loss_map_recon_l"],
                scalar_losses["loss_map_cos_h"],
                scalar_losses["loss_map_cos_l"],
            )

    if total_samples == 0:
        return {k: 0.0 for k in meter}
    return {k: v / total_samples for k, v in meter.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model,
    loader,
    device,
    num_classes: int,
    lambda_cls_mapper: float,
    lambda_map_recon: float,
    lambda_map_cos: float,
):
    model.eval()
    total_samples = 0
    meter = {
        "loss": 0.0,
        "loss_cls_mapper": 0.0,
        "loss_cls_hsi_mapped": 0.0,
        "loss_cls_lidar_mapped": 0.0,
        "loss_cls_enhanced": 0.0,
        "loss_map_recon": 0.0,
        "loss_map_cos": 0.0,
        "loss_map_recon_h": 0.0,
        "loss_map_recon_l": 0.0,
        "loss_map_cos_h": 0.0,
        "loss_map_cos_l": 0.0,
    }
    class_loss_sum = np.zeros((num_classes,), dtype=np.float64)
    class_count = np.zeros((num_classes,), dtype=np.int64)
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Eval-S2-Mapper", leave=False)
    for batch in pbar:
        hsi = batch["hsi"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model.forward_mapper_only(hsi, lidar)
        losses = total_stage2_mapper_loss(
            outputs=outputs,
            labels=labels,
            lambda_cls_mapper=lambda_cls_mapper,
            lambda_map_recon=lambda_map_recon,
            lambda_map_cos=lambda_map_cos,
        )
        ce_per_sample = F.cross_entropy(outputs["logits"], labels, reduction="none")

        preds = outputs["logits"].argmax(dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        labels_np = labels.detach().cpu().numpy().astype(np.int64)
        ce_np = ce_per_sample.detach().cpu().numpy().astype(np.float64)
        np.add.at(class_loss_sum, labels_np, ce_np)
        np.add.at(class_count, labels_np, 1)

        bsz = labels.size(0)
        total_samples += bsz
        scalar_losses = _to_scalar_dict(losses)
        for key in meter:
            meter[key] += scalar_losses[key] * bsz

    if total_samples == 0:
        avg_loss = {k: 0.0 for k in meter}
        metrics = {
            "oa": 0.0,
            "aa": 0.0,
            "kappa": 0.0,
            "per_class_acc": np.zeros((num_classes,), dtype=np.float32),
            "per_class_loss": np.zeros((num_classes,), dtype=np.float32),
            "per_class_count": np.zeros((num_classes,), dtype=np.int64),
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=np.int64),
        }
        return avg_loss, metrics

    avg_loss = {k: v / total_samples for k, v in meter.items()}
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_labels, dim=0)
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=num_classes)

    per_class_loss = np.divide(
        class_loss_sum,
        np.maximum(class_count, 1),
        where=class_count > 0,
    )
    per_class_loss = np.where(class_count > 0, per_class_loss, 0.0).astype(np.float32)
    metrics["per_class_loss"] = per_class_loss
    metrics["per_class_count"] = class_count
    return avg_loss, metrics


def save_checkpoint(path: str, model, optimizer, epoch: int, best_metrics: Dict[str, float], args) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "mapper_state_dict": _mapper_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metrics": best_metrics,
        "args": vars(args),
    }
    torch.save(payload, path)


def main(args):
    if args.dataset_name.lower() != "houston":
        raise NotImplementedError(
            f"Current Stage2 implementation supports dataset_name='houston' only, got {args.dataset_name}"
        )
    if not args.stage1_ckpt:
        raise ValueError("Stage2 mapper pretraining requires --stage1_ckpt.")
    if not os.path.exists(args.stage1_ckpt):
        raise FileNotFoundError(f"stage1_ckpt not found: {args.stage1_ckpt}")
    if args.eval_start_epoch < 1:
        raise ValueError(f"eval_start_epoch must be >= 1, got {args.eval_start_epoch}")
    if args.eval_interval < 1:
        raise ValueError(f"eval_interval must be >= 1, got {args.eval_interval}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = build_logger(
        output_dir=args.output_dir,
        filename="train.log",
        logger_name=f"stage2_mapper_{os.path.abspath(args.output_dir)}",
    )
    logger.info("==== Stage2 Mapper Pretraining ====")
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, test_loader, meta = build_houston_loaders(
        data_root=args.data_root,
        patch_size=args.patch_size,
        hsi_pca_dim=args.hsi_pca_dim,
        use_pca=args.use_pca,
        train_per_class=args.train_per_class,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_misalign_aug=args.use_misalign_aug,
    )
    logger.info("Dataset meta: %s", json.dumps(meta, ensure_ascii=False))

    first_batch = next(iter(train_loader))
    logger.info(
        "Smoke batch | hsi=%s lidar=%s label=%s",
        tuple(first_batch["hsi"].shape),
        tuple(first_batch["lidar"].shape),
        tuple(first_batch["label"].shape),
    )

    model = Stage2HSILiDARMissingModalityClassifier(
        num_classes=meta["num_classes"],
        latent_dim=args.latent_dim,
        csm_heads=args.csm_heads,
        csm_depth=args.csm_depth,
        dropout=args.dropout,
        mapper_num_blocks=args.mapper_num_blocks,
    ).to(device)

    load_info = model.load_stage1_checkpoint(
        checkpoint_path=args.stage1_ckpt,
        strict=args.strict_stage1_load,
        map_location="cpu",
    )
    logger.info(
        "Stage1 ckpt loaded | missing=%d ignored_missing=%d unexpected=%d",
        len(load_info["missing_keys"]),
        len(load_info["ignored_missing_keys"]),
        len(load_info["unexpected_keys"]),
    )

    model.set_all_trainable(False)
    model.set_mapper_trainable(True)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters after freezing: %d", trainable_count)

    optimizer = _build_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)
    best_oa = -1.0
    best_kappa = -1.0
    history = []
    confusion_dir = os.path.join(args.output_dir, "confusion_matrices")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_cls_mapper=args.lambda_cls_mapper,
            lambda_map_recon=args.lambda_map_recon,
            lambda_map_cos=args.lambda_map_cos,
            logger=logger,
            epoch=epoch,
            log_batch_interval=args.log_batch_interval,
        )

        do_eval = should_evaluate(epoch=epoch, eval_start_epoch=args.eval_start_epoch, eval_interval=args.eval_interval)
        if do_eval:
            val_loss, val_metrics = evaluate_one_epoch(
                model=model,
                loader=test_loader,
                device=device,
                num_classes=meta["num_classes"],
                lambda_cls_mapper=args.lambda_cls_mapper,
                lambda_map_recon=args.lambda_map_recon,
                lambda_map_cos=args.lambda_map_cos,
            )
            oa = float(val_metrics["oa"])
            aa = float(val_metrics["aa"])
            kappa = float(val_metrics["kappa"])
            class_ids = meta.get("class_ids", list(range(meta["num_classes"])))

            history.append(
                {
                    "epoch": epoch,
                    "evaluated": True,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metrics": {
                        "oa": oa,
                        "aa": aa,
                        "kappa": kappa,
                        "per_class_acc": val_metrics["per_class_acc"].tolist(),
                        "per_class_loss": val_metrics["per_class_loss"].tolist(),
                        "per_class_count": val_metrics["per_class_count"].tolist(),
                    },
                }
            )

            logger.info(
                "Epoch %03d | train_loss=%.6f cls=%.6f cls_h=%.6f cls_l=%.6f cls_enh=%.6f "
                "map_recon=%.6f map_cos=%.6f | "
                "val_loss=%.6f val_cls=%.6f val_cls_h=%.6f val_cls_l=%.6f val_cls_enh=%.6f "
                "val_map_recon=%.6f val_map_cos=%.6f | "
                "OA=%.6f AA=%.6f Kappa=%.6f",
                epoch,
                train_loss["loss"],
                train_loss["loss_cls_mapper"],
                train_loss["loss_cls_hsi_mapped"],
                train_loss["loss_cls_lidar_mapped"],
                train_loss["loss_cls_enhanced"],
                train_loss["loss_map_recon"],
                train_loss["loss_map_cos"],
                val_loss["loss"],
                val_loss["loss_cls_mapper"],
                val_loss["loss_cls_hsi_mapped"],
                val_loss["loss_cls_lidar_mapped"],
                val_loss["loss_cls_enhanced"],
                val_loss["loss_map_recon"],
                val_loss["loss_map_cos"],
                oa,
                aa,
                kappa,
            )
            logger.info("Per-class metrics: class_idx | orig_label | count | acc | ce_loss")
            for cls_idx in range(meta["num_classes"]):
                orig_label = class_ids[cls_idx] if cls_idx < len(class_ids) else cls_idx
                logger.info(
                    "  %02d | %3s | %7d | %.6f | %.6f",
                    cls_idx,
                    str(orig_label),
                    int(val_metrics["per_class_count"][cls_idx]),
                    float(val_metrics["per_class_acc"][cls_idx]),
                    float(val_metrics["per_class_loss"][cls_idx]),
                )

            conf_mat = val_metrics["confusion_matrix"]
            logger.info("Confusion matrix (epoch %03d):\n%s", epoch, format_confusion_matrix(conf_mat))
            save_confusion_matrix(
                conf_mat=conf_mat,
                save_dir=confusion_dir,
                tag=f"epoch_{epoch:03d}_stage2_mapper",
            )

            is_better = (oa > best_oa) or (abs(oa - best_oa) < 1e-12 and kappa > best_kappa)
            if is_better:
                best_oa = oa
                best_kappa = kappa
                save_checkpoint(
                    path=os.path.join(args.output_dir, "best.pt"),
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metrics={"oa": best_oa, "kappa": best_kappa},
                    args=args,
                )
                save_confusion_matrix(
                    conf_mat=conf_mat,
                    save_dir=confusion_dir,
                    tag="best_stage2_mapper",
                )
                logger.info("New best mapper checkpoint at epoch %03d | OA=%.6f Kappa=%.6f", epoch, best_oa, best_kappa)
        else:
            history.append(
                {
                    "epoch": epoch,
                    "evaluated": False,
                    "train_loss": train_loss,
                    "val_loss": None,
                    "val_metrics": None,
                }
            )
            logger.info(
                "Epoch %03d | train_loss=%.6f cls=%.6f cls_h=%.6f cls_l=%.6f cls_enh=%.6f "
                "map_recon=%.6f map_cos=%.6f "
                "(skip eval, start=%d interval=%d)",
                epoch,
                train_loss["loss"],
                train_loss["loss_cls_mapper"],
                train_loss["loss_cls_hsi_mapped"],
                train_loss["loss_cls_lidar_mapped"],
                train_loss["loss_cls_enhanced"],
                train_loss["loss_map_recon"],
                train_loss["loss_map_cos"],
                args.eval_start_epoch,
                args.eval_interval,
            )

        save_checkpoint(
            path=os.path.join(args.output_dir, "last.pt"),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metrics={"oa": best_oa, "kappa": best_kappa},
            args=args,
        )

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Training finished.")
    if best_oa < 0:
        logger.info("No evaluation executed. Please check eval_start_epoch/eval_interval.")
    else:
        logger.info("Best OA=%.6f Best Kappa=%.6f", best_oa, best_kappa)


def build_parser() -> argparse.ArgumentParser:
    cfg = Stage2Config()
    parser = argparse.ArgumentParser(description="Stage2 DirectionalFeatureMapper pretraining")
    lambda_cls_mapper = getattr(cfg, "lambda_cls_mapper", 1.0)
    lambda_map_recon = getattr(cfg, "lambda_map_recon", 1.0)
    lambda_map_cos = getattr(cfg, "lambda_map_cos", 0.5)

    parser.add_argument("--dataset_name", type=str, default=cfg.dataset_name)
    parser.add_argument("--data_root", type=str, default=cfg.data_root)
    parser.add_argument("--patch_size", type=int, default=cfg.patch_size)
    parser.add_argument("--hsi_pca_dim", type=int, default=cfg.hsi_pca_dim)
    parser.add_argument("--use_pca", type=str2bool, default=cfg.use_pca)
    parser.add_argument("--train_per_class", type=int, default=cfg.train_per_class)
    parser.add_argument("--use_misalign_aug", type=str2bool, default=cfg.use_misalign_aug)

    parser.add_argument("--latent_dim", type=int, default=cfg.latent_dim)
    parser.add_argument("--csm_heads", type=int, default=cfg.csm_heads)
    parser.add_argument("--csm_depth", type=int, default=cfg.csm_depth)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--mapper_num_blocks", type=int, default=cfg.mapper_num_blocks)

    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)

    parser.add_argument("--lambda_cls_mapper", type=float, default=lambda_cls_mapper)
    parser.add_argument("--lambda_map_recon", type=float, default=lambda_map_recon)
    parser.add_argument("--lambda_map_cos", type=float, default=lambda_map_cos)

    parser.add_argument("--eval_start_epoch", type=int, default=cfg.eval_start_epoch)
    parser.add_argument("--eval_interval", type=int, default=cfg.eval_interval)

    parser.add_argument("--stage1_ckpt", type=str, default=cfg.stage1_ckpt)
    parser.add_argument("--strict_stage1_load", type=str2bool, default=cfg.strict_stage1_load)

    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--log_batch_interval", type=int, default=cfg.log_batch_interval)
    parser.add_argument("--output_dir", type=str, default=cfg.output_dir)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    main(cli_args)
