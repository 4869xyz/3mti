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
    from .config_stage1 import Stage1Config
    from .dataset_remote import build_houston_loaders
    from .log_utils import build_logger, format_confusion_matrix, save_confusion_matrix
    from .losses import total_loss
    from .metrics import compute_classification_metrics
    from .model_stage1 import Stage1HSILiDARDiffusionClassifier
except ImportError:
    from config_stage1 import Stage1Config
    from dataset_remote import build_houston_loaders
    from log_utils import build_logger, format_confusion_matrix, save_confusion_matrix
    from losses import total_loss
    from metrics import compute_classification_metrics
    from model_stage1 import Stage1HSILiDARDiffusionClassifier


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


def _to_scalar_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().item()) for k, v in loss_dict.items()}


def should_evaluate(epoch: int, eval_start_epoch: int, eval_interval: int) -> bool:
    return epoch >= eval_start_epoch and (epoch - eval_start_epoch) % eval_interval == 0


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_align: float,
    lambda_diff: float,
    logger,
    epoch: int,
    log_batch_interval: int,
):
    model.train()
    total_samples = 0
    meter = {"loss": 0.0, "loss_cls": 0.0, "loss_align": 0.0, "loss_diff": 0.0}

    pbar = tqdm(loader, desc="Train-S1", leave=False)
    for step, batch in enumerate(pbar, start=1):
        hsi = batch["hsi"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(hsi, lidar)
        losses = total_loss(outputs, labels, lambda_align=lambda_align, lambda_diff=lambda_diff)

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
            cls=f"{scalar_losses['loss_cls']:.4f}",
            align=f"{scalar_losses['loss_align']:.4f}",
            diff=f"{scalar_losses['loss_diff']:.4f}",
        )

        if log_batch_interval > 0 and step % log_batch_interval == 0:
            logger.info(
                "Epoch %03d Batch %04d | loss=%.6f cls=%.6f align=%.6f diff=%.6f",
                epoch,
                step,
                scalar_losses["loss"],
                scalar_losses["loss_cls"],
                scalar_losses["loss_align"],
                scalar_losses["loss_diff"],
            )

    if total_samples == 0:
        return {k: 0.0 for k in meter}
    return {k: v / total_samples for k, v in meter.items()}


@torch.no_grad()
def evaluate_one_epoch(
    model,
    loader,
    device,
    lambda_align: float,
    lambda_diff: float,
    num_classes: int,
):
    model.eval()
    total_samples = 0
    meter = {"loss": 0.0, "loss_cls": 0.0, "loss_align": 0.0, "loss_diff": 0.0}
    class_loss_sum = np.zeros((num_classes,), dtype=np.float64)
    class_count = np.zeros((num_classes,), dtype=np.int64)

    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Eval-S1", leave=False)
    for batch in pbar:
        hsi = batch["hsi"].to(device, non_blocking=True)
        lidar = batch["lidar"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = model(hsi, lidar)
        losses = total_loss(outputs, labels, lambda_align=lambda_align, lambda_diff=lambda_diff)
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
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metrics": best_metrics,
        "args": vars(args),
    }
    torch.save(payload, path)


def main(args):
    if args.dataset_name.lower() != "houston":
        raise NotImplementedError(
            f"Current Stage1 implementation supports dataset_name='houston' only, got {args.dataset_name}"
        )
    if args.eval_start_epoch < 1:
        raise ValueError(f"eval_start_epoch must be >= 1, got {args.eval_start_epoch}")
    if args.eval_interval < 1:
        raise ValueError(f"eval_interval must be >= 1, got {args.eval_interval}")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = build_logger(
        output_dir=args.output_dir,
        filename="train.log",
        logger_name=f"stage1_{os.path.abspath(args.output_dir)}",
    )
    logger.info("==== Stage1 Training ====")
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

    model = Stage1HSILiDARDiffusionClassifier(
        num_classes=meta["num_classes"],
        latent_dim=args.latent_dim,
        csm_heads=args.csm_heads,
        csm_depth=args.csm_depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            lambda_align=args.lambda_align,
            lambda_diff=args.lambda_diff,
            logger=logger,
            epoch=epoch,
            log_batch_interval=args.log_batch_interval,
        )

        do_eval = should_evaluate(
            epoch=epoch,
            eval_start_epoch=args.eval_start_epoch,
            eval_interval=args.eval_interval,
        )

        if do_eval:
            val_loss, val_metrics = evaluate_one_epoch(
                model=model,
                loader=test_loader,
                device=device,
                lambda_align=args.lambda_align,
                lambda_diff=args.lambda_diff,
                num_classes=meta["num_classes"],
            )
            oa = float(val_metrics["oa"])
            aa = float(val_metrics["aa"])
            kappa = float(val_metrics["kappa"])
            per_class_acc = val_metrics["per_class_acc"]
            per_class_loss = val_metrics["per_class_loss"]
            per_class_count = val_metrics["per_class_count"]
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
                        "per_class_acc": per_class_acc.tolist(),
                        "per_class_loss": per_class_loss.tolist(),
                        "per_class_count": per_class_count.tolist(),
                    },
                }
            )

            logger.info(
                "Epoch %03d | train_loss=%.6f cls=%.6f align=%.6f diff=%.6f | "
                "val_loss=%.6f val_cls=%.6f val_align=%.6f val_diff=%.6f | OA=%.6f AA=%.6f Kappa=%.6f",
                epoch,
                train_loss["loss"],
                train_loss["loss_cls"],
                train_loss["loss_align"],
                train_loss["loss_diff"],
                val_loss["loss"],
                val_loss["loss_cls"],
                val_loss["loss_align"],
                val_loss["loss_diff"],
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
                    int(per_class_count[cls_idx]),
                    float(per_class_acc[cls_idx]),
                    float(per_class_loss[cls_idx]),
                )

            conf_mat = val_metrics["confusion_matrix"]
            logger.info("Confusion matrix (epoch %03d):\n%s", epoch, format_confusion_matrix(conf_mat))
            save_confusion_matrix(
                conf_mat=conf_mat,
                save_dir=confusion_dir,
                tag=f"epoch_{epoch:03d}_stage1",
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
                    tag="best_stage1",
                )
                logger.info("New best checkpoint at epoch %03d | OA=%.6f Kappa=%.6f", epoch, best_oa, best_kappa)
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
                "Epoch %03d | train_loss=%.6f cls=%.6f align=%.6f diff=%.6f "
                "(skip eval, start=%d interval=%d)",
                epoch,
                train_loss["loss"],
                train_loss["loss_cls"],
                train_loss["loss_align"],
                train_loss["loss_diff"],
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
    cfg = Stage1Config()
    parser = argparse.ArgumentParser(description="Stage1 HSI+LiDAR classification training")

    parser.add_argument("--dataset_name", type=str, default=cfg.dataset_name)
    parser.add_argument("--data_root", type=str, default=cfg.data_root)
    parser.add_argument("--patch_size", type=int, default=cfg.patch_size)
    parser.add_argument("--hsi_pca_dim", type=int, default=cfg.hsi_pca_dim)
    parser.add_argument("--use_pca", type=str2bool, default=cfg.use_pca)
    parser.add_argument("--latent_dim", type=int, default=cfg.latent_dim)

    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)

    parser.add_argument("--lambda_align", type=float, default=cfg.lambda_align)
    parser.add_argument("--lambda_diff", type=float, default=cfg.lambda_diff)

    parser.add_argument("--use_misalign_aug", type=str2bool, default=cfg.use_misalign_aug)
    parser.add_argument("--eval_start_epoch", type=int, default=cfg.eval_start_epoch)
    parser.add_argument("--eval_interval", type=int, default=cfg.eval_interval)
    parser.add_argument("--train_per_class", type=int, default=cfg.train_per_class)

    parser.add_argument("--csm_heads", type=int, default=cfg.csm_heads)
    parser.add_argument("--csm_depth", type=int, default=cfg.csm_depth)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)

    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--log_batch_interval", type=int, default=cfg.log_batch_interval)
    parser.add_argument("--output_dir", type=str, default=cfg.output_dir)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    main(cli_args)
