from typing import Dict, Union

import numpy as np
import torch


def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    num_classes: int,
) -> np.ndarray:
    y_true = _to_numpy(y_true).reshape(-1)
    y_pred = _to_numpy(y_pred).reshape(-1)
    mask = (
        (y_true >= 0)
        & (y_true < num_classes)
        & (y_pred >= 0)
        & (y_pred < num_classes)
    )
    y_true = y_true[mask].astype(np.int64)
    y_pred = y_pred[mask].astype(np.int64)

    if y_true.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    cm = np.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes * num_classes,
    )
    return cm.reshape(num_classes, num_classes)


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    num_classes: int,
) -> Dict[str, Union[float, np.ndarray]]:
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = cm.sum()
    correct = np.trace(cm)

    oa = float(correct / total) if total > 0 else 0.0

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.divide(np.diag(cm), row_sum, where=row_sum > 0)
    per_class = np.where(row_sum > 0, per_class, 0.0)
    valid = row_sum > 0
    aa = float(per_class[valid].mean()) if np.any(valid) else 0.0

    if total > 0:
        pe = float((row_sum * col_sum).sum() / (total * total))
        kappa = float((oa - pe) / (1.0 - pe)) if (1.0 - pe) > 1e-12 else 0.0
    else:
        kappa = 0.0

    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "per_class_acc": per_class,
        "confusion_matrix": cm,
    }
