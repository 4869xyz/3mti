import os
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

try:
    from .augment import MisalignmentAugment
except ImportError:
    from augment import MisalignmentAugment


def apply_pca(x: np.ndarray, num_components: int) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected HSI cube with shape [H, W, C], got {x.shape}")

    h, w, c = x.shape
    if num_components <= 0:
        raise ValueError(f"num_components must be > 0, got {num_components}")
    if num_components >= c:
        return x.astype(np.float32)

    flat = x.reshape(-1, c)
    pca = PCA(n_components=num_components, whiten=True)
    transformed = pca.fit_transform(flat)
    return transformed.reshape(h, w, num_components).astype(np.float32)


def _pick_first_key(mat_obj: Dict[str, np.ndarray], candidates: List[str]) -> np.ndarray:
    for key in candidates:
        if key in mat_obj:
            return mat_obj[key]
    available = sorted([k for k in mat_obj.keys() if not k.startswith("__")])
    raise KeyError(f"Cannot find keys {candidates}. Available keys: {available}")


def load_houston_dataset(data_root: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 Houston 数据：
    - HSI.mat: 高光谱数据 [H, W, C]
    - LiDAR.mat: 激光雷达/DSM 数据 [H, W] 或 [H, W, 1]
    - gt.mat: 整张标签图 [H, W]
        背景为 -1
        有效类别为 1~15
    """
    data_dir = os.path.join(data_root, "houston")
    hsi_path = os.path.join(data_dir, "HSI.mat")
    lidar_path = os.path.join(data_dir, "LiDAR.mat")
    gt_path = os.path.join(data_dir, "gt.mat")

    for p in (hsi_path, lidar_path, gt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    hsi_mat = sio.loadmat(hsi_path)
    lidar_mat = sio.loadmat(lidar_path)
    gt_mat = sio.loadmat(gt_path)

    hsi = _pick_first_key(hsi_mat, ["HSI", "hsi", "Houston", "data"])
    lidar = _pick_first_key(lidar_mat, ["LiDAR", "lidar", "DSM", "data"])
    label_map = _pick_first_key(gt_mat, ["gt", "label", "labels", "GT", "gt_map"])

    hsi = np.asarray(hsi, dtype=np.float32)
    lidar = np.asarray(lidar, dtype=np.float32)
    label_map = np.asarray(label_map, dtype=np.int64)

    if hsi.ndim != 3:
        raise ValueError(f"HSI should have shape [H, W, C], got {hsi.shape}")

    if lidar.ndim == 3:
        if lidar.shape[2] == 1:
            lidar = lidar[..., 0]
        else:
            # 如果意外是多通道，这里退化成均值单通道
            lidar = lidar.mean(axis=2)

    if lidar.ndim != 2:
        raise ValueError(f"LiDAR should have shape [H, W] after processing, got {lidar.shape}")

    if label_map.ndim != 2:
        raise ValueError(f"Label map should have shape [H, W], got {label_map.shape}")

    if hsi.shape[:2] != lidar.shape[:2]:
        raise ValueError(f"HSI and LiDAR spatial size mismatch: {hsi.shape[:2]} vs {lidar.shape[:2]}")

    if hsi.shape[:2] != label_map.shape[:2]:
        raise ValueError(f"HSI and label map spatial size mismatch: {hsi.shape[:2]} vs {label_map.shape[:2]}")

    return hsi, lidar, label_map


def standardize_hsi(hsi: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    h, w, c = hsi.shape
    flat = hsi.reshape(-1, c)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    out = (flat - mean) / std
    return out.reshape(h, w, c).astype(np.float32)


def standardize_lidar(lidar: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = float(lidar.mean())
    std = float(lidar.std())
    std = max(std, eps)
    return ((lidar - mean) / std).astype(np.float32)


def split_fixed_train_test(
    labels: np.ndarray,
    train_per_class: int = 40,
    seed: int = 42,
    ignore_label: int = -1,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    从整张标签图中划分训练/测试：
    - 忽略背景标签 ignore_label（你的数据里是 -1）
    - 对每个有效类别随机抽取 train_per_class 个像素作为训练
    - 剩余像素作为测试
    """
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D label map, got {labels.shape}")
    if train_per_class <= 0:
        raise ValueError(f"train_per_class must be > 0, got {train_per_class}")

    rng = np.random.default_rng(seed)
    flat = labels.reshape(-1)

    unique_vals = np.unique(flat)
    class_ids = sorted([int(x) for x in unique_vals if int(x) != ignore_label])

    if len(class_ids) == 0:
        raise RuntimeError(
            f"No valid classes found after ignoring label {ignore_label}. "
            f"Unique labels = {unique_vals.tolist()}"
        )

    # 进一步约束：有效类别必须为正整数
    invalid_classes = [x for x in class_ids if x <= 0]
    if invalid_classes:
        raise RuntimeError(
            f"Found non-positive valid class ids {invalid_classes}. "
            f"Please check label map and ignore_label setting."
        )

    train_indices: List[int] = []
    test_indices: List[int] = []

    for cls_id in class_ids:
        indices = np.flatnonzero(flat == cls_id)
        rng.shuffle(indices)
        n = int(indices.size)

        if n == 0:
            continue

        # 至少留 1 个样本给测试集
        n_train = min(train_per_class, n - 1)

        # 极端情况保护
        if n_train <= 0:
            # 只有 1 个样本时，训练测试都用它，不推荐，但避免崩
            train_part = indices[:1]
            test_part = indices[:1]
        else:
            train_part = indices[:n_train]
            test_part = indices[n_train:]

            if test_part.size == 0:
                test_part = indices[-1:]
                train_part = indices[:-1]

        train_indices.extend(train_part.tolist())
        test_indices.extend(test_part.tolist())

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices, class_ids


def flat_indices_to_coords(flat_indices: np.ndarray, width: int) -> np.ndarray:
    rows = flat_indices // width
    cols = flat_indices % width
    return np.stack([rows, cols], axis=1).astype(np.int64)


class RemoteSensingDualModalDataset(Dataset):
    def __init__(
        self,
        hsi_data: np.ndarray,
        lidar_data: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        class_to_index: Dict[int, int],
        patch_size: int = 11,
        augmenter=None,
    ) -> None:
        super().__init__()
        if patch_size % 2 == 0:
            raise ValueError(f"patch_size must be odd, got {patch_size}")

        self.hsi_data = np.asarray(hsi_data, dtype=np.float32)
        self.lidar_data = np.asarray(lidar_data, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.coords = np.asarray(coords, dtype=np.int64)
        self.class_to_index = class_to_index
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.augmenter = augmenter

        if self.hsi_data.ndim != 3:
            raise ValueError(f"hsi_data should be [H, W, C], got {self.hsi_data.shape}")
        if self.lidar_data.ndim != 2:
            raise ValueError(f"lidar_data should be [H, W], got {self.lidar_data.shape}")
        if self.labels.ndim != 2:
            raise ValueError(f"labels should be [H, W], got {self.labels.shape}")

        h, w = self.labels.shape
        if self.hsi_data.shape[:2] != (h, w):
            raise ValueError(f"HSI shape mismatch with labels: {self.hsi_data.shape[:2]} vs {(h, w)}")
        if self.lidar_data.shape[:2] != (h, w):
            raise ValueError(f"LiDAR shape mismatch with labels: {self.lidar_data.shape[:2]} vs {(h, w)}")

        self.hsi_padded = np.pad(
            self.hsi_data,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode="reflect",
        )
        self.lidar_padded = np.pad(
            self.lidar_data,
            ((self.pad, self.pad), (self.pad, self.pad)),
            mode="reflect",
        )

    def __len__(self) -> int:
        return int(self.coords.shape[0])

    def __getitem__(self, idx: int):
        row, col = self.coords[idx].tolist()
        row = int(row)
        col = int(col)

        row_p = row + self.pad
        col_p = col + self.pad

        hsi_patch = self.hsi_padded[
            row_p - self.pad : row_p + self.pad + 1,
            col_p - self.pad : col_p + self.pad + 1,
            :,
        ]
        lidar_patch = self.lidar_padded[
            row_p - self.pad : row_p + self.pad + 1,
            col_p - self.pad : col_p + self.pad + 1,
        ]

        # HSI: [H, W, C] -> [C, H, W] -> [1, C, H, W]
        hsi_patch = np.transpose(hsi_patch, (2, 0, 1))
        hsi_tensor = torch.from_numpy(np.ascontiguousarray(hsi_patch)).unsqueeze(0).float()

        # LiDAR: [H, W] -> [1, H, W]
        lidar_tensor = torch.from_numpy(np.ascontiguousarray(lidar_patch)).unsqueeze(0).float()

        if self.augmenter is not None:
            hsi_tensor, lidar_tensor = self.augmenter(hsi_tensor, lidar_tensor)

        raw_label = int(self.labels[row, col])
        if raw_label not in self.class_to_index:
            raise KeyError(
                f"Label {raw_label} at coord ({row}, {col}) not found in class_to_index. "
                f"Please check split results and ignore_label setting."
            )

        label = self.class_to_index[raw_label]

        return {
            "hsi": hsi_tensor,
            "lidar": lidar_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "coord": torch.tensor([row, col], dtype=torch.long),
        }


def build_houston_loaders(
    data_root: str = "data",
    patch_size: int = 11,
    hsi_pca_dim: int = 30,
    use_pca: bool = True,
    train_per_class: int = 40,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    use_misalign_aug: bool = True,
    ignore_label: int = -1,
):
    hsi_data, lidar_data, labels = load_houston_dataset(data_root=data_root)

    if use_pca:
        hsi_data = apply_pca(hsi_data, hsi_pca_dim)

    hsi_data = standardize_hsi(hsi_data)
    lidar_data = standardize_lidar(lidar_data)

    train_indices, test_indices, class_ids = split_fixed_train_test(
        labels=labels,
        train_per_class=train_per_class,
        seed=seed,
        ignore_label=ignore_label,
    )

    if train_indices.size == 0 or test_indices.size == 0:
        raise RuntimeError("Train/test split is empty. Check labels and train_per_class.")

    flat = labels.reshape(-1)
    train_lab = flat[train_indices]
    test_lab = flat[test_indices]

    print("All labels:", np.unique(flat, return_counts=True))
    print("Train labels:", np.unique(train_lab, return_counts=True))
    print("Test labels:", np.unique(test_lab, return_counts=True))
    print("class_ids:", class_ids)

    h, w = labels.shape
    train_coords = flat_indices_to_coords(train_indices, width=w)
    test_coords = flat_indices_to_coords(test_indices, width=w)

    class_to_index = {cid: i for i, cid in enumerate(class_ids)}

    train_aug = MisalignmentAugment() if use_misalign_aug else None

    dataset_train = RemoteSensingDualModalDataset(
        hsi_data=hsi_data,
        lidar_data=lidar_data,
        labels=labels,
        coords=train_coords,
        class_to_index=class_to_index,
        patch_size=patch_size,
        augmenter=train_aug,
    )
    dataset_test = RemoteSensingDualModalDataset(
        hsi_data=hsi_data,
        lidar_data=lidar_data,
        labels=labels,
        coords=test_coords,
        class_to_index=class_to_index,
        patch_size=patch_size,
        augmenter=None,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    meta = {
        "hsi_shape": tuple(hsi_data.shape),
        "lidar_shape": tuple(lidar_data.shape),
        "labels_shape": tuple(labels.shape),
        "num_classes": len(class_ids),
        "class_ids": class_ids,
        "train_samples": len(dataset_train),
        "test_samples": len(dataset_test),
        "train_per_class": train_per_class,
        "ignore_label": ignore_label,
    }

    return train_loader, test_loader, meta
