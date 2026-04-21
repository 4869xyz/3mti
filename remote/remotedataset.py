import os

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA


SUPPORTED_DATASETS = ["augsburg", "muufl", "houston", "trento", "yancheng"]


def apply_pca(x, num_components):
    """Apply PCA on HSI cube [H, W, C] and reduce to num_components channels."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [H, W, C], got {x.shape}")

    h, w, c = x.shape
    if num_components <= 0:
        raise ValueError(f"num_components must be > 0, got {num_components}")
    if num_components >= c:
        return x.astype(np.float32)

    flat = np.reshape(x, (-1, c))
    pca = PCA(n_components=num_components, whiten=True)
    reduced = pca.fit_transform(flat)
    reduced = np.reshape(reduced, (h, w, num_components)).astype(np.float32)
    return reduced


def load_dataset(args):
    dataset_name = args.dataset_name.lower()
    print("Current working directory:", os.getcwd())
    assert dataset_name in SUPPORTED_DATASETS

    if dataset_name == "augsburg":
        hsi_data = sio.loadmat("data/augsburg/data_HS_LR.mat")["data_HS_LR"]
        lidar_data = sio.loadmat("data/augsburg/data_DSM.mat")["data_DSM"]
        labels_train = sio.loadmat("data/augsburg/train_test_gt.mat")["train_data"]
        labels_test = sio.loadmat("data/augsburg/train_test_gt.mat")["test_data"]
        labels = labels_test + labels_train
    elif dataset_name == "muufl":
        data = sio.loadmat("data/muufl/muufl.mat")
        hsi_data = data["hsi"]
        lidar_data = data["lidar_1"][..., 0]
        labels = data["gt"]
        labels[labels == -1] = 0
    elif dataset_name == "houston":
        hsi_data = sio.loadmat("data/houston/Houston.mat")["HSI"]
        lidar_data = sio.loadmat("data/houston/LiDAR.mat")["LiDAR"]
        labels_train = sio.loadmat("data/houston/train_test_gt.mat")["train_data"]
        labels_test = sio.loadmat("data/houston/train_test_gt.mat")["test_data"]
        labels = labels_test + labels_train
    elif dataset_name == "trento":
        hsi_data = sio.loadmat("data/trento/HSI_Trento.mat")["hsi_trento"]
        lidar_data = sio.loadmat("data/trento/Lidar1_Trento.mat")["lidar1_trento"]
        labels = sio.loadmat("data/trento/GT_Trento.mat")["gt_trento"]
    else:
        raise NotImplementedError("yancheng loader is not implemented in remote/remotedataset.py yet")

    return hsi_data, lidar_data, labels


def sampling(proportion, ground_truth):
    """Sample train/test indices by class. proportion is train ratio."""
    train = {}
    test = {}
    m = int(np.max(ground_truth))

    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)

        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0

        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]

    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]

    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def sampling_with_bg(proportion, ground_truth):
    """Sample train/test indices by class, including background class 0."""
    train = {}
    test = {}
    m = int(np.max(ground_truth))

    for i in range(m + 1):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i]
        np.random.shuffle(indexes)

        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0

        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]

    train_indexes = []
    test_indexes = []
    for i in range(m + 1):
        train_indexes += train[i]
        test_indexes += test[i]

    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


def split_traintest(args, groundTruth):
    train = {}
    test = {}
    m = int(np.max(groundTruth))
    dataset_name = args.dataset_name.lower()

    if dataset_name == "augsburg":
        amount = [40 for _ in range(7)]
    elif dataset_name == "muufl":
        amount = [40 for _ in range(11)]
    elif dataset_name == "houston":
        amount = [40 for _ in range(15)]
    elif dataset_name == "trento":
        amount = [40 for _ in range(6)]
    else:
        raise NotImplementedError(f"split_traintest does not support dataset: {dataset_name}")

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)

        nb_val = int(amount[i])
        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]

    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
