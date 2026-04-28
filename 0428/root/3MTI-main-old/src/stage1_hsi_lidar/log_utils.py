import logging
import os
from typing import Optional

import numpy as np


def build_logger(output_dir: str, filename: str = "train.log", logger_name: Optional[str] = None) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, filename)

    name = logger_name if logger_name else f"stage_logger_{os.path.abspath(output_dir)}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to avoid duplicate logs in repeated runs.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def format_confusion_matrix(conf_mat: np.ndarray) -> str:
    return np.array2string(
        conf_mat,
        max_line_width=200,
        separator=", ",
    )


def save_confusion_matrix(conf_mat: np.ndarray, save_dir: str, tag: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{tag}.npy"), conf_mat)
    with open(os.path.join(save_dir, f"{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(format_confusion_matrix(conf_mat))
        f.write("\n")
