# TP4/src/utils.py
from __future__ import annotations
from dataclasses import dataclass
import time
import random
import os
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class Timer:
    t0: float = 0.0
    t1: float = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.t1 = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return float(self.t1 - self.t0)


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).float().mean().item())


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    # pred, y: shape [N], int64
    f1_sum = 0.0
    for c in range(num_classes):
        tp = int(((pred == c) & (y == c)).sum().item())
        fp = int(((pred == c) & (y != c)).sum().item())
        fn = int(((pred != c) & (y == c)).sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1_c = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_sum += f1_c

    return float(f1_sum / num_classes)


def compute_metrics(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> dict:
    pred = torch.argmax(logits, dim=-1)
    return {
        "acc": accuracy(pred, y),
        "macro_f1": macro_f1(pred, y, num_classes),
    }
