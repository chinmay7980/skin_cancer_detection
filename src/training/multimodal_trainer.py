"""Training and evaluation routines for Multimodal Fusion models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_predictions, find_optimal_f1_threshold


def train_one_epoch_multimodal(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Execute one training epoch for a multimodal network."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for imgs, metas, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs, metas)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(1, total_samples)


def evaluate_multimodal(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate multimodal model on validation partition and return metrics and probabilities.

    Guarantees:
    - Confusion matrix is computed at both default threshold (0.5) and optimal F1 threshold.
    - Confusion matrix counts at optimal threshold match F1_at_optimal exactly.
    """
    model.eval()
    all_probs: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for imgs, metas, targets in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)

            logits = model(imgs, metas)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())

    y_prob = np.array(all_probs, dtype=np.float64)
    y_true = np.array(all_targets, dtype=np.int32)

    # 1. Base evaluation at default 0.5 threshold
    eval_results = evaluate_predictions(y_true, y_prob, threshold=0.5)

    # 2. Threshold exploration on validation data to optimize F1
    opt_info = find_optimal_f1_threshold(y_true, y_prob)
    eval_results.update(opt_info)

    return eval_results, y_prob, y_true


def save_multimodal_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_ap: float,
    checkpoint_path: Path | str,
) -> None:
    """Save model weights and training metadata locally."""
    ckpt_file = Path(checkpoint_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_ap": round(float(val_ap), 5),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_file,
    )
