"""Training, evaluation, benchmarking, and checkpointing for CNN models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_predictions, find_optimal_f1_threshold


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    """Run one epoch of training and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (images, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_cnn(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on dataloader and return metrics, probabilities, and true labels."""
    model.eval()
    all_probs = []
    all_targets = []

    for i, (images, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs)
        all_targets.extend(targets.numpy())

    y_prob = np.array(all_probs)
    y_true = np.array(all_targets)

    # Calculate metrics at default 0.5 threshold
    eval_results = evaluate_predictions(y_true, y_prob, threshold=0.5)

    # Find optimal threshold on validation data
    opt_info = find_optimal_f1_threshold(y_true, y_prob)
    eval_results["optimal_threshold"] = opt_info["optimal_threshold"]
    eval_results["f1_at_optimal"] = opt_info["f1_at_optimal"]
    eval_results["precision_at_optimal"] = opt_info["precision_at_optimal"]
    eval_results["recall_at_optimal"] = opt_info["recall_at_optimal"]

    return eval_results, y_prob, y_true


def benchmark_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_batches: int = 30,
    total_train_samples: int = 240638,
    total_val_samples: int = 80211,
) -> Dict[str, Any]:
    """Benchmark data-loading + forward + backward throughput and project epoch runtime."""
    model.train()
    batch_times = []

    # Warm-up 2 batches
    it = iter(dataloader)
    for _ in range(min(2, len(dataloader))):
        try:
            imgs, tgts = next(it)
            imgs, tgts = imgs.to(device), tgts.to(device)
            optimizer.zero_grad()
            l = criterion(model(imgs), tgts)
            l.backward()
            optimizer.step()
        except StopIteration:
            break

    # Timed benchmarking loop
    batches_measured = 0
    start_bench = time.perf_counter()

    for imgs, tgts in dataloader:
        if batches_measured >= n_batches:
            break
        t0 = time.perf_counter()
        imgs, tgts = imgs.to(device), tgts.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), tgts)
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()

        batch_times.append(t1 - t0)
        batches_measured += 1

    total_bench_time = time.perf_counter() - start_bench
    avg_batch_time = np.mean(batch_times)
    batch_size = dataloader.batch_size or 8
    img_per_sec = batch_size / avg_batch_time

    # Project epoch durations
    train_batches = int(np.ceil(total_train_samples / batch_size))
    est_train_epoch_sec = train_batches * avg_batch_time
    est_train_epoch_hours = est_train_epoch_sec / 3600.0

    val_batches = int(np.ceil(total_val_samples / batch_size))
    # Eval is ~3x faster than train (no backprop)
    est_val_pass_sec = val_batches * (avg_batch_time / 3.0)
    est_val_pass_min = est_val_pass_sec / 60.0

    return {
        "batches_measured": batches_measured,
        "batch_size": batch_size,
        "avg_batch_time_sec": round(float(avg_batch_time), 4),
        "throughput_img_per_sec": round(float(img_per_sec), 2),
        "total_train_samples": total_train_samples,
        "projected_train_batches": train_batches,
        "est_train_epoch_sec": round(float(est_train_epoch_sec), 1),
        "est_train_epoch_hours": round(float(est_train_epoch_hours), 2),
        "total_val_samples": total_val_samples,
        "est_val_pass_min": round(float(est_val_pass_min), 2),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_ap: float,
    checkpoint_path: Path | str,
) -> None:
    """Save model checkpoint safely."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ap": val_ap,
        },
        path,
    )
