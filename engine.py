"""Training & evaluation engine for the puan classifier."""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CFG


def _metrics(preds: np.ndarray, truths: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(truths, preds)),
        "f1": float(f1_score(truths, preds, average="macro", zero_division=0)),
        "mae": float(np.abs(preds - truths).mean()),
    }


def epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    desc: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    accum_loss, n = 0.0, 0
    all_preds: List[np.ndarray] = []
    all_truth: List[np.ndarray] = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        accum_loss += float(loss.item()) * bs
        n += bs
        all_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        all_truth.append(y.detach().cpu().numpy())
        pbar.set_postfix(loss=f"{accum_loss / max(1, n):.4f}")

    preds = np.concatenate(all_preds)
    truth = np.concatenate(all_truth)
    m = _metrics(preds, truth)
    m["loss"] = accum_loss / max(1, n)
    return m, preds, truth


def build_optimizer(model: nn.Module) -> AdamW:
    return AdamW(
        [
            {"params": model.backbone.parameters(), "lr": CFG.learning_rate},
            {"params": model.head.parameters(), "lr": CFG.head_learning_rate},
        ],
        weight_decay=CFG.weight_decay,
    )


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    checkpoint_path: Path,
    epochs: int = CFG.epochs,
    freeze_backbone_epochs: int = CFG.freeze_backbone_epochs,
    early_stop_patience: int = CFG.early_stop_patience,
    log_csv: Optional[Path] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    track = CFG.best_metric
    higher_is_better = track == "valid_f1"
    sched_mode = "max" if higher_is_better else "min"

    optimizer = build_optimizer(model)
    scheduler = ReduceLROnPlateau(
        optimizer, mode=sched_mode, factor=CFG.lr_scheduler_factor,
        patience=CFG.lr_scheduler_patience, min_lr=CFG.min_lr,
    )

    if freeze_backbone_epochs > 0:
        model.freeze_backbone()
        print(f"[info] backbone frozen for {freeze_backbone_epochs} epoch(s).")

    history: List[Dict[str, float]] = []
    best_score = float("-inf") if higher_is_better else float("inf")
    best_snapshot: Dict[str, float] = {}
    no_improve = 0

    start = time.time()
    for epoch in range(1, epochs + 1):
        if epoch == freeze_backbone_epochs + 1 and freeze_backbone_epochs > 0:
            model.unfreeze_backbone()
            print("[info] backbone unfrozen, fine-tuning whole network.")

        t0 = time.time()
        tr, _, _ = epoch_pass(model, train_loader, criterion, optimizer, device,
                              desc=f"Ep {epoch:02d} [train]")
        va, _, _ = epoch_pass(model, valid_loader, criterion, None, device,
                              desc=f"Ep {epoch:02d} [valid]")

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch, "lr": cur_lr, "time_sec": round(elapsed, 1),
            "train_loss": tr["loss"], "train_acc": tr["acc"],
            "train_f1": tr["f1"], "train_mae": tr["mae"],
            "valid_loss": va["loss"], "valid_acc": va["acc"],
            "valid_f1": va["f1"], "valid_mae": va["mae"],
        }
        history.append(record)
        print(
            f"[ep {epoch:02d}/{epochs}] lr={cur_lr:.2e}  "
            f"train loss={tr['loss']:.4f} f1={tr['f1']:.4f} acc={tr['acc']:.4f}  "
            f"| valid loss={va['loss']:.4f} f1={va['f1']:.4f} acc={va['acc']:.4f}  "
            f"mae={va['mae']:.3f} ({elapsed:.1f}s)"
        )

        score = va["f1"] if higher_is_better else va["loss"]
        improved = score > best_score if higher_is_better else score < best_score
        scheduler.step(score)

        if improved:
            best_score = score
            best_snapshot = {"epoch": epoch, **va}
            no_improve = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "best_metric": track,
                    "config_snapshot": {
                        "backbone": CFG.backbone,
                        "image_size": CFG.image_size,
                        "num_classes": CFG.num_classes,
                    },
                },
                checkpoint_path,
            )
            print(f"  ↳ saved best ({track}={best_score:.4f}) → {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"[info] early stop @ epoch {epoch}.")
                break

    total_min = (time.time() - start) / 60
    print(f"[done] {total_min:.1f} min, best {track}={best_score:.4f}")

    if log_csv is not None and history:
        log_csv.parent.mkdir(parents=True, exist_ok=True)
        with log_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

    return history, best_snapshot
