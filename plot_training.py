"""Plot train/valid loss and Macro F1 curves from training_log.csv.

Run after train.py (it also auto-calls this on completion):
    python plot_training.py

Reads CFG.output_dir/training_log.csv and writes training_curves.png
next to it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from config import CFG


def _ema(values: pd.Series, alpha: float = 0.4) -> pd.Series:
    """Exponential moving average. Smaller alpha = smoother."""
    return values.ewm(alpha=alpha, adjust=False).mean()


def plot_training_curves(
    log_csv: Optional[Path] = None,
    out_path: Optional[Path] = None,
    ema_alpha: float = 0.4,
) -> Path:
    log_csv = log_csv or CFG.output_dir / "training_log.csv"
    out_path = out_path or CFG.output_dir / "training_curves.png"
    if not log_csv.exists():
        raise FileNotFoundError(f"Missing training log: {log_csv}")

    df = pd.read_csv(log_csv)
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss: faint raw, bold smoothed
    axes[0].plot(epochs, df["train_loss"], color="C0", alpha=0.25, linewidth=1.0)
    axes[0].plot(epochs, df["valid_loss"], color="C1", alpha=0.25, linewidth=1.0)
    axes[0].plot(epochs, _ema(df["train_loss"], ema_alpha),
                 label="train (EMA)", color="C0", linewidth=2.2)
    axes[0].plot(epochs, _ema(df["valid_loss"], ema_alpha),
                 label="valid (EMA)", color="C1", linewidth=2.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Train / Valid Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # F1
    axes[1].plot(epochs, df["train_f1"], color="C0", alpha=0.25, linewidth=1.0)
    axes[1].plot(epochs, df["valid_f1"], color="C1", alpha=0.25, linewidth=1.0)
    axes[1].plot(epochs, _ema(df["train_f1"], ema_alpha),
                 label="train (EMA)", color="C0", linewidth=2.2)
    axes[1].plot(epochs, _ema(df["valid_f1"], ema_alpha),
                 label="valid (EMA)", color="C1", linewidth=2.2)

    best_idx = df["valid_f1"].idxmax()
    best_epoch = int(df.loc[best_idx, "epoch"])
    best_f1 = float(df.loc[best_idx, "valid_f1"])
    axes[1].scatter([best_epoch], [best_f1], color="red", zorder=5, s=60,
                    label=f"best valid (ep {best_epoch}, F1={best_f1:.4f})")
    axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Train / Valid Macro F1")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[info] training curves -> {out_path}")
    return out_path


if __name__ == "__main__":
    plot_training_curves()
