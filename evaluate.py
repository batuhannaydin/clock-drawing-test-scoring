"""Evaluate the best puan classifier on the test split.

Run:
    python evaluate.py            # single forward pass per sample
    python evaluate.py --tta      # average softmax over 5 small jitter views

TTA here is intentionally CONSERVATIVE — only mild rotations (±7°, ±12°)
plus the identity. We do NOT use 90° flips here, because the clinical
puan score is rotation-sensitive (12 position matters).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)
from torchvision.transforms.functional import rotate as tv_rotate

from config import CFG
from dataset import build_dataloaders
from experiments import apply_preset
from model import build_model, load_checkpoint
from utils import describe_device, get_device, set_seed


# Corner fill for TTA rotation, in *normalized* space. Training's
# RandomRotation fills rotated corners with black (pixel 0) before normalizing,
# so the matching value here is (0 - mean) / std per channel. Using the default
# fill=0 would instead paint mean-grey wedges the model never saw in training.
_ROTATE_FILL: List[float] = [(-m) / s for m, s in zip(CFG.mean, CFG.std)]


def _tta_views(x: torch.Tensor) -> List[torch.Tensor]:
    return [
        x,
        tv_rotate(x, angle=7.0, fill=_ROTATE_FILL),
        tv_rotate(x, angle=-7.0, fill=_ROTATE_FILL),
        tv_rotate(x, angle=12.0, fill=_ROTATE_FILL),
        tv_rotate(x, angle=-12.0, fill=_ROTATE_FILL),
    ]


def _predict(model, x: torch.Tensor, use_tta: bool) -> torch.Tensor:
    views = _tta_views(x) if use_tta else [x]
    probs = None
    for v in views:
        p = F.softmax(model(v), dim=1)
        probs = p if probs is None else probs + p
    return probs / len(views)


def _severity_band(puan: np.ndarray) -> np.ndarray:
    low, high = CFG.severity_thresholds
    out = np.full(puan.shape, 1, dtype=np.int64)  # default orta
    out[puan < low] = 0
    out[puan >= high] = 2
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tta", action="store_true",
                        help="Average predictions across 5 small-jitter views.")
    parser.add_argument("--preset", type=str, default=None,
                        help="Load preset's checkpoint and write to its metric file.")
    args = parser.parse_args()

    if args.preset is not None:
        apply_preset(args.preset)
        print(f"[info] preset: {args.preset}")
        print(f"[info] checkpoint: {CFG.checkpoint_path.name}")
        print(f"[info] metrics out: {CFG.test_metrics_path.name}")

    set_seed(CFG.seed)
    CFG.ensure_dirs()
    device = get_device()
    print(f"[info] Device: {describe_device(device)}")
    print(f"[info] TTA: {'enabled (5 small-jitter views)' if args.tta else 'off'}")

    if not CFG.checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CFG.checkpoint_path}")

    _, _, test_loader, _ = build_dataloaders()
    print(f"[info] Test samples: {len(test_loader.dataset)}")

    model = build_model()
    model, meta = load_checkpoint(model, CFG.checkpoint_path, device)
    if meta:
        print(f"[info] Loaded checkpoint: {meta}")
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            probs = _predict(model, x, args.tta)
            preds.append(probs.argmax(dim=1).cpu().numpy())
            truths.append(y.numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    f1_macro = f1_score(truths, preds, average="macro", zero_division=0)
    acc = accuracy_score(truths, preds)
    mae = float(np.abs(preds - truths).mean())
    cm = confusion_matrix(truths, preds, labels=list(range(CFG.num_classes)))

    print("\n=== Per-class metrics (puan 0..5) ===")
    print(f"Macro F1: {f1_macro:.4f}  Acc: {acc:.4f}  MAE: {mae:.3f}")
    print()
    print(classification_report(
        truths, preds,
        labels=list(range(CFG.num_classes)),
        digits=4, zero_division=0,
    ))
    print("Confusion matrix (rows=truth, cols=pred):")
    print(cm)

    # 3-band severity summary
    sev_truth = _severity_band(truths)
    sev_pred = _severity_band(preds)
    sev_acc = accuracy_score(sev_truth, sev_pred)
    sev_f1 = f1_score(sev_truth, sev_pred, average="macro", zero_division=0)
    sev_cm = confusion_matrix(sev_truth, sev_pred, labels=[0, 1, 2])
    print("\n=== 3-band severity (Şiddetli / Orta / Sağlıklı) ===")
    print(f"Acc: {sev_acc:.4f}  Macro F1: {sev_f1:.4f}")
    print(classification_report(
        sev_truth, sev_pred,
        labels=[0, 1, 2],
        target_names=list(CFG.severity_labels),
        digits=4, zero_division=0,
    ))
    print("Severity confusion:")
    print(sev_cm)

    out = CFG.test_metrics_path
    with out.open("w", encoding="utf-8") as f:
        f.write("Puan Classifier — Test Set Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Macro F1: {f1_macro:.4f}  Acc: {acc:.4f}  MAE: {mae:.3f}\n\n")
        f.write(classification_report(
            truths, preds,
            labels=list(range(CFG.num_classes)),
            digits=4, zero_division=0,
        ))
        f.write("\nConfusion (rows=truth, cols=pred):\n")
        f.write(np.array2string(cm) + "\n\n")
        f.write(f"=== 3-band severity ===\n")
        f.write(f"Acc: {sev_acc:.4f}  Macro F1: {sev_f1:.4f}\n")
        f.write(classification_report(
            sev_truth, sev_pred,
            labels=[0, 1, 2],
            target_names=list(CFG.severity_labels),
            digits=4, zero_division=0,
        ))
        f.write(np.array2string(sev_cm) + "\n")
    print(f"\n[info] Metrics report → {out}")

    # ---- JSON sidecar for downstream comparison/PDF script ----
    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        truths, preds, labels=list(range(CFG.num_classes)), zero_division=0,
    )
    sev_p_arr, sev_r_arr, sev_f1_arr, sev_sup_arr = precision_recall_fscore_support(
        sev_truth, sev_pred, labels=[0, 1, 2], zero_division=0,
    )
    summary = {
        "preset": args.preset,
        "checkpoint": str(CFG.checkpoint_path.name),
        "n_test": int(len(truths)),
        "macro_f1": float(f1_macro),
        "accuracy": float(acc),
        "mae": float(mae),
        "per_class": {
            "labels": list(range(CFG.num_classes)),
            "precision": [float(x) for x in p_arr],
            "recall": [float(x) for x in r_arr],
            "f1": [float(x) for x in f1_arr],
            "support": [int(x) for x in sup_arr],
        },
        "confusion": cm.tolist(),
        "severity": {
            "accuracy": float(sev_acc),
            "macro_f1": float(sev_f1),
            "labels": list(CFG.severity_labels),
            "precision": [float(x) for x in sev_p_arr],
            "recall": [float(x) for x in sev_r_arr],
            "f1": [float(x) for x in sev_f1_arr],
            "support": [int(x) for x in sev_sup_arr],
            "confusion": sev_cm.tolist(),
        },
        "tta": bool(args.tta),
    }
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[info] JSON sidecar → {json_out.name}")


if __name__ == "__main__":
    main()
