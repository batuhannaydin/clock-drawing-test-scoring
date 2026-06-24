"""Ablation presets for the puan classifier.

Each preset is a dict of CFG-field overrides. The runner applies them by
setting attributes on the global CFG before building model/data/loss.

Naming convention for files written by train/evaluate:
    outputs/puan/best_model_<name>.pth
    outputs/puan/training_log_<name>.csv
    outputs/puan/test_metrics_<name>.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from config import CFG


TINY_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
SMALL_BACKBONE = "convnext_small.fb_in22k_ft_in1k"


PRESETS: Dict[str, Dict[str, Any]] = {
    # ===== Tiny ablation series — add one mechanism per step =====
    "t1_vanilla": {
        "backbone": TINY_BACKBONE,
        "use_strong_aug": False,
        "use_class_weights": False,
        "use_weighted_sampler": False,
        "label_smoothing": 0.0,
        "focal_gamma": 0.0,
        "ordinal_lambda": 0.0,
        "drop_path_rate": 0.0,
        "freeze_backbone_epochs": 0,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
    },
    "t2_aug": {
        "backbone": TINY_BACKBONE,
        "use_strong_aug": True,
        "use_class_weights": False,
        "use_weighted_sampler": False,
        "label_smoothing": 0.0,
        "focal_gamma": 0.0,
        "ordinal_lambda": 0.0,
        "drop_path_rate": 0.0,
        "freeze_backbone_epochs": 0,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
    },
    "t3_class_balance": {
        "backbone": TINY_BACKBONE,
        "use_strong_aug": True,
        "use_class_weights": True,
        "use_weighted_sampler": True,
        "label_smoothing": 0.0,
        "focal_gamma": 0.0,
        "ordinal_lambda": 0.0,
        "drop_path_rate": 0.0,
        "freeze_backbone_epochs": 0,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
    },
    "t4_label_smooth": {
        "backbone": TINY_BACKBONE,
        "use_strong_aug": True,
        "use_class_weights": True,
        "use_weighted_sampler": True,
        "label_smoothing": 0.1,
        "focal_gamma": 0.0,
        "ordinal_lambda": 0.0,
        "drop_path_rate": 0.0,
        "freeze_backbone_epochs": 0,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
    },
    "t5_focal": {
        "backbone": TINY_BACKBONE,
        "use_strong_aug": True,
        "use_class_weights": True,
        "use_weighted_sampler": True,
        "label_smoothing": 0.1,
        "focal_gamma": 1.5,
        "ordinal_lambda": 0.0,
        "drop_path_rate": 0.0,
        "freeze_backbone_epochs": 0,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
    },
    "t6_full": {
        "backbone": TINY_BACKBONE,
        "use_strong_aug": True,
        "use_class_weights": True,
        "use_weighted_sampler": True,
        "label_smoothing": 0.1,
        "focal_gamma": 1.5,
        "ordinal_lambda": 0.15,
        "drop_path_rate": 0.2,
        "freeze_backbone_epochs": 2,
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
    },
    # ===== Small variants =====
    "s1_full": {
        "backbone": SMALL_BACKBONE,
        "use_strong_aug": True,
        "use_class_weights": True,
        "use_weighted_sampler": True,
        "label_smoothing": 0.1,
        "focal_gamma": 1.5,
        "ordinal_lambda": 0.15,
        "drop_path_rate": 0.4,
        "freeze_backbone_epochs": 4,
        "learning_rate": 5e-5,
        "weight_decay": 1e-3,
    },
    "s2_relaxed": {
        "backbone": SMALL_BACKBONE,
        "use_strong_aug": True,
        "use_class_weights": True,
        "use_weighted_sampler": True,
        "label_smoothing": 0.1,
        "focal_gamma": 1.5,
        "ordinal_lambda": 0.15,
        "drop_path_rate": 0.3,
        "freeze_backbone_epochs": 3,
        "learning_rate": 8e-5,
        "weight_decay": 1e-3,
    },
}

PRESET_ORDER = [
    "t1_vanilla", "t2_aug", "t3_class_balance", "t4_label_smooth",
    "t5_focal", "t6_full", "s1_full", "s2_relaxed",
]


def apply_preset(name: str) -> Dict[str, Any]:
    """Mutate the global CFG to match a preset and set output paths.

    Returns the dict of overrides applied (for logging).
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}. Known: {list(PRESETS)}")
    overrides = PRESETS[name]
    for k, v in overrides.items():
        setattr(CFG, k, v)
    out = CFG.output_dir
    CFG.run_tag = name
    CFG.checkpoint_path = out / f"best_model_{name}.pth"
    CFG.training_log_path = out / f"training_log_{name}.csv"
    CFG.test_metrics_path = out / f"test_metrics_{name}.txt"
    return overrides
