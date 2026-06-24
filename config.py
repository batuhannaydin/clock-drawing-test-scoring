"""Configuration for the direct 6-class CDT puan classifier.

Single classification head predicting the clinical `puan` score (0..5)
from the master archive. Replaces the multi-task setup in _archive_v2/.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


PROJECT_ROOT: Path = Path(__file__).resolve().parent
# Dataset location. Override with the CDT_DATA_ROOT environment variable so the
# repo is portable — the default below is only a local convenience fallback.
MASTER_ROOT: Path = Path(
    os.environ.get("CDT_DATA_ROOT", PROJECT_ROOT / "data" / "master_veri")
)


@dataclass
class Config:
    # ---- Paths ----
    project_root: Path = PROJECT_ROOT
    master_root: Path = MASTER_ROOT
    images_dir: Path = MASTER_ROOT / "resimler"
    labels_csv: Path = MASTER_ROOT / "master_etiketler.csv"

    output_dir: Path = PROJECT_ROOT / "outputs" / "puan"
    # Defaults below reproduce the SELECTED model (`t2_aug`): a ConvNeXt-Tiny
    # fine-tuned with strong augmentation only. So a bare `python train.py`
    # (no preset) reproduces the model reported in the README. Ablation presets
    # in experiments.py override these fields for their respective runs.
    run_tag: str = "t2_aug"   # used to suffix checkpoint, log, and metric files
    checkpoint_path: Path = PROJECT_ROOT / "outputs" / "puan" / "best_model_t2_aug.pth"
    training_log_path: Path = PROJECT_ROOT / "outputs" / "puan" / "training_log_t2_aug.csv"
    test_metrics_path: Path = PROJECT_ROOT / "outputs" / "puan" / "test_metrics_t2_aug.txt"
    split_csv: Path = PROJECT_ROOT / "outputs" / "puan" / "split.csv"
    # Pre-cropped clock PNGs live here, organized by puan class.
    processed_dir: Path = MASTER_ROOT / "processed"

    # ---- Task ----
    num_classes: int = 6   # puan 0..5
    puan_min: int = 0
    puan_max: int = 5

    # ---- Training ----
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    head_learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    # The selected `t2_aug` recipe uses augmentation only — no label smoothing,
    # focal loss, ordinal regularizer, or drop-path. The ablation presets turn
    # these back on to measure their (here, non-positive) marginal effect.
    label_smoothing: float = 0.0
    dropout: float = 0.4
    drop_path_rate: float = 0.0
    focal_gamma: float = 0.0
    # Ordinal expectation regularizer: λ · |E[Y|x] - y|. Penalizes large
    # class jumps (3→5) more than adjacent confusions (4↔5). Off by default.
    ordinal_lambda: float = 0.0
    # Extra multiplier applied to class 0's weight on top of inverse-freq,
    # to push recall on the rarest "Şiddetli" cases.
    class0_weight_multiplier: float = 1.5
    # Extra multiplier for class 4 — sits on a hard boundary with class 5,
    # boost its weight to recover recall without losing precision too much.
    class4_weight_multiplier: float = 1.3

    freeze_backbone_epochs: int = 0
    best_metric: str = "valid_f1"   # "valid_loss" or "valid_f1"
    early_stop_patience: int = 6
    lr_scheduler_patience: int = 2
    lr_scheduler_factor: float = 0.3
    min_lr: float = 1e-6

    # ---- Balanced subset ----
    # Per-class cap. Classes with fewer samples are kept in full
    # (puan=0 has 538, puan=1 has 1956 — both pass through).
    per_class_cap: int = 3000
    test_split: float = 0.10
    valid_split: float = 0.10

    # ---- Backbone ----
    backbone: str = "convnext_tiny.fb_in22k_ft_in1k"
    pretrained: bool = True

    # ---- Mechanism toggles (used by ablation runs in experiments.py) ----
    # Selected model uses augmentation only; class balancing is off by default
    # because, on this dataset, sampler + loss weights over-corrected (see README).
    use_strong_aug: bool = True
    use_class_weights: bool = False
    use_weighted_sampler: bool = False

    # ---- Reproducibility ----
    seed: int = 42

    # ---- ImageNet normalization ----
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ---- Severity bands derived from puan (informational only) ----
    # Used by evaluate.py to also report a coarse 3-class summary.
    severity_thresholds: Tuple[int, int] = (2, 4)   # <2 şiddetli, 2-3 orta, >=4 sağlıklı
    severity_labels: Tuple[str, str, str] = ("Şiddetli", "Orta", "Sağlıklı")

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


CFG = Config()
