"""Dataset for the puan classifier.

Reads the master archive's `master_etiketler.csv` but loads images from the
*pre-cropped* `processed/{0..5}/` folders rather than the raw NHATS scans in
`resimler/`. The cropping (clock region only, form text removed) happens
offline via _crop_puan_345.py + process_images.py.

Builds a stratified per-class-capped split. Persists to `CFG.split_csv`
so train / evaluate share the same split.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from augment import build_basic_train_transform, build_eval_transform, build_train_transform
from config import CFG


Sample = Tuple[Path, int, str]   # (image_path, puan, split)


def _cropped_path(stem: str, puan: int) -> Optional[Path]:
    """Resolve the pre-cropped PNG for a master_etiketler.csv row."""
    candidate = CFG.processed_dir / str(puan) / f"{stem}.png"
    return candidate if candidate.is_file() else None


def _read_master_labels() -> pd.DataFrame:
    df = pd.read_csv(CFG.labels_csv)
    df = df[(df["puan"] >= CFG.puan_min) & (df["puan"] <= CFG.puan_max)].copy()
    df["dosya_adi"] = df["dosya_adi"].astype(str)
    df["stem"] = df["dosya_adi"].apply(lambda f: Path(f).stem)

    # Keep only rows whose pre-cropped PNG actually exists. Each row points
    # to processed/<puan>/<stem>.png — any row without a cropped file is dropped.
    df["cropped_path"] = df.apply(
        lambda r: _cropped_path(r["stem"], int(r["puan"])),
        axis=1,
    )
    have = df["cropped_path"].notna()
    missing = (~have).sum()
    if missing:
        print(f"[warn] {missing}/{len(df)} rows have no pre-cropped image — dropped.")
    return df[have].reset_index(drop=True)


def _build_split(seed: int = CFG.seed) -> pd.DataFrame:
    """Stratified per-class-capped split into train/valid/test.

    Idempotent: if `CFG.split_csv` exists, it is loaded and returned as-is.
    """
    if CFG.split_csv.exists():
        return pd.read_csv(CFG.split_csv)

    df = _read_master_labels()
    rng = np.random.RandomState(seed)

    rows: List[Dict] = []
    for puan, group in df.groupby("puan"):
        records = group[["stem", "cropped_path"]].to_dict("records")
        rng.shuffle(records)
        # Cap classes that have more than per_class_cap.
        if len(records) > CFG.per_class_cap:
            records = records[: CFG.per_class_cap]
        n = len(records)
        n_test = max(1, int(round(n * CFG.test_split)))
        n_valid = max(1, int(round(n * CFG.valid_split)))
        n_train = n - n_test - n_valid
        for r in records[:n_train]:
            rows.append({"stem": r["stem"], "image_path": str(r["cropped_path"]),
                         "puan": int(puan), "split": "train"})
        for r in records[n_train:n_train + n_valid]:
            rows.append({"stem": r["stem"], "image_path": str(r["cropped_path"]),
                         "puan": int(puan), "split": "valid"})
        for r in records[n_train + n_valid:]:
            rows.append({"stem": r["stem"], "image_path": str(r["cropped_path"]),
                         "puan": int(puan), "split": "test"})

    out = pd.DataFrame(rows)
    CFG.split_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CFG.split_csv, index=False)
    return out


class PuanDataset(Dataset):
    def __init__(
        self,
        samples: List[Sample],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.samples = samples
        self.transform = transform or build_eval_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, puan, _ = self.samples[idx]
        img = Image.open(path)
        # Convert bitmap / grayscale → RGB so the ImageNet backbone is happy.
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(puan, dtype=torch.long)


def _split_samples(df: pd.DataFrame) -> Dict[str, List[Sample]]:
    out: Dict[str, List[Sample]] = {"train": [], "valid": [], "test": []}
    for _, r in df.iterrows():
        out[r["split"]].append((Path(r["image_path"]), int(r["puan"]), r["split"]))
    return out


def _class_weights(samples: List[Sample]) -> torch.Tensor:
    """Inverse-frequency class weights, normalized to mean=1.

    Class 0 ("Şiddetli", rarest puan) gets an additional fixed multiplier
    so the loss prioritizes recall on severe cases.
    """
    counts = Counter(p for _, p, _ in samples)
    total = sum(counts.values())
    K = CFG.num_classes
    w = torch.zeros(K, dtype=torch.float32)
    for c in range(K):
        n = counts.get(c, 0)
        w[c] = total / (K * n) if n > 0 else 0.0
    w[0] = w[0] * CFG.class0_weight_multiplier
    w[4] = w[4] * CFG.class4_weight_multiplier
    # Normalize so the mean is 1.0 (keeps loss scale comparable to unweighted).
    nz = w[w > 0]
    if len(nz) > 0:
        w = w * (len(nz) / nz.sum())
    return w


def _sampler_for(samples: List[Sample]) -> WeightedRandomSampler:
    """WeightedRandomSampler that oversamples minority puan classes."""
    cw = _class_weights(samples)
    sample_w = torch.tensor([cw[p].item() for _, p, _ in samples], dtype=torch.float64)
    return WeightedRandomSampler(sample_w, num_samples=len(samples), replacement=True)


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, Optional[torch.Tensor]]:
    df = _build_split()
    by_split = _split_samples(df)

    train_tf = build_train_transform() if CFG.use_strong_aug else build_basic_train_transform()
    train_ds = PuanDataset(by_split["train"], transform=train_tf)
    valid_ds = PuanDataset(by_split["valid"], transform=build_eval_transform())
    test_ds = PuanDataset(by_split["test"], transform=build_eval_transform())

    cw = _class_weights(by_split["train"]) if CFG.use_class_weights else None

    train_loader_kwargs = dict(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if CFG.use_weighted_sampler:
        train_loader_kwargs["sampler"] = _sampler_for(by_split["train"])
    else:
        train_loader_kwargs["shuffle"] = True
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader, test_loader, cw


def split_distribution() -> Dict[str, Dict[int, int]]:
    df = _build_split()
    out: Dict[str, Dict[int, int]] = {}
    for s in ("train", "valid", "test"):
        sub = df[df["split"] == s]
        out[s] = dict(sorted(Counter(sub["puan"]).items()))
    return out
