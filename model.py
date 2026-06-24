"""ConvNeXt-tiny backbone + single 6-class puan head."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn

from config import CFG


class PuanClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = CFG.backbone,
        num_classes: int = CFG.num_classes,
        pretrained: bool = CFG.pretrained,
        dropout: float = CFG.dropout,
        drop_path_rate: float = CFG.drop_path_rate,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )
        feat_dim: int = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model() -> PuanClassifier:
    return PuanClassifier()


def load_checkpoint(
    model: nn.Module, path: Path, device: torch.device,
) -> Tuple[nn.Module, Optional[dict]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        meta = {k: v for k, v in ckpt.items() if k != "model_state"}
    else:
        model.load_state_dict(ckpt)
        meta = None
    model.to(device)
    return model, meta
