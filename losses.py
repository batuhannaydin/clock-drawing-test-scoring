"""Loss for the 6-class puan classifier.

Class-weighted focal cross-entropy + ordinal expectation regularizer.

  L = FocalCE(logits, y) + λ · | E[Y | x] - y |

The expectation E[Y | x] = Σ_k k · softmax(logits)[k] is a soft, ordinal
read-off of the prediction. Penalizing |E - y| punishes *large* gaps
(e.g. 2 → 5) much more than adjacent confusions (4 ↔ 5), pushing the
model toward an ordinal-aware decision boundary without forcing a
rank-consistent head.

With gamma=0 and ordinal_lambda=0 this reduces to plain CE.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


class FocalCrossEntropy(nn.Module):
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = CFG.label_smoothing,
        gamma: float = CFG.focal_gamma,
        ordinal_lambda: float = CFG.ordinal_lambda,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.ordinal_lambda = ordinal_lambda
        if class_weights is not None:
            self.register_buffer("weight", class_weights.clone())
        else:
            self.weight = None

    def _focal_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.gamma <= 0.0:
            return F.cross_entropy(
                logits, target, weight=self.weight,
                label_smoothing=self.label_smoothing,
            )
        ce = F.cross_entropy(
            logits, target, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-8, 1.0)
            modulator = (1.0 - p_t) ** self.gamma
        if self.weight is not None:
            # `ce` already carries the class weight (F.cross_entropy with `weight`
            # returns weight[target] * nll per sample). To stay consistent with
            # PyTorch's weighted-mean convention we normalize by the sum of the
            # per-sample weights, not by the batch size.
            w_t = self.weight[target]
            return (modulator * ce).sum() / w_t.sum().clamp(min=1e-8)
        return (modulator * ce).mean()

    def _ordinal_penalty(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 between expected class and true class — penalizes large jumps."""
        K = logits.size(1)
        classes = torch.arange(K, dtype=logits.dtype, device=logits.device)
        probs = F.softmax(logits, dim=1)
        expected = (probs * classes).sum(dim=1)
        return (expected - target.float()).abs().mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self._focal_ce(logits, target)
        if self.ordinal_lambda > 0.0:
            loss = loss + self.ordinal_lambda * self._ordinal_penalty(logits, target)
        return loss
