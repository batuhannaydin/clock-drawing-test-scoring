"""Train / eval transforms for the puan classifier.

Same family as the v2 multi-task pipeline — strong-but-clinically-safe
on train, deterministic resize+normalize on eval.
"""
from __future__ import annotations

from torchvision import transforms

from config import CFG


def build_train_transform(image_size: int = CFG.image_size) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(
            degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=5,
        ),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.mean, std=CFG.std),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])


def build_eval_transform(image_size: int = CFG.image_size) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.mean, std=CFG.std),
    ])


def build_basic_train_transform(image_size: int = CFG.image_size) -> transforms.Compose:
    """No-augmentation training transform (vanilla baseline)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG.mean, std=CFG.std),
    ])
