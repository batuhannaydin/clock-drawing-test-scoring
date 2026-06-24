"""Train the 6-class CDT puan classifier.

Run:
    python train.py                         # uses CFG defaults
    python train.py --preset t1_vanilla     # uses an ablation preset from experiments.py
"""
from __future__ import annotations

import argparse

from config import CFG
from dataset import build_dataloaders, split_distribution
from engine import train_one_run
from experiments import apply_preset
from losses import FocalCrossEntropy
from model import build_model
from utils import describe_device, get_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=str, default=None,
                        help="Ablation preset name from experiments.PRESETS")
    args = parser.parse_args()

    if args.preset is not None:
        overrides = apply_preset(args.preset)
        print(f"[info] preset: {args.preset}")
        for k, v in overrides.items():
            print(f"  {k}: {v}")

    set_seed(CFG.seed)
    CFG.ensure_dirs()
    device = get_device()
    print(f"[info] Device: {describe_device(device)}")
    print(f"[info] Checkpoint → {CFG.checkpoint_path.name}")

    dist = split_distribution()
    print(f"[info] Split distribution by puan class:")
    for split, counts in dist.items():
        total = sum(counts.values())
        print(f"  {split:6s} = {total}  {counts}")

    train_loader, valid_loader, _, class_weights = build_dataloaders()
    print(f"[info] Train batches={len(train_loader)}  Valid batches={len(valid_loader)}")
    if class_weights is not None:
        print(f"[info] Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
        cw_device = class_weights.to(device)
    else:
        print("[info] Class weights: DISABLED (plain unweighted CE)")
        cw_device = None

    model = build_model().to(device)
    criterion = FocalCrossEntropy(class_weights=cw_device).to(device)

    train_one_run(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=CFG.checkpoint_path,
        log_csv=CFG.training_log_path,
    )


if __name__ == "__main__":
    main()
