"""Surface likely-mislabeled samples via "confident learning".

After train.py finishes:
    python clean_labels.py [--top-n 300]

For every sample in the split (train + valid + test), we run the trained
model in eval mode and compute the probability it assigns to the *labeled*
class. Samples where the model is highly confident in a *different*
class are likely mislabels.

Outputs
    outputs/puan/label_suspects.csv      sorted, all rows scored
    outputs/puan/_suspect_review/        top-N image copies for visual review
    outputs/puan/_suspect_review/README.txt
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from augment import build_eval_transform
from config import CFG
from dataset import PuanDataset, _build_split
from model import build_model, load_checkpoint
from utils import describe_device, get_device, set_seed


def _all_samples(df: pd.DataFrame) -> list:
    out = []
    for _, r in df.iterrows():
        out.append((Path(r["image_path"]), int(r["puan"]), r["split"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=300,
                        help="Copy top-N most suspicious images for review.")
    parser.add_argument("--exclude-pairs", nargs="*", default=[],
                        metavar="LABEL,PRED",
                        help="Exclude these (labeled,predicted) confusion "
                             "pairs from the top-N selection. CSV remains full. "
                             "Example: --exclude-pairs 4,5 5,4")
    args = parser.parse_args()

    excluded_pairs: set[tuple[int, int]] = set()
    for raw in args.exclude_pairs:
        try:
            a, b = raw.split(",")
            excluded_pairs.add((int(a), int(b)))
        except (ValueError, AttributeError):
            raise SystemExit(f"bad --exclude-pairs entry: {raw!r} (want 'L,P')")
    if excluded_pairs:
        print(f"[info] excluding pairs from top-N: {sorted(excluded_pairs)}")

    set_seed(CFG.seed)
    CFG.ensure_dirs()
    device = get_device()
    print(f"[info] Device: {describe_device(device)}")

    df = _build_split()
    samples = _all_samples(df)
    print(f"[info] Total samples to score: {len(samples)}")

    ds = PuanDataset(samples, transform=build_eval_transform())
    loader = DataLoader(
        ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True,
    )

    model = build_model()
    model, meta = load_checkpoint(model, CFG.checkpoint_path, device)
    print(f"[info] Loaded: {meta}")
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            all_probs.append(F.softmax(model(x), dim=1).cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    pred_class = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    truth = np.array([s[1] for s in samples])
    prob_given_to_truth = probs[np.arange(len(truth)), truth]

    # Suspicion: low prob assigned to the labeled class.
    # Tie-break: high confidence in a *different* class (i.e., big margin).
    margin = pred_conf - prob_given_to_truth

    out_df = pd.DataFrame({
        "image_path": [str(s[0]) for s in samples],
        "stem": [Path(s[0]).stem for s in samples],
        "split": [s[2] for s in samples],
        "labeled_puan": truth,
        "predicted_puan": pred_class,
        "prob_labeled": prob_given_to_truth,
        "prob_predicted": pred_conf,
        "margin": margin,
        "distance": np.abs(pred_class - truth),
    })

    # Two-stage sort: bigger margin AND bigger label-vs-pred distance go first.
    out_df = out_df.sort_values(
        ["distance", "margin"], ascending=[False, False],
    ).reset_index(drop=True)

    suspects_csv = CFG.output_dir / "label_suspects.csv"
    out_df.to_csv(suspects_csv, index=False)
    print(f"\n[info] All scores: {suspects_csv}")

    # Headline numbers
    disagree = (pred_class != truth).sum()
    print(f"[info] Total disagreement: {disagree}/{len(samples)} ({100*disagree/len(samples):.1f}%)")
    severe = (np.abs(pred_class - truth) >= 2).sum()
    print(f"[info] |pred-truth| >= 2: {severe}  (very likely mislabel)")
    very_severe = (np.abs(pred_class - truth) >= 3).sum()
    print(f"[info] |pred-truth| >= 3: {very_severe}  (almost certainly mislabel)")

    # Copy top-N suspects to a review folder
    review_dir = CFG.output_dir / "_suspect_review"
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True)

    if excluded_pairs:
        before = len(out_df)
        pair_col = list(zip(out_df["labeled_puan"].astype(int),
                            out_df["predicted_puan"].astype(int)))
        keep_mask = [p not in excluded_pairs for p in pair_col]
        filtered = out_df.loc[keep_mask].reset_index(drop=True)
        dropped = before - len(filtered)
        print(f"[info] excluded {dropped} rows matching {sorted(excluded_pairs)} "
              f"from review pool (CSV still has all {before} rows)")
    else:
        filtered = out_df

    top = filtered.head(args.top_n)
    manifest = [
        "# Top {} likely-mislabeled images (confident learning)\n".format(len(top)),
        "# For each: model predicts a DIFFERENT puan with high confidence\n",
        "# Action: open image, decide if labeled_puan or predicted_puan is correct\n",
        "# Format: <idx>  labeled=X  pred=Y (p=Z)  margin=M  | filename\n",
    ]
    for i, r in top.iterrows():
        src = Path(r["image_path"])
        if not src.exists():
            continue
        dst = review_dir / f"{i+1:03d}__truth{r['labeled_puan']}_pred{r['predicted_puan']}__{r['stem']}.png"
        try:
            img = Image.open(src)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Downsize aggressively so the review folder stays browsable.
            img.thumbnail((512, 512))
            img.save(dst)
        except Exception as e:
            print(f"  failed on {src.name}: {e}")
            continue
        manifest.append(
            f"{i+1:03d}  labeled={r['labeled_puan']}  pred={r['predicted_puan']} "
            f"(p={r['prob_predicted']:.2f})  margin={r['margin']:.2f}  | "
            f"{r['stem']}\n"
        )

    (review_dir / "README.txt").write_text("".join(manifest), encoding="utf-8")
    print(f"[info] Top-{args.top_n} review folder: {review_dir}")
    print("[info] Open each image, decide which puan is correct. "
          "Then we'll patch master_etiketler.csv and retrain.")


if __name__ == "__main__":
    main()
