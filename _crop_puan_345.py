"""Batch-crop the clock region from puan 3-5 raw NHATS scans.

Reads master_etiketler.csv, samples N images per class (per_class_cap with
some headroom), runs the same OpenCV-based clock cropper used for puan 0-2,
and writes 224x224 PNGs to processed/{3,4,5}/.

No rotation handling — that's deliberately left for the training pipeline.
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("C:/Users/batuh/Desktop/master_veri/master_veri")
RESIM = ROOT / "resimler"
PROC = ROOT / "processed"
TARGET_SIZE = 224
PER_CLASS_CAP = 3500   # cap from CFG.per_class_cap (3000) + headroom
SEED = 42


def crop_clock(img_path: Path, target_size: int = TARGET_SIZE) -> np.ndarray | None:
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    h_img, w_img = arr.shape
    _, bw = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(bw, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    img_area = h_img * w_img
    cands = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = max(w, h) / (min(w, h) + 1)
        if area < img_area * 0.01 or area > img_area * 0.7 or aspect > 3.0:
            continue
        cands.append((cv2.contourArea(c) * (1.0 / (aspect ** 0.5)), x, y, w, h))
    if not cands:
        return None
    cands.sort(key=lambda v: v[0], reverse=True)
    _, x, y, w, h = cands[0]
    pad = int(max(w, h) * 0.1)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    cropped = arr[y1:y2, x1:x2]
    ch, cw = cropped.shape
    if ch != cw:
        size = max(ch, cw)
        sq = np.full((size, size), 255, dtype=np.uint8)
        sq[(size - ch) // 2:(size - ch) // 2 + ch, (size - cw) // 2:(size - cw) // 2 + cw] = cropped
        cropped = sq
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)


def main():
    df = pd.read_csv(ROOT / "master_etiketler.csv")
    rng = random.Random(SEED)
    total_start = time.time()
    grand_total = 0
    grand_fail = 0

    for puan in (3, 4, 5):
        out_dir = PROC / str(puan)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Skip files we've already cropped (idempotent reruns)
        existing = {f.stem for f in out_dir.iterdir() if f.suffix == ".png"}

        files = df[df["puan"] == puan]["dosya_adi"].tolist()
        rng.shuffle(files)
        files = files[:PER_CLASS_CAP]

        print(f"\n[puan={puan}] target={len(files)} files  (already cropped: {len(existing)})")
        t0 = time.time()
        saved, failed, skipped = 0, 0, 0
        for i, fname in enumerate(files, 1):
            stem = Path(fname).stem
            if stem in existing:
                skipped += 1
                continue
            src = RESIM / fname
            if not src.exists():
                failed += 1
                continue
            try:
                arr = crop_clock(src)
                if arr is None:
                    failed += 1
                    continue
                out_path = out_dir / f"{stem}.png"
                Image.fromarray(arr).save(out_path)
                saved += 1
            except Exception as e:
                failed += 1
            if i % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / max(1, saved) * (len(files) - i)
                print(f"  {i:4d}/{len(files)}  saved={saved} fail={failed} skip={skipped}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
        total_t = time.time() - t0
        print(f"[done puan={puan}] saved={saved}  failed={failed}  skipped={skipped}  "
              f"time={total_t:.0f}s")
        grand_total += saved
        grand_fail += failed

    print(f"\n[ALL DONE] saved={grand_total}  failed={grand_fail}  "
          f"total_time={(time.time() - total_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
