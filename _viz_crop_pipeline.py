"""Generate a 10-step visualization of the crop_clock pipeline for the report.

Produces a 2x5 grid PNG showing each intermediate stage on a real CDT scan.
Output: outputs/puan/crop_pipeline_steps.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

RESIM = Path("C:/Users/batuh/Desktop/master_veri/master_veri/resimler")
OUT = Path("C:/Users/batuh/Desktop/güncel/outputs/puan/crop_pipeline_steps.png")
TARGET_SIZE = 224

# Pick a representative example — a puan-3 scan we know was cropped successfully.
EXAMPLE_STEM = "10000022_R1"


def run_pipeline(img_path: Path):
    stages = {}

    # 1. Load grayscale
    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    h_img, w_img = arr.shape
    stages["1_original"] = arr.copy()

    # 2. Inverse threshold @ 127
    _, bw = cv2.threshold(arr, 127, 255, cv2.THRESH_BINARY_INV)
    stages["2_threshold"] = bw.copy()

    # 3. Morphological dilation (25x25 elliptical, 3 iter)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    dilated = cv2.dilate(bw, kernel, iterations=3)
    stages["3_dilated"] = dilated.copy()

    # 4. External contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stages["4_contours"] = contours

    # 5. Filter candidates by area & aspect ratio
    img_area = h_img * w_img
    cands = []
    rejected = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = max(w, h) / (min(w, h) + 1)
        if area < img_area * 0.01 or area > img_area * 0.7 or aspect > 3.0:
            rejected.append((x, y, w, h))
            continue
        score = cv2.contourArea(c) * (1.0 / (aspect ** 0.5))
        cands.append((score, x, y, w, h))
    stages["5_candidates"] = (cands, rejected)

    # 6. Highest-scoring candidate
    cands.sort(key=lambda v: v[0], reverse=True)
    best = cands[0]
    _, bx, by, bw_, bh_ = best
    stages["6_best"] = (bx, by, bw_, bh_)

    # 7. Raw bounding box on original
    stages["7_bbox"] = (bx, by, bw_, bh_)

    # 8. Add 10% padding
    pad = int(max(bw_, bh_) * 0.1)
    x1, y1 = max(0, bx - pad), max(0, by - pad)
    x2, y2 = min(w_img, bx + bw_ + pad), min(h_img, by + bh_ + pad)
    cropped = arr[y1:y2, x1:x2]
    stages["8_padded"] = (x1, y1, x2 - x1, y2 - y1, cropped.copy())

    # 9. Square via white padding (centered)
    ch, cw = cropped.shape
    if ch != cw:
        size = max(ch, cw)
        sq = np.full((size, size), 255, dtype=np.uint8)
        sq[(size - ch) // 2:(size - ch) // 2 + ch, (size - cw) // 2:(size - cw) // 2 + cw] = cropped
        squared = sq
    else:
        squared = cropped.copy()
    stages["9_squared"] = squared.copy()

    # 10. Resize to 224x224 with INTER_AREA
    final = cv2.resize(squared, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    stages["10_final"] = final.copy()

    return stages


def make_figure(stages, src_name: str, out_path: Path):
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.suptitle(
        f"Saat Kırpma Pipeline'ı — 10 Adım  (örnek: {src_name})",
        fontsize=16, fontweight="bold", y=0.995,
    )

    panels = [
        ("1. Gri tonlama yükleme", "PIL convert('L')"),
        ("2. Ters ikilileştirme", "THRESH_BINARY_INV, eşik=127"),
        ("3. Morfolojik genişletme", "eliptik 25×25, 3 iter."),
        ("4. Dış konturlar", "RETR_EXTERNAL"),
        ("5. Aday filtresi", "alan ∈ [1%, 70%], aspect ≤ 3"),
        ("6. En yüksek skorlu aday", "skor = alan × (1/√aspect)"),
        ("7. Kapsayıcı dikdörtgen", "bounding box"),
        ("8. %10 dolgu eklenmiş", "pad = 0.1·max(w,h)"),
        ("9. Kare hâle getirme", "beyaz (255) kenar dolgusu"),
        ("10. Son ölçek", f"INTER_AREA → {TARGET_SIZE}×{TARGET_SIZE}"),
    ]

    for idx, (title, subtitle) in enumerate(panels):
        ax = axes[idx // 5, idx % 5]
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        if idx == 0:
            ax.imshow(stages["1_original"], cmap="gray", vmin=0, vmax=255)
        elif idx == 1:
            ax.imshow(stages["2_threshold"], cmap="gray", vmin=0, vmax=255)
        elif idx == 2:
            ax.imshow(stages["3_dilated"], cmap="gray", vmin=0, vmax=255)
        elif idx == 3:
            # original + all external contours overlaid
            ax.imshow(stages["1_original"], cmap="gray", vmin=0, vmax=255)
            for c in stages["4_contours"]:
                pts = c.squeeze()
                if pts.ndim == 2:
                    ax.plot(pts[:, 0], pts[:, 1], color="tab:red", lw=0.8)
            ax.text(0.02, 0.98, f"n={len(stages['4_contours'])}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2))
        elif idx == 4:
            # original + rejected (red dashed) + accepted candidates (green solid)
            cands, rejected = stages["5_candidates"]
            ax.imshow(stages["1_original"], cmap="gray", vmin=0, vmax=255)
            for x, y, w, h in rejected:
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.2,
                                               ec="tab:red", fc="none", ls="--"))
            for _, x, y, w, h in cands:
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.6,
                                               ec="tab:green", fc="none"))
            ax.text(0.02, 0.98, f"kabul={len(cands)} / red={len(rejected)}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=9, color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=2))
        elif idx == 5:
            # best candidate highlighted
            cands, rejected = stages["5_candidates"]
            bx, by, bw_, bh_ = stages["6_best"]
            ax.imshow(stages["1_original"], cmap="gray", vmin=0, vmax=255)
            for _, x, y, w, h in cands[1:]:
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.0,
                                               ec="tab:green", fc="none", alpha=0.4))
            ax.add_patch(patches.Rectangle((bx, by), bw_, bh_, lw=2.4,
                                           ec="gold", fc="none"))
        elif idx == 6:
            bx, by, bw_, bh_ = stages["7_bbox"]
            ax.imshow(stages["1_original"], cmap="gray", vmin=0, vmax=255)
            ax.add_patch(patches.Rectangle((bx, by), bw_, bh_, lw=2.0,
                                           ec="tab:orange", fc="none"))
        elif idx == 7:
            x1, y1, w, h, _ = stages["8_padded"]
            bx, by, bw_, bh_ = stages["7_bbox"]
            ax.imshow(stages["1_original"], cmap="gray", vmin=0, vmax=255)
            ax.add_patch(patches.Rectangle((bx, by), bw_, bh_, lw=1.2,
                                           ec="tab:orange", fc="none", ls="--"))
            ax.add_patch(patches.Rectangle((x1, y1), w, h, lw=2.0,
                                           ec="tab:blue", fc="none"))
        elif idx == 8:
            ax.imshow(stages["9_squared"], cmap="gray", vmin=0, vmax=255)
        elif idx == 9:
            ax.imshow(stages["10_final"], cmap="gray", vmin=0, vmax=255)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[ok] saved → {out_path}")


def main():
    src = RESIM / f"{EXAMPLE_STEM}.tif"
    assert src.exists(), f"missing: {src}"
    stages = run_pipeline(src)
    make_figure(stages, src.name, OUT)


if __name__ == "__main__":
    main()
