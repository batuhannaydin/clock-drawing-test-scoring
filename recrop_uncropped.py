"""Re-crop the 14 'kırpılmamış' images using the standard crop_clock pipeline.

Reads the originals from master_veri/resimler/<stem>.tif and writes properly
cropped 224x224 grayscale PNGs to master_veri/processed/_kirpilmamis/,
overwriting the prior badly-formatted versions. Also drops an
_assignments.txt template the user fills in with the target class per image.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from _crop_puan_345 import crop_clock


MASTER_ROOT = Path("C:/Users/batuh/Desktop/master_veri/master_veri")
RESIM = MASTER_ROOT / "resimler"
UNCROPPED_DIR = MASTER_ROOT / "processed" / "_kirpilmamis"


def main() -> None:
    stems = sorted(p.stem for p in UNCROPPED_DIR.glob("*.png"))
    print(f"[info] {len(stems)} stems in {UNCROPPED_DIR}")

    ok, failed = [], []
    for stem in stems:
        src = RESIM / f"{stem}.tif"
        if not src.exists():
            print(f"  [missing tif] {src}")
            failed.append(stem)
            continue
        arr = crop_clock(src)
        if arr is None:
            print(f"  [crop failed] {stem}")
            failed.append(stem)
            continue
        out = UNCROPPED_DIR / f"{stem}.png"
        Image.fromarray(arr).save(out)
        ok.append(stem)
        print(f"  [ok] {stem} → 224x224 L")

    # Assignments template
    txt = UNCROPPED_DIR / "_assignments.txt"
    lines = [
        "# After cropping, decide which puan class each image belongs to.\n",
        "# Edit the line after `-->` with a single digit 0..5,\n",
        "# or write `sil` to drop the image entirely.\n",
        "# Lines under [failed] couldn't be re-cropped — review manually.\n",
        "\n[ok]\n",
    ]
    for s in ok:
        lines.append(f"{s} --> \n")
    if failed:
        lines.append("\n[failed]\n")
        for s in failed:
            lines.append(f"{s} --> \n")
    txt.write_text("".join(lines), encoding="utf-8")

    print(f"\n[done] ok={len(ok)} failed={len(failed)}")
    print(f"[info] assignments template → {txt}")


if __name__ == "__main__":
    main()
