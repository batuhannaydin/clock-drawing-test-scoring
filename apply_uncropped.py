"""Apply the user's _assignments.txt for re-cropped 'kırpılmamış' images.

For each entry `<stem> --> <decision>`:
  * digit 0..5  : move processed/_kirpilmamis/<stem>.png -> processed/<digit>/<stem>.png
                  AND update master_etiketler.csv puan = digit
  * sil / SİL   : delete processed/_kirpilmamis/<stem>.png

Skips stems with no decision (blank after -->) or unrecognized tokens.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

MASTER_ROOT = Path("C:/Users/batuh/Desktop/master_veri/master_veri")
UNCROPPED_DIR = MASTER_ROOT / "processed" / "_kirpilmamis"
ASSIGN_TXT = UNCROPPED_DIR / "_assignments.txt"
LABELS_CSV = MASTER_ROOT / "master_etiketler.csv"

LINE_RE = re.compile(r"^(\S+)\s*-+>\s*(.*)$")


def main() -> None:
    assert ASSIGN_TXT.exists(), f"missing: {ASSIGN_TXT}"
    assert LABELS_CSV.exists(), f"missing: {LABELS_CSV}"

    decisions = {}   # stem -> "sil" or int
    skipped_blank = []
    skipped_unknown = []

    for raw in ASSIGN_TXT.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        stem, decision = m.group(1), m.group(2).strip()
        if not decision:
            skipped_blank.append(stem)
            continue
        d_norm = decision.upper().replace("İ", "I")
        if d_norm == "SIL":
            decisions[stem] = "sil"
        elif decision in {"0", "1", "2", "3", "4", "5"}:
            decisions[stem] = int(decision)
        else:
            skipped_unknown.append((stem, decision))

    print(f"[info] {len(decisions)} actionable decisions parsed")
    if skipped_blank:
        print(f"[warn] {len(skipped_blank)} stems with blank decision: {skipped_blank}")
    if skipped_unknown:
        print(f"[warn] {len(skipped_unknown)} stems with unknown tokens: {skipped_unknown}")

    df = pd.read_csv(LABELS_CSV)
    df["stem_lookup"] = df["dosya_adi"].astype(str).apply(lambda f: Path(f).stem)

    n_deleted, n_moved, n_csv_updated, n_missing_file, n_missing_csv = 0, 0, 0, 0, 0

    for stem, decision in decisions.items():
        src = UNCROPPED_DIR / f"{stem}.png"
        if not src.exists():
            n_missing_file += 1
            print(f"  [skip] not in _kirpilmamis: {stem}")
            continue
        if decision == "sil":
            src.unlink()
            n_deleted += 1
            print(f"  [del] {stem}")
        else:
            target = int(decision)
            dst_dir = MASTER_ROOT / "processed" / str(target)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{stem}.png"
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            n_moved += 1
            mask = df["stem_lookup"] == stem
            if mask.any():
                df.loc[mask, "puan"] = target
                n_csv_updated += int(mask.sum())
                print(f"  [mv] {stem} -> processed/{target}/ + CSV puan={target}")
            else:
                n_missing_csv += 1
                print(f"  [warn] stem not in CSV: {stem}")

    df.drop(columns=["stem_lookup"]).to_csv(LABELS_CSV, index=False)

    print("\n=== Summary ===")
    print(f"  deleted              : {n_deleted}")
    print(f"  moved                : {n_moved}")
    print(f"  CSV rows updated     : {n_csv_updated}")
    print(f"  source missing       : {n_missing_file}")
    print(f"  not in CSV           : {n_missing_csv}")


if __name__ == "__main__":
    main()
