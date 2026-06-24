"""Execute the manual review decisions in _suspect_review/README.txt.

Parses each line's `--> <note>` and performs:
  * "sil"                  : delete processed/<labeled>/<stem>.png
  * "X'e taşı/yerleştir"   : move processed/<labeled>/<stem>.png  ->  processed/X/<stem>.png
                             AND patch master_etiketler.csv puan = X
  * target == labeled      : no-op

Backs up master_etiketler.csv before patching.
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_README = Path("outputs/puan/_suspect_review/README.txt")
MASTER_ROOT = Path("C:/Users/batuh/Desktop/master_veri/master_veri")
PROCESSED = MASTER_ROOT / "processed"
UNCROPPED_DIR = PROCESSED / "_kirpilmamis"
LABELS_CSV = MASTER_ROOT / "master_etiketler.csv"

LINE_RE = re.compile(
    r"^(\d+)\s+labeled=(\d)\s+pred=(\d)\s+\(p=[\d.]+\)\s+margin=[\d.]+\s+\|\s+(\S+)\s+-+>\s+(.+)$"
)
TARGET_RE = re.compile(r"(?<!\d)([0-5])(?!\d)")


def parse_actions(readme_path: Path):
    actions = []
    for raw in readme_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        idx, labeled, _pred, stem, note = m.groups()
        labeled = int(labeled)
        note_l = note.lower()

        if "kırpılmamış" in note_l or "kirpilmamis" in note_l:
            actions.append({"idx": idx, "stem": stem, "labeled": labeled,
                            "action": "uncropped"})
            continue

        if "sil" in note_l:
            actions.append({"idx": idx, "stem": stem, "labeled": labeled,
                            "action": "delete"})
            continue

        tm = TARGET_RE.search(note_l)
        if tm is None:
            actions.append({"idx": idx, "stem": stem, "labeled": labeled,
                            "action": "unknown", "note": note})
            continue
        target = int(tm.group(1))
        if target == labeled:
            actions.append({"idx": idx, "stem": stem, "labeled": labeled,
                            "target": target, "action": "skip"})
        else:
            actions.append({"idx": idx, "stem": stem, "labeled": labeled,
                            "target": target, "action": "move"})
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_README,
                        help="Annotated review file (defaults to README.txt).")
    args = parser.parse_args()

    readme_path = args.input
    assert readme_path.exists(), f"input not found: {readme_path}"
    assert LABELS_CSV.exists(), f"labels CSV not found: {LABELS_CSV}"
    assert PROCESSED.is_dir(), f"processed dir missing: {PROCESSED}"
    print(f"[info] reading annotations from: {readme_path}")

    actions = parse_actions(readme_path)
    print(f"[info] parsed {len(actions)} annotated lines\n")
    counts = Counter(a["action"] for a in actions)
    print(f"[info] action breakdown: {dict(counts)}")

    unknowns = [a for a in actions if a["action"] == "unknown"]
    if unknowns:
        print("[warn] unparsed instructions:")
        for a in unknowns:
            print(f"  {a['idx']} {a['stem']}: {a['note']}")

    move_breakdown = Counter(
        (a["labeled"], a.get("target")) for a in actions if a["action"] == "move"
    )
    print(f"[info] move transitions: {dict(move_breakdown)}\n")

    # Backup labels CSV
    backup = LABELS_CSV.with_suffix(".csv.bak_pre_suspect_review")
    if not backup.exists():
        shutil.copy2(LABELS_CSV, backup)
        print(f"[info] CSV backup -> {backup.name}")
    else:
        print(f"[info] CSV backup already exists: {backup.name}")

    df = pd.read_csv(LABELS_CSV)
    df["stem_lookup"] = df["dosya_adi"].astype(str).apply(lambda f: Path(f).stem)
    stem_index = df.set_index("stem_lookup", drop=False)

    n_deleted, n_moved, n_skip, n_missing_file, n_missing_csv = 0, 0, 0, 0, 0
    n_uncropped = 0
    csv_updates = []  # (stem, new_puan)

    for a in actions:
        stem = a["stem"]
        labeled = a["labeled"]
        src = PROCESSED / str(labeled) / f"{stem}.png"

        if a["action"] == "delete":
            if src.exists():
                src.unlink()
                n_deleted += 1
            else:
                n_missing_file += 1
                print(f"  [skip-del] not found: {src}")
            continue

        if a["action"] == "uncropped":
            UNCROPPED_DIR.mkdir(parents=True, exist_ok=True)
            dst = UNCROPPED_DIR / f"{stem}.png"
            if not src.exists():
                n_missing_file += 1
                print(f"  [skip-uncrop] src not found: {src}")
                continue
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            n_uncropped += 1
            continue

        if a["action"] == "skip":
            n_skip += 1
            continue

        if a["action"] == "move":
            target = a["target"]
            dst_dir = PROCESSED / str(target)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{stem}.png"
            if not src.exists():
                n_missing_file += 1
                print(f"  [skip-move] src not found: {src}")
                continue
            if dst.exists():
                # overwrite to be safe; previous version is in the wrong folder anyway
                dst.unlink()
            shutil.move(str(src), str(dst))
            n_moved += 1
            csv_updates.append((stem, target))

    # Apply CSV updates
    n_csv_updated = 0
    for stem, new_puan in csv_updates:
        if stem in stem_index.index:
            mask = df["stem_lookup"] == stem
            df.loc[mask, "puan"] = new_puan
            n_csv_updated += mask.sum()
        else:
            n_missing_csv += 1
            print(f"  [warn] stem not in CSV: {stem}")

    df = df.drop(columns=["stem_lookup"])
    df.to_csv(LABELS_CSV, index=False)

    print("\n=== Summary ===")
    print(f"  deleted files       : {n_deleted}")
    print(f"  moved files         : {n_moved}")
    print(f"  uncropped→_kirpilmamis: {n_uncropped}")
    print(f"  CSV rows updated    : {n_csv_updated}")
    print(f"  no-op (target=label): {n_skip}")
    print(f"  source missing      : {n_missing_file}")
    print(f"  not in CSV          : {n_missing_csv}")
    print(f"\n  CSV  -> {LABELS_CSV}")
    print(f"  Backup -> {backup}")


if __name__ == "__main__":
    main()
