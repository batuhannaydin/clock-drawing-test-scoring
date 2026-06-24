"""Run the full ablation sweep defined in experiments.PRESET_ORDER.

For each preset:
  1. subprocess: python train.py --preset <name>
  2. subprocess: python evaluate.py --preset <name>

Writes outputs/puan/_sweep_status.json so we can resume if interrupted.

Run:
    python run_all_experiments.py                      # all presets
    python run_all_experiments.py --from t3_class_balance   # resume from a preset
    python run_all_experiments.py --only t1_vanilla,s2_relaxed
    python run_all_experiments.py --skip-train         # eval only (assumes ckpts exist)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiments import PRESET_ORDER, PRESETS

PROJECT_ROOT = Path(__file__).resolve().parent
STATUS_PATH = PROJECT_ROOT / "outputs" / "puan" / "_sweep_status.json"


def _read_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return {"runs": {}}


def _write_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str))


def _run_one(preset: str, skip_train: bool) -> dict:
    log_dir = PROJECT_ROOT / "outputs" / "puan" / "_sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"preset": preset, "started": datetime.now().isoformat()}

    if not skip_train:
        train_log = log_dir / f"{preset}_train.log"
        print(f"\n{'=' * 72}\n[{preset}] training → {train_log.name}\n{'=' * 72}")
        t0 = time.time()
        with train_log.open("w", encoding="utf-8") as fout:
            proc = subprocess.run(
                [sys.executable, "-u", "train.py", "--preset", preset],
                stdout=fout, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT),
            )
        elapsed = (time.time() - t0) / 60
        record["train_min"] = round(elapsed, 1)
        record["train_returncode"] = proc.returncode
        if proc.returncode != 0:
            record["status"] = "train_failed"
            print(f"[{preset}] TRAIN FAILED (returncode={proc.returncode}). See {train_log}.")
            return record
        print(f"[{preset}] train done in {elapsed:.1f} min")

    eval_log = log_dir / f"{preset}_eval.log"
    print(f"\n[{preset}] evaluating → {eval_log.name}")
    t0 = time.time()
    with eval_log.open("w", encoding="utf-8") as fout:
        proc = subprocess.run(
            [sys.executable, "-u", "evaluate.py", "--preset", preset],
            stdout=fout, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT),
        )
    record["eval_min"] = round((time.time() - t0) / 60, 1)
    record["eval_returncode"] = proc.returncode
    if proc.returncode != 0:
        record["status"] = "eval_failed"
        print(f"[{preset}] EVAL FAILED (returncode={proc.returncode}). See {eval_log}.")
        return record
    print(f"[{preset}] eval done in {record['eval_min']} min")
    record["status"] = "ok"
    record["finished"] = datetime.now().isoformat()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_preset", default=None,
                        help="Start the sweep at this preset (resume).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated preset names — run just these.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, run evaluate only (ckpts must exist).")
    args = parser.parse_args()

    if args.only:
        targets = [t.strip() for t in args.only.split(",") if t.strip()]
        for t in targets:
            if t not in PRESETS:
                raise SystemExit(f"unknown preset: {t}")
    else:
        targets = list(PRESET_ORDER)
        if args.from_preset:
            if args.from_preset not in targets:
                raise SystemExit(f"unknown preset: {args.from_preset}")
            targets = targets[targets.index(args.from_preset):]

    print(f"[sweep] {len(targets)} preset(s): {targets}")
    print(f"[sweep] skip_train={args.skip_train}")

    status = _read_status()
    t_total = time.time()
    for preset in targets:
        rec = _run_one(preset, args.skip_train)
        status["runs"][preset] = rec
        _write_status(status)
        if rec.get("status") not in ("ok",):
            print(f"\n[sweep] STOPPING at {preset} (status={rec.get('status')})")
            sys.exit(1)

    total_min = (time.time() - t_total) / 60
    print(f"\n[sweep] ALL DONE — total {total_min:.1f} min")
    print(f"[sweep] status JSON → {STATUS_PATH}")


if __name__ == "__main__":
    main()
