# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Predict the LOCKED internal test with the saved fold checkpoints of finished runs (no retraining).
Run this once, for the frozen configurations only; then: 05_evaluate.py --split test (fold-ensemble average)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.train import predict_test_from_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    for r in a.runs:
        folds = sorted((cfg["work_dir"] / "results" / r).glob("fold*"))
        if len(folds) != 5 or not all((f / "best.pt").exists() for f in folds):
            raise SystemExit(f"{r}: expected 5 finished folds with best.pt, found {len(folds)}")
        for f in folds:
            n = predict_test_from_checkpoint(f, cfg["work_dir"], workers=a.workers)
            print(f"{r} {f.name}: {n} test events predicted", flush=True)


if __name__ == "__main__":
    main()
