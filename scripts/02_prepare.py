# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Phase 2 - events, patient-wise folds (+ locked internal test), leakage check, EEG window cache."""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.data import build_cache, build_events, check_no_leakage, make_folds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["work_dir"]
    work.mkdir(parents=True, exist_ok=True)

    ev = build_events(pd.read_csv(cfg["data_dir"] / "train.csv"))
    ev["fold"] = make_folds(ev, cfg["n_folds"], cfg["seed"])
    check_no_leakage(ev, ev.fold.values)
    print("events per split x stratum (0: >=10 votes, 1: 3-7, 2: 1-2; fold -1 = locked internal test):")
    print(pd.crosstab(ev.fold, ev.stratum, margins=True).to_string())
    print("patients per split:", ev.groupby("fold").patient_id.nunique().to_dict())
    ev.to_parquet(work / "events.parquet", index=False)

    cache = work / "windows.npy"
    if cache.exists() and not a.overwrite:
        print(f"{cache} exists, skipping (use --overwrite to rebuild)")
        return
    T = cfg["span_s"] * cfg["fs_out"]
    print(f"caching {len(ev)} windows of 18 x {T} float16 (~{len(ev) * 18 * T * 2 / 1e9:.1f} GB) -> {cache}")
    build_cache(ev, cfg["data_dir"] / "train_eegs", cache, cfg["span_s"], cfg["fs_out"], a.workers)
    print("done")


if __name__ == "__main__":
    main()
