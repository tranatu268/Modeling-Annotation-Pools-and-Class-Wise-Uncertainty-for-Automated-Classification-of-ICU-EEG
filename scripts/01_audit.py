# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Phase 1 - data audit: columns, rater IDs, vote histogram, strata table, file integrity. Writes work_dir/audit/."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.data import CLASSES, VOTES, build_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n_files", type=int, default=200, help="EEG files sampled for the integrity check")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["work_dir"] / "audit"
    out.mkdir(parents=True, exist_ok=True)

    tr = pd.read_csv(cfg["data_dir"] / "train.csv")
    print("columns:", list(tr.columns))
    rater = [c for c in tr.columns if any(w in c.lower() for w in ["rater", "annot", "reader", "scorer"])]
    print("rater-ID columns:", rater or "none -> RaTE pools are vote strata")

    tr["n"] = tr[VOTES].sum(axis=1)
    hist = tr.n.value_counts().sort_index().rename_axis("n_votes").rename("rows")
    hist.to_csv(out / "vote_hist_rows.csv")
    print("\nrows per vote count:\n", hist.to_string())

    ev = build_events(tr)
    print(f"\nrows {len(tr)}  eegs {tr.eeg_id.nunique()}  patients {tr.patient_id.nunique()}  events {len(ev)}")
    rows = []
    for s, name in [(0, ">=10"), (1, "3-7"), (2, "1-2")]:
        e = ev[ev.stratum == s]
        dom = np.bincount(e[VOTES].values.argmax(1), minlength=6) / len(e)
        soft = (e[VOTES].values / e.n.values[:, None]).mean(0)
        rows.append({"stratum": name, "patients": e.patient_id.nunique(), "eegs": e.eeg_id.nunique(),
                     "rows": int(e.n_rows.sum()), "events": len(e),
                     **{f"dominant_{c}": round(d, 4) for c, d in zip(CLASSES, dom)},
                     **{f"soft_{c}": round(v, 4) for c, v in zip(CLASSES, soft)}})
    strata = pd.DataFrame(rows)
    strata.to_csv(out / "strata.csv", index=False)
    print("\n", strata.T.to_string())
    pats = ev.groupby("patient_id").stratum.nunique()
    print(f"\npatients present in >1 stratum: {(pats > 1).sum()} / {len(pats)}")

    eeg_dir = cfg["data_dir"] / "train_eegs"
    have = {int(p.stem) for p in eeg_dir.glob("*.parquet")}
    missing = set(tr.eeg_id) - have
    print(f"EEG files: {len(have)}  missing for train.csv: {len(missing)}")

    rng = np.random.default_rng(0)
    chk = []
    for eid in rng.choice(sorted(set(tr.eeg_id) & have), min(a.n_files, len(have)), replace=False):
        x = pd.read_parquet(eeg_dir / f"{eid}.parquet")
        v = x.drop(columns="EKG").values
        chk.append({"eeg_id": eid, "seconds": len(x) / 200, "nan_frac": np.isnan(v).mean(),
                    "max_abs_uV": np.nanmax(np.abs(v)), "flat_channels": int((np.nanstd(v, 0) < 1e-6).sum())})
    chk = pd.DataFrame(chk)
    chk.to_csv(out / "eeg_integrity_sample.csv", index=False)
    print("\nEEG integrity (sample):\n", chk.describe().T[["mean", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
