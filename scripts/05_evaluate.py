# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Evaluate runs on >=10-vote events (out-of-fold, or the locked test) with patient-bootstrap CIs.
With --ref, every run is also compared to the reference run (paired bootstrap difference and p-value)."""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.metrics import patient_bootstrap  # noqa: E402
from hms.results import load_preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = cfg["work_dir"] / "results"

    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    preds = {r: load_preds(results, r, a.split) for r in a.runs + ([a.ref] if a.ref else [])}
    ids = ev.index[ev.stratum == 0]
    for p in preds.values():  # evaluate every run on exactly the same events
        ids = ids.intersection(p.index)
    t = ev.loc[ids, VOTES].values.astype(float)
    t /= t.sum(1, keepdims=True)
    pats = ev.loc[ids, "patient_id"].values
    P = {r: p.loc[ids, [f"p_{c}" for c in CLASSES]].values for r, p in preds.items()}
    print(f"{a.split}: {len(ids)} events with >=10 votes from {len(set(pats))} patients")

    rows = []
    for r in a.runs:
        res = patient_bootstrap(t, P[r], pats, B=a.B, p_ref=P[a.ref] if a.ref and r != a.ref else None)
        for metric, d in res.items():
            rows.append({"run": r, "metric": metric, **d})
    table = pd.DataFrame(rows)
    table.insert(1, "ref", a.ref or "")
    show = ["kl", "bal_acc", "cohen_kappa", "macro_f1", "ece", "ece_soft", "brier", "risk", "recall_lrda",
            "auroc_seizure"]
    print(table[table.metric.isin(show)].drop(columns="ref").round(4).to_string(index=False))
    # short file name (Windows paths are limited to 260 characters); run names are stored inside the file
    tag = hashlib.md5("|".join(a.runs + [a.ref or ""]).encode()).hexdigest()[:8]
    out = results / f"eval_{a.split}_{tag}.csv"
    table.to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
