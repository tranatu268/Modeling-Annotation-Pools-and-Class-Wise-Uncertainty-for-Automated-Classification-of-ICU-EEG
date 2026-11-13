# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Phase 8 - conformal prediction sets and deferral on out-of-fold >=10-vote predictions of one run."""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms import conformal  # noqa: E402
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.metrics import RISK, kaggle_kl  # noqa: E402
from hms.results import load_preds  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="val: out-of-fold predictions; test: locked internal test (fold-ensemble average). "
                         "Calibration and evaluation patients are drawn from the same split in both cases.")
    a = ap.parse_args()
    cfg = load_config(a.config)
    run_dir = cfg["work_dir"] / "results" / a.run
    sfx = "" if a.split == "val" else "_test"

    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    pr = load_preds(cfg["work_dir"] / "results", a.run, a.split)
    ids = pr.index.intersection(ev.index[ev.stratum == 0])
    print(f"{a.split}: {len(ids)} events with >=10 votes, {ev.loc[ids, 'patient_id'].nunique()} patients")
    p = pr.loc[ids, [f"p_{c}" for c in CLASSES]].values
    t = ev.loc[ids, VOTES].values.astype(float)
    t /= t.sum(1, keepdims=True)
    pats = ev.loc[ids, "patient_id"].values

    rows = [{"alpha": al, "mode": m, **conformal.run(p, t, pats, al, a.repeats, mode=m)}
            for al in a.alphas for m in conformal.MODES]
    table = pd.DataFrame(rows).rename(columns={f"coverage_class{k}": f"cov_{c}" for k, c in enumerate(CLASSES)})
    table.to_csv(run_dir / f"conformal{sfx}.csv", index=False)
    print(table.round(4).to_string(index=False))

    # risk-coverage: abstain on the least confident events first
    order = np.argsort(-p.max(1))
    kl, risk = kaggle_kl(t, p)[order], np.einsum("nj,jk,nk->n", t, RISK, p)[order]
    cov = np.arange(1, len(order) + 1) / len(order)
    rc = pd.DataFrame({"coverage": cov, "kl": np.cumsum(kl) / np.arange(1, len(kl) + 1),
                       "risk": np.cumsum(risk) / np.arange(1, len(risk) + 1)})
    rc.iloc[::max(1, len(rc) // 200)].to_csv(run_dir / f"risk_coverage{sfx}.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(rc.coverage, rc.risk)
    ax.set_xlabel("coverage (fraction not deferred)")
    ax.set_ylabel("mean clinical risk")
    fig.tight_layout()
    fig.savefig(run_dir / f"risk_coverage{sfx}.pdf")


if __name__ == "__main__":
    main()
