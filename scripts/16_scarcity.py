# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Stress test: is the pool model more useful when high-agreement (>=10-vote) labels are scarce?

Runs trained with --hi_frac f keep only a fraction f of the >=10-vote PATIENTS in each training fold (all low-vote
events are kept; the patient subset depends only on the seed, so both methods see the same subset). Validation is
always on all >=10-vote events of the held-out fold, so every fraction is scored on the same 5,073 events.
For each fraction: out-of-fold KL of two-stage training and of the pool-aware models (mean over available seeds),
and the paired patient-bootstrap difference against two-stage training at the same fraction. f = 1 uses the main runs.

    python scripts\\16_scarcity.py                   # whatever runs exist; missing ones are listed
Writes work/results/scarcity.csv (and work/figdata/scarcity.csv for MATLAB).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.data import VOTES  # noqa: E402
from hms.metrics import fast_metrics, kaggle_kl  # noqa: E402
from hms.results import PCOLS, load_preds  # noqa: E402

SEEDS = (42, 1, 2)
BASE = {  # frozen full-data run names (seed placeholder last)
    "cnn1d": {"two-stage": "cnn1d_kl_mx0.2_ema0.999_am_e25_last_s{}",
              "pool-heads": "cnn1d_pheads_k1000_mx0.2_ema0.999_am_e30_last_s{}",
              "RaTE": "cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s{}"},
    "spec2d": {"two-stage": "spec2d-tf_efficientnetv2_b0_kl_mx0.4_ema0.999_last_s{}",
               "pool-heads": "spec2d-tf_efficientnetv2_b0_pheads_k1000_mx0.4_ema0.999_e20_last_s{}",
               "RaTE": "spec2d-tf_efficientnetv2_b0_rate_k1000_mx0.4_rlr0.001_ema0.999_e20_last_s{}"},
}
REF = "two-stage"


def run_name(pattern, frac, seed):
    """Name written by 04_train.py: the '_hi<f>' token comes right before '_s<seed>'."""
    base = pattern.format(seed)
    if frac >= 1:
        return base
    head, s = base.rsplit("_s", 1)
    return f"{head}_hi{frac:g}_s{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fracs", default="0.1,0.25,0.5,1")
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = cfg["work_dir"] / "results"
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    fracs = [float(f) for f in a.fracs.split(",")]
    rng = np.random.default_rng(0)
    rows, missing = [], []
    for bb, methods in BASE.items():
        avail = {}
        for f in fracs:
            for m, pat in methods.items():
                runs = [run_name(pat, f, s) for s in SEEDS]
                have = [r for r in runs if len(list((results / r).glob("fold*/preds.parquet"))) == 5]
                missing += [r for r in runs if r not in have and (f < 1 or s_is_main(r))]
                if have:
                    avail[(f, m)] = have
        if not avail:
            continue
        preds = {r: load_preds(results, r, "val") for rs in avail.values() for r in rs}
        ids = ev.index[ev.stratum == 0]
        for p in preds.values():
            ids = ids.intersection(p.index)
        t = ev.loc[ids, VOTES].values.astype(float)
        t /= t.sum(1, keepdims=True)
        pats = ev.loc[ids, "patient_id"].values
        uniq, inv = np.unique(pats, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        boots = [np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(a.B)]

        def kl_events(runs):  # per-event KL averaged over seeds
            return np.mean([kaggle_kl(t, preds[r].loc[ids, PCOLS].values) for r in runs], 0)

        for f in fracs:
            if (f, REF) not in avail:
                continue
            ref = kl_events(avail[(f, REF)])
            for m in methods:
                if (f, m) not in avail:
                    continue
                kl = kl_events(avail[(f, m)])
                risk = np.mean([fast_metrics(t, preds[r].loc[ids, PCOLS].values)["risk"] for r in avail[(f, m)]])
                row = {"backbone": bb, "frac": f, "method": m, "seeds": len(avail[(f, m)]), "events": len(ids),
                       "kl": kl.mean(), "risk": risk}
                if m != REF:
                    d = kl - ref
                    b = np.array([d[i].mean() for i in boots])
                    row.update(d_kl=d.mean(), d_lo=np.percentile(b, 2.5), d_hi=np.percentile(b, 97.5),
                               p=min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
                rows.append(row)
    if missing:
        print("missing runs (train them with scripts\\run_scarcity.ps1):\n  " + "\n  ".join(sorted(set(missing))))
    if not rows:
        return
    tab = pd.DataFrame(rows)
    tab.to_csv(results / "scarcity.csv", index=False)
    (cfg["work_dir"] / "figdata").mkdir(exist_ok=True)
    tab.to_csv(cfg["work_dir"] / "figdata" / "scarcity.csv", index=False)
    tab["cell"] = tab.apply(lambda r: f"{r.kl:.4f} (n={r.seeds})" + (
        "" if pd.isna(r.get("d_kl")) else f"  Δ {r.d_kl:+.4f} [{r.d_lo:+.4f}, {r.d_hi:+.4f}] p={r.p:.3f}"), axis=1)
    pd.set_option("display.width", 250)
    for bb, d in tab.groupby("backbone"):
        print(f"\n=== {bb}: out-of-fold KL on all >=10-vote development events; Δ vs two-stage at the same fraction ===")
        print(d.pivot_table(index="frac", columns="method", values="cell", aggfunc="first").to_string())
    print(f"\n-> {results / 'scarcity.csv'}")


def s_is_main(run):
    return run.endswith("_s42")  # at f = 1 only seed 42 is required (the main runs have 3 seeds anyway)


if __name__ == "__main__":
    main()
