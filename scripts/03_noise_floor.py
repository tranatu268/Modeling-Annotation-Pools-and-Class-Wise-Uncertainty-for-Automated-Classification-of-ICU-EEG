# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Phase 3 (gate G1) - irreducible KL on >=10-vote events (development folds only). Writes work_dir/noise_floor/.
Compare the 'empirical_bayes' mean with the out-of-fold KL of your models from 05_evaluate.py."""
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
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.noise_floor import floor_asymptotic, floor_empirical_bayes, floor_plugin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["work_dir"] / "noise_floor"
    out.mkdir(parents=True, exist_ok=True)

    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet")
    hi = ev[(ev.stratum == 0) & (ev.fold >= 0)].reset_index(drop=True)  # never touch the locked test
    c = hi[VOTES].values.astype(float)
    df = pd.DataFrame({"event_id": hi.event_id, "n": hi.n, "dominant": np.array(CLASSES)[c.argmax(1)]})
    df["empirical_bayes"], alpha = floor_empirical_bayes(c)
    for s in [0.1, 0.5]:
        df[f"plugin_a{s}"] = floor_plugin(c, a=s)
    df["asymptotic"] = floor_asymptotic(c)
    df.to_csv(out / "per_event.csv", index=False)

    cols = ["empirical_bayes", "plugin_a0.1", "plugin_a0.5", "asymptotic"]
    summary = df[cols].agg(["mean", "median"]).T
    summary.to_csv(out / "summary.csv")
    print(f">=10-vote events: {len(df)}; fitted Dirichlet prior alpha = {np.round(alpha, 3)}")
    print(f"expected KL of a PERFECT model:\n{summary.round(4).to_string()}")
    print("\nby dominant class (empirical_bayes):\n", df.groupby("dominant").empirical_bayes.mean().round(4).to_string())

    by_n = df.groupby("n")[cols].mean()
    by_n.to_csv(out / "by_n.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for k in cols:
        ax.plot(by_n.index, by_n[k], marker="o", ms=3, label=k)
    ax.set_xlabel("number of expert votes n")
    ax.set_ylabel("expected KL of a perfect model (nats)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "noise_floor_by_n.pdf")


if __name__ == "__main__":
    main()
