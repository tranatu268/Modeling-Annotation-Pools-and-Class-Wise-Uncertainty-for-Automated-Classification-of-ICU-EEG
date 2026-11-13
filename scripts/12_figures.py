# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Paper figures from the FROZEN development results (out-of-fold, >=10-vote events). CPU only.

  Fig. 3  rater-pool shifts: RaTE mu (every fold x seed) and pool-head shifts (mean +- input-dependent SD),
          with the raw log-ratio of class prevalence between strata as a reference
  Fig. 5  class-wise reliability against the expert vote fraction (equal-count bins) + class-wise calibration error
  Fig. 6  conformal sets vs nominal coverage: per-class coverage (marginal vs label-conditional) and deferral/set size

Writes PDF (vector, for the paper) and PNG (preview) to <work_dir>/figures/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import conformal, load_config  # noqa: E402
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.results import PCOLS, load_preds  # noqa: E402

LABELS = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
SEEDS = (42, 1, 2)
RUNS = {
    "rate": {"cnn1d": "cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s{}",
             "spec2d": "spec2d-tf_efficientnetv2_b0_rate_k1000_mx0.4_rlr0.001_ema0.999_e20_last_s{}"},
    "pheads": {"cnn1d": "cnn1d_pheads_k1000_mx0.2_ema0.999_am_e30_last_s{}",
               "spec2d": "spec2d-tf_efficientnetv2_b0_pheads_k1000_mx0.4_ema0.999_e20_last_s{}"},
    "kl": {"cnn1d": "cnn1d_kl_mx0.2_ema0.999_am_e25_last_s{}",
           "spec2d": "spec2d-tf_efficientnetv2_b0_kl_mx0.4_ema0.999_last_s{}"},
}
NICE = {"rate": "RaTE", "pheads": "pool-heads", "kl": "two-stage"}


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3(cfg, results, out):
    import torch
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet")
    dev = ev[ev.fold >= 0]
    share = np.stack([(dev[dev.stratum == s][VOTES].values / dev[dev.stratum == s].n.values[:, None]).mean(0)
                      for s in range(3)])
    raw = np.log(share[1:]) - np.log(share[:1])
    raw -= raw.mean(1, keepdims=True)  # centred like mu

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.15), sharey=True)
    x = np.arange(6)
    series = [("rate", "cnn1d", -0.27), ("rate", "spec2d", -0.09), ("pheads", "cnn1d", 0.09), ("pheads", "spec2d", 0.27)]
    rows = []
    for pool, ax in zip((1, 2), axes):
        ax.bar(x, raw[pool - 1], 0.8, color="0.88", zorder=0, label="raw prevalence log-ratio")
        for j, (meth, bb, dx) in enumerate(series):
            vals, sds = [], []
            for s in SEEDS:
                for f in sorted((results / RUNS[meth][bb].format(s)).glob("fold*")):
                    if meth == "rate":
                        mu = torch.load(f / "rate_params.pt")["mu"].numpy()
                        vals.append(mu[pool] - mu[pool].mean())
                    else:
                        d = torch.load(f / "pool_shift.pt")
                        vals.append(d["mean"][pool - 1].numpy())
                        sds.append(d["std"][pool - 1].numpy())
            vals = np.array(vals)
            for k in range(6):
                rows.append({"pool": pool, "method": meth, "backbone": bb, "class": CLASSES[k],
                             "mean": vals[:, k].mean(), "sd_across_fits": vals[:, k].std(ddof=1),
                             "input_sd": np.mean(sds, 0)[k] if sds else np.nan})
            lab = f"{NICE[meth]} ({bb})"
            if meth == "rate":
                ax.plot(np.repeat(x + dx, len(vals)), vals.T.ravel(), "o", ms=2, color=OKABE[j], alpha=0.6, label=lab)
            else:
                ax.errorbar(x + dx, vals.mean(0), yerr=np.mean(sds, 0), fmt="s", ms=3, color=OKABE[j], capsize=1.5,
                            elinewidth=0.7, label=lab + ", mean ± input SD")
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_xticks(x, LABELS)
        ax.set_title({1: "Pool of 3–7 votes", 2: "Pool of 1–2 votes"}[pool])
    axes[0].set_ylabel("shift vs ≥10-vote panel (log-odds)")
    axes[1].legend(frameon=False, loc="upper right", ncol=1)
    fig.tight_layout()
    save(fig, out, "fig3_pool_shifts")
    pd.DataFrame(rows).to_csv(out / "fig3_pool_shifts.csv", index=False)


def classwise_reliability(t, p, k, bins=10):
    order = np.argsort(p[:, k])
    groups = np.array_split(order, bins)
    mp = np.array([p[g, k].mean() for g in groups])
    mt = np.array([t[g, k].mean() for g in groups])
    w = np.array([len(g) for g in groups]) / len(order)
    return mp, mt, float((w * np.abs(mp - mt)).sum())


def load_dev(cfg, results, run_names):
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    preds = {r: load_preds(results, r, "val") for r in run_names}
    ids = ev.index[ev.stratum == 0]
    for p in preds.values():
        ids = ids.intersection(p.index)
    t = ev.loc[ids, VOTES].values.astype(float)
    t /= t.sum(1, keepdims=True)
    return t, {r: p.loc[ids, PCOLS].values for r, p in preds.items()}, ev.loc[ids, "patient_id"].values


def fig5(cfg, results, out, compare):
    """One row of six class panels (fits a two-column figure ~1.8 in high). Method names in the first panel,
    class-wise calibration error of each method written inside every panel in the method's colour."""
    names = [RUNS[m][bb].format(42) for m, bb in compare]
    t, P, _ = load_dev(cfg, results, names)
    fig, axes = plt.subplots(1, 6, figsize=(7.16, 1.75))
    rows = []
    for k, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], ":", color="0.5", lw=0.6)
        for j, ((m, bb), r) in enumerate(zip(compare, names)):
            mp, mt, e = classwise_reliability(t, P[r], k)
            ax.plot(mp, mt, "o-", ms=1.8, lw=0.8, color=OKABE[j], label=f"{NICE[m]} ({bb})")
            ax.text(0.97, 0.05 + 0.11 * j, f"{e:.3f}", transform=ax.transAxes, ha="right", fontsize=5.5,
                    color=OKABE[j])
            rows.append({"class": CLASSES[k], "method": m, "backbone": bb, "classwise_ce": e})
        hi = max(0.05, float(np.quantile(np.concatenate([P[r][:, k] for r in names]), 0.995)))
        ax.set_xlim(0, min(1, hi * 1.05))
        ax.set_ylim(0, min(1, hi * 1.05))
        ax.set_title(LABELS[k], pad=2)
        ax.tick_params(labelsize=5.5, length=2, pad=1)
        ax.set_xlabel("predicted prob.", fontsize=6, labelpad=1)
    axes[0].set_ylabel("fraction of votes", fontsize=6, labelpad=1)
    axes[0].legend(frameon=False, loc="upper left", fontsize=5, handlelength=1.0, borderaxespad=0.1)
    fig.tight_layout(w_pad=0.4)
    save(fig, out, "fig5_reliability")
    pd.DataFrame(rows).to_csv(out / "fig5_reliability.csv", index=False)


def fig6(cfg, results, out, main, repeats):
    name = RUNS[main[0]][main[1]].format(42)
    t, P, pats = load_dev(cfg, results, [name])
    p = P[name]
    alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    rows = [{"alpha": al, "mode": m, **conformal.run(p, t, pats, al, repeats, mode=m)}
            for al in alphas for m in ("marginal", "label")]
    df = pd.DataFrame(rows)
    df.to_csv(out / "fig6_conformal.csv", index=False)
    nominal = 1 - np.array(alphas)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    ax = axes[0]
    ax.plot(nominal, nominal, ":", color="0.4", lw=0.8, label="nominal")
    for k in range(6):
        for m, ls in (("marginal", "--"), ("label", "-")):
            d = df[df["mode"] == m]
            ax.plot(1 - d.alpha, d[f"coverage_class{k}"], ls, color=OKABE[k], lw=0.9,
                    label=LABELS[k] if m == "label" else None)
    ax.plot([], [], "--", color="0.3", label="marginal")
    ax.plot([], [], "-", color="0.3", label="label-conditional")
    ax.set_xlabel("nominal coverage 1 − α")
    ax.set_ylabel("expert votes covered, per class")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax = axes[1]
    for m, c in (("marginal", OKABE[0]), ("label", OKABE[1])):
        d = df[df["mode"] == m]
        ax.plot(1 - d.alpha, d.defer_rate, "o-", ms=2.5, color=c, label=f"deferral, {m}")
    ax.set_xlabel("nominal coverage 1 − α")
    ax.set_ylabel("fraction of segments deferred")
    ax2 = ax.twinx()
    for m, c in (("marginal", OKABE[0]), ("label", OKABE[1])):
        d = df[df["mode"] == m]
        ax2.plot(1 - d.alpha, d.set_size, "s:", ms=2, color=c, alpha=0.7, label=f"set size, {m}")
    ax2.set_ylabel("mean set size")
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout()
    save(fig, out, "fig6_conformal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--figs", default="3,5,6")
    ap.add_argument("--repeats", type=int, default=50, help="conformal calibration/test splits for Fig. 6")
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = cfg["work_dir"] / "results"
    out = cfg["work_dir"] / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    figs = a.figs.split(",")
    if "3" in figs:
        fig3(cfg, results, out)
        print("Fig. 3 done", flush=True)
    if "5" in figs:
        fig5(cfg, results, out, [("kl", "spec2d"), ("pheads", "spec2d"), ("rate", "cnn1d")])
        print("Fig. 5 done", flush=True)
    if "6" in figs:
        fig6(cfg, results, out, ("pheads", "spec2d"), a.repeats)
        print("Fig. 6 done", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
