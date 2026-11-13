# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Model vs experts on >=10-vote events, from vote counts only (no rater IDs needed). CPU only.

Writes work_dir/expert_agreement/<tag>/:
  agreement.csv      overall and per-class: expert-expert vs model-expert agreement, Cohen's kappa, specific
                     agreement, with patient-bootstrap 95% CIs and model-minus-expert differences
  confusion_*.csv    expert vote vs leave-one-out consensus, and model vs the same consensus (weights, row %, col %)
  operating.csv      per class: model AUROC/AUPRC vs leave-one-out consensus, average-expert and model (FPR, TPR, PPV)
  fig_kappa.pdf, fig_confusion.pdf, fig_roc.pdf
"""
import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import agreement as ag  # noqa: E402
from hms import load_config  # noqa: E402
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.results import PCOLS, load_preds  # noqa: E402

LABELS = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]


OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
NICE = [("pheads", "pool-heads"), ("rate", "RaTE"), ("kl_w", "vote-weighted KL"), ("divo", "DiVo"),
        ("kl", "two-stage")]  # kl_w before kl: "_kl_" is also a substring of "_kl_w_"


def short(name):
    """Readable label, e.g. 'pool-heads (spec2d)' or 'RaTE (cnn1d)'."""
    bb = "spec2d" if name.startswith("spec2d") else name.split("_")[0]
    for tok, nice in NICE:
        if f"_{tok}_" in name:
            return f"{nice} ({bb})"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = cfg["work_dir"] / "results"
    tag = hashlib.md5("|".join(a.runs + [a.split]).encode()).hexdigest()[:8]
    out = cfg["work_dir"] / "expert_agreement" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.txt").write_text("\n".join([a.split] + a.runs))

    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    preds = {r: load_preds(results, r, a.split) for r in a.runs}
    ids = ev.index[ev.stratum == 0]
    for p in preds.values():
        ids = ids.intersection(p.index)
    c = ev.loc[ids, VOTES].values.astype(int)
    pats = ev.loc[ids, "patient_id"].values
    print(f"{a.split}: {len(ids)} events with >=10 votes, {len(set(pats))} patients, "
          f"median {int(np.median(c.sum(1)))} votes/event")

    rows, ops = [], []
    for r in a.runs:
        P = preds[r].loc[ids, PCOLS].values
        res = ag.agreement_with_ci(c, P, pats, B=a.B)
        for key, d in res.items():
            vals = np.atleast_1d(d["value"])
            names = ["all"] if len(vals) == 1 else CLASSES
            for i, cl in enumerate(names):
                rows.append({"run": r, "stat": key, "class": cl, "value": vals[i],
                             "lo": np.atleast_1d(d["lo"])[i], "hi": np.atleast_1d(d["hi"])[i]})
        tab = ag.loo_table(c, P)
        for who in ("expert", "model"):
            M = ag.confusion(tab, who)
            if who == "expert" and r != a.runs[0]:
                continue  # the expert matrix does not depend on the run
            stem = "confusion_expert" if who == "expert" else f"confusion_model_{r}"
            for norm, D in (("weight", M), ("row_pct", 100 * M / M.sum(1, keepdims=True)),
                            ("col_pct", 100 * M / M.sum(0, keepdims=True))):
                pd.DataFrame(D, index=[f"cons_{x}" for x in CLASSES],
                             columns=[f"{who}_{x}" for x in CLASSES]).to_csv(out / f"{stem}_{norm}.csv")
        for o in ag.operating_points(tab, P):
            ops.append({"run": r, "class": CLASSES[o.pop("class")], **o})

    table = pd.DataFrame(rows)
    table.to_csv(out / "agreement.csv", index=False)
    opst = pd.DataFrame(ops)
    opst.to_csv(out / "operating.csv", index=False)

    overall = table[table["class"] == "all"].pivot_table(index="stat", columns="run", values="value")
    print("\nOverall (point estimates):\n", overall.round(4).to_string())
    for r in a.runs:
        t = table[table.run == r]
        print(f"\n[{r}] per class:")
        piv = t[t.stat.isin(["kappa_ee_k", "kappa_me_k", "kappa_me_k-minus-ee", "psa_ee", "psa_me"])]
        show = piv.assign(txt=piv.apply(lambda x: f"{x.value:.3f} [{x.lo:.3f}, {x.hi:.3f}]", axis=1))
        print(show.pivot_table(index="class", columns="stat", values="txt", aggfunc="first")
              .reindex(CLASSES).to_string())
    print("\nOperating points vs leave-one-out consensus:\n",
          opst.round(3).to_string(index=False))

    # ---- figures -------------------------------------------------------------------------------------------
    # (1) per-class kappa: expert-expert (grey) vs each run
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    width = 0.8 / (1 + len(a.runs))
    x = np.arange(6)
    t0 = table[(table.run == a.runs[0]) & (table.stat == "kappa_ee_k")].set_index("class").loc[CLASSES]
    ax.barh(x, t0.value, width, xerr=[t0.value - t0.lo, t0.hi - t0.value], color="0.7", label="Expert–expert",
            error_kw={"lw": 0.6})
    for j, r in enumerate(a.runs, 1):
        tr = table[(table.run == r) & (table.stat == "kappa_me_k")].set_index("class").loc[CLASSES]
        ax.barh(x + j * width, tr.value, width, xerr=[tr.value - tr.lo, tr.hi - tr.value],
                label=f"Model–expert ({short(r)})", color=OKABE[j - 1], error_kw={"lw": 0.6})
    ax.set_yticks(x + width * len(a.runs) / 2, LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's κ (one-vs-rest)")
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=1)
    fig.tight_layout()
    fig.savefig(out / "fig_kappa.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_kappa.png", dpi=200, bbox_inches="tight")

    # (2) row-normalised confusion vs leave-one-out consensus: experts and first run
    Me = pd.read_csv(out / "confusion_expert_row_pct.csv", index_col=0).values
    Mm = pd.read_csv(out / f"confusion_model_{a.runs[0]}_row_pct.csv", index_col=0).values
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.9))
    for axx, M, title in ((axes[0], Me, "Individual expert"), (axes[1], Mm, f"Model ({short(a.runs[0])})")):
        axx.imshow(M, cmap="Blues", vmin=0, vmax=100)
        for i in range(6):
            for j in range(6):
                axx.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6,
                         color="white" if M[i, j] > 60 else "black")
        axx.set_xticks(range(6), LABELS, fontsize=6, rotation=45)
        axx.set_yticks(range(6), LABELS, fontsize=6)
        axx.set_title(title, fontsize=8)
        axx.set_xlabel("Decision", fontsize=7)
    axes[0].set_ylabel("Leave-one-out consensus", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_confusion.pdf")

    # (3) per-class ROC of each run vs leave-one-out consensus, with the average-expert operating point.
    # One row of six panels; FPR axis zoomed to [0, 0.5] where all expert operating points lie.
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 6, figsize=(7.16, 1.8), sharey=True)
    tab0 = ag.loo_table(c, preds[a.runs[0]].loc[ids, PCOLS].values)
    for k, axx in enumerate(axes):
        pos, ev_i, w = tab0[:, 2] == k, tab0[:, 0].astype(int), tab0[:, 4]
        for j, r in enumerate(a.runs):
            P = preds[r].loc[ids, PCOLS].values
            fpr, tpr, _ = roc_curve(pos, P[ev_i, k], sample_weight=w)
            auc = opst[(opst.run == r) & (opst["class"] == CLASSES[k])].auroc.iloc[0]
            axx.plot(fpr, tpr, lw=0.9, color=OKABE[j], label=short(r))
            axx.text(0.97, 0.05 + 0.1 * j, f"AUC {auc:.3f}", transform=axx.transAxes, ha="right", fontsize=5.2,
                     color=OKABE[j])
        o = opst[(opst.run == a.runs[0]) & (opst["class"] == CLASSES[k])].iloc[0]
        axx.plot(o.fpr_expert, o.tpr_expert, "kx", ms=5, mew=1.1, label="average expert")
        axx.set_xlim(0, 0.5)
        axx.set_ylim(0, 1.0)
        axx.set_title(LABELS[k], fontsize=7, pad=2)
        axx.tick_params(labelsize=5.5, length=2, pad=1)
        axx.set_xlabel("FPR", fontsize=6, labelpad=1)
        for side in ("top", "right"):
            axx.spines[side].set_visible(False)
    axes[0].set_ylabel("TPR", fontsize=6, labelpad=1)
    axes[0].legend(frameon=False, fontsize=5, loc="center right", handlelength=1.0, borderaxespad=0.1)
    fig.tight_layout(w_pad=0.3)
    fig.savefig(out / "fig_roc.pdf")
    fig.savefig(out / "fig_roc.png", dpi=200)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
