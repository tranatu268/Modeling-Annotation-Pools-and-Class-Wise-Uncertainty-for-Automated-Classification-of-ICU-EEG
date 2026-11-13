# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Supplementary tables (LaTeX + CSV) from the frozen results. No training; reads only saved outputs.

    python scripts\\17_supp_tables.py      # -> paper/supp_tables.tex  and  work/results/supp/*.csv

  S1  per-seed metrics, development (out-of-fold, >=10-vote events)
  S2  per-seed metrics, locked internal test (fold-ensemble mean)
  S3  per-class recall, AUROC and AUPRC (development, mean over seeds)
  S4  prediction sets at every alpha, both rules, development and test
  S5  divergence to held-out votes of low-vote events, all heads, development
  S6  the same on the locked test (post hoc)
  S7  scarcity stress test (fraction of >=10-vote patients kept for training)
  S8  label-noise floor by number of votes (all estimators)
  S9  I-CARE feasibility run: deferral and top class per model, input gain and hospital (no outcome used)
Tables whose source file is missing are skipped with a message.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.metrics import all_metrics  # noqa: E402
from hms.results import PCOLS, load_preds  # noqa: E402

SEEDS = (42, 1, 2)
C1 = "cnn1d_{}_mx0.2_ema0.999_am_e{}_last_s{}"
S2 = "spec2d-tf_efficientnetv2_b0_{}_mx0.4_ema0.999{}_last_s{}"
FROZEN = {  # same runs as 11_tables.py
    "1-D CNN": {
        "Two-stage": [C1.format("kl", 25, s) for s in SEEDS],
        "Vote-weighted KL": [C1.format("kl_w", 30, 42)],
        "DiVo": [C1.format("divo", 30, 42)],
        "RaTE": [f"cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s{s}" for s in SEEDS],
        "Pool-heads": [C1.format("pheads_k1000", 30, s) for s in SEEDS],
    },
    "Spectrogram CNN": {
        "Two-stage": [S2.format("kl", "", s) for s in SEEDS],
        "Vote-weighted KL": [S2.format("kl_w", "_e20", 42)],
        "DiVo": [S2.format("divo", "_e20", 42)],
        "RaTE": [f"spec2d-tf_efficientnetv2_b0_rate_k1000_mx0.4_rlr0.001_ema0.999_e20_last_s{s}" for s in SEEDS],
        "Pool-heads": [S2.format("pheads_k1000", "_e20", s) for s in SEEDS],
    },
}
CONFORMAL_RUNS = {"Spectrogram pool-heads": FROZEN["Spectrogram CNN"]["Pool-heads"][0],
                  "1-D CNN RaTE": FROZEN["1-D CNN"]["RaTE"][0]}
LABELS = ["SZ", "LPD", "GPD", "LRDA", "GRDA", "Other"]
METHOD_NAMES = {"two-stage": "Two-stage", "vote-weighted KL": "Vote-weighted KL", "DiVo (learned kappa)": "DiVo",
                "RaTE": "RaTE", "pool-heads": "Pool-heads"}


def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def fp(p):
    return "--" if pd.isna(p) else "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ci(d, lo, hi, nd=3):
    if pd.isna(d):
        return "--"
    return f"${d:+.{nd}f}$ [${lo:+.{nd}f}$, ${hi:+.{nd}f}$]"


def table(caption, label, header, rows, colspec, note=None, size="\\footnotesize"):
    out = ["\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    if note:
        out += ["\\vspace{2pt}", "", f"\\parbox{{\\linewidth}}{{\\scriptsize {note}}}"]
    out += ["\\end{table}", ""]
    return "\n".join(out)


def targets(ev, ids):
    t = ev.loc[ids, VOTES].values.astype(float)
    return t / t.sum(1, keepdims=True)


def per_run_metrics(ev, results, split):
    rows = []
    for bb, methods in FROZEN.items():
        preds = {r: load_preds(results, r, split) for runs in methods.values() for r in runs}
        ids = ev.index[ev.stratum == 0]
        for p in preds.values():
            ids = ids.intersection(p.index)
        if len(ids) == 0:
            print(f"  {bb}: no {split} predictions with >=10 votes -> skipped")
            continue
        t = targets(ev, ids)
        y = t.argmax(1)
        for m, runs in methods.items():
            for r in runs:
                p = preds[r].loc[ids, PCOLS].values
                met = all_metrics(t, p)
                for k, c in enumerate(CLASSES):
                    met[f"auprc_{c}"] = average_precision_score(y == k, p[:, k]) if (y == k).any() else np.nan
                rows.append({"backbone": bb, "method": m, "seed": int(r.rsplit("_s", 1)[1]), "events": len(ids),
                             **met})
    return pd.DataFrame(rows)


def s1_s2(df, split):
    keys = ["kl", "bal_acc", "cohen_kappa", "macro_f1", "ece_soft", "brier", "risk"]
    rows, last = [], None
    for r in df.itertuples():
        if last is not None and r.backbone != last:
            rows.append("MID")
        rows.append([r.backbone if r.backbone != last else "", r.method, str(r.seed)]
                    + [f(getattr(r, k), 4 if k == "kl" else 3) for k in keys])
        last = r.backbone
    name = "development (out-of-fold)" if split == "val" else "locked internal test (mean of the five fold models)"
    n = int(df.events.iloc[0])
    return table(f"Per-seed results on events with at least ten votes, {name}; {n:,} events",
                 f"tab:s_{split}", ["Backbone", "Training", "Seed", "KL", "Bal.\\ acc.", "$\\kappa$", "Macro-F1",
                                    "ECE$_\\mathrm{soft}$", "Brier", "Risk"], rows, "@{}llrccccccc@{}",
                 "Balanced accuracy, $\\kappa$ and macro-F1 use the majority vote. Risk: expected cost under the "
                 "fixed asymmetric cost matrix. Lower is better for KL, ECE, Brier and risk.")


def s3(df):
    rows = []
    for metric, name in (("recall", "Recall"), ("auroc", "AUROC"), ("auprc", "AUPRC")):
        if rows:
            rows.append("MID")
        g = df.groupby(["backbone", "method"], sort=False)[[f"{metric}_{c}" for c in CLASSES]].mean()
        first, last = True, None
        for (bb, m), v in g.iterrows():
            rows.append([name if first else "", bb if bb != last else "", m] + [f(x) for x in v.values])
            first, last = False, bb
    return table("Per-class results on development events with at least ten votes (one-versus-rest against the "
                 "majority vote; mean over seeds)", "tab:s_perclass",
                 ["Metric", "Backbone", "Training"] + LABELS, rows, "@{}lll" + "c" * 6 + "@{}",
                 "Single-pool ablations (vote-weighted KL, DiVo) have one seed.", size="\\scriptsize")


def s4(results):
    rows = []
    for name, run in CONFORMAL_RUNS.items():
        for split, fn in (("Development", "conformal.csv"), ("Test", "conformal_test.csv")):
            path = results / run / fn
            if not path.exists():
                print(f"  S4: missing {path}")
                continue
            d = pd.read_csv(path)
            if rows:
                rows.append("MID")
            first = True
            for r in d.itertuples():
                cov = [getattr(r, f"cov_{c}") for c in CLASSES]
                rows.append([name if first else "", split if first else "", f"{r.alpha:g}",
                             {"marginal": "Marginal", "predicted": "Mondrian", "label": "Label-cond."}[r.mode],
                             f(r.coverage), f(r.set_size, 2), f(r.defer_rate), f(min(cov))] + [f(x) for x in cov])
                first = False
    if not rows:
        return ""
    return table("Prediction sets at every nominal level (100 random 50/50 patient splits into calibration and "
                 "evaluation halves within each split; seed 42)", "tab:s_conformal",
                 ["Model", "Split", "$\\alpha$", "Rule", "Cov.", "Size", "Defer", "Min."] + LABELS, rows,
                 "@{}llclcccc" + "c" * 6 + "@{}",
                 "Cov.: patient-weighted coverage of expert votes; class columns: coverage of the votes of each "
                 "class; Min.: lowest class coverage; Defer: fraction of segments whose set has two or more "
                 "classes including seizure or LPD.", size="\\scriptsize")


def s5(results):
    parts = []
    for split, title in (("val", "development (out-of-fold)"), ("test", "locked test (post hoc)")):
        path = results / f"pool_likelihood_{split}.csv"
        if not path.exists():
            print(f"  S5: missing {path}")
            continue
        d = pd.read_csv(path)
        d = d[d.pool > 0]
        rows, last = [], (None, None)
        for r in d.itertuples():
            key = (r.backbone, r.pool)
            if last[0] is not None and key != last:
                rows.append("MID")
            head = "$\\pi_s$" if r.head == "own head" else "$\\pi_0$"
            rows.append([{"cnn1d": "1-D CNN", "spec2d": "Spectrogram"}.get(r.backbone, r.backbone)
                         if r.backbone != last[0] else "",
                         f"{r.pool} ({r.events:,} / {r.patients:,})" if key != last else "",
                         METHOD_NAMES.get(r.method, r.method), head, f(r.kl),
                         ci(r.d_vs_two_stage, r.d_lo, r.d_hi),
                         ci(getattr(r, "d_vs_pool0_head", np.nan), getattr(r, "d0_lo", np.nan),
                            getattr(r, "d0_hi", np.nan)),
                         fp(getattr(r, "p_holm_vs_pool0_head", np.nan))])
            last = key
        parts.append(table(f"Divergence to held-out votes of low-vote events, {title}",
                           f"tab:s_heldout_{split}",
                           ["Backbone", "Pool (events / patients)", "Training", "Head", "KL",
                            "$\\Delta$ vs two-stage $\\pi_0$ [95\\% CI]", "$\\Delta$ vs own $\\pi_0$ [95\\% CI]",
                            "$p_\\mathrm{Holm}$"], rows, "@{}llllcccc@{}",
                           "KL between the votes of each event and the predicted distribution, mean over seeds. "
                           "$\\pi_s$: distribution of the event's own pool; $\\pi_0$: the high-agreement head. "
                           "$p_\\mathrm{Holm}$: own head versus own $\\pi_0$, Holm-adjusted over the four "
                           "pool-aware comparisons of each backbone.", size="\\scriptsize"))
    return "\n".join(parts)


def s6(results):
    path = results / "scarcity.csv"
    if not path.exists():
        print(f"  S6: missing {path}")
        return ""
    d = pd.read_csv(path)
    d = d[(d.backbone == "cnn1d") & d.method.isin(["two-stage", "pool-heads"])].sort_values(["frac", "method"],
                                                                                         ascending=[True, False])
    rows, last = [], None
    for r in d.itertuples():
        rows.append([f"{r.frac:.0%}".replace("%", "\\%") if r.frac != last else "", METHOD_NAMES.get(r.method),
                     str(r.seeds), f(r.kl, 4), ci(getattr(r, "d_kl", np.nan), getattr(r, "d_lo", np.nan),
                                                  getattr(r, "d_hi", np.nan), 4), f(r.risk)])
        last = r.frac
    return table("Scarcity stress test with the 1-D CNN: fraction of patients with $\\geq$10-vote events kept for "
                 f"training (all low-vote events kept); evaluation on all {int(d.events.iloc[0]):,} development events "
                 "with at least ten votes", "tab:s_scarcity", ["Kept", "Training", "Seeds", "KL", "$\\Delta$ KL [95\\% CI]",
                                                  "Risk"], rows, "@{}llcccc@{}",
                 "$\\Delta$: pool-heads minus two-stage at the same fraction, paired patient-level bootstrap; the "
                 "interval does not include variation between training runs.")


def s7(cfg):
    pe_path = cfg["work_dir"] / "noise_floor" / "per_event.csv"
    if not pe_path.exists():
        print(f"  S7: missing {pe_path}")
        return ""
    pe = pd.read_csv(pe_path)
    names = {"empirical_bayes": "Empirical Bayes", "plugin_a0.1": "Plug-in ($a=0.1$)",
             "plugin_a0.5": "Plug-in ($a=0.5$)", "asymptotic": "Asymptotic"}
    cols = [c for c in names if c in pe]
    g = pe.groupby("n")[cols].mean()
    g["events"] = pe.groupby("n").size()
    rows = [[str(n), f"{int(v.events):,}"] + [f(v[c]) for c in cols] for n, v in g.iterrows()]
    rows += ["MID", ["All", f"{len(pe):,}"] + [f(pe[c].mean()) for c in cols]]
    return table("Label-noise floor (expected KL of a model that predicts the true panel distribution exactly) by "
                 "number of votes, development events with at least ten votes", "tab:s_floor",
                 ["Votes", "Events"] + [names[c] for c in cols], rows, "@{}rr" + "c" * len(cols) + "@{}",
                 "Plug-in: votes smoothed with $a$ pseudo-counts per class; asymptotic: $(K_\\mathrm{eff}-1)/(2n)$. "
                 "The empirical-Bayes estimator is used in the main text.")


def s9(cfg):
    """I-CARE feasibility (no outcome used): deferral and predicted classes per model, input gain and hospital."""
    ic = cfg["work_dir"] / "icare"
    paths = {1: ic / "icare_summary.csv", 2: ic / "icare_summary_g2.csv"}
    if not all(q.exists() for q in paths.values()):
        print(f"  S9: missing {[str(q) for q in paths.values() if not q.exists()]}")
        return ""
    d = pd.concat([pd.read_csv(q).assign(gain=g) for g, q in paths.items()])
    d = d[d["by"].isin(["all", "hospital"])]
    name = lambda r: "Spectrogram pool-heads" if r.startswith("spec2d") else "1-D CNN RaTE" if "rate" in r else r
    d["model"] = d["run"].map(name)
    qc = ""
    pq = sorted(ic.glob("preds_*_s42.parquet"))
    if pq:
        w = pd.read_parquet(pq[0], columns=["patient", "sd_uv", "clip_frac", "qc_pass"])
        flat = int((w.sd_uv < 0.1).sum())
        qc = (f"{w.patient.nunique()} patients with EEG 12--72~h after ROSC, {len(w):,} windows; excluded: "
              f"{flat:,} flat-line ($<$0.1~$\\mu$V) and {int((~w.qc_pass).sum()) - flat:,} high-amplitude or clipped "
              f"windows. ")
    rows, last = [], None
    for m in d.model.unique():
        for g in (1, 2):
            sub = d[(d.model == m) & (d.gain == g)]
            if last is not None:
                rows.append("MID")
            first = True
            for r in sub.itertuples():
                grp = "All" if r.by == "all" else f"Hospital {r.group}"
                rows.append([m if first and g == 1 else "", f"$\\times${g}" if first else "", grp, str(r.patients),
                             f"{r.windows:,}", f(r.defer), f(r.set_size, 2), f(r.max_prob), f(r.fold_agree)]
                            + [f"{100 * getattr(r, 'argmax_' + c):.1f}" for c in CLASSES])
                first = False
            last = m
        ref = d[(d.model == m) & (d.gain == 1) & (d["by"] == "all")].iloc[0]
        rows += ["MID", ["", "", "HMS dev.", "--", "--", f(ref.hms_defer), f(ref.hms_set_size, 2), "--", "--"]
                 + ["--"] * 6]
    return table("Feasibility run on post-cardiac-arrest EEG from the I-CARE database (no outcome variable used): "
                 "label-conditional prediction sets ($\\alpha=0.2$, thresholds calibrated on the HMS development "
                 "events) and top predicted class, at the recorded amplitude ($\\times$1) and with the input "
                 "amplitude doubled ($\\times$2)", "tab:s_icare",
                 ["Model", "Gain", "Group", "Pat.", "Windows", "Defer", "Size", "Max $p$", "Fold agr."] + LABELS,
                 rows, "@{}lllrrcccc" + "c" * 6 + "@{}",
                 qc + "Defer, set size, maximum probability and fold agreement (fraction of the five fold models "
                 "whose top class equals the ensemble's) are patient-weighted means over quality-controlled "
                 "windows; class columns: percentage of windows with that top class. HMS dev.: the same "
                 "thresholds on the HMS development events with at least ten votes. Hospital C contributed no "
                 "patients. The physical amplitude scale of the I-CARE recordings is not documented in their "
                 "headers, hence the gain sensitivity analysis.", size="\\scriptsize")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="default: <repo>/paper/supp_tables.tex")
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = cfg["work_dir"] / "results"
    csv_dir = results / "supp"
    csv_dir.mkdir(exist_ok=True)
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    parts, dev = [], None
    for split in ("val", "test"):
        df = per_run_metrics(ev, results, split)
        if df.empty:
            continue
        df.to_csv(csv_dir / f"per_run_{split}.csv", index=False)
        parts.append(s1_s2(df, split))
        if split == "val":
            dev = df
        print(f"S{1 if split == 'val' else 2} done", flush=True)
    if dev is not None:
        parts.append(s3(dev))
    parts += [s4(results), s5(results), s6(results), s7(cfg), s9(cfg)]
    out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "paper" / "supp_tables.tex"
    out.write_text("% generated by scripts/17_supp_tables.py -- do not edit by hand\n\n" + "\n".join(p for p in parts if p))
    print(f"-> {out}\n-> {csv_dir}")


if __name__ == "__main__":
    main()
