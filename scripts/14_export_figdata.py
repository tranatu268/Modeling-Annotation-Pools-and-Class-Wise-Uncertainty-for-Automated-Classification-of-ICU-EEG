# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Export the source data of the paper figures as tidy CSV files for the MATLAB plotting scripts (matlab/).

The numbers are computed exactly as in 12_figures.py / 07_expert_agreement.py, from the FROZEN development runs
(out-of-fold predictions, >=10-vote events). MATLAB only draws; it never recomputes a statistic.

    python scripts\\14_export_figdata.py            # -> work/figdata/*.csv
Files:
  pool_shifts_fits.csv   pool, method, backbone, seed, fold, class, shift, input_sd   (one row per fit and class)
  pool_shifts_raw.csv    pool, class, raw_logratio                                     (centred prevalence log-ratio)
  reliability_bins.csv   method, backbone, class, bin, mean_pred, mean_votes, n
  reliability_ce.csv     method, backbone, class, classwise_ce
  conformal.csv          model, alpha, mode, coverage, set_size, defer_rate, cov_<class>
  roc_curves.csv         method, backbone, class, fpr, tpr        (each curve resampled on a fixed FPR grid)
  roc_auc.csv            method, backbone, class, auroc
  expert_points.csv      class, fpr_expert, tpr_expert
  noise_floor_by_n.csv   n_bin, n_lo, n_hi, events, floor_eb, floor_asym, method, kl, kl_lo, kl_hi
  eeg_examples_meta.csv  example, event_id, pool, n, votes_*, p0_*, p1_* (case C only), inset_*, deferred
  eeg_case_<A|B|C>.csv   t (s, 0 = centre of the 10-s labelled epoch), 18 bipolar channels in uV
  pr_curves.csv          method, backbone, class, recall, precision  (interpolated precision on a recall grid)
  pr_auc.csv             method, backbone, class, auprc, prevalence
  expert_pr_points.csv   class, tpr_expert, ppv_expert
  confusion.csv          who (expert | model), norm (row | col), consensus, decision, pct
Select parts with --parts, e.g.  --parts pr,confusion  (default: all).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import agreement as ag  # noqa: E402
from hms import conformal, load_config  # noqa: E402
from hms.data import BIPOLAR, CLASSES, VOTES  # noqa: E402
from hms.results import PCOLS, load_preds  # noqa: E402

SEEDS = (42, 1, 2)
RUNS = {
    ("rate", "cnn1d"): "cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s{}",
    ("rate", "spec2d"): "spec2d-tf_efficientnetv2_b0_rate_k1000_mx0.4_rlr0.001_ema0.999_e20_last_s{}",
    ("pheads", "cnn1d"): "cnn1d_pheads_k1000_mx0.2_ema0.999_am_e30_last_s{}",
    ("pheads", "spec2d"): "spec2d-tf_efficientnetv2_b0_pheads_k1000_mx0.4_ema0.999_e20_last_s{}",
    ("kl", "cnn1d"): "cnn1d_kl_mx0.2_ema0.999_am_e25_last_s{}",
    ("kl", "spec2d"): "spec2d-tf_efficientnetv2_b0_kl_mx0.4_ema0.999_last_s{}",
}
# models shown in the reliability and ROC figures (seed 42), in plotting order
SHOWN = [("kl", "spec2d"), ("pheads", "spec2d"), ("rate", "cnn1d")]
ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
FPR_GRID = np.round(np.linspace(0, 1, 1001), 4)


def pool_shifts(cfg, results):
    import torch
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet")
    dev = ev[ev.fold >= 0]
    share = np.stack([(dev[dev.stratum == s][VOTES].values / dev[dev.stratum == s].n.values[:, None]).mean(0)
                      for s in range(3)])
    raw = np.log(share[1:]) - np.log(share[:1])
    raw -= raw.mean(1, keepdims=True)  # centred like mu
    raw_rows = [{"pool": p + 1, "class": CLASSES[k], "raw_logratio": raw[p, k]} for p in range(2) for k in range(6)]
    rows = []
    for (meth, bb), pat in RUNS.items():
        if meth == "kl":
            continue
        for s in SEEDS:
            for f in sorted((results / pat.format(s)).glob("fold*")):
                fold = int(f.name.replace("fold", ""))
                if meth == "rate":
                    mu = torch.load(f / "rate_params.pt")["mu"].numpy()
                    vals = {p: (mu[p] - mu[p].mean(), np.full(6, np.nan)) for p in (1, 2)}
                else:
                    d = torch.load(f / "pool_shift.pt")
                    vals = {p: (d["mean"][p - 1].numpy(), d["std"][p - 1].numpy()) for p in (1, 2)}
                for p, (m, sd) in vals.items():
                    for k in range(6):
                        rows.append({"pool": p, "method": meth, "backbone": bb, "seed": s, "fold": fold,
                                     "class": CLASSES[k], "shift": float(m[k]), "input_sd": float(sd[k])})
    return pd.DataFrame(rows), pd.DataFrame(raw_rows)


def load_dev(cfg, results, names):
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    preds = {r: load_preds(results, r, "val") for r in names}
    ids = ev.index[ev.stratum == 0]
    for p in preds.values():
        ids = ids.intersection(p.index)
    c = ev.loc[ids, VOTES].values.astype(int)
    return c, {r: p.loc[ids, PCOLS].values for r, p in preds.items()}, ev.loc[ids, "patient_id"].values


def reliability(c, P, bins=10):
    t = c / c.sum(1, keepdims=True)
    rows, ce_rows = [], []
    for (meth, bb), p in P.items():
        for k in range(6):
            groups = np.array_split(np.argsort(p[:, k]), bins)
            mp = np.array([p[g, k].mean() for g in groups])
            mt = np.array([t[g, k].mean() for g in groups])
            w = np.array([len(g) for g in groups]) / len(p)
            for b in range(bins):
                rows.append({"method": meth, "backbone": bb, "class": CLASSES[k], "bin": b + 1,
                             "mean_pred": mp[b], "mean_votes": mt[b], "n": len(groups[b])})
            ce_rows.append({"method": meth, "backbone": bb, "class": CLASSES[k],
                            "classwise_ce": float((w * np.abs(mp - mt)).sum())})
    return pd.DataFrame(rows), pd.DataFrame(ce_rows)


def conformal_curves(c, P, pats, repeats):
    t = c / c.sum(1, keepdims=True)
    rows = []
    for (meth, bb), p in P.items():
        for al in ALPHAS:
            for mode in ("marginal", "label"):
                r = conformal.run(p, t, pats, al, repeats, mode=mode)
                r = {k.replace("coverage_class", "cov_"): v for k, v in r.items()}
                r = {(f"cov_{CLASSES[int(k[4:])]}" if k.startswith("cov_") else k): v for k, v in r.items()}
                rows.append({"model": f"{meth}_{bb}", "alpha": al, "mode": mode, **r})
    return pd.DataFrame(rows)


def roc_data(c, P):
    tab = ag.loo_table(c, next(iter(P.values())))  # expert / consensus columns do not depend on the model
    ev_i, cons, w = tab[:, 0].astype(int), tab[:, 2].astype(int), tab[:, 4]
    curves, aucs, experts = [], [], []
    for k in range(6):
        pos = cons == k
        dec = tab[:, 1].astype(int) == k
        experts.append({"class": CLASSES[k], "fpr_expert": (w * dec * ~pos).sum() / (w * ~pos).sum(),
                        "tpr_expert": (w * dec * pos).sum() / (w * pos).sum()})
        for (meth, bb), p in P.items():
            fpr, tpr, _ = roc_curve(pos, p[ev_i, k], sample_weight=w)
            tpr_grid = np.interp(FPR_GRID, fpr, tpr)  # right-continuous step curve sampled on a fixed grid
            curves += [{"method": meth, "backbone": bb, "class": CLASSES[k], "fpr": f, "tpr": v}
                       for f, v in zip(FPR_GRID, tpr_grid)]
            aucs.append({"method": meth, "backbone": bb, "class": CLASSES[k],
                         "auroc": roc_auc_score(pos, p[ev_i, k], sample_weight=w)})
    return pd.DataFrame(curves), pd.DataFrame(aucs), pd.DataFrame(experts)


REC_GRID = np.round(np.linspace(0, 1, 501), 4)


def pr_data(c, P):
    """One-vs-rest precision-recall against the leave-one-out consensus (same weighted rows as the ROC), with the
    average expert's (recall, precision). Precision is interpolated: the maximum precision at recall >= r."""
    tab = ag.loo_table(c, next(iter(P.values())))
    ev_i, cons, w = tab[:, 0].astype(int), tab[:, 2].astype(int), tab[:, 4]
    curves, aucs, experts = [], [], []
    for k in range(6):
        pos = cons == k
        dec = tab[:, 1].astype(int) == k
        experts.append({"class": CLASSES[k], "tpr_expert": (w * dec * pos).sum() / (w * pos).sum(),
                        "ppv_expert": (w * dec * pos).sum() / (w * dec).sum()})
        for (meth, bb), p in P.items():
            prec, rec, _ = precision_recall_curve(pos, p[ev_i, k], sample_weight=w)
            order = np.argsort(rec)
            rec, prec = rec[order], np.maximum.accumulate(prec[order][::-1])[::-1]  # interpolated precision
            grid = np.interp(REC_GRID, rec, prec)
            curves += [{"method": meth, "backbone": bb, "class": CLASSES[k], "recall": r, "precision": v}
                       for r, v in zip(REC_GRID, grid)]
            aucs.append({"method": meth, "backbone": bb, "class": CLASSES[k],
                         "auprc": average_precision_score(pos, p[ev_i, k], sample_weight=w),
                         "prevalence": (w * pos).sum() / w.sum()})
    return pd.DataFrame(curves), pd.DataFrame(aucs), pd.DataFrame(experts)


def confusion_data(c, p):
    """Weighted confusion of individual expert votes and of the model's top class against the leave-one-out
    consensus of the other votes; row- (recall) and column-normalised (precision) percentages."""
    tab = ag.loo_table(c, p)
    rows = []
    for who in ("expert", "model"):
        M = ag.confusion(tab, who)
        for norm, D in (("row", 100 * M / M.sum(1, keepdims=True)), ("col", 100 * M / M.sum(0, keepdims=True))):
            rows += [{"who": who, "norm": norm, "consensus": CLASSES[i], "decision": CLASSES[j], "pct": D[i, j]}
                     for i in range(6) for j in range(6)]
    return pd.DataFrame(rows)


def noise_floor_by_n(cfg, results, B=300, seed=0):
    """Empirical-Bayes noise floor (from 03_noise_floor.py) and model KL per vote-count bin, same events.
    Model KL is the per-event KL averaged over the three seeds; 95% CI by patient bootstrap within the bin."""
    from hms.metrics import kaggle_kl
    pe = pd.read_csv(cfg["work_dir"] / "noise_floor" / "per_event.csv")
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    pe["patient"] = ev.loc[pe.event_id, "patient_id"].values
    t = ev.loc[pe.event_id, VOTES].values.astype(float)
    t /= t.sum(1, keepdims=True)
    for key in [("kl", "spec2d"), ("pheads", "spec2d"), ("kl", "cnn1d"), ("pheads", "cnn1d")]:
        kls = [kaggle_kl(t, load_preds(results, RUNS[key].format(sd), "val").loc[pe.event_id, PCOLS].values)
               for sd in SEEDS]
        pe[f"kl_{key[0]}_{key[1]}"] = np.mean(kls, 0)
    edges = list(range(10, 20)) + [20]
    pe["n_bin"] = np.where(pe.n >= 20, "20+", pe.n.astype(str))
    rng = np.random.default_rng(seed)
    rows = []
    for lo in edges:
        d = pe[pe.n >= 20] if lo == 20 else pe[pe.n == lo]
        if d.empty:
            continue
        pats = d.patient.unique()
        groups = [d.index[d.patient == q].values for q in pats]
        base = {"n_bin": "20+" if lo == 20 else str(lo), "n_lo": lo, "n_hi": int(d.n.max()), "events": len(d),
                "floor_eb": d.empirical_bayes.mean(), "floor_asym": d.asymptotic.mean()}
        for col in [c for c in pe.columns if c.startswith("kl_")]:
            boots = [pe.loc[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]), col].mean()
                     for _ in range(B)]
            rows.append({**base, "method": col[3:], "kl": d[col].mean(),
                         "kl_lo": np.percentile(boots, 2.5), "kl_hi": np.percentile(boots, 97.5)})
    return pd.DataFrame(rows)


def eeg_examples(cfg, results, alpha=0.2, device=None):
    """Three illustrative development events for the supplementary figure, chosen by fixed rules (no hand picking)
    from the out-of-fold predictions of the spectrogram pool-head model (seed 42):
      A  confident and correct: >=10 votes, set = {majority class}, >=80% of votes agree, class LPD or GPD,
         highest predicted probability of that class;
      B  deferred: >=10 votes, set has >=2 classes incl. seizure or LPD, the two largest vote shares are both
         >=30% and include seizure or LPD; the event with most votes;
      C  pool shift: 3-7 votes, all for seizure; among the 50 events with the lowest pool-0 seizure probability,
         the one with the largest gap between the pool-1 head and the pool-0 head for seizure.
    Label-conditional thresholds are calibrated on all >=10-vote development events (illustration only)."""
    import torch
    from hms.train import load_fold_model
    run = RUNS[("pheads", "spec2d")].format(42)
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    pr = load_preds(results, run, "val")
    dev = ev.loc[pr.index]
    P = pr[PCOLS].values
    t = dev[VOTES].values.astype(float)
    n = t.sum(1)
    t = t / n[:, None]
    hi = (dev.stratum == 0).values
    q = conformal.calibrate(P[hi], t[hi], dev.patient_id.values[hi], alpha, mode="label")
    sets = conformal.predict_sets(P, q, mode="label")
    size = sets.sum(1)
    risky = sets[:, conformal.HIGH_RISK].any(1)
    deferred = (size >= 2) & risky
    top = t.argmax(1)

    def pick(cand, score):
        idx = np.flatnonzero(cand)
        return None if idx.size == 0 else idx[np.argmax(score[idx])]

    iA = pick(hi & (size == 1) & sets[np.arange(len(t)), top] & (t.max(1) >= 0.8) & np.isin(top, [1, 2]),
              P[np.arange(len(t)), top])
    srt = np.sort(t, 1)
    two = np.argsort(-t, 1)[:, :2]
    iB = pick(hi & deferred & (srt[:, -2] >= 0.3) & np.isin(two, conformal.HIGH_RISK).any(1), n)
    candC = np.flatnonzero((dev.stratum == 1).values & (dev.seizure_vote.values == n))
    candC = candC[np.argsort(P[candC, 0])[:50]]

    X = np.load(cfg["work_dir"] / "windows.npy", mmap_mode="r")
    win = int(cfg["win_s"] * cfg["fs_out"])
    s0 = (X.shape[2] - win) // 2

    def sig(i):  # model input window of event row i, back to uV
        return np.asarray(X[dev.index[i], :, s0:s0 + win], dtype=np.float32) * 32

    p1 = {}
    for fold in sorted(set(dev.fold.values[candC])):
        model, _, _, dev_t, _ = load_fold_model(results / run / f"fold{fold}", device)
        model.eval()
        idx = candC[dev.fold.values[candC] == fold]
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(np.stack([sig(i) / 32 for i in idx])).to(dev_t))
        prob = logits.float().softmax(-1).cpu().numpy()  # B x pools x K
        for i, pr_i in zip(idx, prob):
            p1[i] = pr_i[1]
    iC = max(p1, key=lambda i: p1[i][0] - P[i, 0]) if p1 else None

    meta, chans = [], ["%s_%s" % ab for ab in BIPOLAR]
    tt = np.arange(win) / cfg["fs_out"] - cfg["win_s"] / 2
    for case, i in (("A", iA), ("B", iB), ("C", iC)):
        if i is None:
            print(f"  case {case}: no event satisfies the selection rule, skipped")
            continue
        row = {"example": case, "event_id": int(dev.index[i]), "pool": int(dev.stratum.values[i]), "n": int(n[i]),
               "deferred": int(deferred[i])}
        for k, c in enumerate(CLASSES):
            row[f"votes_{c}"] = int(round(t[i, k] * n[i]))
            row[f"p0_{c}"] = P[i, k]
            row[f"p1_{c}"] = p1[i][k] if case == "C" else np.nan
            row[f"inset_{c}"] = int(sets[i, k])
        meta.append(row)
        pd.DataFrame(np.column_stack([tt, sig(i).T]), columns=["t"] + chans).round(3).to_csv(
            cfg["work_dir"] / "figdata" / f"eeg_case_{case}.csv", index=False)
    return pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--repeats", type=int, default=50, help="conformal calibration/evaluation splits")
    ap.add_argument("--skip_shifts", action="store_true", help="skip the pool-shift export (needs torch)")
    ap.add_argument("--skip_eeg", action="store_true", help="skip the EEG examples (needs torch + windows.npy)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--parts", default="all",
                    help="comma list of: shifts,reliability,conformal,roc,floor,eeg,pr,confusion (default all)")
    a = ap.parse_args()
    parts = {"shifts", "reliability", "conformal", "roc", "floor", "eeg", "pr", "confusion"} \
        if a.parts == "all" else set(a.parts.split(","))
    cfg = load_config(a.config)
    results = cfg["work_dir"] / "results"
    out = cfg["work_dir"] / "figdata"
    out.mkdir(parents=True, exist_ok=True)

    if "shifts" in parts and not a.skip_shifts:
        fits, raw = pool_shifts(cfg, results)
        fits.to_csv(out / "pool_shifts_fits.csv", index=False)
        raw.to_csv(out / "pool_shifts_raw.csv", index=False)
        print(f"pool shifts: {len(fits)} rows", flush=True)

    names = {key: RUNS[key].format(42) for key in SHOWN}
    c, Pn, pats = load_dev(cfg, results, list(names.values()))
    P = {key: Pn[names[key]] for key in SHOWN}
    print(f">=10-vote development events: {len(c)}, patients: {len(set(pats))}", flush=True)

    if "reliability" in parts:
        bins, ce = reliability(c, P)
        bins.to_csv(out / "reliability_bins.csv", index=False)
        ce.to_csv(out / "reliability_ce.csv", index=False)
        print("reliability done", flush=True)

    if "conformal" in parts:
        conformal_curves(c, {("pheads", "spec2d"): P[("pheads", "spec2d")]}, pats, a.repeats).to_csv(
            out / "conformal.csv", index=False)
        print("conformal done", flush=True)

    if "roc" in parts:
        curves, aucs, experts = roc_data(c, P)
        curves.to_csv(out / "roc_curves.csv", index=False)
        aucs.to_csv(out / "roc_auc.csv", index=False)
        experts.to_csv(out / "expert_points.csv", index=False)
        print("ROC done", flush=True)

    if "pr" in parts:
        curves, aucs, experts = pr_data(c, P)
        curves.to_csv(out / "pr_curves.csv", index=False)
        aucs.to_csv(out / "pr_auc.csv", index=False)
        experts.to_csv(out / "expert_pr_points.csv", index=False)
        print("PR done", flush=True)

    if "confusion" in parts:
        confusion_data(c, P[("pheads", "spec2d")]).to_csv(out / "confusion.csv", index=False)
        print("confusion done", flush=True)

    if "floor" in parts:
        noise_floor_by_n(cfg, results).to_csv(out / "noise_floor_by_n.csv", index=False)
        print("noise floor by n done", flush=True)

    if "eeg" in parts and not a.skip_eeg:
        meta = eeg_examples(cfg, results, device=a.device)
        meta.to_csv(out / "eeg_examples_meta.csv", index=False)
        print(meta[["example", "event_id", "pool", "n", "deferred"]].to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
