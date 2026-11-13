# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Re-analyses asked for in the internal review of 2026-11-12. No training: saved predictions only.

    python scripts\\18_review_analyses.py                      # all parts, B = 10000
    python scripts\\18_review_analyses.py --parts main paired  # some parts

Parts (output: work/results/review/*.csv, and a printed summary):
  main      decision-level cost (Bayes decision under the cost matrix), seizure sensitivity, cost-2 errors,
            KL with B >= 10000, p = (r+1)/(B+1), Holm, patient-weighted KL, leave-one-large-patient-out,
            two-level (seed, patient) bootstrap, KL difference by number of votes; expert benchmark
  paired    recordings labelled by pool 0 AND pool 1: votes of windows <= 5 s apart, and KL of each model's
            pi_0 at the pool-1 window to the votes of the matched >=10-vote panel
  shift     counterfactual standardization of the pool effect (replaces the "60-77%" split)
  conformal deployment evaluation (thresholds from the development folds applied to the locked test),
            variability over calibration splits, patient-level n_eff, subsample-once, empty sets,
            safety of the reported (non-deferred) stream
  icare     deferral by amplitude quintile and time since ROSC (no outcome read)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.conformal import HIGH_RISK, calibrate, evaluate_sets, patient_weights, predict_sets  # noqa: E402
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.metrics import RISK, kaggle_kl  # noqa: E402
from hms.results import PCOLS  # noqa: E402

SEEDS = (42, 1, 2)
C1 = "cnn1d_{}_mx0.2_ema0.999_am_e{}_last_s{}"
S2 = "spec2d-tf_efficientnetv2_b0_{}_mx0.4_ema0.999{}_last_s{}"
FROZEN = {  # same runs as 11_tables.py
    "cnn1d": {
        "two-stage": [C1.format("kl", 25, s) for s in SEEDS],
        "vote-weighted KL": [C1.format("kl_w", 30, 42)],
        "DiVo": [C1.format("divo", 30, 42)],
        "RaTE": [f"cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s{s}" for s in SEEDS],
        "pool-heads": [C1.format("pheads_k1000", 30, s) for s in SEEDS],
    },
    "spec2d": {
        "two-stage": [S2.format("kl", "", s) for s in SEEDS],
        "vote-weighted KL": [S2.format("kl_w", "_e20", 42)],
        "DiVo": [S2.format("divo", "_e20", 42)],
        "RaTE": [f"spec2d-tf_efficientnetv2_b0_rate_k1000_mx0.4_rlr0.001_ema0.999_e20_last_s{s}" for s in SEEDS],
        "pool-heads": [S2.format("pheads_k1000", "_e20", s) for s in SEEDS],
    },
}
REF = "two-stage"
MAIN3 = ("two-stage", "RaTE", "pool-heads")
CONF_RUNS = {"spec2d pool-heads": FROZEN["spec2d"]["pool-heads"], "cnn1d RaTE": FROZEN["cnn1d"]["RaTE"],
             "spec2d two-stage": FROZEN["spec2d"]["two-stage"]}


# ------------------------------------------------------------------------------------------------ helpers
def fold_preds(results, run):
    """All rows of all fold models: event_id, p_*, split, fold (test rows: one per fold model)."""
    files = sorted((results / run).glob("fold*/preds.parquet"))
    if not files:
        raise FileNotFoundError(f"no preds for run {run}")
    return pd.concat([pd.read_parquet(f) for f in files])


def split_preds(df, split):
    d = df[df.split == split]
    if split == "test":  # fold-ensemble mean, as in the manuscript
        d = d.groupby("event_id", as_index=False)[PCOLS].mean()
    return d.set_index("event_id")[PCOLS]


def pool_preds(results, run, split):
    """event_id -> (3 pools x 6) array: out-of-fold (val) or fold-ensemble mean (test). Needs 15_pool_likelihood."""
    parts = [np.load(f) for f in sorted((results / run).glob(f"fold*/pool_preds_{split}.npz"))]
    if not parts:
        raise FileNotFoundError(f"no pool_preds_{split}.npz for {run}: run scripts/15_pool_likelihood.py first")
    if split == "val":
        return pd.Series(list(np.concatenate([z["P"] for z in parts])), index=np.concatenate([z["ids"] for z in parts]))
    return pd.Series(list(np.mean([z["P"] for z in parts], 0)), index=parts[0]["ids"])


def targets(ev, ids):
    c = ev.loc[ids, VOTES].values.astype(float)
    return c, c / c.sum(1, keepdims=True)


def boot_weights(pats, B, rng):
    """(B x patients) multinomial resampling counts, and the patient index of every event."""
    uniq, inv = np.unique(pats, return_inverse=True)
    return rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B).astype(np.float32), inv


def psum(x, inv):
    return np.bincount(inv, weights=x, minlength=inv.max() + 1)


def boot_ratio(num, den, W, inv):
    """Bootstrap distribution of sum(num) / sum(den) over patients (den = 1 per event -> event mean)."""
    return (W @ psum(num, inv)) / (W @ psum(den, inv))


def summ(point, b):
    """Point estimate, percentile 95% CI and two-sided p = (r + 1) / (B + 1) for H0: value = 0."""
    b = b[np.isfinite(b)]
    r = min((b <= 0).sum(), (b >= 0).sum())
    return {"est": point, "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
            "p": min(1.0, 2 * (r + 1) / (len(b) + 1))}


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def decide(p, bayes):
    """Single-class decision: argmax, or the Bayes decision under the cost matrix (argmin_k sum_j p_j C_jk)."""
    return (p @ RISK).argmin(1) if bayes else p.argmax(1)


def event_stats(c, t, p):
    """Per-event numerators (dict name -> (num, den)); every statistic is sum(num) / sum(den)."""
    one = np.ones(len(t))
    out = {"kl": (kaggle_kl(t, p), one), "soft_risk": (np.einsum("nj,jk,nk->n", t, RISK, p), one)}
    hr = t[:, HIGH_RISK].sum(1)
    for name, bayes in (("bayes", True), ("argmax", False)):
        k = decide(p, bayes)
        out[f"cost_{name}"] = ((t * RISK[:, k].T).sum(1), one)
        out[f"cost2_{name}"] = (hr * np.isin(k, [4, 5]), hr)        # seizure/LPD votes answered GRDA/Other
        for j, cls in enumerate(CLASSES):                               # vote-level sensitivity of every class
            out[f"sens_{cls}_{name}"] = (t[:, j] * (k == j), t[:, j])
    return out


def expert_stats(c):
    """The same statistics for a random expert of the panel judged by another expert of the same panel."""
    n = c.sum(1)
    pair = n * (n - 1)
    t = c / n[:, None]
    cc = c[:, :, None] * c[:, None, :] - np.einsum("nj,jk->njk", c, np.eye(6))   # ordered pairs (true j, says k)
    out = {"cost_expert": ((cc * RISK).sum((1, 2)) / pair, np.ones(len(c))),
           "cost2_expert": (cc[:, HIGH_RISK][:, :, [4, 5]].sum((1, 2)) / pair, t[:, HIGH_RISK].sum(1))}
    for j, cls in enumerate(CLASSES):
        out[f"sens_{cls}_expert"] = (cc[:, j, j] / pair, t[:, j])
    return out


# --------------------------------------------------------------------------------------------------- main
def part_main(cfg, ev, out, B):
    results = cfg["work_dir"] / "results"
    rows, sens_rows, extra = [], [], []
    for split in ("val", "test"):
        for bb, methods in FROZEN.items():
            preds = {r: split_preds(fold_preds(results, r), split) for runs in methods.values() for r in runs}
            ids = ev.index[ev.stratum == 0]
            for p in preds.values():
                ids = ids.intersection(p.index)
            c, t = targets(ev, ids)
            pats = ev.loc[ids, "patient_id"].values
            rng = np.random.default_rng(0)
            W, inv = boot_weights(pats, B, rng)
            per_seed = {m: [event_stats(c, t, preds[r].loc[ids].values) for r in runs] for m, runs in methods.items()}
            stat = {m: {k: (np.mean([s[k][0] for s in ss], 0), ss[0][k][1]) for k in ss[0]}
                    for m, ss in per_seed.items()}
            ex = expert_stats(c)
            print(f"[{split} {bb}] {len(ids)} events, {len(set(pats))} patients", flush=True)
            names = [k for k in stat[REF] if not k.startswith("sens_") or k.startswith("sens_seizure")]
            for k in names:
                bref = boot_ratio(*stat[REF][k], W, inv)
                block = []
                for m in methods:
                    num, den = stat[m][k]
                    bm = boot_ratio(num, den, W, inv)
                    row = {"split": split, "backbone": bb, "metric": k, "method": m, "seeds": len(methods[m]),
                           "value": num.sum() / den.sum(), "v_lo": np.percentile(bm, 2.5), "v_hi": np.percentile(bm, 97.5)}
                    if m != REF:
                        d = summ(row["value"] - stat[REF][k][0].sum() / stat[REF][k][1].sum(), bm - bref)
                        row.update(d_vs_two_stage=d["est"], d_lo=d["lo"], d_hi=d["hi"], p=d["p"])
                    ek = k.rsplit("_", 1)[0] + "_expert"
                    if ek in ex:   # model minus the average expert on the same events
                        be = boot_ratio(*ex[ek], W, inv)
                        e = summ(row["value"] - ex[ek][0].sum() / ex[ek][1].sum(), bm - be)
                        row.update(expert=ex[ek][0].sum() / ex[ek][1].sum(), d_vs_expert=e["est"], e_lo=e["lo"], e_hi=e["hi"])
                    block.append(row)
                ps = [r["p"] for r in block if "p" in r]
                for r, q in zip([r for r in block if "p" in r], holm(ps)):
                    r["p_holm"] = q
                rows += block
            for m in MAIN3:   # sensitivity of every class at the default operating points (seed mean)
                for j, cls in enumerate(CLASSES):
                    sens_rows.append({"split": split, "backbone": bb, "method": m, "class": cls,
                                      **{w: stat[m][f"sens_{cls}_{w}"][0].sum() / t[:, j].sum() for w in ("argmax", "bayes")},
                                      "expert": ex[f"sens_{cls}_expert"][0].sum() / t[:, j].sum()})

            # --- sensitivity of the KL comparison to the structure of the evaluation set
            kl = {m: stat[m]["kl"][0] for m in methods}
            uniq, cnt = np.unique(pats, return_counts=True)
            big = uniq[np.argsort(-cnt)][:10]
            pw = patient_weights(pats)
            for m in ("RaTE", "pool-heads"):
                d = kl[m] - kl[REF]
                base = {"split": split, "backbone": bb, "method": m}
                extra.append({**base, "analysis": "event mean", **summ(d.mean(), boot_ratio(d, np.ones(len(d)), W, inv))})
                extra.append({**base, "analysis": "patient-weighted mean",
                              **summ((d * pw).sum() / pw.sum(), boot_ratio(d * pw, pw, W, inv))})
                keep = pats != big[0]
                Wk, invk = boot_weights(pats[keep], B, np.random.default_rng(1))
                extra.append({**base, "analysis": f"without the largest patient ({cnt.max()} events)",
                              **summ(d[keep].mean(), boot_ratio(d[keep], np.ones(keep.sum()), Wk, invk))})
                loo = [d[pats != b].mean() for b in big]
                extra.append({**base, "analysis": "leave one of the 10 largest patients out: range of the difference",
                              "est": np.mean(loo), "lo": min(loo), "hi": max(loo), "p": np.nan})
                # two-level bootstrap: seeds of both methods resampled, then patients
                Sm = np.stack([psum(s["kl"][0], inv) for s in per_seed[m]])
                Sr = np.stack([psum(s["kl"][0], inv) for s in per_seed[REF]])
                n = W @ psum(np.ones(len(d)), inv)
                pick = lambda S: np.stack([(W[b] @ S[rng.integers(0, len(S), len(S))].mean(0)) for b in range(B)])  # noqa: E731
                extra.append({**base, "analysis": "two-level bootstrap (seeds, patients)",
                              **summ(d.mean(), (pick(Sm) - pick(Sr)) / n)})
                if split == "val":
                    nv = ev.loc[ids, "n"].values
                    for lab, sel in (("10-11 votes", nv <= 11), ("12-16 votes", (nv >= 12) & (nv <= 16)), (">=17 votes", nv >= 17)):
                        extra.append({**base, "analysis": f"{lab} ({sel.sum()} events)",
                                      **summ(d[sel].mean(), boot_ratio(d * sel, sel.astype(float), W, inv))})
            if bb == "cnn1d":
                extra.append({"split": split, "backbone": "-", "method": "-", "analysis": "structure: events / patients / largest "
                              "patient share / top-10 share", "est": len(ids), "lo": len(uniq), "hi": cnt.max() / len(ids),
                              "p": np.sort(cnt)[-10:].sum() / len(ids)})
    main = pd.DataFrame(rows)
    main.to_csv(out / "decision_cost.csv", index=False)
    pd.DataFrame(sens_rows).to_csv(out / "sensitivity_default.csv", index=False)
    pd.DataFrame(extra).to_csv(out / "kl_sensitivity.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    show = main[main.method.isin(MAIN3) & main.metric.isin(["kl", "soft_risk", "cost_bayes", "cost_argmax", "cost2_bayes",
                                                              "sens_seizure_bayes", "sens_seizure_argmax"])]
    print(show[["split", "backbone", "metric", "method", "value", "d_vs_two_stage", "d_lo", "d_hi", "p", "p_holm",
                "expert", "d_vs_expert", "e_lo", "e_hi"]].round(4).to_string(index=False))
    print(pd.DataFrame(sens_rows).round(3).to_string(index=False))
    print(pd.DataFrame(extra).round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------- paired
def match_pairs(ev, max_gap=5.0):
    """Every pool-0 event with the nearest pool-1 event of the same recording (gap between labelled offsets)."""
    rows = []
    for _, g in ev[ev.stratum <= 1].groupby("eeg_id"):
        a, b = g[g.stratum == 0], g[g.stratum == 1]
        if len(a) == 0 or len(b) == 0:
            continue
        for i, ra in a.iterrows():
            gaps = [np.abs(np.subtract.outer(ra.offsets, rb.offsets)).min() for _, rb in b.iterrows()]
            j = int(np.argmin(gaps))
            rows.append({"e0": i, "e1": b.index[j], "gap": gaps[j], "center_gap": abs(ra.center_offset - b.iloc[j].center_offset),
                         "eeg_id": ra.eeg_id, "patient_id": ra.patient_id, "fold": ra.fold})
    pairs = pd.DataFrame(rows)
    pairs = pairs[pairs.gap <= max_gap].sort_values("gap").drop_duplicates("e1").drop_duplicates("e0")
    return pairs.reset_index(drop=True)


def part_paired(cfg, ev, out, B):
    results = cfg["work_dir"] / "results"
    allp = match_pairs(ev)
    rows = []
    for name, pairs in (("development", allp[allp.fold >= 0]), ("test", allp[allp.fold < 0])):
        if len(pairs) < 10:
            continue
        c0, t0 = targets(ev, pairs.e0)
        c1, t1 = targets(ev, pairs.e1)
        W, inv = boot_weights(pairs.patient_id.values, B, np.random.default_rng(0))
        one = np.ones(len(pairs))
        base = {"set": name, "pairs": len(pairs), "recordings": pairs.eeg_id.nunique(), "patients": pairs.patient_id.nunique(),
                "median_gap_s": pairs.gap.median(), "median_center_gap_s": pairs.center_gap.median()}
        for j, cls in enumerate(CLASSES):
            d = t1[:, j] - t0[:, j]
            rows.append({**base, "quantity": f"vote share {cls}: pool 0 / pool 1 / difference", "a": t0[:, j].mean(),
                         "b": t1[:, j].mean(), **summ(d.mean(), boot_ratio(d, one, W, inv))})
        same = (t0.argmax(1) == t1.argmax(1)).astype(float)
        rows.append({**base, "quantity": "same majority class", "a": same.mean(), **summ(same.mean(), boot_ratio(same, one, W, inv))})
        unan = c1[:, 0] == c1.sum(1)
        rows.append({**base, "quantity": "pool 1 unanimous seizure: pairs / panel majority seizure / panel no seizure vote",
                     "a": unan.sum(), "b": (t0[unan].argmax(1) == 0).mean(), "est": (c0[unan, 0] == 0).mean()})
        # models: does pi_0 at the pool-1 window predict what the >=10-vote panel said about (almost) the same window?
        split = "val" if name == "development" else "test"
        for bb, methods in FROZEN.items():
            kl = {}
            for m in MAIN3:
                P = [np.stack(pool_preds(results, r, split).loc[pairs.e1].values) for r in methods[m]]
                kl[m, 0] = np.mean([kaggle_kl(t0, p[:, 0]) for p in P], 0)
                kl[m, 1] = np.mean([kaggle_kl(t0, p[:, 1]) for p in P], 0)
                kl[m, "own"] = np.mean([kaggle_kl(t1, p[:, 1]) for p in P], 0)
            for m in MAIN3:
                r = {**base, "backbone": bb, "quantity": f"KL(panel votes || pi_0 at the pool-1 window), {m}", "a": kl[m, 0].mean()}
                if m != REF:
                    d = kl[m, 0] - kl[REF, 0]
                    r.update(summ(d.mean(), boot_ratio(d, one, W, inv)))
                    d1 = kl[m, 0] - kl[m, 1]
                    rows.append({**base, "backbone": bb, "quantity": f"KL(panel || pi_0) - KL(panel || pi_1), {m}",
                                 "a": kl[m, 0].mean(), "b": kl[m, 1].mean(), **summ(d1.mean(), boot_ratio(d1, one, W, inv))})
                    rows.append({**base, "backbone": bb, "quantity": f"KL(pool-1 votes || pi_1), {m}", "a": kl[m, "own"].mean()})
                rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "paired_pools.csv", index=False)
    allp.to_csv(out / "paired_pools_pairs.csv", index=False)
    print(tab.round(4).to_string(index=False))


# -------------------------------------------------------------------------------------------------- shift
def part_shift(cfg, ev, out, B):
    """Mean predicted class shares on the SAME pool-1 events under pi_1 and pi_0, and pi_0 on pool-0 events."""
    results = cfg["work_dir"] / "results"
    rows = []
    dev = ev[ev.fold >= 0]
    for bb, methods in FROZEN.items():
        for m in ("RaTE", "pool-heads"):
            P = [pool_preds(results, r, "val") for r in methods[m]]
            ids1, ids0 = dev.index[dev.stratum == 1].intersection(P[0].index), dev.index[dev.stratum == 0].intersection(P[0].index)
            A = np.mean([np.stack(p.loc[ids1].values) for p in P], 0)      # pool-1 events x 3 heads x 6
            Z = np.mean([np.stack(p.loc[ids0].values) for p in P], 0)
            t1, t0 = targets(ev, ids1)[1], targets(ev, ids0)[1]
            W, inv = boot_weights(ev.loc[ids1, "patient_id"].values, B, np.random.default_rng(0))
            one = np.ones(len(ids1))
            for j, cls in enumerate(CLASSES):
                d = A[:, 1, j] - A[:, 0, j]
                rows.append({"backbone": bb, "method": m, "class": cls, "votes_pool1": t1[:, j].mean(), "pi1_on_pool1": A[:, 1, j].mean(),
                             "pi0_on_pool1": A[:, 0, j].mean(), "pi0_on_pool0": Z[:, 0, j].mean(), "votes_pool0": t0[:, j].mean(),
                             "pool_effect": d.mean(), "pe_lo": np.percentile(boot_ratio(d, one, W, inv), 2.5),
                             "pe_hi": np.percentile(boot_ratio(d, one, W, inv), 97.5),
                             "case_mix_effect": A[:, 0, j].mean() - Z[:, 0, j].mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "pool_standardization.csv", index=False)
    print(tab.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- conformal
def n_eff_patient(t, pats):
    """Effective sample size of every class when the patient (not the event) is the unit."""
    w = patient_weights(pats)[:, None] * t
    _, inv = np.unique(pats, return_inverse=True)
    s = np.stack([psum(w[:, k], inv) for k in range(6)], 1)
    return s.sum(0) ** 2 / (s ** 2).sum(0)


def calibrate_once(p, t, pats, alpha, rng):
    """'Subsample once': one (event, vote) draw per calibration patient -> exchangeable draws, exact split conformal
    per class (threshold 1 when the class has too few draws)."""
    uniq, inv = np.unique(pats, return_inverse=True)
    order = rng.permutation(len(pats))
    first = order[np.unique(inv[order], return_index=True)[1]]            # one random event per patient
    y = np.array([rng.choice(6, p=t[i]) for i in first])
    q = np.ones(6)
    for k in range(6):
        s = np.sort(1 - p[first[y == k], k])
        r = int(np.ceil((1 - alpha) * (len(s) + 1)))
        if 0 < r <= len(s):
            q[k] = s[r - 1]
    return q


def part_conformal(cfg, ev, out, B, alphas=(0.1, 0.2), repeats=100):
    results = cfg["work_dir"] / "results"
    dep, var, safe = [], [], []
    for name, runs in CONF_RUNS.items():
        for run in runs:
            seed = int(run.rsplit("_s", 1)[1])
            df = fold_preds(results, run)
            df = df[df.event_id.isin(ev.index[ev.stratum == 0])]
            val, test = df[df.split == "val"], df[df.split == "test"]
            pv, (cv, tv), gv = val[PCOLS].values, targets(ev, val.event_id), ev.loc[val.event_id, "patient_id"].values
            for alpha in alphas:
                # (1) variability over random 50/50 patient splits of the development set
                rng = np.random.default_rng(0)
                uniq = np.unique(gv)
                reps = []
                for _ in range(repeats):
                    cal = np.isin(gv, rng.choice(uniq, len(uniq) // 2, replace=False))
                    reps.append(evaluate_sets(predict_sets(pv[~cal], calibrate(pv[cal], tv[cal], gv[cal], alpha)), tv[~cal], gv[~cal]))
                reps = pd.DataFrame(reps)
                for k in reps:
                    var.append({"model": name, "seed": seed, "alpha": alpha, "stat": k, "mean": reps[k].mean(), "sd": reps[k].std(),
                                "p05": reps[k].quantile(.05), "p95": reps[k].quantile(.95)})
                # (2) deployment: thresholds of every fold model from its own out-of-fold events, applied to its test predictions
                for mode in ("label", "marginal", "label_once"):
                    per_fold, sets_all, rows_all = [], [], []
                    for f, tf in test.groupby("fold"):
                        vf = val[val.fold == f]
                        p, (_, t), g = vf[PCOLS].values, targets(ev, vf.event_id), ev.loc[vf.event_id, "patient_id"].values
                        if mode == "label_once":
                            q = np.median([calibrate_once(p, t, g, alpha, np.random.default_rng(i)) for i in range(51)], 0)
                        else:
                            q = calibrate(p, t, g, alpha, mode)
                        sets = predict_sets(tf[PCOLS].values, q, "marginal" if mode == "marginal" else "label")
                        tt, gt = targets(ev, tf.event_id)[1], ev.loc[tf.event_id, "patient_id"].values
                        per_fold.append({**evaluate_sets(sets, tt, gt), **{f"q{k}": q[k] for k in range(6)},
                                         **{f"neff_event{k}": (patient_weights(g)[:, None] * t)[:, k].sum() ** 2
                                            / ((patient_weights(g)[:, None] * t)[:, k] ** 2).sum() for k in range(6)},
                                         **{f"neff_patient{k}": v for k, v in enumerate(n_eff_patient(t, g))}})
                        sets_all.append(sets)
                        rows_all.append(tf.event_id.values)
                    pf = pd.DataFrame(per_fold)
                    row = {"model": name, "seed": seed, "alpha": alpha, "mode": mode, **pf.mean().to_dict(),
                           **{f"{k}_minfold": pf[k].min() for k in pf if k.startswith("coverage")}}
                    # patient-bootstrap CI on the test set of the fold-averaged class coverage and deferral
                    ids = np.concatenate(rows_all)
                    S = np.concatenate(sets_all)
                    tt, gt = targets(ev, ids)[1], ev.loc[ids, "patient_id"].values
                    w = patient_weights(gt) * 5          # every test patient appears once per fold model
                    W, inv = boot_weights(gt, min(B, 2000), np.random.default_rng(0))
                    for k in range(6):
                        b = boot_ratio(w * tt[:, k] * S[:, k], w * tt[:, k], W, inv)
                        row[f"coverage_class{k}_lo"], row[f"coverage_class{k}_hi"] = np.percentile(b, [2.5, 97.5])
                    defer = ((S.sum(1) >= 2) & S[:, HIGH_RISK].any(1)).astype(float)
                    row["defer_lo"], row["defer_hi"] = np.percentile(boot_ratio(w * defer, w, W, inv), [2.5, 97.5])
                    dep.append(row)
                    if mode == "label":   # (3) what is in the reported (non-deferred) stream
                        c = ev.loc[ids, VOTES].values.astype(float)
                        n = c.sum(1)
                        agree = ((c * (c - 1)).sum(1) / (n * (n - 1)))
                        maj = tt.argmax(1)
                        rep = defer == 0
                        r = {"model": name, "seed": seed, "alpha": alpha, "reported_fraction": (w * rep).sum() / w.sum(),
                             "expert_pair_agreement_reported": (w * agree)[rep].sum() / w[rep].sum(),
                             "expert_pair_agreement_deferred": (w * agree)[~rep].sum() / w[~rep].sum()}
                        for k in HIGH_RISK:
                            m = rep & (maj == k)
                            r[f"reported_majority_{CLASSES[k]}_events"] = m.sum() / 5
                            r[f"reported_majority_{CLASSES[k]}_not_in_set"] = (~S[m, k]).mean() if m.any() else np.nan
                            r[f"{CLASSES[k]}_votes_in_reported_stream"] = (w * tt[:, k] * rep).sum() / (w * tt[:, k]).sum()
                            r[f"{CLASSES[k]}_votes_reported_and_missed"] = (w * tt[:, k] * rep * ~S[:, k]).sum() / (w * tt[:, k]).sum()
                        safe.append(r)
            print(f"conformal: {run}", flush=True)
    dep, var, safe = pd.DataFrame(dep), pd.DataFrame(var), pd.DataFrame(safe)
    dep.to_csv(out / "conformal_deployment.csv", index=False)
    var.to_csv(out / "conformal_split_variability.csv", index=False)
    safe.to_csv(out / "conformal_reported_stream.csv", index=False)
    cov = [f"coverage_class{k}" for k in range(6)]
    g = dep.groupby(["model", "alpha", "mode"], sort=False)[["coverage"] + cov + ["set_size", "defer_rate", "empty_rate"]].mean()
    print("\nDeployment (dev thresholds -> locked test), mean over seeds and fold models:\n", g.round(3).to_string())
    print("\nWorst fold model (min over folds, mean over seeds):\n",
          dep.groupby(["model", "alpha", "mode"], sort=False)[[c + "_minfold" for c in cov]].mean().round(3).to_string())
    print("\nn_eff per class, calibration folds (event level as in the code / patient level):\n",
          dep[dep["mode"] == "label"].groupby("model", sort=False)[[f"neff_event{k}" for k in range(6)] + [f"neff_patient{k}" for k in range(6)]]
          .mean().round(0).to_string())
    v = var[var.stat.str.startswith("coverage_class") | var.stat.isin(["defer_rate", "set_size", "empty_rate"])]
    print("\nDevelopment, 100 random calibration splits (mean over seeds of mean / 5th / 95th percentile):\n",
          v.groupby(["model", "alpha", "stat"], sort=False)[["mean", "p05", "p95"]].mean().round(3).to_string())
    print("\nReported stream on the test set (mean over seeds):\n", safe.groupby(["model", "alpha"], sort=False).mean(numeric_only=True)
          .drop(columns="seed").round(3).T.to_string())


# -------------------------------------------------------------------------------------------------- icare
def part_icare(cfg, ev, out, B):
    ic = cfg["work_dir"] / "icare"
    rows = []
    for f in sorted(ic.glob("preds_*_s42.parquet")):
        d = pd.read_parquet(f)
        d = d[d.qc_pass]
        model = "spec2d pool-heads" if f.name.startswith("preds_spec2d") else "cnn1d RaTE"
        d["amp"] = pd.qcut(d.sd_uv, 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
        for by in ("amp", "hour_bin"):
            for g, x in d.groupby(by, observed=True):
                rows.append({"model": model, "by": by, "group": str(g), "windows": len(x), "patients": x.patient.nunique(),
                             "median_sd_uv": x.sd_uv.median(), "defer_window_mean": x.defer.mean(),
                             "defer_patient_weighted": x.groupby("patient").defer.mean().mean(), "p_other": x.p_other.mean(),
                             "max_prob": x.max_prob.mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "icare_by_amplitude.csv", index=False)
    print(tab.round(3).to_string(index=False))


# ---------------------------------------------------------------------------------------------- ablation
ABL = {  # seed-42 ablation runs (scripts/run_ablation.ps1); references are the seed-42 main runs
    "cnn1d": {"pool 0 only": "cnn1d_kl_mx0.2_ema0.999_am_e30_last_pool0_s42", "one stage, all events": "cnn1d_kl1stage_e30_last_s42",
              "pool-heads, pool 2 merged": "cnn1d_pheads_k1000_mx0.2_ema0.999_am_e30_last_poolmerge2_s42"},
    "spec2d": {"pool 0 only": "spec2d-tf_efficientnetv2_b0_kl_mx0.4_ema0.999_e20_last_pool0_s42",
               "one stage, all events": "spec2d_kl1stage_e20_last_s42",
               "pool-heads, pool 2 merged": "spec2d-tf_efficientnetv2_b0_pheads_k1000_mx0.4_ema0.999_e20_last_poolmerge2_s42"},
}


def part_ablation(cfg, ev, out, B):
    results = cfg["work_dir"] / "results"
    rows = []
    for split in ("val", "test"):
        for bb, abl in ABL.items():
            runs = {"two-stage": FROZEN[bb]["two-stage"][0], "pool-heads": FROZEN[bb]["pool-heads"][0], **abl}
            preds = {m: split_preds(fold_preds(results, r), split) for m, r in runs.items()}
            ids = ev.index[ev.stratum == 0]
            for p in preds.values():
                ids = ids.intersection(p.index)
            c, t = targets(ev, ids)
            W, inv = boot_weights(ev.loc[ids, "patient_id"].values, B, np.random.default_rng(0))
            st = {m: event_stats(c, t, p.loc[ids].values) for m, p in preds.items()}
            for k in ("kl", "cost_bayes", "sens_seizure_bayes", "sens_seizure_argmax"):
                for m in runs:
                    row = {"split": split, "backbone": bb, "metric": k, "method": m, "value": st[m][k][0].sum() / st[m][k][1].sum()}
                    for ref in ("two-stage", "pool-heads"):
                        if m != ref:
                            d = summ(row["value"] - st[ref][k][0].sum() / st[ref][k][1].sum(),
                                     boot_ratio(*st[m][k], W, inv) - boot_ratio(*st[ref][k], W, inv))
                            row.update({f"d_vs_{ref}": d["est"], f"lo_{ref}": d["lo"], f"hi_{ref}": d["hi"], f"p_{ref}": d["p"]})
                    rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ablation.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    print(tab.round(4).to_string(index=False))




PARTS = {"main": part_main, "paired": part_paired, "shift": part_shift, "conformal": part_conformal, "icare": part_icare, "ablation": part_ablation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["work_dir"] / "results" / "review"
    out.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id", drop=False)
    for part in a.parts:
        print(f"\n================ {part} ================", flush=True)
        PARTS[part](cfg, ev, out, a.B)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
