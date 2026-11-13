# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Table II (main rows, 3 seeds) and Table III (ablation ladder) for the FROZEN configurations.

For every (backbone, method): metrics of each seed's out-of-fold (or locked-test) predictions on >=10-vote events,
mean and SD over seeds; difference of the seed-averaged metric against two-stage training with a paired
patient-level bootstrap (the same patient resample for every method and seed). Holm correction across the
method comparisons of each backbone is applied to the KL p-values.

    python scripts\\11_tables.py                 # development cross-validation (out-of-fold)
    python scripts\\11_tables.py --split test    # locked internal test: run ONCE, after the paper text is frozen
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.data import VOTES  # noqa: E402
from hms.metrics import fast_metrics as all_metrics  # noqa: E402  (numpy version, identical values, ~20x faster)
from hms.results import PCOLS, load_preds  # noqa: E402

SEEDS = (42, 1, 2)
C1 = "cnn1d_{}_mx0.2_ema0.999_am_e{}_last_s{}"
S2 = "spec2d-tf_efficientnetv2_b0_{}_mx0.4_ema0.999{}_last_s{}"
FROZEN = {  # backbone -> method -> run names (one per seed)
    "cnn1d": {
        "two-stage": [C1.format("kl", 25, s) for s in SEEDS],
        "vote-weighted KL": [C1.format("kl_w", 30, 42)],
        "DiVo (learned kappa)": [C1.format("divo", 30, 42)],
        "RaTE": [f"cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s{s}" for s in SEEDS],
        "pool-heads": [C1.format("pheads_k1000", 30, s) for s in SEEDS],
    },
    "spec2d": {
        "two-stage": [S2.format("kl", "", s) for s in SEEDS],
        "vote-weighted KL": [S2.format("kl_w", "_e20", 42)],
        "DiVo (learned kappa)": [S2.format("divo", "_e20", 42)],
        "RaTE": [f"spec2d-tf_efficientnetv2_b0_rate_k1000_mx0.4_rlr0.001_ema0.999_e20_last_s{s}" for s in SEEDS],
        "pool-heads": [S2.format("pheads_k1000", "_e20", s) for s in SEEDS],
    },
}
SHOW = ["kl", "bal_acc", "cohen_kappa", "macro_f1", "ece_soft", "brier", "risk", "recall_lrda", "auroc_seizure"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--ref", default="two-stage")
    a = ap.parse_args()
    cfg = load_config(a.config)
    results = cfg["work_dir"] / "results"
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet").set_index("event_id")
    rows = []
    for backbone, methods in FROZEN.items():
        preds = {r: load_preds(results, r, a.split) for runs in methods.values() for r in runs}
        ids = ev.index[ev.stratum == 0]
        for p in preds.values():
            ids = ids.intersection(p.index)
        t = ev.loc[ids, VOTES].values.astype(float)
        t /= t.sum(1, keepdims=True)
        P = {r: p.loc[ids, PCOLS].values for r, p in preds.items()}
        pats = ev.loc[ids, "patient_id"].values
        uniq, inv = np.unique(pats, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
        print(f"[{backbone}] {a.split}: {len(ids)} events, {len(uniq)} patients", flush=True)

        def seed_mean(method, idx):
            ms = [all_metrics(t[idx], P[r][idx]) for r in methods[method]]
            return {k: float(np.mean([m[k] for m in ms])) for k in SHOW}

        full = np.arange(len(ids))
        point = {m: seed_mean(m, full) for m in methods}
        per_seed = {m: [all_metrics(t, P[r]) for r in methods[m]] for m in methods}
        rng = np.random.default_rng(0)
        boots = {m: [] for m in methods}
        for _ in range(a.B):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
            for m in methods:
                boots[m].append(seed_mean(m, idx))
        for m in methods:
            for k in SHOW:
                b = np.array([x[k] for x in boots[m]])
                row = {"backbone": backbone, "method": m, "seeds": len(methods[m]), "metric": k,
                       "mean": point[m][k], "sd_seeds": float(np.std([s[k] for s in per_seed[m]], ddof=1))
                       if len(per_seed[m]) > 1 else np.nan,
                       "lo": np.nanpercentile(b, 2.5), "hi": np.nanpercentile(b, 97.5)}
                if m != a.ref:
                    d = b - np.array([x[k] for x in boots[a.ref]])
                    row.update(diff=point[m][k] - point[a.ref][k], diff_lo=np.nanpercentile(d, 2.5),
                               diff_hi=np.nanpercentile(d, 97.5),
                               p=min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))
                rows.append(row)
    table = pd.DataFrame(rows)
    for bb in FROZEN:
        sel = (table.backbone == bb) & (table.metric == "kl") & table.p.notna()
        table.loc[sel, "p_holm"] = holm(table.loc[sel, "p"].values)
    out = cfg["work_dir"] / "results" / f"tables_{a.split}.csv"
    table.to_csv(out, index=False)

    def fmt(r):
        s = f"{r['mean']:.4f}" + (f" ± {r['sd_seeds']:.4f}" if r["seeds"] > 1 else "")
        if pd.notna(r.get("diff")):
            s += f"  (Δ {r['diff']:+.4f} [{r['diff_lo']:+.4f}, {r['diff_hi']:+.4f}])"
        return s
    table["cell"] = table.apply(fmt, axis=1)
    for bb in FROZEN:
        print(f"\n=== {bb} ({a.split}) — mean ± SD over seeds; Δ vs {a.ref} with patient-bootstrap 95% CI ===")
        print(table[table.backbone == bb].pivot_table(index="method", columns="metric", values="cell",
                                                      aggfunc="first").reindex(list(FROZEN[bb]))[SHOW].to_string())
        k = table[(table.backbone == bb) & (table.metric == "kl") & table.p.notna()]
        print("KL p (raw / Holm):", {m: f"{p:.3f} / {q:.3f}" for m, p, q in zip(k.method, k.p, k.p_holm)})
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
