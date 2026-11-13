# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Held-out test of the rater-pool model on the LOW-vote events (pools 1 and 2), which the main tables never use.

For every frozen run and fold, the fold model predicts its validation events of all three pools (or, with
--split test, the locked-test events, averaged over the five fold models) with the head of EVERY pool:
  pool-heads: softmax of each head;  RaTE: pi_s with fixed Monte-Carlo noise;  single-pool models: pi_0 for all.
On the events of pool s we compare the divergence KL(c/n || pi) of
  (i)  the model's own pool-s distribution pi_s           ("own head")
  (ii) the same model's pool-0 distribution pi_0          ("pool-0 head")
  (iii) the two-stage curriculum's pi_0                    (reference)
If the learned pool shifts are real properties of the raters, pi_s must predict the held-out votes of pool-s
patients better than pi_0, and better than the two-stage model.
Per-event KL is averaged over seeds; 95% CIs from a paired patient-level bootstrap; Holm over the pool-aware
comparisons (2 methods x 2 low-vote pools) of each backbone. Predictions are cached next to each fold model.

    python scripts\\15_pool_likelihood.py                  # development folds (out-of-fold)
    python scripts\\15_pool_likelihood.py --split test     # locked test: a post-hoc analysis, report it as such
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.data import VOTES, EventDataset  # noqa: E402
from hms.metrics import kaggle_kl  # noqa: E402
from hms.train import load_fold_model, predict_pools  # noqa: E402

SEEDS = (42, 1, 2)
C1 = "cnn1d_{}_mx0.2_ema0.999_am_e{}_last_s{}"
S2 = "spec2d-tf_efficientnetv2_b0_{}_mx0.4_ema0.999{}_last_s{}"
FROZEN = {  # same runs as 11_tables.py
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
POOL_AWARE = ("RaTE", "pool-heads")
REF = "two-stage"


def fold_pool_preds(fold_dir, work, ev, split, device, workers):
    """N x 3 x K pool distributions of one fold model for its val events (or the locked test), cached."""
    cache = fold_dir / f"pool_preds_{split}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["ids"], z["P"]
    model, rate, cfg, dev_t, use_amp = load_fold_model(fold_dir, device)
    df = ev[ev.fold == (cfg["fold"] if split == "val" else -1)]
    ds = EventDataset(work / "windows.npy", df, int(cfg["win_s"] * cfg["fs_out"]), train=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=workers)
    P, ids = predict_pools(model, rate, dl, dev_t, use_amp, cfg.get("M_eval", 256))
    np.savez_compressed(cache, ids=ids, P=P.astype(np.float32))
    return ids, P


def run_pool_preds(results, work, ev, run, split, device, workers):
    """event_id -> 3 x K array: out-of-fold (val) or fold-ensemble mean (test)."""
    folds = sorted((results / run).glob("fold*"))
    if not folds:
        raise FileNotFoundError(f"no fold models for {run}")
    parts = [fold_pool_preds(f, work, ev, split, device, workers) for f in folds]
    if split == "val":
        ids = np.concatenate([p[0] for p in parts])
        P = np.concatenate([p[1] for p in parts])
    else:
        ids = parts[0][0]
        assert all(np.array_equal(ids, p[0]) for p in parts)
        P = np.mean([p[1] for p in parts], 0)
    return pd.Series(list(P), index=ids)


def boot_mean(x, pats, B, rng):
    """Patient-level bootstrap of the mean of per-event values x (may be 2-D: events x series)."""
    uniq, inv = np.unique(pats, return_inverse=True)
    x = np.atleast_2d(x.T).T
    sums = np.stack([np.bincount(inv, weights=x[:, j], minlength=len(uniq)) for j in range(x.shape[1])], 1)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
    idx = rng.integers(0, len(uniq), (B, len(uniq)))
    return np.stack([sums[i].sum(0) / cnt[i].sum() for i in idx])  # B x series


def summarise(diff, pats, B, rng):
    b = boot_mean(diff, pats, B, rng)[:, 0]
    p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return diff.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), p


def holm(p):
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--backbones", default="cnn1d,spec2d")
    a = ap.parse_args()
    if a.split == "test":
        print("NOTE: locked-test analysis added after the test set was opened -> report it as post hoc.")
    cfg = load_config(a.config)
    work, results = cfg["work_dir"], cfg["work_dir"] / "results"
    ev = pd.read_parquet(work / "events.parquet")
    evi = ev.set_index("event_id")
    rng = np.random.default_rng(0)
    rows = []
    for bb in a.backbones.split(","):
        methods = FROZEN[bb]
        preds = {}
        for m, runs in methods.items():
            for r in tqdm(runs, desc=f"{bb} {m}", unit="run"):
                preds[r] = run_pool_preds(results, work, ev, r, a.split, a.device, a.workers)
        ids = None
        for s in preds.values():
            ids = s.index if ids is None else ids.intersection(s.index)
        sub = evi.loc[ids]
        for pool in (0, 1, 2):
            e = sub.index[sub.stratum == pool]
            if len(e) == 0:
                continue
            c = sub.loc[e, VOTES].values.astype(float)
            t = c / c.sum(1, keepdims=True)
            pats = sub.loc[e, "patient_id"].values

            def kl_head(run, head):
                P = np.stack(preds[run].loc[e].values)[:, head]
                return kaggle_kl(t, P)
            ref = np.mean([kl_head(r, 0) for r in methods[REF]], 0)
            print(f"[{bb}] {a.split} pool {pool}: {len(e)} events, {len(set(pats))} patients", flush=True)
            for m, runs in methods.items():
                heads = [("pool-0 head", 0)] + ([("own head", pool)] if (m in POOL_AWARE and pool > 0) else [])
                for hname, h in heads:
                    kl = np.mean([kl_head(r, h) for r in runs], 0)
                    row = {"backbone": bb, "pool": pool, "method": m, "head": hname, "seeds": len(runs),
                           "events": len(e), "patients": len(set(pats)), "kl": kl.mean()}
                    if m != REF or h != 0:
                        d, lo, hi, p = summarise(kl - ref, pats, a.B, rng)
                        row.update(d_vs_two_stage=d, d_lo=lo, d_hi=hi, p_vs_two_stage=p)
                    if hname == "own head":
                        own0 = np.mean([kl_head(r, 0) for r in runs], 0)
                        d, lo, hi, p = summarise(kl - own0, pats, a.B, rng)
                        row.update(d_vs_pool0_head=d, d0_lo=lo, d0_hi=hi, p_vs_pool0_head=p)
                    rows.append(row)
    tab = pd.DataFrame(rows)
    sel = tab["head"] == "own head"
    for bb in tab.backbone.unique():
        s = sel & (tab.backbone == bb)
        tab.loc[s, "p_holm_vs_two_stage"] = holm(tab.loc[s, "p_vs_two_stage"].values)
        tab.loc[s, "p_holm_vs_pool0_head"] = holm(tab.loc[s, "p_vs_pool0_head"].values)
    out = results / f"pool_likelihood_{a.split}.csv"
    tab.to_csv(out, index=False)

    def cell(r, d, lo, hi):
        return "" if pd.isna(r.get(d)) else f"{r[d]:+.4f} [{r[lo]:+.4f}, {r[hi]:+.4f}]"
    tab["vs two-stage"] = tab.apply(lambda r: cell(r, "d_vs_two_stage", "d_lo", "d_hi"), axis=1)
    tab["own vs pool-0 head"] = tab.apply(lambda r: cell(r, "d_vs_pool0_head", "d0_lo", "d0_hi"), axis=1)
    pd.set_option("display.width", 250)
    for (bb, pool), d in tab.groupby(["backbone", "pool"]):
        print(f"\n=== {bb}, pool {pool} ({d.events.iloc[0]} events, {d.patients.iloc[0]} patients), "
              f"KL to the held-out votes (lower is better) ===")
        print(d[["method", "head", "seeds", "kl", "vs two-stage", "own vs pool-0 head"]]
              .round(4).to_string(index=False))
    own = tab[sel][["backbone", "pool", "method", "p_vs_two_stage", "p_holm_vs_two_stage", "p_vs_pool0_head",
                    "p_holm_vs_pool0_head"]]
    print("\nOwn-head p-values (raw / Holm per backbone):\n", own.round(4).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
