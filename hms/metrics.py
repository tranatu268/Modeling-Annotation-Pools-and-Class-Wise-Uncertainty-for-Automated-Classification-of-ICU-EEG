# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Metrics for both evaluation traditions, calibration, risk-weighted error, patient-level bootstrap."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, recall_score, roc_auc_score

from .data import CLASSES


def kaggle_kl(t, p, eps=1e-15):
    """Per-sample KL(t || p) exactly as the HMS competition metric: p clipped to [eps, 1-eps], t == 0 terms are 0."""
    p = np.clip(p, eps, 1 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, t * np.log(t / p), 0.0).sum(1)


def _binned_gap(conf, hit, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return e


def ece(p, y, bins=15):
    """Top-label expected calibration error against majority labels (hard-label tradition)."""
    return _binned_gap(p.max(1), (p.argmax(1) == y).astype(float), bins)


def ece_soft(p, t, bins=15):
    """Calibration against the expert distribution: confidence max_k p_k vs the fraction of experts who chose
    the predicted class, t[argmax p]. Zero for a model that outputs the true vote distribution."""
    pred = p.argmax(1)
    return _binned_gap(p.max(1), t[np.arange(len(t)), pred], bins)


# Cost of predicting column k when the expert says row j. Missing seizure/LPD (calling them GRDA/Other) costs most.
RISK = np.array([
    # sz   lpd  gpd  lrda grda other   (predicted)
    [0.0, 0.5, 1.0, 1.0, 2.0, 2.0],  # true seizure
    [0.5, 0.0, 0.5, 0.5, 2.0, 2.0],  # true lpd
    [1.0, 0.5, 0.0, 1.0, 0.5, 1.0],  # true gpd
    [1.0, 0.5, 1.0, 0.0, 0.5, 1.0],  # true lrda
    [1.0, 1.0, 0.5, 0.5, 0.0, 0.5],  # true grda
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # true other
])


def all_metrics(t, p):
    """t: soft targets (N x 6, rows sum to 1), p: predicted probabilities. Returns a flat dict."""
    y, yhat = t.argmax(1), p.argmax(1)
    out = {
        "kl": kaggle_kl(t, p).mean(),
        "bal_acc": balanced_accuracy_score(y, yhat),
        "cohen_kappa": cohen_kappa_score(y, yhat),
        "macro_f1": f1_score(y, yhat, average="macro"),
        "weighted_f1": f1_score(y, yhat, average="weighted"),
        "ece": ece(p, y),
        "ece_soft": ece_soft(p, t),
        "brier": ((p - t) ** 2).sum(1).mean(),
        "risk": np.einsum("nj,jk,nk->n", t, RISK, p).mean(),
    }
    rec = recall_score(y, yhat, average=None, labels=range(6), zero_division=0)
    for k, c in enumerate(CLASSES):
        out[f"recall_{c}"] = rec[k]
        out[f"auroc_{c}"] = roc_auc_score(y == k, p[:, k]) if 0 < (y == k).sum() < len(y) else np.nan
    return out


def fast_metrics(t, p):
    """Numpy-only subset of all_metrics (identical values) for bootstrap loops: kl, bal_acc, cohen_kappa,
    macro_f1, ece_soft, brier, risk, recall_lrda, auroc_seizure."""
    from scipy.stats import rankdata
    y, yhat = t.argmax(1), p.argmax(1)
    C = np.bincount(y * 6 + yhat, minlength=36).reshape(6, 6).astype(float)
    tp, sup, pred = np.diag(C), C.sum(1), C.sum(0)
    n = C.sum()
    present = sup > 0
    pe = (sup * pred).sum() / n ** 2
    used = (sup + pred) > 0
    f1 = np.where(used, 2 * tp / np.maximum(2 * tp + (pred - tp) + (sup - tp), 1e-12), 0.0)
    pos = y == 0
    r = rankdata(p[:, 0])
    npos, nneg = pos.sum(), (~pos).sum()
    return {
        "kl": kaggle_kl(t, p).mean(),
        "bal_acc": (tp[present] / sup[present]).mean(),
        "cohen_kappa": (tp.sum() / n - pe) / (1 - pe),
        "macro_f1": f1[used].mean(),
        "ece_soft": ece_soft(p, t),
        "brier": ((p - t) ** 2).sum(1).mean(),
        "risk": np.einsum("nj,jk,nk->n", t, RISK, p).mean(),
        "recall_lrda": tp[3] / sup[3] if sup[3] > 0 else 0.0,
        "auroc_seizure": (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else np.nan,
    }


def patient_bootstrap(t, p, patients, B=1000, seed=0, p_ref=None):
    """Resample patients with replacement. Returns point estimates and 95% CIs; if p_ref is given, also the
    paired difference (p - p_ref) in every metric with a two-sided bootstrap p-value."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patients, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    point = all_metrics(t, p)
    keys = list(point)
    boots, diffs = [], []
    for _ in range(B):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        m = all_metrics(t[idx], p[idx])
        boots.append([m[k] for k in keys])
        if p_ref is not None:
            r = all_metrics(t[idx], p_ref[idx])
            diffs.append([m[k] - r[k] for k in keys])
    boots = np.array(boots)
    res = {k: {"value": point[k], "lo": np.nanpercentile(boots[:, i], 2.5), "hi": np.nanpercentile(boots[:, i], 97.5)}
           for i, k in enumerate(keys)}
    if p_ref is not None:
        diffs, ref = np.array(diffs), all_metrics(t, p_ref)
        for i, k in enumerate(keys):
            d = diffs[:, i]
            res[k].update(diff=point[k] - ref[k], diff_lo=np.nanpercentile(d, 2.5), diff_hi=np.nanpercentile(d, 97.5),
                          p_value=min(1.0, 2 * min(np.nanmean(d <= 0), np.nanmean(d >= 0))))
    return res
