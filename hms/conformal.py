# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Weighted split conformal prediction sets when the ground truth is a distribution of expert votes.

Estimand: a new patient, one of their events at random, one expert vote y ~ t at random:
    P(y in Gamma(x)) >= 1 - alpha.
Every calibration (event i, class k) pair is a weighted sample of that law with weight w_ik = t_ik / n_events(patient),
so all calibration events are used (no label sampling, no discarded events) and test metrics use the same weights.

Modes (all deployable: none needs the votes of the test event):
  marginal  : one threshold for all classes (coverage on average; rare classes can be under-covered)
  predicted : Mondrian by predicted class
  label     : label-conditional (Vovk): threshold q_k from the class-k pairs; class k enters the set when
              1 - p_k <= q_k. Coverage holds within every class - the clinical miss rate per class is <= alpha.
A group whose effective sample size n_eff = (sum w)^2 / sum w^2 is below (1 - alpha) / alpha cannot be calibrated;
its threshold is 1 (the class is always included), never a silent fallback to another group's threshold."""
import numpy as np

HIGH_RISK = [0, 1]  # seizure, lpd
MODES = ("marginal", "predicted", "label")


def patient_weights(patients):
    _, inv, cnt = np.unique(patients, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def _weighted_quantile(s, w, alpha):
    if w.sum() <= 0:
        return 1.0
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    if n_eff < (1 - alpha) / alpha:
        return 1.0
    level = min(1.0, (1 - alpha) * (1 + 1 / n_eff))  # finite-sample correction with the effective sample size
    order = np.argsort(s)
    cw = np.cumsum(w[order]) / w.sum()
    return float(s[order][min(np.searchsorted(cw, level - 1e-12), len(s) - 1)])


def calibrate(p, t, patients, alpha, mode="label"):
    """Returns a length-6 threshold vector; its meaning depends on mode (see predict_sets)."""
    wi = patient_weights(patients)
    S = 1 - p                        # score of every (event, class) pair
    W = wi[:, None] * t              # its weight
    if mode == "marginal":
        return np.full(6, _weighted_quantile(S.ravel(), W.ravel(), alpha))
    if mode == "predicted":
        g = p.argmax(1)
        return np.array([_weighted_quantile(S[g == k].ravel(), W[g == k].ravel(), alpha) for k in range(6)])
    return np.array([_weighted_quantile(S[:, k], W[:, k], alpha) for k in range(6)])


def predict_sets(p, qhat, mode="label"):
    if mode == "predicted":  # threshold of the predicted class applies to every candidate
        return (1 - p) <= qhat[p.argmax(1)][:, None]
    return (1 - p) <= qhat[None, :]  # marginal (all equal) or label-conditional (one per candidate class)


def evaluate_sets(sets, t, patients):
    """Patient-weighted: expected coverage, mean set size, deferral and empty-set rates, and per-class coverage
    over expert votes (of all votes for class k, the fraction whose label k is in the set = 1 - miss rate)."""
    w = patient_weights(patients)
    w = w / w.sum()
    cov = (sets * t).sum(1)
    defer = (sets.sum(1) >= 2) & sets[:, HIGH_RISK].any(1)
    out = {"coverage": (w * cov).sum(), "set_size": (w * sets.sum(1)).sum(), "defer_rate": (w * defer).sum(),
           "empty_rate": (w * (sets.sum(1) == 0)).sum()}
    for k in range(6):
        mass = (w * t[:, k]).sum()
        out[f"coverage_class{k}"] = (w * t[:, k] * sets[:, k]).sum() / mass if mass > 0 else np.nan
    return out


def run(p, t, patients, alpha=0.1, repeats=100, seed=0, mode="label"):
    """Repeated random 50/50 patient splits into calibration / test. Returns mean of each statistic."""
    assert mode in MODES
    rng = np.random.default_rng(seed)
    uniq = np.unique(patients)
    rows = []
    for _ in range(repeats):
        cal = np.isin(patients, rng.choice(uniq, len(uniq) // 2, replace=False))
        qhat = calibrate(p[cal], t[cal], patients[cal], alpha, mode)
        rows.append(evaluate_sets(predict_sets(p[~cal], qhat, mode), t[~cal], patients[~cal]))
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
