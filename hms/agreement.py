# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Model-vs-expert and expert-vs-expert agreement computed exactly from anonymous vote counts.

HMS has no rater IDs, but every quantity below only needs the vote counts c_i of each event:
  * expert-expert : two distinct experts drawn at random from the same event's panel
  * model-expert  : the model's top class vs one expert drawn at random from that panel
  * leave-one-out : each expert vote is compared with the consensus of the OTHER n-1 votes, and the model is
                    compared with exactly the same leave-one-out consensuses, so humans and model face the same
                    reference (a consensus that includes the rater's own vote would flatter the experts)
All headline statistics are ratios of sums of per-event terms, so patient-level bootstrap is a matrix product."""
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

K = 6


def per_event_terms(c, p):
    """c: N x K vote counts, p: N x K model probabilities. Returns dict of N-length arrays (or N x K)."""
    c = c.astype(float)
    n = c.sum(1, keepdims=True)
    t = c / n
    yhat = np.eye(K)[p.argmax(1)]
    pair = n * (n - 1)
    return {
        "one": np.ones(len(c)),
        "ee": (c * (c - 1)).sum(1) / pair[:, 0],               # P(two random experts agree)
        "me": (yhat * t).sum(1),                               # P(model agrees with a random expert)
        "s": t,                                                # P(random expert says k)
        "b": c * (c - 1) / pair,                               # P(both experts say k)
        "nb": (n - c) * (n - c - 1) / pair,                    # P(neither says k)
        "y": yhat,                                             # model says k
        "my": yhat * t,                                        # model says k and expert says k
        "mn": (1 - yhat) * (1 - t),                            # model not k and expert not k
    }


def _flatten(terms):
    """Stack all per-event terms into one N x M matrix; return it with {name: slice or column index}."""
    cols, index, at = [], {}, 0
    for k, v in terms.items():
        if v.ndim == 1:
            cols.append(v[:, None])
            index[k] = at
            at += 1
        else:
            cols.append(v)
            index[k] = slice(at, at + v.shape[1])
            at += v.shape[1]
    return np.hstack(cols), index


def stats_from_sums(S):
    """S: dict of summed terms (scalars or length-K arrays). Returns agreement, kappa and specific agreement."""
    N = S["one"]
    pi, q = S["s"] / N, S["y"] / N
    pe_ee, pe_me = (pi ** 2).sum(), (q * pi).sum()
    out = {"agree_ee": S["ee"] / N, "agree_me": S["me"] / N,
           "kappa_ee": (S["ee"] / N - pe_ee) / (1 - pe_ee), "kappa_me": (S["me"] / N - pe_me) / (1 - pe_me)}
    pe_ee_k = pi ** 2 + (1 - pi) ** 2
    pe_me_k = q * pi + (1 - q) * (1 - pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["psa_ee"] = S["b"] / S["s"]                        # specific agreement for class k (experts)
        out["psa_me"] = 2 * S["my"] / (S["y"] + S["s"])         # specific agreement for class k (model)
        out["kappa_ee_k"] = ((S["b"] + S["nb"]) / N - pe_ee_k) / (1 - pe_ee_k)
        out["kappa_me_k"] = ((S["my"] + S["mn"]) / N - pe_me_k) / (1 - pe_me_k)
    return out


def agreement_with_ci(c, p, patients, B=1000, seed=0):
    """Point estimates, patient-bootstrap 95% CIs, and model-minus-expert differences with CIs."""
    F, index = _flatten(per_event_terms(c, p))
    uniq, inv = np.unique(patients, return_inverse=True)
    G = np.zeros((len(uniq), F.shape[1]))
    np.add.at(G, inv, F)                                       # per-patient sums

    def stats(row):
        return stats_from_sums({k: row[ix] for k, ix in index.items()})

    point = stats(G.sum(0))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B)
    boots = [stats(row) for row in W @ G]
    res = {}
    for key, val in point.items():
        arr = np.array([b[key] for b in boots])
        res[key] = {"value": val, "lo": np.nanpercentile(arr, 2.5, axis=0), "hi": np.nanpercentile(arr, 97.5, axis=0)}
    for a, b in [("agree_me", "agree_ee"), ("kappa_me", "kappa_ee"), ("psa_me", "psa_ee"), ("kappa_me_k", "kappa_ee_k")]:
        d = np.array([x[a] - x[b] for x in boots])
        res[f"{a}-minus-ee"] = {"value": point[a] - point[b], "lo": np.nanpercentile(d, 2.5, axis=0),
                                "hi": np.nanpercentile(d, 97.5, axis=0)}
    return res


def loo_table(c, p):
    """One row per (event, dropped expert class j, leave-one-out consensus m) with weight
    (c_ij / n_i) * (1 / #ties). Columns: event, expert (= j), consensus (= m), model (= argmax p), weight."""
    rows = []
    pred = p.argmax(1)
    for i, ci in enumerate(c.astype(int)):
        n = ci.sum()
        for j in np.flatnonzero(ci):
            rest = ci.copy()
            rest[j] -= 1
            winners = np.flatnonzero(rest == rest.max())
            for m in winners:
                rows.append((i, j, m, pred[i], ci[j] / n / len(winners)))
    return np.array(rows, dtype=float)


def confusion(table, who):
    """Weighted confusion matrix: rows = leave-one-out consensus, columns = expert vote or model prediction."""
    col = 1 if who == "expert" else 3
    M = np.zeros((K, K))
    np.add.at(M, (table[:, 2].astype(int), table[:, col].astype(int)), table[:, 4])
    return M


def operating_points(table, p):
    """Per class: AUROC/AUPRC of the model score against the leave-one-out consensus, and the (FPR, TPR) of the
    average expert and of the model's argmax decision on exactly the same weighted rows."""
    ev, cons, w = table[:, 0].astype(int), table[:, 2].astype(int), table[:, 4]
    out = []
    for k in range(K):
        pos = cons == k
        row = {"class": k}
        for who, col in (("expert", 1), ("model", 3)):
            dec = table[:, col].astype(int) == k
            row[f"tpr_{who}"] = (w * dec * pos).sum() / (w * pos).sum()
            row[f"fpr_{who}"] = (w * dec * ~pos).sum() / (w * ~pos).sum()
            row[f"ppv_{who}"] = (w * dec * pos).sum() / max((w * dec).sum(), 1e-12)
        row["auroc"] = roc_auc_score(pos, p[ev, k], sample_weight=w)
        row["auprc"] = average_precision_score(pos, p[ev, k], sample_weight=w)
        # is the average expert below the model's curves? model TPR at the expert's FPR, model PPV at its TPR
        fpr, tpr, _ = roc_curve(pos, p[ev, k], sample_weight=w)
        row["tpr_model_at_expert_fpr"] = float(np.interp(row["fpr_expert"], fpr, tpr))
        prec, rec, _ = precision_recall_curve(pos, p[ev, k], sample_weight=w)
        row["ppv_model_at_expert_tpr"] = float(prec[rec >= row["tpr_expert"]].max())
        out.append(row)
    return out
