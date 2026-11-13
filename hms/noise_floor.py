# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Irreducible KL: expected KL(c*/n || pi) of a PERFECT model that predicts the true vote distribution pi,
when the label is only a finite panel of n votes.

Main estimator: empirical Bayes. Fit a Dirichlet prior over pi by maximising the Dirichlet-multinomial marginal
likelihood of all >=10-vote events, then average over the posterior Dir(c + alpha). In simulation it is unbiased
when pi really is Dirichlet and ~+10% when pi is a mixture (tests/test_metrics.py). The plug-in estimator depends on
its smoothing a (sensitivity range) and the asymptotic (K_eff - 1)/(2n) underestimates by 30-45%."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .metrics import kaggle_kl


def fit_dirichlet_prior(counts):
    n = counts.sum(1)

    def nll(log_a):
        a = np.exp(log_a)
        return -(gammaln(a.sum()) - gammaln(n + a.sum()) + (gammaln(counts + a) - gammaln(a)).sum(1)).sum()

    return np.exp(minimize(nll, np.zeros(counts.shape[1]), method="L-BFGS-B").x)


def floor_empirical_bayes(counts, B=300, seed=0):
    """Returns (per-event expected KL, fitted prior alpha)."""
    rng = np.random.default_rng(seed)
    alpha = fit_dirichlet_prior(counts)
    out = np.empty(len(counts))
    for i, c in enumerate(counts):
        pi = rng.dirichlet(c + alpha, B)
        sims = np.stack([rng.multinomial(int(c.sum()), p) for p in pi]) / c.sum()
        out[i] = kaggle_kl(sims, pi).mean()
    return out, alpha


def floor_plugin(counts, a=0.5, B=300, seed=0):
    """Per event: pi~ = (c + a) / (n + K a), mean KL(c*/n || pi~) for c* ~ Mult(n, pi~)."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(counts))
    for i, c in enumerate(counts):
        n = int(c.sum())
        pi = (c + a) / (n + a * len(c))
        sims = rng.multinomial(n, pi, size=B) / n
        out[i] = kaggle_kl(sims, np.broadcast_to(pi, sims.shape)).mean()
    return out


def floor_asymptotic(counts):
    """(K_eff - 1) / (2 n), K_eff = classes with >= 1 vote. Biased low; reported only as a reference."""
    return ((counts > 0).sum(1) - 1) / (2 * counts.sum(1))
