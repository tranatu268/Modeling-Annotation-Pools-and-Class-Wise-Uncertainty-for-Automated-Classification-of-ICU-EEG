# Authors: Tran Anh Tu, Huynh Hoang Nhut
import numpy as np
from scipy.special import rel_entr

from hms import conformal
from hms.metrics import all_metrics, kaggle_kl, patient_bootstrap
from hms.noise_floor import floor_asymptotic, floor_empirical_bayes, floor_plugin

rng = np.random.default_rng(0)


def _votes(N=400, n=15):
    pi = rng.dirichlet(np.ones(6) * 0.7, N)
    c = np.stack([rng.multinomial(n, p) for p in pi]).astype(float)
    return pi, c, c / c.sum(1, keepdims=True)


def test_kaggle_kl_matches_rel_entr():
    _, _, t = _votes()
    p = rng.dirichlet(np.ones(6), len(t))
    assert np.allclose(kaggle_kl(t, p), rel_entr(t, np.clip(p, 1e-15, 1 - 1e-15)).sum(1))


def test_fast_metrics_equal_sklearn_versions():
    from hms.metrics import fast_metrics
    for N in (50, 3000):
        pi, _, t = _votes(N=N)
        p = 0.6 * pi + 0.4 * rng.dirichlet(np.ones(6), N)
        p[: N // 10] = np.eye(6)[rng.integers(0, 6, N // 10)] * 0.9 + 0.1 / 6  # ties in scores, odd predictions
        slow, fast = all_metrics(t, p), fast_metrics(t, p)
        for k, v in fast.items():
            assert np.isclose(v, slow[k], atol=1e-10), k


def test_perfect_prediction_has_zero_metrics():
    _, _, t = _votes()
    m = all_metrics(t, t)
    assert abs(m["kl"]) < 1e-12 and m["brier"] < 1e-12 and m["ece_soft"] < 1e-12


def test_soft_ece_flags_overconfidence():
    _, _, t = _votes()
    sharp = np.eye(6)[t.argmax(1)] * 0.94 + 0.01  # nearly one-hot at the majority class
    assert all_metrics(t, sharp)["ece_soft"] > 0.1


def _true_floor(pi, n, B=300):
    return np.mean([kaggle_kl(rng.multinomial(n, p, size=B) / n, np.broadcast_to(p, (B, 6))).mean() for p in pi])


def test_empirical_bayes_floor_is_unbiased_when_prior_is_dirichlet():
    for n in [10, 15, 25]:
        pi, c, _ = _votes(N=400, n=n)
        est, _ = floor_empirical_bayes(c, B=200)
        assert abs(est.mean() / _true_floor(pi, n) - 1) < 0.05


def test_empirical_bayes_floor_bias_on_mixture_is_small():
    pi = np.concatenate([rng.dirichlet(np.ones(6) * 0.1, 300), rng.dirichlet(np.ones(6) * 3, 100)])
    c = np.stack([rng.multinomial(15, p) for p in pi]).astype(float)
    ratio = floor_empirical_bayes(c, B=200)[0].mean() / _true_floor(pi, 15)
    assert 0.95 < ratio < 1.2


def test_asymptotic_floor_is_biased_low():
    pi, c, _ = _votes(N=400, n=15)
    assert floor_asymptotic(c).mean() < 0.8 * _true_floor(pi, 15) < floor_plugin(c, a=0.5).mean() * 1.2


def test_bootstrap_paired_difference_sign():
    pi, _, t = _votes()
    pats = np.repeat(np.arange(100), 4)
    noisy = 0.5 * pi + 0.5 / 6
    res = patient_bootstrap(t, pi, pats, B=50, p_ref=noisy)
    assert res["kl"]["diff"] < 0 and res["kl"]["diff_hi"] < 0


def test_conformal_coverage_when_calibrated():
    pi, _, t = _votes(N=2000)
    pats = np.arange(len(t))
    for alpha in [0.1, 0.2]:
        for mode in conformal.MODES:
            r = conformal.run(pi, t, pats, alpha=alpha, repeats=20, mode=mode)
            assert r["coverage"] > 1 - alpha - 0.03, mode


def test_label_conditional_fixes_rare_class_undercoverage():
    # imbalanced, partly ambiguous classes; class 0 is rare (like seizure in the >=10-vote stratum)
    pi = rng.dirichlet(np.array([0.15, 1, 1, 1, 1, 4.0]), 4000)
    c = np.stack([rng.multinomial(15, q) for q in pi]).astype(float)
    t = c / c.sum(1, keepdims=True)
    p = 0.7 * pi + 0.3 * pi.mean(0)  # a model shrunk toward the prior under-covers rare classes
    pats = np.arange(len(t))
    marg = conformal.run(p, t, pats, alpha=0.2, repeats=20, mode="marginal")
    lab = conformal.run(p, t, pats, alpha=0.2, repeats=20, mode="label")
    assert lab["coverage_class0"] > marg["coverage_class0"] + 0.05
    assert all(lab[f"coverage_class{k}"] > 0.8 - 0.03 for k in range(6))  # every class at nominal


def test_weighted_conformal_with_repeated_patients_and_rare_class():
    # 300 patients with 1-20 events each (like HMS), rare class 0 with only ~5% of the vote mass
    n_ev = rng.integers(1, 21, 300)
    pats = np.repeat(np.arange(300), n_ev)
    pi = rng.dirichlet(np.array([0.1, 1, 1, 1, 1, 4.0]), len(pats))
    t = np.stack([rng.multinomial(15, q) for q in pi]) / 15
    p = 0.7 * pi + 0.3 * pi.mean(0)
    r = conformal.run(p, t, pats, alpha=0.2, repeats=20, mode="label")
    assert abs(r["coverage"] - 0.8) < 0.04
    assert all(r[f"coverage_class{k}"] > 0.8 - 0.04 for k in range(6))


def test_uncalibratable_group_always_included():
    assert conformal._weighted_quantile(np.array([0.3, 0.5]), np.array([1.0, 1.0]), alpha=0.05) == 1.0
