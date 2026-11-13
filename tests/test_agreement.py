# Authors: Tran Anh Tu, Huynh Hoang Nhut
import numpy as np

from hms import agreement as ag

rng = np.random.default_rng(0)


def _panel(N=3000, n=15, conc=0.7):
    pi = rng.dirichlet(np.ones(6) * conc, N)
    c = np.stack([rng.multinomial(n, q) for q in pi])
    return pi, c


def test_model_that_is_one_more_expert_matches_expert_agreement():
    # a "model" that answers like one more random expert from the same population must agree with experts
    # exactly as often as experts agree with each other (both are unbiased for sum_k pi_k^2)
    pi, c = _panel()
    yhat = np.array([rng.choice(6, p=q) for q in pi])
    p = np.eye(6)[yhat] * 0.9 + 0.1 / 6
    r = ag.agreement_with_ci(c, p, np.arange(len(c)), B=200)
    d = r["agree_me-minus-ee"]
    assert d["lo"] < 0 < d["hi"] and abs(d["value"]) < 0.02


def test_oracle_model_beats_experts():
    pi, c = _panel()
    r = ag.agreement_with_ci(c, pi, np.arange(len(c)), B=200)  # argmax of the true distribution
    assert r["agree_me-minus-ee"]["lo"] > 0 and r["kappa_me-minus-ee"]["lo"] > 0


def test_unanimous_panels_give_perfect_expert_agreement():
    c = np.zeros((60, 6), int)
    c[np.arange(60), np.arange(60) % 6] = 12
    s = ag.stats_from_sums({k: v.sum(0) for k, v in ag.per_event_terms(c, np.eye(6)[np.arange(60) % 6]).items()})
    assert np.isclose(s["agree_ee"], 1) and np.isclose(s["kappa_ee"], 1) and np.allclose(s["psa_ee"], 1)


def test_loo_consensus_excludes_own_vote():
    # 6 votes A, 5 votes B: an A-voter sees 5 A vs 5 B (tie), a B-voter sees 6 A vs 4 B
    c = np.array([[6, 5, 0, 0, 0, 0]])
    tab = ag.loo_table(c, np.array([[0.5, 0.5, 0, 0, 0, 0]]))
    M = ag.confusion(tab, "expert")
    assert np.isclose(tab[:, 4].sum(), 1)
    assert np.isclose(M[0, 0], 6 / 11 * 0.5) and np.isclose(M[1, 0], 6 / 11 * 0.5)  # A-voter: half agree (tie)
    assert np.isclose(M[0, 1], 5 / 11) and np.isclose(M[1, 1], 0)                    # B-voter always disagrees


def test_operating_points_ranges():
    pi, c = _panel(N=800)
    ops = ag.operating_points(ag.loo_table(c, pi), pi)
    for o in ops:
        assert 0.5 < o["auroc"] <= 1 and 0 <= o["fpr_expert"] <= 1 and 0 <= o["tpr_expert"] <= 1
