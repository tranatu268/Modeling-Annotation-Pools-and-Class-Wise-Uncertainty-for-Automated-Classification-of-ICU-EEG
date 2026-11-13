# Authors: Tran Anh Tu, Huynh Hoang Nhut
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog

from hms.losses import RaTE, cost_acns, cost_covote, dirmult_nll, sinkhorn_cost, soft_kl

torch.manual_seed(0)


def _batch(B=64, n_max=20):
    counts = torch.distributions.Multinomial(n_max, torch.rand(6)).sample((B,)).double()
    logp = F.log_softmax(torch.randn(B, 6, dtype=torch.float64), -1)
    return counts, logp


def test_divo_limit_is_vote_weighted_cross_entropy():
    # kappa -> inf: DirMult NLL -> -sum_k c_k log p_k = n * CE(c/n, p)  (Kaggle vote weighting is the limit)
    counts, logp = _batch()
    nll = dirmult_nll(counts, logp, torch.full((len(counts),), 1e9, dtype=torch.float64))
    ce = -(counts * logp).sum(1)
    assert torch.allclose(nll, ce, rtol=1e-4, atol=1e-4)


def test_vote_weighted_kl_differs_from_ce_by_constant():
    counts, logp = _batch()
    w = counts.sum(1)
    t = counts / w[:, None]
    ent = (t * t.clamp_min(1e-12).log()).sum(1)
    assert torch.allclose(soft_kl(logp, counts, weight=w), ((-(counts * logp).sum(1) + w * ent).sum() / w.sum()))


def test_divo_gradients_finite_with_zero_counts():
    counts, _ = _batch()
    counts[:, 0] = 0
    logits = torch.randn(len(counts), 6, dtype=torch.float64, requires_grad=True)
    kappa = torch.full((len(counts),), 5.0, dtype=torch.float64, requires_grad=True)
    dirmult_nll(counts, F.log_softmax(logits, -1), kappa).mean().backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(kappa.grad).all()


def test_divo_prefers_true_distribution():
    p_true = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    counts = torch.distributions.Multinomial(15, p_true).sample((4000,))
    k = torch.full((4000,), 20.0)
    good = dirmult_nll(counts, p_true.log().expand(4000, 6), k).mean()
    bad = dirmult_nll(counts, torch.full((4000, 6), 1 / 6).log(), k).mean()
    assert good < bad


def test_rate_without_spread_is_softmax():
    rate = RaTE(M=8)
    with torch.no_grad():
        rate.L_raw.fill_(0)
        rate.L_raw.diagonal(dim1=1, dim2=2).fill_(-50)  # spread ~1e-3
    z = torch.randn(10, 6)
    out = rate(z, torch.zeros(10, dtype=torch.long))
    assert torch.allclose(out, F.log_softmax(z, -1), atol=1e-2)


def test_rate_pool0_ignores_mu():
    rate = RaTE(M=16)
    z, s = torch.randn(10, 6), torch.zeros(10, dtype=torch.long)
    g = torch.Generator().manual_seed(1)
    a = rate(z, s, generator=g)
    with torch.no_grad():
        rate.mu[0] += 5.0
    g.manual_seed(1)
    assert torch.allclose(a, rate(z, s, generator=g))


def test_rate_output_is_distribution_and_pools_differ():
    rate = RaTE(M=16)
    with torch.no_grad():
        rate.mu[1, 0] = 2.0
    z = torch.randn(10, 6)
    p0 = rate(z, torch.zeros(10, dtype=torch.long)).exp()
    p1 = rate(z, torch.ones(10, dtype=torch.long)).exp()
    assert torch.allclose(p0.sum(1), torch.ones(10), atol=1e-5)
    assert (p1[:, 0] > p0[:, 0]).all()


def test_pool_heads_start_shared_and_nest_constant_shift():
    from hms.models import Heads
    h = Heads(8, n_pools=3)
    x = torch.randn(5, 8)
    z, _ = h(x)
    assert z.shape == (5, 3, 6) and torch.allclose(z[:, 0], z[:, 1]) and torch.allclose(z[:, 0], z[:, 2])
    with torch.no_grad():
        h.delta.bias[:6] = torch.tensor([1.4, -0.2, -0.3, -0.1, 0.2, -1.0])  # a RaTE-like constant shift, pool 1
    z, _ = h(x)
    assert torch.allclose(z[:, 1] - z[:, 0], h.delta.bias[:6].expand(5, 6)) and torch.allclose(z[:, 2], z[:, 0])


def _exact_ot(a, b, C):
    K = len(a)
    A_eq = np.vstack([np.kron(np.eye(K), np.ones(K)), np.kron(np.ones(K), np.eye(K))])
    return linprog(C.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]), bounds=(0, None)).fun


def test_sinkhorn_close_to_exact_ot():
    C = cost_acns().double()
    a = torch.distributions.Dirichlet(torch.ones(6)).sample((5,)).double()
    b = torch.distributions.Dirichlet(torch.ones(6)).sample((5,)).double()
    approx = sinkhorn_cost(a, b, C, eps=0.01, iters=500)
    exact = [_exact_ot(a[i].numpy(), b[i].numpy(), C.numpy()) for i in range(5)]
    assert np.allclose(approx.numpy(), exact, atol=2e-2)


def test_costs_are_valid():
    for C in [cost_acns(), cost_covote(torch.rand(100, 6) * 10)]:
        assert torch.allclose(C, C.T) and (C.diagonal() == 0).all() and (C >= 0).all()
