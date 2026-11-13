# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Losses: soft-label KL (baselines), DiVo (Dirichlet-multinomial), RaTE (random-threshold rater pools), Sinkhorn OT."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_kl(logp, counts, weight=None):
    """KL(c/n || p) per sample, averaged (optionally weighted). logp: B x K log-probabilities."""
    t = counts / counts.sum(1, keepdim=True)
    kl = (t * (t.clamp_min(1e-12).log() - logp)).sum(1)
    return kl.mean() if weight is None else (weight * kl).sum() / weight.sum()


def dirmult_nll(counts, logp, kappa):
    """-log DirMult(c | n, alpha = kappa * p), dropping the multinomial coefficient (constant in the parameters).
    As kappa -> inf this tends to -sum_k c_k log p_k = n * CE(c/n, p): vote-count weighting is the limiting case."""
    alpha = kappa[:, None] * logp.exp().clamp_min(1e-8)
    n, a0 = counts.sum(1), alpha.sum(1)
    ll = (torch.lgamma(a0) - torch.lgamma(n + a0)
          + (torch.lgamma(counts + alpha) - torch.lgamma(alpha)).sum(1))
    return -ll


class RaTE(nn.Module):
    """Random-threshold expert pools. A rater from pool s adds a threshold shift b ~ N(mu_s, L_s L_s^T)
    to the shared consensus logits z(x); pi_s(x) = E_b softmax(z + b). Pool 0 (>=10 votes, like the test) has mu = 0.
    The pool index is used only to pick parameters in the loss; it is never a model input."""

    def __init__(self, n_pools=3, K=6, M=32):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(n_pools, K))
        self.L_raw = nn.Parameter(torch.zeros(n_pools, K, K))
        self.M = M

    def scale_tril(self):
        d = F.softplus(torch.diagonal(self.L_raw, dim1=1, dim2=2) - 2.0) + 1e-3  # small initial spread
        return torch.tril(self.L_raw, -1) + torch.diag_embed(d)

    def forward(self, z, s, M=None, generator=None):
        """Returns log pi_s(x), B x K."""
        M = M or self.M
        mu = torch.cat([torch.zeros_like(self.mu[:1]), self.mu[1:]])  # identifiability: mu_0 = 0
        eps = torch.randn(M, *z.shape, device=z.device, dtype=z.dtype, generator=generator)
        b = mu[s] + torch.einsum("bij,mbj->mbi", self.scale_tril()[s], eps)
        b = b - b.mean(-1, keepdim=True)  # sum-to-zero: a common shift of all logits is not identifiable
        return (torch.logsumexp(F.log_softmax(z + b, -1), 0) - torch.log(torch.tensor(float(M)))).clamp_max(0)


def sinkhorn_cost(a, b, C, eps=0.05, iters=100):
    """Entropic OT cost <P, C> between rows of a and b (B x K) for ground cost C (K x K), log-domain Sinkhorn."""
    log_a, log_b = a.clamp_min(1e-12).log(), b.clamp_min(1e-12).log()
    Mk = -C / eps
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(iters):
        f = eps * (log_a - torch.logsumexp(Mk[None] + g[:, None, :] / eps, dim=2))
        g = eps * (log_b - torch.logsumexp(Mk[None] + f[:, :, None] / eps, dim=1))
    P = torch.exp(Mk[None] + (f[:, :, None] + g[:, None, :]) / eps)
    return (P * C).sum((1, 2))


def cost_acns():
    """Ground cost from ACNS 2021 axes. Order: seizure, lpd, gpd, lrda, grda, other.
    LPD/GPD/LRDA/GRDA = {lateralized, generalized} x {periodic discharges, rhythmic delta}: 0.5 per differing axis.
    Seizure sits next to LPD on the IIIC continuum. Values are a starting point to be reviewed by a neurologist."""
    C = torch.tensor([
        # sz    lpd   gpd   lrda  grda  other
        [0.00, 0.50, 0.75, 0.75, 1.00, 1.00],  # seizure
        [0.50, 0.00, 0.50, 0.50, 1.00, 1.00],  # lpd
        [0.75, 0.50, 0.00, 1.00, 0.50, 1.00],  # gpd
        [0.75, 0.50, 1.00, 0.00, 0.50, 1.00],  # lrda
        [1.00, 1.00, 0.50, 0.50, 0.00, 1.00],  # grda
        [1.00, 1.00, 1.00, 1.00, 1.00, 0.00],  # other
    ])
    return C


def cost_covote(counts):
    """Data-driven cost: 1 - cosine similarity between class columns of the soft-label matrix.
    Classes that tend to receive votes on the same samples are close. Compute on TRAINING folds only."""
    t = counts / counts.sum(1, keepdim=True)
    t = t / t.norm(dim=0, keepdim=True).clamp_min(1e-12)
    C = (1 - t.T @ t).clamp_min(0)
    C.fill_diagonal_(0)
    return C / C.max()
