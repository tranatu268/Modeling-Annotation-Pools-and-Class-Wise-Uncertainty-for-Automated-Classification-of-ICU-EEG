# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Backbones. Each returns (logits[B,6], kappa[B]); kappa is the Dirichlet concentration used by DiVo/RaTE
(ignored by the KL baselines)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Heads(nn.Module):
    """Class logits and kappa = 1 + softplus(raw) (initialised at kappa_init).
    n_pools > 1 (pool-heads): logits become B x n_pools x K with logits_s = shared(h) + delta_s(h), delta_0 = 0.
    delta_s starts at zero (and is shrunk by weight decay), so the model nests the shared head (delta = 0) and RaTE's
    constant per-pool shift (delta_s(h) = bias only)."""

    def __init__(self, d, K=6, kappa_init=10.0, n_pools=1):
        super().__init__()
        self.K, self.n_pools = K, n_pools
        self.cls, self.kappa = nn.Linear(d, K), nn.Linear(d, 1)
        nn.init.zeros_(self.kappa.weight)
        nn.init.constant_(self.kappa.bias, math.log(math.expm1(kappa_init - 1)))
        if n_pools > 1:
            self.delta = nn.Linear(d, (n_pools - 1) * K)
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

    def forward(self, h):
        z, kappa = self.cls(h), 1 + F.softplus(self.kappa(h).squeeze(1))
        if self.n_pools > 1:
            d = self.delta(h).view(len(h), self.n_pools - 1, self.K)
            z = torch.cat([z[:, None], z[:, None] + d], 1)
        return z, kappa


def _block(ci, co, stride):
    return nn.Sequential(nn.Conv1d(ci, co, 7, stride, 3, bias=False), nn.BatchNorm1d(co), nn.GELU(),
                         nn.Conv1d(co, co, 7, 1, 3, bias=False), nn.BatchNorm1d(co), nn.GELU())


class CNN1D(nn.Module):
    """Channel-shared temporal CNN (stride 50) -> per-time-step channel mixing -> BiGRU -> mean/max pooling."""

    def __init__(self, n_ch=18, width=64, hidden=128, n_pools=1):
        super().__init__()
        self.enc = nn.Sequential(_block(1, 32, 2), _block(32, width, 5), _block(width, width, 5))
        self.mix = nn.Sequential(nn.Linear(n_ch * width, 2 * hidden), nn.GELU())
        self.gru = nn.GRU(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.heads = Heads(4 * hidden, n_pools=n_pools)

    def forward(self, x):  # x: B x C x T
        B, C, T = x.shape
        h = self.enc(x.reshape(B * C, 1, T))               # (B*C) x W x T'
        h = h.reshape(B, C * h.shape[1], -1).transpose(1, 2)  # B x T' x (C*W)
        h, _ = self.gru(self.mix(h))
        return self.heads(torch.cat([h.mean(1), h.amax(1)], 1))


class Spec2D(nn.Module):
    """Log-power STFT of each bipolar channel, averaged within the 5 montage chains (LL, RL, LP, RP, central),
    stacked vertically into one image (5*F x T') for an ImageNet-pretrained timm CNN."""
    CHAINS = [range(0, 4), range(4, 8), range(8, 12), range(12, 16), range(16, 18)]

    def __init__(self, fs=100, name="efficientnet_b0", pretrained=True, fmax=40.0, drop_rate=0.0, drop_path=0.0,
                 n_pools=1):
        super().__init__()
        import timm
        self.n_fft, self.hop = 2 * fs, fs // 4          # 0.5 Hz bins, 0.25 s hop
        self.n_freq = int(fmax / (fs / self.n_fft)) + 1
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.cnn = timm.create_model(name, pretrained=pretrained, in_chans=1, num_classes=0,
                                     drop_rate=drop_rate, drop_path_rate=drop_path)
        self.heads = Heads(self.cnn.num_features, n_pools=n_pools)

    def forward(self, x):
        B, C, T = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            S = torch.stft(x.float().reshape(B * C, T), self.n_fft, self.hop, window=self.window, return_complex=True)
            P = torch.log1p(S[:, :self.n_freq].abs() ** 2).reshape(B, C, self.n_freq, -1)
        img = torch.cat([P[:, list(c)].mean(1) for c in self.CHAINS], 1).flip(1)  # B x 5F x T'
        img = (img - img.mean((1, 2), keepdim=True)) / (img.std((1, 2), keepdim=True) + 1e-6)
        return self.heads(self.cnn(img[:, None]))


def build_model(name, fs=100, pretrained=True, drop_rate=0.0, drop_path=0.0, n_pools=1):
    if name == "cnn1d":
        return CNN1D(n_pools=n_pools)
    if name.startswith("spec2d"):  # e.g. spec2d or spec2d:tf_efficientnetv2_b3
        return Spec2D(fs=fs, name=name.split(":")[1] if ":" in name else "efficientnet_b0", pretrained=pretrained,
                      drop_rate=drop_rate, drop_path=drop_path, n_pools=n_pools)
    raise ValueError(name)
