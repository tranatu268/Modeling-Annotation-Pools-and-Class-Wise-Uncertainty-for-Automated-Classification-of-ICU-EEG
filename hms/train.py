# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""One training run = one backbone x one loss x one validation fold. Writes best.pt, log.csv, preds.parquet."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CLASSES, VOTES, EventDataset
from .losses import RaTE, cost_acns, cost_covote, dirmult_nll, sinkhorn_cost, soft_kl
from .metrics import kaggle_kl
from .models import build_model

LOSSES = ["kl", "kl_w", "divo", "rate", "pheads"]
# pheads: DiVo + one residual head per vote stratum (input-dependent shift); prediction uses the >=10-vote head
# kl   : soft-label KL (with stage2_epochs > 0: the two-stage top-Kaggle recipe)
# kl_w : soft-label KL weighted by vote count n (the limit of DiVo as kappa -> inf)
# divo : Dirichlet-multinomial likelihood on raw vote counts
# rate : DiVo + random-threshold rater pools per vote stratum
# Any loss can add stage 2 (same loss on >=10-vote events) via stage2_epochs.


@torch.no_grad()
def predict_loader(model, rate, dl, device, use_amp, M_eval):
    """Deterministic prediction (fixed Monte-Carlo noise for RaTE). Pool-heads and RaTE predict for pool 0, the
    >=10-vote regime of the test set. Returns probabilities, kappa, event ids, and the centred low-vote head shifts
    of pool-heads (None otherwise)."""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    P, K, ids, D = [], [], [], []
    for x, _, _, eid in dl:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, kappa = model(x.to(device, non_blocking=True))
        logits = logits.float()
        if logits.dim() == 3:  # pool-heads: keep the shift of each low-vote head, predict with the >=10-vote head
            d = logits[:, 1:] - logits[:, :1]
            D.append((d - d.mean(-1, keepdim=True)).cpu())  # centred like RaTE's mu
            logits = logits[:, 0]
        if rate:
            p = rate(logits, torch.zeros(len(x), dtype=torch.long, device=device), M=M_eval, generator=gen).exp()
        else:
            p = logits.softmax(-1)
        P.append(p.cpu().numpy()), K.append(kappa.float().cpu().numpy()), ids.append(eid.numpy())
    return np.concatenate(P), np.concatenate(K), np.concatenate(ids), (torch.cat(D) if D else None)


@torch.no_grad()
def predict_pools(model, rate, dl, device, use_amp, M_eval, n_pools=3):
    """Predicted vote distribution of EVERY rater pool, N x n_pools x K, with event ids.
    Pool-heads: softmax of each head; RaTE: pi_s with fixed Monte-Carlo noise; single-pool models: pi_0 repeated."""
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    P, ids = [], []
    for x, _, _, eid in dl:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, _ = model(x.to(device, non_blocking=True))
        logits = logits.float()
        if logits.dim() == 3:  # pool-heads
            p = logits.softmax(-1)
        elif rate:
            p = torch.stack([rate(logits, torch.full((len(x),), s, dtype=torch.long, device=device), M=M_eval,
                                  generator=gen).exp() for s in range(n_pools)], 1)
        else:
            p = logits.softmax(-1)[:, None].expand(-1, n_pools, -1)
        P.append(p.cpu().numpy())
        ids.append(eid.numpy())
    return np.concatenate(P), np.concatenate(ids)


def load_fold_model(fold_dir, device=None):
    """Rebuild a finished fold's model (and RaTE) from config.json + best.pt.
    Returns (model, rate or None, cfg, device, use_amp)."""
    fold_dir = Path(fold_dir)
    cfg = json.loads((fold_dir / "config.json").read_text())
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = cfg.get("amp", True) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = build_model(cfg["backbone"], fs=cfg["fs_out"], pretrained=False, drop_rate=cfg.get("drop_rate", 0.0),
                        drop_path=cfg.get("drop_path", 0.0), n_pools=3 if cfg["loss"] == "pheads" else 1).to(device)
    rate = RaTE(n_pools=3, M=cfg.get("M_train", 32)).to(device) if cfg["loss"] == "rate" else None
    ck = torch.load(fold_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    if rate:
        rate.load_state_dict(ck["rate"])
    return model, rate, cfg, device, use_amp


def predict_test_from_checkpoint(fold_dir, work_dir, device=None, workers=0):
    """Predict the locked internal test (fold -1) with a finished fold's best.pt and its config.json, and replace
    any 'test' rows in that fold's preds.parquet. Identical to training with predict_test=True, without retraining."""
    fold_dir, work_dir = Path(fold_dir), Path(work_dir)
    model, rate, cfg, device, use_amp = load_fold_model(fold_dir, device)
    ev = pd.read_parquet(work_dir / "events.parquet")
    test = ev[ev.fold == -1]
    ds = EventDataset(work_dir / "windows.npy", test, int(cfg["win_s"] * cfg["fs_out"]), train=False)
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 32), shuffle=False, num_workers=workers)
    p, kap, ids, _ = predict_loader(model, rate, dl, device, use_amp, cfg.get("M_eval", 256))
    f = pd.DataFrame(p, columns=[f"p_{c}" for c in CLASSES])
    f.insert(0, "event_id", ids)
    f["kappa"], f["split"], f["fold"] = kap, "test", cfg["fold"]
    old = pd.read_parquet(fold_dir / "preds.parquet")
    pd.concat([old[old.split != "test"], f]).to_parquet(fold_dir / "preds.parquet", index=False)
    return len(f)


class EMA:
    """Exponential moving average of parameters AND floating-point buffers (BatchNorm running stats), so the
    averaged weights are always evaluated with matching normalisation statistics.
    reset() is called at the start of every phase: the shadow restarts from the live weights and the decay is
    capped so that the EMA horizon 1/(1-decay) is at most 20% of the phase's optimizer steps."""

    def __init__(self, modules, decay):
        self.base_decay = decay
        self.tensors = ([p for m in modules for p in m.parameters() if p.requires_grad]
                        + [b for m in modules for b in m.buffers() if b.dtype.is_floating_point])
        self.reset(steps=None)

    @torch.no_grad()
    def reset(self, steps):
        self.decay = self.base_decay if steps is None else min(self.base_decay, 1 - 5 / max(steps, 5))
        self.shadow = [t.detach().clone() for t in self.tensors]

    @torch.no_grad()
    def update(self):
        for s, t in zip(self.shadow, self.tensors):
            s.lerp_(t.detach(), 1 - self.decay)

    @torch.no_grad()
    def swap(self):  # exchange live and shadow tensors; call once before and once after evaluation
        for s, t in zip(self.shadow, self.tensors):
            tmp = t.detach().clone()
            t.copy_(s)
            s.copy_(tmp)


def run(work_dir, out_dir, fold, backbone="cnn1d", loss="divo", ot_lambda=0.0, ot_cost="acns",
        epochs=15, stage2_epochs=0, batch_size=32, lr=1e-3, weight_decay=1e-2, fs_out=100, win_s=50,
        workers=4, seed=42, amp=True, pretrained=True, M_train=32, M_eval=256, predict_test=False,
        hi_frac=1.0, kappa_fixed=None, mixup=0.0, ema_decay=0.0, aug_mask=False,
        drop_rate=0.0, drop_path=0.0, select="best", rate_lr=None, mixup_within=False, device=None,
        max_steps=None, pools="all"):
    """select='best': checkpoint with the lowest validation KL (optimistic, more so for longer final phases);
    select='last': the final checkpoint, no validation peeking (use for the reported tables)."""
    assert loss in LOSSES and select in ("best", "last")
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in locals().items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    work_dir, out_dir = Path(work_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    ev = pd.read_parquet(work_dir / "events.parquet")
    cache = work_dir / "windows.npy"
    tr = ev[(ev.fold >= 0) & (ev.fold != fold)]
    if hi_frac < 1.0:  # stress test: keep only a fraction of >=10-vote PATIENTS in training
        hi_pat = tr.loc[tr.stratum == 0, "patient_id"].unique()
        keep = np.random.default_rng(seed).choice(hi_pat, int(round(hi_frac * len(hi_pat))), replace=False)
        tr = tr[(tr.stratum != 0) | tr.patient_id.isin(keep)]
    if pools == "0":  # ablation: train on the >=10-vote events only
        tr = tr[tr.stratum == 0]
    elif pools == "merge2":  # ablation: 1-2-vote events share the head / shift of the >=10-vote pool
        tr = tr.assign(stratum=tr.stratum.replace(2, 0))
    else:
        assert pools == "all", pools
    va = ev[ev.fold == fold]
    va_hi = va[va.stratum == 0]
    win = int(win_s * fs_out)

    model = build_model(backbone, fs=fs_out, pretrained=pretrained, drop_rate=drop_rate, drop_path=drop_path,
                        n_pools=3 if loss == "pheads" else 1).to(device)
    rate = RaTE(n_pools=3, M=M_train).to(device) if loss == "rate" else None
    params = list(model.parameters()) + (list(rate.parameters()) if rate else [])
    C = None
    if ot_lambda > 0:
        C = cost_acns() if ot_cost == "acns" else cost_covote(torch.tensor(tr[VOTES].values, dtype=torch.float32))
        C = C.to(device)
        np.savetxt(out_dir / "ot_cost.txt", C.cpu().numpy(), fmt="%.3f")

    def loader(df, train):
        ds = EventDataset(cache, df, win, train, seed=seed, mask=aug_mask)
        return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=workers, drop_last=train,
                          pin_memory=device.type == "cuda", persistent_workers=workers > 0)

    def objective(logits, kappa, counts, s):
        logits, kappa = logits.float(), kappa.float()
        if kappa_fixed is not None:  # diagnostic: constant concentration instead of the learned kappa(x)
            kappa = torch.full_like(kappa, kappa_fixed)
        if loss == "pheads":  # each sample is scored by the head of its own vote stratum
            logits = logits[torch.arange(len(s), device=logits.device), s]
        if loss == "kl":
            logp = F.log_softmax(logits, -1)
            L = soft_kl(logp, counts)
        elif loss == "kl_w":
            logp = F.log_softmax(logits, -1)
            L = soft_kl(logp, counts, weight=counts.sum(1))
        else:
            logp = rate(logits, s) if rate else F.log_softmax(logits, -1)
            L = dirmult_nll(counts, logp, kappa).mean() / counts.sum(1).mean()
        if C is not None:
            L = L + ot_lambda * sinkhorn_cost(counts / counts.sum(1, keepdim=True), logp.exp(), C).mean()
        return L

    def predict(dl):
        P, K, ids, predict.shifts = predict_loader(model, rate, dl, device, use_amp, M_eval)
        return P, K, ids

    def val_kl(dl, df):
        p, _, ids = predict(dl)
        t = df.set_index("event_id").loc[ids, VOTES].values.astype(float)
        return kaggle_kl(t / t.sum(1, keepdims=True), p).mean()

    phases = [(tr, epochs)]
    if stage2_epochs > 0:  # stage 2: same loss, >=10-vote events only
        phases.append((tr[tr.stratum == 0], stage2_epochs))
    dl_va_hi = loader(va_hi, train=False)
    best, log, step = np.inf, [], 0
    ema = EMA([model] + ([rate] if rate else []), ema_decay) if ema_decay > 0 else None
    for ph, (df, n_ep) in enumerate(phases):
        dl = loader(df, train=True)
        if rate and rate_lr is not None:  # own group: threshold shifts are not shrunk and learn at their own rate
            groups = [{"params": list(model.parameters())},
                      {"params": list(rate.parameters()), "lr": rate_lr, "weight_decay": 0.0}]
            max_lr = [lr, rate_lr]
        else:
            groups, max_lr = params, lr
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=n_ep * len(dl), pct_start=0.1)
        if ema:
            ema.reset(steps=n_ep * (min(len(dl), max_steps) if max_steps else len(dl)))
            print(f"  phase {ph}: EMA decay {ema.decay:.5f}", flush=True)
        for ep in range(n_ep):
            model.train()
            t0, tot, nb = time.time(), 0.0, 0
            bar = tqdm(dl, desc=f"fold {fold} ph{ph} ep{ep}", unit="batch", leave=False, mininterval=1)
            for x, counts, s, _ in bar:
                x, counts, s = x.to(device, non_blocking=True), counts.to(device), s.to(device)
                if mixup > 0:  # one lambda per batch; vote-count vectors are mixed like the signals
                    lam = float(np.random.beta(mixup, mixup))
                    perm = torch.randperm(len(x), device=device)
                    if mixup_within:  # partners only from the same vote stratum, so the pool stays exact
                        perm = torch.arange(len(x), device=device)
                        for v in s.unique():
                            idx = (s == v).nonzero(as_tuple=True)[0]
                            perm[idx] = idx[torch.randperm(len(idx), device=device)]
                    x = lam * x + (1 - lam) * x[perm]
                    counts = lam * counts + (1 - lam) * counts[perm]
                    if lam < 0.5:  # the pool index follows the dominant sample
                        s = s[perm]
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits, kappa = model(x)
                L = objective(logits, kappa, counts, s)
                opt.zero_grad(set_to_none=True)
                L.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                if ema:
                    ema.update()
                tot, nb, step = tot + L.item(), nb + 1, step + 1
                if nb % 20 == 0:
                    bar.set_postfix(loss=f"{tot / nb:.4f}")
                if max_steps and nb >= max_steps:
                    break
            bar.close()
            if ema:
                ema.swap()  # validate, select and checkpoint with the EMA weights
            kl = val_kl(dl_va_hi, va_hi)
            log.append({"phase": ph, "epoch": ep, "train_loss": tot / max(nb, 1), "val_kl_hi": kl,
                        "seconds": time.time() - t0})
            print(f"[fold {fold}] phase {ph} epoch {ep}: loss {tot / max(nb, 1):.4f}  val KL(>=10 votes) {kl:.4f}",
                  flush=True)
            pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)
            # model selection on the validation fold; in stage-2 runs only stage-2 epochs are eligible
            if ph == len(phases) - 1 and (kl < best or select == "last"):
                best = kl
                torch.save({"model": model.state_dict(), "rate": rate.state_dict() if rate else None},
                           out_dir / "best.pt")
            if ema:
                ema.swap()

    ck = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    if rate:
        rate.load_state_dict(ck["rate"])
    outs = [("val", va)] + ([("test", ev[ev.fold == -1])] if predict_test else [])
    frames = []
    for split, df in outs:
        p, kap, ids = predict(loader(df, train=False))
        if split == "val" and predict.shifts is not None:  # comparable to RaTE's mu: mean (and spread) per pool
            torch.save({"mean": predict.shifts.mean(0), "std": predict.shifts.std(0)}, out_dir / "pool_shift.pt")
        f = pd.DataFrame(p, columns=[f"p_{c}" for c in CLASSES])
        f.insert(0, "event_id", ids)
        f["kappa"], f["split"], f["fold"] = kap, split, fold
        frames.append(f)
    pd.concat(frames).to_parquet(out_dir / "preds.parquet", index=False)
    if rate:
        torch.save({"mu": rate.mu.detach().cpu(), "scale_tril": rate.scale_tril().detach().cpu()},
                   out_dir / "rate_params.pt")
    return best
