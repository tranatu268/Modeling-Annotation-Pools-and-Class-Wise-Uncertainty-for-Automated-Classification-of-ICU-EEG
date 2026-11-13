# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Train one configuration on one or more folds. Results go to work_dir/results/<name>/fold<k>/."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402
from hms.train import LOSSES, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", default=None, help="run name (default: <backbone>_<loss>[_ot...][_hi...])")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--backbone", default="cnn1d", help="cnn1d | spec2d | spec2d:<timm model name>")
    ap.add_argument("--loss", default="divo", choices=LOSSES)
    ap.add_argument("--ot_lambda", type=float, default=0.0)
    ap.add_argument("--ot_cost", default="acns", choices=["acns", "covote"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--stage2_epochs", type=int, default=None,
                    help="epochs of stage 2 on >=10-vote events (default: 5 for kl, 0 otherwise)")
    ap.add_argument("--kappa_fixed", type=float, default=None, help="diagnostic: constant DiVo/RaTE concentration")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup Beta(a,a) on signals and vote counts (try 0.2)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay for weights, e.g. 0.999 (0 = off)")
    ap.add_argument("--aug_mask", action="store_true", help="random time masking up to 10%% of the window")
    ap.add_argument("--drop_rate", type=float, default=0.0, help="spec2d only: timm head dropout")
    ap.add_argument("--drop_path", type=float, default=0.0, help="spec2d only: timm stochastic depth")
    ap.add_argument("--rate_lr", type=float, default=None,
                    help="RaTE parameters in their own group with this lr and no weight decay (default: shared)")
    ap.add_argument("--mixup_within", action="store_true", help="mixup partners only from the same vote stratum")
    ap.add_argument("--select", default="best", choices=["best", "last"],
                    help="checkpoint: best validation KL, or last epoch (no validation peeking; for reported tables)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hi_frac", type=float, default=1.0, help="stress test: fraction of >=10-vote patients kept")
    ap.add_argument("--pools", default="all", choices=["all", "0", "merge2"],
                    help="ablation: 0 = train on >=10-vote events only; merge2 = 1-2-vote events use pool 0")
    ap.add_argument("--predict_test", action="store_true", help="also predict the LOCKED internal test (final only)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None, help="debug: stop each epoch after this many steps")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.stage2_epochs is None:
        a.stage2_epochs = 5 if a.loss == "kl" else 0

    name = a.name or "_".join(filter(None, [
        a.backbone.replace(":", "-"), a.loss,
        f"k{a.kappa_fixed:g}" if a.kappa_fixed is not None else "",
        f"ot{a.ot_cost}{a.ot_lambda:g}" if a.ot_lambda > 0 else "",
        "st2" if a.stage2_epochs > 0 and a.loss != "kl" else "",
        f"mx{a.mixup:g}" if a.mixup > 0 else "", "w" if a.mixup_within else "",
        f"rlr{a.rate_lr:g}" if a.rate_lr is not None else "", f"ema{a.ema:g}" if a.ema > 0 else "",
        "am" if a.aug_mask else "", f"e{a.epochs}" if a.epochs != 15 else "",
        "last" if a.select == "last" else "",
        f"hi{a.hi_frac:g}" if a.hi_frac < 1 else "", f"pool{a.pools}" if a.pools != "all" else "", f"s{a.seed}"]))
    for k in [int(f) for f in a.folds.split(",")]:
        best = run(work_dir=cfg["work_dir"], out_dir=cfg["work_dir"] / "results" / name / f"fold{k}", fold=k,
                   backbone=a.backbone, loss=a.loss, ot_lambda=a.ot_lambda, ot_cost=a.ot_cost, epochs=a.epochs,
                   stage2_epochs=a.stage2_epochs, batch_size=a.batch_size, lr=a.lr, weight_decay=a.weight_decay,
                   fs_out=cfg["fs_out"], win_s=cfg["win_s"], workers=a.workers, seed=a.seed, amp=not a.no_amp,
                   pretrained=not a.no_pretrained, predict_test=a.predict_test, hi_frac=a.hi_frac,
                   kappa_fixed=a.kappa_fixed, mixup=a.mixup, ema_decay=a.ema, aug_mask=a.aug_mask,
                   drop_rate=a.drop_rate, drop_path=a.drop_path, select=a.select, rate_lr=a.rate_lr,
                   mixup_within=a.mixup_within, max_steps=a.max_steps, pools=a.pools)
        print(f"== {name} fold {k}: {a.select} val KL (>=10 votes) = {best:.4f}")


if __name__ == "__main__":
    main()
