# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Technical feasibility of the frozen HMS models on I-CARE (external, post-cardiac-arrest EEG).

Runs every fold model of each run on the downloaded I-CARE windows and reports ONLY technical quantities:
  - throughput; fold-model agreement (argmax agreement, SD of probabilities across folds)
  - predicted class distribution (mean probability, argmax share), overall / by hospital / by hour since ROSC
  - confidence (max probability, entropy)
  - label-conditional conformal sets (alpha, default 0.2) with thresholds calibrated per fold model on that fold's
    HMS out-of-fold >=10-vote predictions: deferral, set size and empty-set rate on I-CARE vs the HMS reference.
    A large change of deferral is a dataset-shift indicator, not a performance estimate.

Outcome and CPC are NEVER read here (patients.csv is loaded with usecols=Patient, Hospital), so the outcome
analysis stays independent of every choice made in this check.

    python scripts\\13_icare_infer.py
    python scripts\\13_icare_infer.py --gain 2      # unit sensitivity: multiply I-CARE amplitudes by 2
Writes <icare>/preds_<run>[_g<gain>].parquet (one row per window, fold-ensemble mean) and icare_summary*.csv.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import conformal, load_config  # noqa: E402
from hms.data import CLASSES, VOTES  # noqa: E402
from hms.train import load_fold_model, predict_loader  # noqa: E402

RUNS = ["spec2d-tf_efficientnetv2_b0_pheads_k1000_mx0.4_ema0.999_e20_last_s42",
        "cnn1d_rate_k1000_mx0.2_rlr0.001_ema0.999_am_e30_last_s42"]
PCOLS = [f"p_{c}" for c in CLASSES]
CLIP = 1024 / 32  # the cached windows are clipped at +-1024 uV and divided by 32
QC_CLIP, QC_SD = 0.01, 150.0  # artefact rule, fixed before any outcome analysis: >1% of samples at the clip, or a
                              # median bipolar SD > 150 uV (HMS 90th percentile ~31 uV). Low amplitude is NOT
                              # excluded: suppression is a real post-arrest finding.
QC_FLAT = 0.1  # technical flat line (added after the first full run, before any outcome analysis): median bipolar
               # SD < 0.1 uV = digital zero / disconnected amplifier, far below the ~1-2 uV noise of suppressed EEG


def hms_reference(work_dir, fold_dir, alpha):
    """Label-conditional thresholds from this fold model's own out-of-fold >=10-vote predictions, plus the
    HMS deferral / set size / class distribution obtained with those thresholds (the reference for I-CARE)."""
    ev = pd.read_parquet(work_dir / "events.parquet").set_index("event_id")
    pr = pd.read_parquet(fold_dir / "preds.parquet")
    pr = pr[pr.split == "val"].set_index("event_id")
    ids = pr.index.intersection(ev.index[ev.stratum == 0])
    p = pr.loc[ids, PCOLS].values
    t = ev.loc[ids, VOTES].values.astype(float)
    t /= t.sum(1, keepdims=True)
    pats = ev.loc[ids, "patient_id"].values
    q = conformal.calibrate(p, t, pats, alpha, mode="label")
    ref = conformal.evaluate_sets(conformal.predict_sets(p, q, mode="label"), t, pats)
    ref["argmax_share"] = np.bincount(p.argmax(1), minlength=6) / len(p)
    ref["max_prob"] = float(np.median(p.max(1)))
    return q, ref


def iter_chunks(blocks, wdir, gain, chunk):
    """Concatenate block files into chunks of >= `chunk` windows. Yields (x float32, row-info DataFrame)."""
    xs, infos, n = [], [], 0
    for b in blocks.itertuples():
        x = np.load(wdir / f"{b.patient}_{int(b.hour):03d}.npy").astype(np.float32)
        clip_frac = (np.abs(x) >= CLIP - 1e-3).mean((1, 2))  # QC on the recorded amplitude, before any gain
        ch_sd = x.std(2) * 32
        sd_uv = np.array([np.median(c[c > 0]) if (c > 0).any() else 0.0 for c in ch_sd])
        if gain != 1:
            x = np.clip(x * gain, -CLIP, CLIP)
        xs.append(x)
        infos.append(pd.DataFrame({"patient": b.patient, "hour": int(b.hour), "window": np.arange(len(x)),
                                   "clip_frac": clip_frac, "sd_uv": sd_uv,
                                   "qc_pass": (clip_frac <= QC_CLIP) & (sd_uv <= QC_SD) & (sd_uv >= QC_FLAT)}))
        n += len(x)
        if n >= chunk:
            yield np.concatenate(xs), pd.concat(infos, ignore_index=True)
            xs, infos, n = [], [], 0
    if xs:
        yield np.concatenate(xs), pd.concat(infos, ignore_index=True)


def patient_weighted(df, col):
    """Every patient counts equally (patients contribute different numbers of windows)."""
    return df.groupby("patient")[col].mean().mean()


def describe(df, by=None):
    rows = []
    groups = [("all", df)] if by is None else list(df.groupby(by, observed=True))  # skip empty bins
    for g, d in groups:
        r = {"group": g, "patients": d.patient.nunique(), "windows": len(d)}
        for k, c in enumerate(CLASSES):
            r[f"p_{c}"] = patient_weighted(d, f"p_{c}")
            d = d.assign(**{f"am_{c}": (d.argmax_cls == k).astype(float)})
            r[f"argmax_{c}"] = patient_weighted(d, f"am_{c}")
        for col in ("max_prob", "entropy", "defer", "set_size", "empty", "fold_agree", "fold_sd"):
            r[col] = patient_weighted(d, col)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--icare", default=None, help="default: <work_dir>/icare")
    ap.add_argument("--runs", nargs="+", default=RUNS)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--gain", type=float, default=1.0, help="amplitude multiplier (unit sensitivity analysis)")
    ap.add_argument("--chunk", type=int, default=256, help="windows per forward chunk")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit_blocks", type=int, default=None, help="smoke test on the first N blocks")
    a = ap.parse_args()
    cfg = load_config(a.config)
    work = cfg["work_dir"]
    ic = Path(a.icare) if a.icare else work / "icare"
    wdir = ic / "windows"

    blocks = pd.read_csv(ic / "blocks.csv", dtype={"patient": str})
    blocks = blocks[blocks.n_windows > 0]
    blocks = blocks[[(wdir / f"{p}_{int(h):03d}.npy").exists() for p, h in zip(blocks.patient, blocks.hour)]]
    blocks = blocks.sort_values(["patient", "hour"]).reset_index(drop=True)
    if a.limit_blocks:
        blocks = blocks.head(a.limit_blocks)
    hosp = pd.read_csv(ic / "patients.csv", dtype={"Patient": str}, usecols=["Patient", "Hospital"])  # no outcome
    print(f"I-CARE: {blocks.patient.nunique()} patients, {len(blocks)} blocks, "
          f"{int(blocks.n_windows.sum())} windows of {cfg['win_s']} s; gain {a.gain}", flush=True)
    sfx = "" if a.gain == 1 else f"_g{a.gain:g}"

    summaries = []
    for run in a.runs:
        folds = sorted((work / "results" / run).glob("fold*"))
        if not folds:
            print(f"!! no fold models for {run}, skipped")
            continue
        models, qs, refs = [], [], []
        for f in folds:
            model, rate, mcfg, device, use_amp = load_fold_model(f, a.device)
            q, ref = hms_reference(work, f, a.alpha)
            models.append((model, rate, mcfg)), qs.append(q), refs.append(ref)
        win = int(models[0][2]["win_s"] * models[0][2]["fs_out"])

        out, t0, n_done = [], time.perf_counter(), 0
        bar = tqdm(total=int(blocks.n_windows.sum()), desc=run[:40], unit="win")
        for x, info in iter_chunks(blocks, wdir, a.gain, a.chunk):
            assert x.shape[1:] == (18, win), f"window shape {x.shape[1:]} != (18, {win})"
            xt = torch.from_numpy(x)
            batches = [(xt[i:i + a.batch], None, None, torch.arange(i, min(i + a.batch, len(x))))
                       for i in range(0, len(x), a.batch)]
            P, S = [], []
            for (model, rate, mcfg), q in zip(models, qs):
                p, _, _, _ = predict_loader(model, rate, batches, device, use_amp, mcfg.get("M_eval", 256))
                P.append(p)
                S.append(conformal.predict_sets(p, q, mode="label"))
            P, S = np.stack(P), np.stack(S)  # folds x windows x classes
            pm = P.mean(0)
            info[PCOLS] = pm
            info["argmax_cls"] = pm.argmax(1)
            info["max_prob"] = pm.max(1)
            info["entropy"] = -(pm * np.log(np.clip(pm, 1e-12, 1))).sum(1)
            info["fold_agree"] = (P.argmax(2) == pm.argmax(1)[None]).mean(0)
            info["fold_sd"] = P.std(0).mean(1)
            sizes = S.sum(2)
            info["defer"] = ((sizes >= 2) & S[:, :, conformal.HIGH_RISK].any(2)).mean(0)  # fraction of fold models
            info["set_size"] = sizes.mean(0)
            info["empty"] = (sizes == 0).mean(0)
            out.append(info)
            n_done += len(x)
            bar.update(len(x))
        bar.close()
        dt = time.perf_counter() - t0
        df = pd.concat(out, ignore_index=True).merge(hosp, left_on="patient", right_on="Patient", how="left")
        df = df.drop(columns="Patient")
        df["Hospital"] = df.Hospital.fillna("unknown")  # metadata missing (interrupted download): re-run 09
        df["hour_bin"] = pd.cut(df.hour, [0, 24, 48, 1e9], right=False, labels=["<24 h", "24-47 h", ">=48 h"])
        df.to_parquet(ic / f"preds_{run}{sfx}.parquet", index=False)
        n_bad = int((~df.qc_pass).sum())
        bad_pat = df[~df.qc_pass].groupby("patient").size()
        df = df[df.qc_pass]  # every summary below uses QC-passed windows only

        ref = {k: float(np.mean([r[k] for r in refs])) for k in ("defer_rate", "set_size", "empty_rate", "max_prob")}
        ref_share = np.mean([r["argmax_share"] for r in refs], 0)
        print(f"\n=== {run} ({len(folds)} fold models) ===")
        print(f"throughput: {n_done / dt:.1f} windows/s for all folds together ({dt:.0f} s)")
        print(f"QC (>{QC_CLIP:.0%} clipped, SD>{QC_SD:.0f} uV or SD<{QC_FLAT} uV): excluded {n_bad} of {n_done} windows "
              f"from {len(bad_pat)} patients {bad_pat.to_dict()}; unknown hospital: "
              f"{df[df.Hospital == 'unknown'].patient.nunique()} patients")
        print(f"conformal label-conditional, alpha={a.alpha}:  HMS (calibration, >=10 votes) defer "
              f"{ref['defer_rate']:.3f}, set size {ref['set_size']:.2f}, empty {ref['empty_rate']:.3f}, "
              f"median max-p {ref['max_prob']:.2f}")
        print("HMS argmax share:   ", dict(zip(CLASSES, ref_share.round(3))))
        tab = {"all": describe(df), "hospital": describe(df, "Hospital"), "hour": describe(df, "hour_bin")}
        for name, t in tab.items():
            t.insert(0, "by", name)
            t.insert(0, "run", run)
            summaries.append(t)
        show = ["group", "patients", "windows"] + [f"argmax_{c}" for c in CLASSES] + \
               ["max_prob", "defer", "set_size", "empty", "fold_agree"]
        pd.set_option("display.width", 250)
        for name, t in tab.items():
            print(f"\n-- I-CARE by {name} (patient-weighted) --")
            print(t[show].round(3).to_string(index=False))
        summaries[-3]["hms_defer"] = ref["defer_rate"]
        summaries[-3]["hms_set_size"] = ref["set_size"]

    if summaries:
        path = ic / f"icare_summary{sfx}.csv"
        pd.concat(summaries, ignore_index=True).to_csv(path, index=False)
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
