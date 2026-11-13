# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Unit check before running models on external EEG: compare the bipolar standard deviation (uV, 0.5-40 Hz) of
HMS training windows with that of downloaded I-CARE blocks, by hospital and hour. If I-CARE were not in uV, every
hospital would be off by a similar large factor (x10, x1000). Outcome is NOT read (pre-registered outcome analysis)."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--icare", default=None, help="default: <work_dir>/icare")
    ap.add_argument("--n_hms", type=int, default=2000, help="HMS events sampled for the reference")
    a = ap.parse_args()
    cfg = load_config(a.config)

    X = np.load(cfg["work_dir"] / "windows.npy", mmap_mode="r")
    ev = pd.read_parquet(cfg["work_dir"] / "events.parquet")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(X), min(a.n_hms, len(X)), replace=False))
    T, w = X.shape[2], int(cfg["win_s"] * cfg["fs_out"])
    s = (T - w) // 2
    sd = []
    for i in idx:  # cached HMS windows are band-passed uV / 32; undo the scaling (clipping is rare)
        x = np.asarray(X[i, :, s:s + w], dtype=np.float32) * 32
        ch = x.std(1)
        sd.append(np.median(ch[ch > 0]) if (ch > 0).any() else np.nan)
    hms = pd.DataFrame({"sd": sd, "stratum": ev.stratum.values[idx]})
    dom = ev[[c for c in ev.columns if c.endswith("_vote")]].values.argmax(1)[idx]
    hms["dominant"] = np.array(["seizure", "lpd", "gpd", "lrda", "grda", "other"])[dom]

    q = [0.1, 0.25, 0.5, 0.75, 0.9]
    print("HMS bipolar SD (uV) quantiles:", hms.sd.quantile(q).round(1).to_dict())
    print("HMS median by dominant class:", hms.groupby("dominant").sd.median().round(1).to_dict())

    ic = Path(a.icare) if a.icare else cfg["work_dir"] / "icare"
    blocks = pd.read_csv(ic / "blocks.csv", dtype={"patient": str})
    pat = pd.read_csv(ic / "patients.csv", dtype={"Patient": str}, usecols=["Patient", "Hospital"])
    b = blocks[blocks.n_windows > 0].merge(pat, left_on="patient", right_on="Patient", how="left")
    print(f"\nI-CARE: {b.patient.nunique()} patients, {len(b)} blocks")
    print("I-CARE bipolar SD (uV) quantiles:", b.bipolar_sd_uv.quantile(q).round(1).to_dict())
    b["late"] = np.where(b.hour >= 48, "48-72 h", "12-47 h")
    print("\nmedian bipolar SD (uV) by hospital x time:")
    print(b.pivot_table(index="Hospital", columns="late", values="bipolar_sd_uv", aggfunc="median").round(1)
          .to_string())
    print("\nblocks per hospital:", b.groupby("Hospital").size().to_dict(),
          "\nsampling rates per hospital:", b.groupby("Hospital").fs.unique().apply(lambda v: sorted(map(float, v)))
          .to_dict())
    print("\nfraction of samples at the +-1024 uV clip: mean", round(float(b.frac_clipped.mean()), 5),
          "| median", round(float(b.frac_clipped.median()), 5),
          "| blocks > 1%:", int((b.frac_clipped > 0.01).sum()), "of", len(b))
    print("by hospital (mean):", b.groupby("Hospital").frac_clipped.mean().round(4).to_dict())
    print("worst blocks:\n" + b.nlargest(5, "frac_clipped")[["patient", "hour", "Hospital", "fs", "bipolar_sd_uv",
                                                               "frac_clipped", "missing"]].to_string(index=False))


if __name__ == "__main__":
    main()
