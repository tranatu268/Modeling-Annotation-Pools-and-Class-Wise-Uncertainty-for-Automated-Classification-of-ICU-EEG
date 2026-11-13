# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Download a selection of I-CARE EEG and convert it to HMS model input. Resumable: finished blocks are skipped.

Selection (fixed BEFORE looking at any model output): for every patient, EEG hours in [hours_from, hours_to);
at most max_hours of them, spread evenly; from each hour the middle `minutes`. Each block is standardized exactly
like HMS training data and cut into non-overlapping 50 s windows.

Output (default work_dir/icare/):
  windows/<pid>_<hour>.npy   float16 (n_windows, 18, 5000)
  windows/<pid>_<hour>.json  block info (sampling rate, missing electrodes, amplitude, units, ...)
  patients.csv               clinical metadata per patient (hospital, age, sex, ROSC, TTM, outcome, CPC, ...)
  blocks.csv                 one row per finished block (rebuilt at the end of every run)

First run:   python scripts\09_icare_fetch.py --probe
Small test:  python scripts\09_icare_fetch.py --limit 5
Full run:    python scripts\09_icare_fetch.py --workers 4
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms import icare, load_config  # noqa: E402
from hms.external import standardize_block  # noqa: E402


def choose_hours(records, h0, h1, k):
    by_hour = {}
    for r in records:  # one record per hour: the first segment that starts in that hour
        h = icare.record_hour(r)
        if h0 <= h < h1 and h not in by_hour:
            by_hour[h] = r
    hours = sorted(by_hour)
    if len(hours) > k:
        hours = [hours[i] for i in np.unique(np.round(np.linspace(0, len(hours) - 1, k)).astype(int))]
    return [(h, by_hour[h]) for h in hours]


def do_patient(pid, a, wdir):
    meta = icare.read_metadata(pid)
    meta["Patient"] = pid
    mdir = wdir.parent / "meta"  # saved at once, so an interrupted run still has every patient's metadata
    mdir.mkdir(exist_ok=True)
    (mdir / f"{pid}.json").write_text(json.dumps(meta))
    done = 0
    for h, rec in choose_hours(icare.list_eeg_records(pid), a.hours_from, a.hours_to, a.max_hours):
        stem = wdir / f"{pid}_{h:03d}"
        if stem.with_suffix(".json").exists():
            done += 1
            continue
        x, header, ranged = icare.fetch_block(pid, rec, a.minutes)
        info = {"patient": pid, "hour": h, "record": rec, "fs": header["fs"], "units": header["signals"][0]["units"],
                "utility_freq": header["utility_freq"], "range_request": ranged}
        if x is None:
            info.update(n_windows=0, skipped="recording shorter than half the block")
        else:
            wins, missing, sd = standardize_block(x, [s["name"] for s in header["signals"]], header["fs"])
            np.save(stem.with_suffix(".npy"), wins)
            info.update(n_windows=len(wins), missing=missing, bipolar_sd_uv=round(sd, 2),
                        frac_clipped=float((np.abs(wins.astype(np.float32)) >= 1024 / 32).mean()))
        stem.with_suffix(".json").write_text(json.dumps(info))
        done += 1
    return meta, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=None, help="default: <work_dir>/icare")
    ap.add_argument("--hours_from", type=int, default=12)
    ap.add_argument("--hours_to", type=int, default=72)
    ap.add_argument("--max_hours", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--patients", default="all", help="'all' or comma-separated ids, e.g. 0284,0582")
    ap.add_argument("--limit", type=int, default=None, help="only the first N patients (testing)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="check access and formats on one patient, save nothing")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(a.out) if a.out else cfg["work_dir"] / "icare"
    wdir = out / "windows"
    wdir.mkdir(parents=True, exist_ok=True)

    pids = (icare.list_patients() if a.patients == "all"  # zfill: PowerShell turns 0311,0319 into 311 319
            else [p.strip().zfill(4) for p in a.patients.replace(" ", ",").split(",") if p.strip()])
    print(f"{len(pids)} patients listed (first: {pids[:5]})")
    if a.limit:
        pids = pids[: a.limit]

    if a.probe:
        pid = pids[0]
        recs = icare.list_eeg_records(pid)
        print(f"patient {pid}: {len(recs)} EEG records, hours {[icare.record_hour(r) for r in recs][:12]} ...")
        print("metadata:", icare.read_metadata(pid))
        sel = choose_hours(recs, a.hours_from, a.hours_to, a.max_hours)
        print("selected (hour, record):", sel)
        if sel:
            x, header, ranged = icare.fetch_block(pid, sel[0][1], 1)
            print(f"header: fs={header['fs']} nsig={header['nsig']} units={header['signals'][0]['units']} "
                  f"channels={[s['name'] for s in header['signals']]}")
            print(f"HTTP Range supported: {ranged}")
            if x is not None:
                wins, missing, sd = standardize_block(x, [s["name"] for s in header["signals"]], header["fs"])
                print(f"1-min test block -> windows {wins.shape}, missing {missing}, bipolar SD {sd:.1f} "
                      f"(HMS training EEG is in uV; typical bipolar SD is tens of uV)")
        return

    metas, blocks_done, failed = [], 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool, tqdm(total=len(pids), unit="patient", desc="I-CARE") as bar:
        futs = {pool.submit(do_patient, p, a, wdir): p for p in pids}
        for f in as_completed(futs):
            p = futs[f]
            try:
                meta, n = f.result()
                metas.append(meta)
                blocks_done += n
            except Exception as e:  # keep going; a re-run retries only what is missing
                failed += 1
                tqdm.write(f"{p}: FAILED {type(e).__name__}: {e}")
            mb = icare.bytes_downloaded() / 1e6
            bar.set_postfix(blocks=blocks_done, failed=failed, MB=f"{mb:.0f}",
                            MBps=f"{mb / max(time.time() - t0, 1e-6):.1f}")
            bar.update()

    old = out / "patients.csv"
    pat = pd.DataFrame(metas + [json.loads(j.read_text()) for j in sorted((out / "meta").glob("*.json"))])
    if old.exists():
        pat = pd.concat([pd.read_csv(old, dtype={"Patient": str}), pat]).drop_duplicates("Patient", keep="last")
    pat = pat.drop_duplicates("Patient", keep="last")
    pat.sort_values("Patient").to_csv(old, index=False)
    blocks = pd.DataFrame([json.loads(j.read_text()) for j in sorted(wdir.glob("*.json"))])
    blocks.to_csv(out / "blocks.csv", index=False)
    ok = blocks[blocks.n_windows > 0] if len(blocks) else blocks
    print(f"\npatients with metadata: {len(pat)}; blocks: {len(blocks)} ({len(ok)} usable, "
          f"{int(ok.n_windows.sum()) if len(ok) else 0} windows)")
    if len(ok):
        print("sampling rates:", ok.fs.value_counts().to_dict())
        print("bipolar SD (uV) quantiles 10/50/90%:", ok.bipolar_sd_uv.quantile([.1, .5, .9]).round(1).to_dict())
        print(f"downloaded this run: {icare.bytes_downloaded() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
