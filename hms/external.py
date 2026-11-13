# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""External EEG (e.g. I-CARE, PhysioNet WFDB) -> exactly the HMS model input.

HMS pipeline (hms.data.extract_window): monopolar 200 Hz -> 18 double-banana channels -> NaN to 0 ->
band-pass 0.5-40 Hz (filtfilt) -> resample to fs_out -> clip +-1024 uV, /32 -> float16.
Here the same steps run on a longer block recorded at any sampling rate, which is first resampled to 200 Hz."""
import re
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly, sosfiltfilt

from .data import BIPOLAR, FS, SOS

# 10-10 names used by some hospitals -> the 10-20 names used by HMS
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}

_SIG = re.compile(r"^(\S+)\s+(\d+)(?:\+(\d+))?\s+([-\d.eE+]+)(?:\(([-\d]+)\))?(?:/(\S+))?\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+(.+)$")


def parse_header(text):
    """Minimal WFDB header parser for single-segment records with format 16 (+byte offset)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rec = lines[0].split()
    out = {"record": rec[0], "nsig": int(rec[1]), "fs": float(rec[2]), "nsamp": int(rec[3]), "signals": [],
           "utility_freq": None}
    for ln in lines[1:]:
        if ln.startswith("#"):
            m = re.match(r"#\s*Utility frequency:\s*(\d+)", ln)
            if m:
                out["utility_freq"] = int(m.group(1))
            continue
        m = _SIG.match(ln)
        if not m:
            raise ValueError(f"cannot parse signal line: {ln}")
        file, fmt, offset, gain, baseline, units, name = m.groups()
        if fmt != "16":
            raise ValueError(f"only WFDB format 16 is supported, got {fmt}")
        gain = float(gain) or 200.0  # WFDB: gain 0 means 200 adu per unit
        out["signals"].append({"file": file, "offset": int(offset or 0), "gain": gain,
                               "baseline": int(baseline) if baseline is not None else 0,
                               "units": units or "", "name": name.strip()})
    assert len(out["signals"]) == out["nsig"]
    return out


def decode_block(raw, header):
    """Bytes of consecutive samples (little-endian int16, all channels per sample) -> physical (nsig x n)."""
    nsig = header["nsig"]
    adc = np.frombuffer(raw, dtype="<i2")
    adc = adc[: len(adc) // nsig * nsig].reshape(-1, nsig).T.astype(np.float64)
    g = np.array([s["gain"] for s in header["signals"]])[:, None]
    b = np.array([s["baseline"] for s in header["signals"]])[:, None]
    return (adc - b) / g


def to_bipolar(x, names):
    """Monopolar (nch x n) with channel names -> 18 double-banana channels; a pair with a missing electrode is 0.
    Returns (bipolar, sorted list of missing electrodes)."""
    alias = {k.upper(): v.upper() for k, v in ALIASES.items()}
    idx = {}
    for i, n in enumerate(names):
        key = n.strip().upper()
        idx.setdefault(alias.get(key, key), i)
    need = sorted({e for pair in BIPOLAR for e in pair})
    missing = [e for e in need if e.upper() not in idx]
    out = np.zeros((len(BIPOLAR), x.shape[1]))
    for k, (a, c) in enumerate(BIPOLAR):
        if a.upper() in idx and c.upper() in idx:
            out[k] = x[idx[a.upper()]] - x[idx[c.upper()]]
    return np.nan_to_num(out), missing


def standardize_block(x, names, fs, fs_out=100, win_s=50, margin_s=5):
    """Monopolar block (nch x n, uV) at any fs -> (n_windows, 18, win_s*fs_out) float16 HMS-ready windows,
    non-overlapping, after dropping margin_s seconds at both ends (filter edge effects). Also returns the
    missing electrodes and the median per-channel standard deviation of the band-passed bipolar signal in uV."""
    bip, missing = to_bipolar(x, names)
    if fs != FS:
        r = Fraction(FS, int(round(fs))).limit_denominator(1000)
        bip = resample_poly(bip, r.numerator, r.denominator, axis=1)
    bip = sosfiltfilt(SOS, bip, axis=1)
    sd = float(np.median(bip.std(1)[np.abs(bip).sum(1) > 0])) if np.abs(bip).sum() > 0 else 0.0
    if fs_out != FS:
        bip = resample_poly(bip, fs_out, FS, axis=1)
    m, w = int(margin_s * fs_out), int(win_s * fs_out)
    bip = bip[:, m: bip.shape[1] - m]
    n = bip.shape[1] // w
    wins = bip[:, : n * w].reshape(len(BIPOLAR), n, w).transpose(1, 0, 2)
    return (np.clip(wins, -1024, 1024) / 32).astype(np.float16), missing, sd
