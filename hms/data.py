# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Data: events (de-duplicated label windows), patient-wise folds, EEG window cache, Dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm

CLASSES = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
VOTES = [f"{c}_vote" for c in CLASSES]
FS = 200  # HMS raw sampling rate

# Double-banana montage: LL, RL, LP, RP chains (4 each) + central (Fz-Cz, Cz-Pz) = 18 channels
BIPOLAR = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
           ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
           ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
           ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
           ("Fz", "Cz"), ("Cz", "Pz")]
LR_SWAP = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 16, 17]  # left <-> right hemisphere
SOS = butter(4, [0.5, 40.0], btype="band", fs=FS, output="sos")


def stratum(n):
    """Vote-count pool. 0: >=10 votes (same regime as the hidden test), 1: 3-7, 2: 1-2."""
    n = np.asarray(n)
    return np.where(n >= 10, 0, np.where(n >= 3, 1, 2))


def build_events(train: pd.DataFrame) -> pd.DataFrame:
    """Merge rows of the same eeg_id with identical vote vectors into one event (3rd-place 'reverse augmentation')."""
    df = train.copy()
    df["n"] = df[VOTES].sum(axis=1)
    ev = (df.groupby(["eeg_id"] + VOTES, sort=False)
            .agg(patient_id=("patient_id", "first"),
                 n_patients=("patient_id", "nunique"),
                 n=("n", "first"),
                 n_rows=("n", "size"),
                 offsets=("eeg_label_offset_seconds", lambda s: sorted(s)))
            .reset_index())
    assert (ev.n_patients == 1).all(), "an eeg_id maps to several patients"
    ev["center_offset"] = ev.offsets.map(lambda o: o[len(o) // 2])
    ev["stratum"] = stratum(ev.n.values)
    ev["event_id"] = np.arange(len(ev))
    return ev.drop(columns="n_patients")


def make_folds(ev: pd.DataFrame, n_folds=5, seed=42) -> np.ndarray:
    """fold = -1 for a locked internal test set (~20% of patients), 0..n_folds-1 otherwise. Grouped by patient."""
    y = ev[VOTES].values.argmax(1) * 3 + ev.stratum.values
    groups = ev.patient_id.values
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, _ = next(outer.split(ev, y, groups))
    fold = np.full(len(ev), -1)
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(inner.split(dev, y[dev], groups[dev])):
        fold[dev[va]] = k
    return fold


def check_no_leakage(ev: pd.DataFrame, fold: np.ndarray):
    """Raise if any patient or eeg_id appears in more than one split."""
    for key in ["patient_id", "eeg_id"]:
        per_key = pd.Series(fold).groupby(ev[key].values).nunique()
        bad = per_key[per_key > 1]
        if len(bad):
            raise AssertionError(f"{len(bad)} {key}s appear in several splits, e.g. {bad.index[:5].tolist()}")


def read_bipolar(path) -> np.ndarray:
    eeg = pd.read_parquet(path)
    x = np.stack([eeg[a].values - eeg[b].values for a, b in BIPOLAR]).astype(np.float64)
    return np.nan_to_num(x)


def extract_window(x: np.ndarray, center_offset: float, span_s: float, fs_out: int) -> np.ndarray:
    """Band-pass 0.5-40 Hz, resample to fs_out, return span_s seconds centred on the 50 s label window.
    Samples outside the recording are zero. Output is clipped to +-1024 uV and divided by 32 (3rd-place scaling)."""
    mid = int(round((center_offset + 25.0) * FS))
    half, pad = int(span_s / 2 * FS), 10 * FS  # 10 s margin absorbs filter edge effects
    lo, hi = mid - half - pad, mid + half + pad
    seg = np.zeros((x.shape[0], hi - lo))
    a, b = max(lo, 0), min(hi, x.shape[1])
    seg[:, a - lo:b - lo] = x[:, a:b]
    seg = sosfiltfilt(SOS, seg, axis=1)
    if fs_out != FS:
        seg = resample_poly(seg, fs_out, FS, axis=1)
    p = pad * fs_out // FS
    seg = seg[:, p:p + int(span_s * fs_out)]
    return (np.clip(seg, -1024, 1024) / 32).astype(np.float16)


def _windows_for_eeg(args):
    path, idx, offsets, span_s, fs_out = args
    x = read_bipolar(path)
    return idx, [extract_window(x, o, span_s, fs_out) for o in offsets]


def build_cache(ev: pd.DataFrame, eeg_dir, out_path, span_s=60, fs_out=100, workers=4):
    """Write all event windows into one float16 memmap of shape (n_events, 18, span_s*fs_out)."""
    from concurrent.futures import ProcessPoolExecutor
    T = int(span_s * fs_out)
    X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(ev), len(BIPOLAR), T))
    jobs = [(Path(eeg_dir) / f"{eeg_id}.parquet", g.event_id.values, g.center_offset.values, span_s, fs_out)
            for eeg_id, g in ev.groupby("eeg_id")]
    with ProcessPoolExecutor(workers) as pool:
        for idx, wins in tqdm(pool.map(_windows_for_eeg, jobs, chunksize=8), total=len(jobs), unit="EEG",
                              desc="cache"):
            X[idx] = np.stack(wins)
    X.flush()


class EventDataset(Dataset):
    """Returns (x[18, win], counts[6], stratum, event_id). Training: random time shift + L/R hemisphere swap."""

    def __init__(self, cache_path, ev: pd.DataFrame, win_samples: int, train: bool, seed=0, mask=False):
        self.cache_path, self.win, self.train, self.mask = str(cache_path), win_samples, train, mask
        self.rows = ev.event_id.values
        self.counts = ev[VOTES].values.astype(np.float32)
        self.strata = ev.stratum.values.astype(np.int64)
        self.X, self.rng, self.seed = None, None, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.X is None:  # opened lazily so the dataset pickles to Windows DataLoader workers
            self.X = np.load(self.cache_path, mmap_mode="r")
            wi = get_worker_info()  # a different augmentation stream per worker
            self.rng = np.random.default_rng(self.seed + (wi.id if wi else 0))
        x = self.X[self.rows[i]]
        T = x.shape[1]
        s = self.rng.integers(0, T - self.win + 1) if self.train else (T - self.win) // 2
        x = np.asarray(x[:, s:s + self.win], dtype=np.float32)
        if self.train and self.rng.random() < 0.5:
            x = x[LR_SWAP]  # class labels are laterality-agnostic, so votes are unchanged
        if self.train and self.mask and self.rng.random() < 0.5:  # time masking: zero up to 10% of the window
            L = int(self.rng.integers(1, max(2, self.win // 10)))
            a = int(self.rng.integers(0, self.win - L))
            x[:, a:a + L] = 0
        return x, self.counts[i], self.strata[i], self.rows[i]
