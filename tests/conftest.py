# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Synthetic HMS-like dataset (40 patients, 2 events per EEG) so the whole pipeline runs in seconds on CPU."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hms.data import VOTES  # noqa: E402

CHANNELS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
            "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]


@pytest.fixture(scope="session")
def fake_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("hms")
    (root / "train_eegs").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(40):
        eeg_id = 1000 + pid
        sig = rng.normal(0, 20, (120 * 200, 20)).astype(np.float32)
        sig[rng.integers(0, len(sig), 50), 3] = np.nan  # a few NaNs, like the real data
        pd.DataFrame(sig, columns=CHANNELS).to_parquet(root / "train_eegs" / f"{eeg_id}.parquet")
        for e, (n, offsets) in enumerate([(int(rng.integers(10, 16)), [0.0, 4.0, 8.0]), (3, [60.0, 66.0])]):
            votes = rng.multinomial(n, rng.dirichlet(np.ones(6) * 0.5))
            for sub, off in enumerate(offsets):
                rows.append({"eeg_id": eeg_id, "eeg_sub_id": sub, "eeg_label_offset_seconds": off,
                             "spectrogram_id": 5000 + pid, "spectrogram_sub_id": sub,
                             "spectrogram_label_offset_seconds": off, "label_id": len(rows), "patient_id": 900 + pid,
                             "expert_consensus": "x", **dict(zip(VOTES, votes))})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    work = root / "work"
    cfg = root / "config.yaml"
    cfg.write_text(f"data_dir: {root.as_posix()}\nwork_dir: {work.as_posix()}\n"
                   "fs_out: 100\nspan_s: 60\nwin_s: 50\nn_folds: 5\nseed: 42\n")
    return root, cfg
