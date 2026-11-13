# Authors: Tran Anh Tu, Huynh Hoang Nhut
"""Load out-of-fold (val) or locked-test predictions of a run."""
import pandas as pd

from .data import CLASSES

PCOLS = [f"p_{c}" for c in CLASSES]


def load_preds(results, name, split):
    files = sorted((results / name).glob("fold*/preds.parquet"))
    if not files:
        raise FileNotFoundError(f"no preds for run {name}")
    df = pd.concat([pd.read_parquet(f) for f in files])
    df = df[df.split == split]
    if split == "test":  # one test prediction per fold model -> average the fold ensemble
        df = df.groupby("event_id", as_index=False)[PCOLS].mean()
    return df.set_index("event_id")
