"""Cleaning, alias handling and real-dataset loading."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALIASES, FEATURES, map_label


def signed_log1p(X):
    """Compress heavy-tailed network features (bytes/s, durations...)."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {c: ALIASES[c] for c in df.columns if ALIASES.get(c)}
    return df.rename(columns=rename)


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only model features, numeric, with inf/NaN handled."""
    df = normalise_columns(df)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def load_real_dataset(path: str, per_class_cap: int = 30000, seed: int = 42) -> pd.DataFrame:
    """Load CICIDS2017/2018 CSV file(s) (file or folder) -> features + 7-class 'Label'."""
    p = Path(path)
    files = sorted(p.glob("*.csv")) if p.is_dir() else [p]
    frames = []
    for f in files:
        d = normalise_columns(pd.read_csv(f, low_memory=False, encoding="latin1"))
        label_col = "Label" if "Label" in d.columns else d.columns[-1]
        d["Label"] = d[label_col].map(map_label)
        missing = [c for c in FEATURES if c not in d.columns]
        if missing:
            print(f"  skip {f.name}: missing {missing}")
            continue
        frames.append(d[FEATURES + ["Label"]])
    if not frames:
        raise SystemExit("No usable CSV files found.")
    df = pd.concat(frames, ignore_index=True)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = pd.concat([X, df["Label"]], axis=1).dropna().drop_duplicates()
    parts = [g.sample(min(len(g), per_class_cap), random_state=seed) for _, g in df.groupby("Label")]
    return pd.concat(parts, ignore_index=True)
