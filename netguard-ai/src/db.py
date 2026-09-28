"""Tiny SQLite prediction history."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB = Path("data/history.db")


def _conn():
    DB.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS predictions(
        ts TEXT, source TEXT, prediction TEXT, confidence REAL, threat REAL)""")
    return c


def log(source: str, results: pd.DataFrame, limit: int = 500):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(now, source, r.Prediction, float(r.Confidence), float(r.Threat))
            for r in results.head(limit).itertuples()]
    with _conn() as c:
        c.executemany("INSERT INTO predictions VALUES (?,?,?,?,?)", rows)


def recent(n=300) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query(
            "SELECT ts AS Time, source AS Source, prediction AS Prediction, confidence AS Confidence, "
            "threat AS Threat FROM predictions ORDER BY rowid DESC LIMIT ?", c, params=(n,))


def clear():
    with _conn() as c:
        c.execute("DELETE FROM predictions")
