"""Synthetic CICIDS-style flow generator (demo, live simulator, offline training).

Each attack family gets behavioural signatures similar to what CICIDS2017 shows, with heavy
overlap plus a share of 'stealthy' attack flows that look benign, so the task is not trivial.
"""
import numpy as np
import pandas as pd

from .config import CLASSES, FEATURES

SCENARIOS = {
    "Normal business day": {"Benign": 0.93, "DoS/DDoS": 0.02, "Probe": 0.02, "Brute Force": 0.01,
                            "Web Attack": 0.01, "Botnet": 0.005, "Other": 0.005},
    "DDoS storm": {"Benign": 0.35, "DoS/DDoS": 0.60, "Probe": 0.02, "Brute Force": 0.01,
                   "Web Attack": 0.01, "Botnet": 0.005, "Other": 0.005},
    "Recon / port-scan wave": {"Benign": 0.45, "DoS/DDoS": 0.02, "Probe": 0.48, "Brute Force": 0.02,
                               "Web Attack": 0.01, "Botnet": 0.01, "Other": 0.01},
    "Credential-stuffing attack": {"Benign": 0.5, "DoS/DDoS": 0.02, "Probe": 0.05, "Brute Force": 0.38,
                                   "Web Attack": 0.03, "Botnet": 0.01, "Other": 0.01},
    "Botnet outbreak": {"Benign": 0.55, "DoS/DDoS": 0.05, "Probe": 0.03, "Brute Force": 0.02,
                        "Web Attack": 0.02, "Botnet": 0.30, "Other": 0.03},
    "Full-spectrum attack": {"Benign": 0.30, "DoS/DDoS": 0.18, "Probe": 0.14, "Brute Force": 0.12,
                             "Web Attack": 0.10, "Botnet": 0.10, "Other": 0.06},
}
TRAIN_MIX = {"Benign": 0.40, "DoS/DDoS": 0.17, "Probe": 0.12, "Brute Force": 0.10,
             "Web Attack": 0.08, "Botnet": 0.08, "Other": 0.05}


def _ln(rng, mu, sigma, k):
    return rng.lognormal(mu, sigma, k)


def _pick(rng, ports, probs, k):
    p = np.array(probs, dtype=float)
    return rng.choice(ports, size=k, p=p / p.sum())


def _profile(rng, cls, k):
    b = lambda p: (rng.random(k) < p).astype(float)
    if cls == "Benign":
        return dict(port=_pick(rng, [80, 443, 53, 22, 8080, 123, 445, 3389, 25, 21],
                               [22, 40, 14, 4, 4, 3, 4, 3, 3, 3], k),
                    dur=_ln(rng, 11.5, 2.6, k), fp=1 + _ln(rng, 1.6, 1.0, k), bp=1 + _ln(rng, 1.6, 1.1, k),
                    fl=_ln(rng, 4.6, 1.2, k), bl=_ln(rng, 5.6, 1.5, k),
                    syn=b(0.25), ack=b(0.85), psh=b(0.45), rst=b(0.05), fin=b(0.35),
                    win=np.clip(_ln(rng, 9.3, 1.4, k), 0, 65535), idle=_ln(rng, 10, 3, k))
    if cls == "DoS/DDoS":
        slow = rng.random(k) < 0.35
        return dict(port=_pick(rng, [80, 443, 8080, 53], [70, 20, 6, 4], k),
                    dur=np.where(slow, _ln(rng, 17.0, 0.8, k), _ln(rng, 10.2, 1.6, k)),
                    fp=np.where(slow, 2 + _ln(rng, 1.0, 0.5, k), 3 + _ln(rng, 1.8, 0.8, k)),
                    bp=np.where(slow, _ln(rng, 0.3, 0.7, k), 1 + _ln(rng, 1.0, 0.8, k)),
                    fl=_ln(rng, 3.0, 1.0, k), bl=np.where(slow, _ln(rng, 2.0, 1.2, k), _ln(rng, 5.8, 1.3, k)),
                    syn=b(0.55), ack=b(0.5), psh=b(0.35), rst=b(0.3), fin=b(0.1),
                    win=np.where(slow, 29200.0, _ln(rng, 8.2, 1.2, k)),
                    idle=np.where(slow, _ln(rng, 16, 1, k), _ln(rng, 8, 2, k)))
    if cls == "Probe":
        return dict(port=np.where(rng.random(k) < 0.85, rng.integers(1, 65535, k),
                                  _pick(rng, [22, 80, 443, 445, 3389], [1] * 5, k)),
                    dur=_ln(rng, 8.7, 1.1, k), fp=1 + rng.poisson(0.8, k), bp=rng.poisson(0.6, k).astype(float),
                    fl=_ln(rng, 1.2, 1.0, k), bl=_ln(rng, 0.8, 1.2, k),
                    syn=b(0.9), ack=b(0.25), psh=b(0.05), rst=b(0.6), fin=b(0.05),
                    win=_pick(rng, [1024, 2048, 29200, 5840, 14600], [30, 20, 25, 15, 10], k).astype(float),
                    idle=_ln(rng, 5, 2, k))
    if cls == "Brute Force":
        return dict(port=_pick(rng, [22, 21, 3389, 23], [50, 35, 10, 5], k),
                    dur=_ln(rng, 13.0, 0.8, k), fp=8 + _ln(rng, 2.2, 0.5, k), bp=8 + _ln(rng, 2.1, 0.5, k),
                    fl=_ln(rng, 3.4, 0.6, k), bl=_ln(rng, 3.6, 0.7, k),
                    syn=b(0.3), ack=b(0.95), psh=b(0.9), rst=b(0.1), fin=b(0.7),
                    win=np.clip(_ln(rng, 8.9, 0.9, k), 0, 65535), idle=_ln(rng, 9, 2, k))
    if cls == "Web Attack":
        return dict(port=_pick(rng, [80, 443, 8080, 8000], [55, 25, 12, 8], k),
                    dur=_ln(rng, 12.4, 1.1, k), fp=4 + _ln(rng, 1.8, 0.7, k), bp=3 + _ln(rng, 1.7, 0.8, k),
                    fl=_ln(rng, 5.4, 0.8, k), bl=_ln(rng, 6.6, 1.0, k),
                    syn=b(0.2), ack=b(0.9), psh=b(0.95), rst=b(0.08), fin=b(0.4),
                    win=np.clip(_ln(rng, 9.6, 0.9, k), 0, 65535), idle=_ln(rng, 10, 2, k))
    if cls == "Botnet":
        return dict(port=np.where(rng.random(k) < 0.6,
                                  _pick(rng, [8080, 6667, 8888, 443, 4444], [30, 25, 15, 20, 10], k),
                                  rng.integers(1024, 65535, k)),
                    dur=_ln(rng, 13.6, 1.0, k), fp=3 + _ln(rng, 1.6, 0.6, k), bp=3 + _ln(rng, 1.4, 0.6, k),
                    fl=_ln(rng, 4.5, 0.6, k), bl=_ln(rng, 4.9, 0.8, k),
                    syn=b(0.3), ack=b(0.9), psh=b(0.7), rst=b(0.06), fin=b(0.3),
                    win=np.clip(_ln(rng, 8.6, 0.9, k), 0, 65535), idle=_ln(rng, 15.5, 0.8, k))
    # Other: infiltration / exfiltration
    return dict(port=_pick(rng, [443, 445, 139, 8443], [45, 30, 15, 10], k),
                dur=_ln(rng, 15.8, 1.3, k), fp=6 + _ln(rng, 2.4, 0.8, k), bp=25 + _ln(rng, 3.4, 0.7, k),
                fl=_ln(rng, 4.0, 0.8, k), bl=_ln(rng, 7.0, 0.5, k),
                syn=b(0.2), ack=b(0.95), psh=b(0.6), rst=b(0.05), fin=b(0.3),
                win=np.clip(_ln(rng, 9.8, 0.8, k), 0, 65535), idle=_ln(rng, 14, 1.5, k))


def _finalise(rng, d):
    dur = np.maximum(d["dur"], 1.0)
    fp, bp = np.maximum(np.round(d["fp"]), 1), np.maximum(np.round(d["bp"]), 0)
    flen, blen = fp * d["fl"], bp * d["bl"]
    tot = fp + bp
    sec = dur / 1e6
    return pd.DataFrame({
        "Destination Port": np.asarray(d["port"], dtype=float), "Flow Duration": dur,
        "Total Fwd Packets": fp, "Total Backward Packets": bp,
        "Total Length Fwd Packets": flen, "Total Length Bwd Packets": blen,
        "Fwd Packet Length Mean": d["fl"], "Bwd Packet Length Mean": np.where(bp > 0, d["bl"], 0.0),
        "Flow Bytes/s": (flen + blen) / sec, "Flow Packets/s": tot / sec,
        "Flow IAT Mean": dur / np.maximum(tot - 1, 1),
        "Fwd IAT Mean": dur / np.maximum(fp - 1, 1) * rng.uniform(0.7, 1.3, len(dur)),
        "SYN Flag Count": d["syn"], "ACK Flag Count": d["ack"], "PSH Flag Count": d["psh"],
        "RST Flag Count": d["rst"], "FIN Flag Count": d["fin"],
        "Average Packet Size": (flen + blen) / np.maximum(tot, 1),
        "Init Win Bytes Forward": np.asarray(d["win"], dtype=float), "Idle Mean": d["idle"],
    })[FEATURES]


def generate(n=20000, mix=None, seed=42, stealth=0.07) -> pd.DataFrame:
    """Generate n labelled flows. `stealth` = share of attacks that mimic benign traffic."""
    rng = np.random.default_rng(seed)
    mix = mix or TRAIN_MIX
    probs = np.array([mix.get(c, 0) for c in CLASSES], dtype=float)
    labels = rng.choice(CLASSES, size=n, p=probs / probs.sum())
    frames = []
    for c in CLASSES:
        k = int((labels == c).sum())
        if k == 0:
            continue
        df = _finalise(rng, _profile(rng, c, k))
        if c != "Benign" and stealth > 0:
            m = rng.random(k) < stealth
            if m.any():
                df.loc[m, :] = _finalise(rng, _profile(rng, "Benign", int(m.sum()))).values
        df["Label"] = c
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def generate_scenario(n, scenario="Normal business day", seed=None):
    seed = int(np.random.randint(1, 10**9)) if seed is None else seed
    return generate(n, SCENARIOS[scenario], seed=seed)
