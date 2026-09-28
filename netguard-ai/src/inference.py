"""Model loading, prediction, Threat Score and occlusion-based explanations."""
import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, GROUPS, SEVERITY


def load_bundle(path="models/netguard_model.joblib"):
    return joblib.load(path)


def predict_df(bundle, X: pd.DataFrame):
    """Return (results DataFrame, probability matrix)."""
    P = bundle["pipeline"].predict_proba(X[FEATURES])
    classes = bundle["classes"]
    idx = P.argmax(1)
    sev = np.array([SEVERITY[c] for c in classes], dtype=float)
    out = pd.DataFrame({
        "Prediction": [classes[i] for i in idx],
        "Confidence": np.round(P.max(1) * 100, 2),
        "Threat": np.round(P @ sev, 1),
    })
    return out, P


def explain_flow(bundle, row: pd.DataFrame):
    """Occlusion explanation: how much does the predicted-class probability drop when a feature
    (or a whole feature family) is replaced by its *benign median*? Model-agnostic, no extra deps."""
    pipe, ref, classes = bundle["pipeline"], bundle["benign_median"], bundle["classes"]
    row = row[FEATURES].reset_index(drop=True)
    base = pipe.predict_proba(row)[0]
    ci = int(base.argmax())

    def occlude(cols):
        r = row.copy()
        for c in cols:
            r[c] = ref[c]
        return r

    feat_rows = pd.concat([occlude([f]) for f in FEATURES], ignore_index=True)
    grp_rows = pd.concat([occlude(v) for v in GROUPS.values()], ignore_index=True)
    fe = base[ci] - pipe.predict_proba(feat_rows)[:, ci]
    ge = base[ci] - pipe.predict_proba(grp_rows)[:, ci]
    feat = pd.DataFrame({"Feature": FEATURES, "Impact": fe, "Value": row.iloc[0].values})
    feat = feat.reindex(feat.Impact.abs().sort_values(ascending=False).index)
    grp = pd.DataFrame({"Family": list(GROUPS), "Impact": ge})
    return classes[ci], feat, grp
