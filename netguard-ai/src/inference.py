"""Model loading, prediction, Threat Score and occlusion-based explanations."""
import joblib
import numpy as np
import pandas as pd

from .config import FEATURES, GROUPS, SEVERITY


def train_fallback_bundle():
    """Quickly train a model bundle if pickled model has version incompatibilities."""
    import os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from .data_gen import generate
    from .preprocessing import signed_log1p

    df = generate(15000, seed=42).drop_duplicates().reset_index(drop=True)
    classes = [c for c in CLASSES if c in set(df["Label"])]
    y = df["Label"].map({c: i for i, c in enumerate(classes)}).values
    X = df[FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = Pipeline([
        ("log", FunctionTransformer(signed_log1p, validate=True)),
        ("scale", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=60, max_depth=16, random_state=42, n_jobs=-1))
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    comp = pd.DataFrame([{
        "Model": "Random Forest",
        "Accuracy": accuracy_score(yte, pred),
        "Weighted F1": f1_score(yte, pred, average="weighted"),
        "Macro F1": f1_score(yte, pred, average="macro"),
        "Train time (s)": 1.0
    }])

    sub = np.random.RandomState(0).choice(len(Xte), min(1000, len(Xte)), replace=False)
    pi = permutation_importance(pipe, Xte.iloc[sub], yte[sub], n_repeats=2, scoring="f1_weighted", random_state=0)
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean}).sort_values("Importance", ascending=False)
    report = classification_report(yte, pred, target_names=classes, output_dict=True)

    bundle = dict(
        pipeline=pipe, classes=classes, features=FEATURES, best_model="Random Forest",
        source="Synthetic CICIDS-style flows (auto-trained)",
        comparison=comp, importance=importance,
        report=pd.DataFrame(report).T.loc[classes],
        confusion=confusion_matrix(yte, pred), n_train=len(Xtr), n_test=len(Xte),
        benign_median=X[df["Label"] == "Benign"].median(),
        class_median={c: X[df["Label"] == c].median() for c in classes},
    )
    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, "models/netguard_model.joblib", compress=3)
    return bundle


def load_bundle(path="models/netguard_model.joblib"):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Warning: Could not unpickle model ({e}). Re-training compatible model bundle on the fly...")
        return train_fallback_bundle()


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
