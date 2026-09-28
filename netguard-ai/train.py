"""Train, compare and save the NetGuard AI model.

    python train.py                       # synthetic CICIDS-style data (works offline)
    python train.py --data path/to/CICIDS2017/   # real dataset (file or folder of CSVs)
"""
import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import CLASSES, FEATURES
from src.data_gen import generate
from src.preprocessing import load_real_dataset, signed_log1p

warnings.filterwarnings("ignore")


def make_pipe(clf):
    return Pipeline([("log", FunctionTransformer(signed_log1p, validate=True)),
                     ("scale", StandardScaler()), ("clf", clf)])


def candidates(fast):
    c = {
        "Logistic Regression": LogisticRegression(max_iter=400, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=60 if fast else 120, max_depth=16,
                                                min_samples_leaf=2, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=42),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(max_iter=150 if fast else 250,
                                                                     learning_rate=0.1, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        c["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.1,
                                     subsample=0.9, n_jobs=-1, eval_metric="mlogloss")
    except Exception:
        pass
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="CICIDS2017/2018 CSV file or folder (optional)")
    ap.add_argument("--n", type=int, default=30000, help="synthetic rows if no --data")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()

    if a.data:
        print(f"Loading real dataset from {a.data} ...")
        df, source = load_real_dataset(a.data), f"Real dataset: {a.data}"
    else:
        print(f"Generating {a.n:,} synthetic CICIDS-style flows ...")
        df, source = generate(a.n, seed=42), "Synthetic CICIDS-style flows (demo)"
    df = df.drop_duplicates().reset_index(drop=True)

    classes = [c for c in CLASSES if c in set(df["Label"])]
    y = df["Label"].map({c: i for i, c in enumerate(classes)}).values
    X = df[FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train {len(Xtr):,} | Test {len(Xte):,} | classes: {classes}\n")

    rows, fitted = [], {}
    for name, clf in candidates(a.fast).items():
        t = time.time()
        pipe = make_pipe(clf).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        r = dict(Model=name, Accuracy=accuracy_score(yte, pred),
                 **{"Weighted F1": f1_score(yte, pred, average="weighted"),
                    "Macro F1": f1_score(yte, pred, average="macro")},
                 **{"Train time (s)": time.time() - t})
        rows.append(r), fitted.__setitem__(name, pipe)
        print(f"{name:30s} acc={r['Accuracy']:.4f}  wF1={r['Weighted F1']:.4f}  ({r['Train time (s)']:.1f}s)")

    comp = pd.DataFrame(rows).sort_values("Weighted F1", ascending=False).reset_index(drop=True)
    best = comp.loc[0, "Model"]
    pipe = fitted[best]
    pred = pipe.predict(Xte)
    print(f"\nBest model by weighted F1: {best}\n")
    print(classification_report(yte, pred, target_names=classes, digits=4))

    print("Computing permutation importance ...")
    sub = np.random.RandomState(0).choice(len(Xte), min(2500, len(Xte)), replace=False)
    pi = permutation_importance(pipe, Xte.iloc[sub], yte[sub], n_repeats=3, scoring="f1_weighted",
                                random_state=0, n_jobs=-1)
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean}) \
        .sort_values("Importance", ascending=False)

    report = classification_report(yte, pred, target_names=classes, output_dict=True)
    bundle = dict(
        pipeline=pipe, classes=classes, features=FEATURES, best_model=best, source=source,
        comparison=comp, importance=importance,
        report=pd.DataFrame(report).T.loc[classes],
        confusion=confusion_matrix(yte, pred), n_train=len(Xtr), n_test=len(Xte),
        benign_median=X[df["Label"] == "Benign"].median(),
        class_median={c: X[df["Label"] == c].median() for c in classes},
    )
    joblib.dump(bundle, "models/netguard_model.joblib", compress=3)
    json.dump(dict(best_model=best, source=source, accuracy=float(accuracy_score(yte, pred)),
                   weighted_f1=float(f1_score(yte, pred, average="weighted")),
                   macro_f1=float(f1_score(yte, pred, average="macro"))),
              open("models/metrics.json", "w"), indent=2)
    print("Saved models/netguard_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
