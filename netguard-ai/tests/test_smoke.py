import numpy as np
from src.data_gen import generate
from src.inference import explain_flow, load_bundle, predict_df
from src.preprocessing import clean_features


def test_pipeline():
    b = load_bundle()
    d = generate(300, seed=3)
    res, P = predict_df(b, clean_features(d))
    assert len(res) == 300 and np.allclose(P.sum(1), 1)
    assert res.Threat.between(0, 100).all()
    cls, feat, grp = explain_flow(b, d.drop(columns="Label").iloc[[0]])
    assert cls in b["classes"] and len(feat) == 20 and len(grp) == 4
