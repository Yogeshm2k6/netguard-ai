"""NetGuard AI - Network Attack Classification & Threat Intelligence Console."""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import db
from src.config import CLASSES, COLORS, FEATURES, PLAYBOOK, threat_level
from src.data_gen import SCENARIOS, generate, generate_scenario
from src.inference import explain_flow, load_bundle, predict_df
from src.preprocessing import clean_features

MODEL_PATH = Path("models/netguard_model.joblib")
st.set_page_config(page_title="NetGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 0%,#0f2740 0%,#0a1220 45%,#070b14 100%);}
.hero{padding:18px 26px;border-radius:16px;border:1px solid #1f3a5a;
  background:linear-gradient(120deg,#0d2a47cc,#12203acc);margin-bottom:10px}
.hero h1{margin:0;font-size:2.0rem;color:#e6f1ff;letter-spacing:.5px}
.hero p{margin:4px 0 0;color:#8fb3d9}
.kpi{border:1px solid #1f3a5a;border-radius:14px;padding:14px 16px;background:#0c1a2ecc}
.kpi .l{color:#7fa2c8;font-size:.78rem;text-transform:uppercase;letter-spacing:1px}
.kpi .v{font-size:1.8rem;font-weight:700;color:#e6f1ff}
.badge{display:inline-block;padding:4px 14px;border-radius:999px;font-weight:700;color:#0a1220}
.pill{border-left:4px solid;padding:10px 14px;border-radius:8px;background:#0c1a2e;margin:6px 0}
</style>""", unsafe_allow_html=True)

PLOT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=40, b=10))


@st.cache_resource(show_spinner="Loading model...")
def get_bundle():
    if not MODEL_PATH.exists():
        with st.spinner("First run: training model (~20s)..."):
            subprocess.run([sys.executable, "train.py", "--fast"], check=True)
    return load_bundle(str(MODEL_PATH))


bundle = get_bundle()


def kpi(label, value, color="#e6f1ff"):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v" style="color:{color}">{value}</div></div>'


def donut(counts: pd.Series, title):
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.58,
                           marker_colors=[COLORS[c] for c in counts.index], sort=False))
    fig.update_layout(title=title, height=330, **PLOT)
    return fig


def gauge(score):
    lvl, col = threat_level(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, title={"text": f"Threat Score - {lvl}"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": col},
               "steps": [{"range": [0, 15], "color": "#12351f"}, {"range": [15, 45], "color": "#3a2f10"},
                         {"range": [45, 75], "color": "#3d2410"}, {"range": [75, 100], "color": "#3d1414"}]}))
    fig.update_layout(height=280, **PLOT)
    return fig


# ------------------------------------------------------------------ sidebar / header
with st.sidebar:
    st.markdown("## 🛡️ NetGuard AI")
    st.caption("Machine Learning-Based Network Attack Classification & Detection System")
    st.markdown(f"**Active model:** {bundle['best_model']}")
    st.markdown(f"**Data:** {bundle['source']}")
    rep = bundle["comparison"].iloc[0]
    st.metric("Accuracy", f"{rep['Accuracy']*100:.2f}%")
    st.metric("Weighted F1", f"{rep['Weighted F1']*100:.2f}%")
    st.divider()
    alert_thr = st.slider("Alert threshold (Threat Score)", 10, 90, 45)
    st.caption("Flows above this score are raised as alerts.")

st.markdown('<div class="hero"><h1>🛡️ NETGUARD AI</h1>'
            '<p>Real-time network attack classification · explainable predictions · threat scoring · response playbooks</p></div>',
            unsafe_allow_html=True)

tab_live, tab_one, tab_batch, tab_lab, tab_hist, tab_about = st.tabs(
    ["🛰️ Live SOC Console", "🔍 Flow Analyzer", "📂 Batch Analysis", "🧠 Model Lab", "🗂️ History", "ℹ️ About"])

# ------------------------------------------------------------------ LIVE SOC
with tab_live:
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    scenario = c1.selectbox("Attack scenario", list(SCENARIOS), index=5)
    batch = c2.slider("Flows / tick", 10, 200, 60)
    ticks = c3.slider("Ticks", 3, 40, 12)
    speed = c4.slider("Speed (s/tick)", 0.1, 1.5, 0.35)
    b1, b2, _ = st.columns([1, 1, 4])
    run = b1.button("▶ Start simulation", type="primary", width="stretch")
    if b2.button("↺ Reset", width="stretch"):
        st.session_state.pop("stream", None)

    ph_k = st.empty()
    l, r = st.columns([3, 2])
    ph_line, ph_pie = l.empty(), r.empty()
    ph_feed = st.empty()

    def render_live():
        s = st.session_state.get("stream")
        if s is None or s.empty:
            ph_k.info("Choose a scenario and press **Start simulation** to watch NetGuard classify traffic live.")
            return
        att = s[s.Prediction != "Benign"]
        acc = (s.Prediction == s.Truth).mean() * 100
        with ph_k.container():
            k = st.columns(5)
            k[0].markdown(kpi("Flows analysed", f"{len(s):,}"), unsafe_allow_html=True)
            k[1].markdown(kpi("Attacks flagged", f"{len(att):,}", "#ef4444"), unsafe_allow_html=True)
            k[2].markdown(kpi("Attack rate", f"{len(att)/len(s)*100:.1f}%", "#f59e0b"), unsafe_allow_html=True)
            k[3].markdown(kpi("Avg threat", f"{s.Threat.mean():.1f}"), unsafe_allow_html=True)
            k[4].markdown(kpi("Live accuracy", f"{acc:.1f}%", "#22c55e"), unsafe_allow_html=True)
        tl = s.groupby("Tick").agg(Threat=("Threat", "mean")).reset_index()
        f = px.area(tl, x="Tick", y="Threat", title="Network threat level over time")
        f.update_traces(line_color="#ef4444", fillcolor="rgba(239,68,68,.25)")
        f.update_layout(height=330, yaxis_range=[0, 100], **PLOT)
        ph_line.plotly_chart(f, width="stretch", key=f"line{time.time_ns()}")
        ph_pie.plotly_chart(donut(s.Prediction.value_counts().reindex(CLASSES).dropna(), "Traffic composition"),
                            width="stretch", key=f"pie{time.time_ns()}")
        hot = s[s.Threat >= alert_thr].tail(8).iloc[::-1][["Tick", "Prediction", "Confidence", "Threat", "Truth"]]
        with ph_feed.container():
            st.markdown("#### 🚨 Latest alerts")
            st.dataframe(hot, hide_index=True, width="stretch")

    if run:
        st.session_state.stream = pd.DataFrame(columns=["Tick", "Prediction", "Confidence", "Threat", "Truth"])
        for t in range(1, ticks + 1):
            d = generate_scenario(batch, scenario)
            res, _ = predict_df(bundle, d)
            res["Truth"], res["Tick"] = d["Label"].values, t
            st.session_state.stream = pd.concat([st.session_state.stream, res], ignore_index=True)
            render_live()
            time.sleep(speed)
        db.log("live-sim", st.session_state.stream)
    else:
        render_live()

# ------------------------------------------------------------------ SINGLE FLOW
with tab_one:
    def load_scenario():
        cls = st.session_state.get("preset", "Benign")
        d = generate(400, {c: float(c == cls) for c in CLASSES}, seed=int(time.time()) % 10**6, stealth=0)
        row = d.iloc[np.random.randint(len(d))]
        for f in FEATURES:
            st.session_state[f"in_{f}"] = float(row[f])

    if "in_Destination Port" not in st.session_state:
        st.session_state["preset"] = "Benign"
        load_scenario()

    st.markdown("Load a **realistic flow profile**, tweak any value, and watch the prediction react (what-if analysis).")
    p1, p2 = st.columns([2, 1])
    p1.selectbox("Traffic profile", CLASSES, key="preset")
    p2.button("🎲 Load random flow of this type", on_click=load_scenario, width="stretch")

    cols = st.columns(4)
    vals = {}
    for i, f in enumerate(FEATURES):
        vals[f] = cols[i % 4].number_input(f, key=f"in_{f}", format="%.3f" if "Mean" in f or "/s" in f else "%.1f")
    if st.button("🔍 Analyse flow", type="primary"):
        row = pd.DataFrame([vals])
        res, P = predict_df(bundle, row)
        pred, conf, threat = res.Prediction[0], res.Confidence[0], res.Threat[0]
        lvl, lcol = threat_level(threat)
        g1, g2 = st.columns([1, 1.4])
        g1.plotly_chart(gauge(threat), width="stretch")
        with g2:
            st.markdown(f"### Prediction: <span class='badge' style='background:{COLORS[pred]}'>{pred}</span>"
                        f" &nbsp; Confidence **{conf:.1f}%**", unsafe_allow_html=True)
            pf = pd.DataFrame({"Class": bundle["classes"], "Probability": P[0] * 100}).sort_values("Probability")
            fig = px.bar(pf, x="Probability", y="Class", orientation="h", color="Class",
                         color_discrete_map=COLORS, title="Class probabilities (%)")
            fig.update_layout(height=250, showlegend=False, **PLOT)
            st.plotly_chart(fig, width="stretch")
        st.markdown(f"<div class='pill' style='border-color:{lcol}'><b>🛠 Response playbook ({lvl}):</b> "
                    f"{PLAYBOOK[pred]}</div>", unsafe_allow_html=True)

        st.markdown("#### 🧠 Why this prediction? (explainable AI)")
        _, feat, grp = explain_flow(bundle, row)
        e1, e2 = st.columns([1.5, 1])
        top = feat.head(8).iloc[::-1]
        fig = px.bar(top, x="Impact", y="Feature", orientation="h", title=f"Top drivers of '{pred}'",
                     color=top.Impact.apply(lambda v: "Supports" if v > 0 else "Opposes"),
                     color_discrete_map={"Supports": "#ef4444" if pred != "Benign" else "#22c55e", "Opposes": "#38bdf8"})
        fig.update_layout(height=340, legend_title="", **PLOT)
        e1.plotly_chart(fig, width="stretch")
        fig = px.bar(grp.sort_values("Impact"), x="Impact", y="Family", orientation="h", title="Behaviour families")
        fig.update_traces(marker_color="#a855f7")
        fig.update_layout(height=340, **PLOT)
        e2.plotly_chart(fig, width="stretch")
        st.caption("Impact = drop in the predicted class probability when a feature (or family) is replaced "
                   "with its typical *benign* value. Bigger = more responsible for the verdict.")
        db.log("single", res)

# ------------------------------------------------------------------ BATCH
with tab_batch:
    st.markdown("Upload a CSV of flows (CICIDS-style column names, extra columns are ignored).")
    up = st.file_uploader("Network traffic CSV", type="csv")
    demo = generate_scenario(400, "Full-spectrum attack", seed=7)
    st.download_button("⬇ Download sample CSV", demo.drop(columns="Label").to_csv(index=False),
                       "sample_traffic.csv", "text/csv")
    use_demo = st.checkbox("…or analyse the built-in sample")
    raw = pd.read_csv(up) if up is not None else (demo.drop(columns="Label") if use_demo else None)
    if raw is not None:
        try:
            X = clean_features(raw)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        res, P = predict_df(bundle, X)
        out = pd.concat([raw.reset_index(drop=True), res], axis=1)
        out.insert(0, "Flow", np.arange(1, len(out) + 1))
        att = res[res.Prediction != "Benign"]
        k = st.columns(5)
        k[0].markdown(kpi("Flows", f"{len(res):,}"), unsafe_allow_html=True)
        k[1].markdown(kpi("Benign", f"{(res.Prediction=='Benign').sum():,}", "#22c55e"), unsafe_allow_html=True)
        k[2].markdown(kpi("Attacks", f"{len(att):,}", "#ef4444"), unsafe_allow_html=True)
        k[3].markdown(kpi("Attack rate", f"{len(att)/len(res)*100:.1f}%", "#f59e0b"), unsafe_allow_html=True)
        k[4].markdown(kpi("Avg confidence", f"{res.Confidence.mean():.1f}%"), unsafe_allow_html=True)
        a, b = st.columns(2)
        a.plotly_chart(donut(res.Prediction.value_counts().reindex(CLASSES).dropna(), "Attack distribution"),
                       width="stretch")
        fig = px.histogram(res, x="Threat", nbins=30, title="Threat score distribution", color_discrete_sequence=["#ef4444"])
        fig.update_layout(height=330, **PLOT)
        b.plotly_chart(fig, width="stretch")
        if len(att):
            st.info(f"Most common attack: **{att.Prediction.value_counts().idxmax()}** "
                    f"· {int((res.Threat >= alert_thr).sum())} flows above alert threshold ({alert_thr}).")
        st.markdown("#### Top 15 riskiest flows")
        st.dataframe(out.sort_values("Threat", ascending=False).head(15)[["Flow", "Prediction", "Confidence", "Threat"] + FEATURES[:4]],
                     hide_index=True, width="stretch")
        st.download_button("⬇ Download full results (CSV)", out.to_csv(index=False), "netguard_results.csv", "text/csv",
                           type="primary")
        db.log("batch", res)

# ------------------------------------------------------------------ MODEL LAB
with tab_lab:
    st.markdown(f"**Data source:** {bundle['source']} · train {bundle['n_train']:,} / test {bundle['n_test']:,} rows. "
                "Model chosen by **weighted F1**, not just accuracy.")
    comp = bundle["comparison"].copy()
    st.dataframe(comp.style.format({c: "{:.4f}" for c in comp.columns if c != "Model"} | {"Train time (s)": "{:.1f}"})
                 .highlight_max(subset=["Weighted F1"], color="#14532d"), hide_index=True, width="stretch")
    x1, x2 = st.columns(2)
    cm = bundle["confusion"].astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig = px.imshow(cmn, x=bundle["classes"], y=bundle["classes"], text_auto=".2f", color_continuous_scale="Blues",
                    title="Normalised confusion matrix", labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(height=430, **PLOT)
    x1.plotly_chart(fig, width="stretch")
    rp = bundle["report"].reset_index().rename(columns={"index": "Class"})
    fig = px.bar(rp.melt(id_vars="Class", value_vars=["precision", "recall", "f1-score"]), x="Class", y="value",
                 color="variable", barmode="group", title="Per-class precision / recall / F1")
    fig.update_layout(height=430, yaxis_range=[0, 1], legend_title="", **PLOT)
    x2.plotly_chart(fig, width="stretch")
    imp = bundle["importance"].head(12).iloc[::-1]
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Global feature importance (permutation, ΔF1)")
    fig.update_traces(marker_color="#38bdf8")
    fig.update_layout(height=420, **PLOT)
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ HISTORY
with tab_hist:
    h = db.recent(300)
    if h.empty:
        st.info("No predictions logged yet. Run the simulator, analyse a flow or upload a CSV.")
    else:
        st.dataframe(h, hide_index=True, width="stretch")
        fig = px.histogram(h, x="Prediction", color="Prediction", color_discrete_map=COLORS, title="Logged predictions")
        fig.update_layout(height=320, showlegend=False, **PLOT)
        st.plotly_chart(fig, width="stretch")
        if st.button("🗑 Clear history"):
            db.clear()
            st.rerun()

# ------------------------------------------------------------------ ABOUT
with tab_about:
    st.markdown("""
### NetGuard AI: Machine Learning-Based Network Attack Classification and Detection System
**Pipeline:** data → cleaning (inf/NaN/duplicates) → signed-log transform → scaling → model comparison
(Logistic Regression, Random Forest, Gradient Boosting, XGBoost if installed) → best model by weighted F1 →
Streamlit console.

**What makes it different**
- **Threat Score (0-100):** class probabilities weighted by attack severity, so an uncertain botnet call ranks above a confident port-scan.
- **Occlusion-based explainability:** shows which features / behaviour families drove each verdict (no SHAP dependency).
- **Live SOC simulator:** replay attack scenarios (DDoS storm, botnet outbreak…) with live accuracy against ground truth.
- **Response playbooks:** every class maps to concrete mitigation steps.
- **What-if analysis:** edit any flow value and re-classify.
- **SQLite history + CSV export.**

**Using real data:** download CICIDS2017 *MachineLearningCVE* CSVs and run
`python train.py --data path/to/csvs/` — labels are mapped automatically to the 7 classes and the app picks up the new model.

⚠️ The bundled model is trained on *synthetic* CICIDS-style flows so the project runs offline out of the box; retrain on real data before drawing security conclusions.
""")
