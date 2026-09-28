# 🛡️ NetGuard AI
**Machine Learning-Based Network Attack Classification and Detection System**

Classifies network flows into **Benign, DoS/DDoS, Probe, Brute Force, Web Attack, Botnet, Other** and wraps the
model in a SOC-style Streamlit console.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
A trained model is included (`models/`). To retrain: `python train.py` (add `--fast` for a quicker run).

## Unique features
| Feature | What it does |
|---|---|
| **Live SOC Console** | Replays scenarios (DDoS storm, botnet outbreak, port-scan wave…) with live threat chart, alert feed and live accuracy |
| **Threat Score (0-100)** | Probabilities weighted by attack severity → LOW / MEDIUM / HIGH / CRITICAL |
| **Explainable AI** | Occlusion analysis shows which features and behaviour families drove a verdict (no SHAP needed) |
| **What-if analyzer** | Load a realistic flow, edit any value, re-classify |
| **Response playbooks** | Mitigation steps per attack type |
| **Batch analysis** | Upload CSV → predictions, dashboard, download results |
| **Model Lab** | Model comparison (weighted F1), confusion matrix, per-class metrics, feature importance |
| **History** | SQLite log of every prediction |

## Use the real CICIDS2017 dataset
1. Download the *MachineLearningCVE* CSVs (Canadian Institute for Cybersecurity).
2. `python train.py --data path/to/MachineLearningCVE/`
   Labels are auto-mapped (`DoS Hulk`/`DDoS` → DoS/DDoS, `PortScan` → Probe, `FTP/SSH-Patator` → Brute Force,
   `Web Attack …` → Web Attack, `Bot` → Botnet, rest → Other), inf/NaN/duplicates removed, classes capped for balance.
3. Restart the app - it uses the new model automatically.

> The bundled model is trained on **synthetic CICIDS-style flows** so the project runs offline. Retrain on real data
> before drawing real security conclusions.

## Project structure
```
app.py            Streamlit console          train.py   train + compare + save
src/config.py     features, classes, playbooks, severity
src/data_gen.py   synthetic flow generator & scenarios
src/preprocessing.py  cleaning, aliases, real-data loader
src/inference.py  prediction, threat score, explanations
src/db.py         SQLite history
models/           netguard_model.joblib, metrics.json
data/sample_traffic.csv    sample for the Batch tab
tests/test_smoke.py        pytest smoke test
```

## Deploy (Streamlit Community Cloud)
Push to GitHub → share.streamlit.io → select repo → main file `app.py`. Or `docker build -t netguard . && docker run -p 8501:8501 netguard`.

## Mini-project report outline
Abstract · Problem · Dataset · Preprocessing · Models & metrics (weighted F1) · Explainability · System design · Results · Future work (real-time packet capture with Scapy/CICFlowMeter, deep learning, SHAP).
