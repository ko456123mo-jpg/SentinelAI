# 🛡️ SentinelAI — AI Cybersecurity Assistant

**Unified Threat Detection & Response Platform** — an end-to-end AI system that
detects attacks in network traffic, spam/phishing in messages and malware in
files, then *decides* the response automatically.

**Project level:** Level 3 — Expert (all mandatory components + Deep Learning +
NLP + Computer Vision + Reinforcement Learning + AI Agent).

---

## 👨‍💻 Developer

| | |
|---|---|
| **Name** | **Mohammed Moneer Al-absi** |
| **Student ID** | 2023050086 |
| **Section** | Cybersecurity (CS) |
| **Supervisor** | Eng. Sondos Saif |
| **Project** | AI Final Project — 2026 |

---

## ⚡ Quick Start

> **Requires Python 3.10 – 3.13** (TensorFlow does not support 3.14+ yet).

### Windows — one click
1. **Code → Download ZIP** → extract the folder
2. Open the folder → **double-click `run_windows.bat`**
   (first run installs everything automatically, then the browser opens at `http://localhost:7860`)

### Linux / Kali — one command
```bash
git clone https://github.com/ko456123mo-jpg/SentinelAI.git
cd SentinelAI
bash run_kali.sh     # finds/creates the environment, installs, starts the console
```

### Manual (any OS)
```bash
pip install -r requirements.txt
python -m src.webapp               # web console → http://localhost:7860
python -m src.main --stages agent  # run the AI agent on 27 live events
python -m src.main                 # retrain everything from scratch (~3 min)
python -m unittest discover -s tests   # 17/17 tests must pass
```

> All datasets, trained models (17 artifacts) and results ship **inside the repo** —
> the console works immediately after install, no training needed.

---

## 1. Problem definition (Stage 1)

| Item | Description |
|------|-------------|
| **Problem** | SOC/security teams face thousands of heterogeneous events per day (network flows, messages, files). Manual triage is slow and inconsistent, novel (zero-day) behaviour is missed by signature tools, and the *response* decision (block? isolate? monitor?) is left to overworked analysts. |
| **Why it matters** | Slow or wrong responses let attacks through (false negatives) or disrupt legitimate users (false positives). |
| **Who uses it** | Security analysts, network/SOC operators. |
| **What the AI does** | (1) *Classifies* each network flow (Benign/DDoS/PortScan/BruteForce/Botnet), (2) *detects anomalies* without labels (zero-day candidates), (3) *detects spam/phishing* text, (4) *identifies the malware family* from byte-plot images, (5) *chooses the response* with a reinforcement-learned policy, (6) *writes an incident report* with recommendations. |

## 2. Data sources (Stage 2)

| Dataset | Type | Size | Source |
|---------|------|------|--------|
| Network flows | tabular, 16 features, 5 classes | 25,000+ rows | **Generated** — seeded statistical models per class, fully documented in `src/data_collection.py` (allowed by the project requirements) |
| SMS Spam | text, ham/spam | 5,574 messages | **REAL public dataset** — UCI ML Repository “SMS Spam Collection” (Almeida & Hidalgo) |
| Malware images | 48×48 grayscale byte-plots, 4 families | 720 images | **Generated** — malware-visualization approach (Nataraj et al., 2011) |
| Threat-intel feed | malware domain blocklist | 18,000+ domains | **REAL public feed** downloaded over HTTPS at runtime (web/network programming) — stamparm/blackbook |

## 3. Pipeline (11 stages of the project outline)

```
Stage 1  Problem definition .......... README §1 + report
Stage 2  Data collection ............. src/data_collection.py
Stage 3  Preprocessing ............... src/preprocessing.py   (duplicates, missing,
                                    one-hot, log1p, scaling, SelectKBest, 80/20 split)
Stage 4  Supervised learning ......... src/supervised_model.py (LogReg, KNN, Tree,
                                    RF, SVM, NaiveBayes + 5-fold CV + GridSearch
                                    + class-imbalance experiment)
Stage 5  Evaluation .................. metrics + confusion matrix + ROC + FP/FN
Stage 6  Unsupervised learning ....... src/unsupervised_model.py (K-Means, DBSCAN,
                                    Hierarchical/Ward + dendrogram, PCA,
                                    IsolationForest trained on benign only)
Stage 7  Deep learning ............... src/deep_learning.py (MLP vs classic ML)
Stage 8  NLP ......................... src/nlp.py (TF-IDF + LogReg/NB + LSTM
                                    sequence model, real data)
Stage 9  Computer vision ............. src/computer_vision.py (CNN + HOG/SVM
                                    baseline comparison, 4 families)
Stage 10 Reinforcement learning ...... src/reinforcement_learning.py
                                    (hand-implemented Q-Learning response policy)
Stage 11 AI agent .................... src/agent.py (SentinelAgent: routing,
                                    second opinion, risk score, RL decision, report)
```

## 4. Project structure

```
AI_Project/
├── data/
│   ├── raw/            network_flows.csv, sms_spam.csv, malware_images/
│   ├── processed/      flows_train.csv, flows_test.csv
│   └── test/           held-out events used by the agent demo
├── src/                one module per pipeline stage (+ main.py orchestrator)
├── models/             persisted models (.joblib / .keras) + rl_policy.csv
├── results/
│   ├── figures/        all generated charts (~30 PNG)
│   ├── predictions/    exported predictions, misclassified cases, incidents
│   └── reports/        per-stage JSON metrics + automated security report
├── tests/              unit + integration tests (unittest)
├── docs/               full documentation report + presentation + study guide
├── requirements.txt
└── README.md
```

## 5. How to run

```bash
pip install -r requirements.txt

# 1) download the real SMS dataset (optional but recommended -
#    a documented fallback corpus is used automatically if missing)
mkdir -p data/raw && cd data/raw
curl -LO https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip && cd ../..

# 2) run the whole pipeline (data -> models -> figures -> agent report)
python -m src.main

# partial runs
python -m src.main --stages collect,preprocess,supervised
python -m src.main --skip cv,dl

# tests
python -m unittest discover -s tests -v
```

## 6. The agent workflow (Stage 11)

```
event (flow / message / file image)
   └─> ROUTE: choose the right engine
        ├─ flow    -> supervised classifier  +  IsolationForest 2nd opinion
        ├─ message -> TF-IDF + LogisticRegression (spam engine)
        └─ image   -> CNN family classifier
   └─> ENRICH: threat-intel IOC matching (18k real blacklisted domains)
   └─> FUSE: prediction + confidence + anomaly flag + IOC -> risk 0-100
   └─> DECIDE: RL policy (learned Q-table) -> Monitor | Alert | Block | Isolate
   └─> ACT: incident record + aggregate security report + recommendations
```

## 7. Key design decisions (rubric: justify choices)

- **Median imputation** for missing numeric values → robust to heavy right tails.
- **log1p** on byte/packet counters → compresses skew before scaling.
- **StandardScaler fitted on train only** → no information leakage to the test set.
- **SelectKBest (mutual information, k=12)** → removes noisy features, speeds SVM.
- **Stratified 80/20 split** → preserves the intentional class imbalance.
- **SVM fitted on a capped 10k subsample** → O(n²) training cost (documented trade-off).
- **IsolationForest trained on benign traffic only** → realistic one-class SOC setup that can catch *unseen* attack behaviour.
- **Hand-written Q-learning update** → the learning-from-rewards process is fully explainable, not hidden inside a library.
- **MLP compared against classic ML on identical splits** → fair comparison required by the outline.

## 8. Results

All metrics are written to `results/reports/*.json` after each run and
summarized in `results/reports/pipeline_summary.json`; the automated agent
report is `results/reports/security_report.md`. Highlights are listed in
`docs/` (report & presentation) — see `docs/SentinelAI_Report.docx`.

## 9. References

1. Almeida, T. & Hidalgo, J. — *SMS Spam Collection*, UCI ML Repository.
2. Nataraj, L. et al. (2011) — *Malware Images: Visualization and Automatic
   Classification*, VizSec.
3. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12 (2011).
4. Chollet, F. et al. — Keras (TensorFlow), 2015.
5. Sutton, R. & Barto, A. — *Reinforcement Learning: An Introduction* (2018),
   Q-learning chapter.
6. Liu, F. T. et al. (2008) — *Isolation Forest*, ICDM.
