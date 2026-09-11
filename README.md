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

> ⚠️ **Kali 2025.2+ ships Python 3.14**, which TensorFlow does not support yet.
> `run_kali.sh` will detect this and print the fix — or see the full
> step-by-step guide: **`docs/KALI_SETUP_AR.md`** (Arabic, incl. troubleshooting).

### Manual (any OS)
```bash
pip install -r requirements.txt
python -m src.webapp               # web console → http://localhost:7860
python -m src.main --stages agent  # run the AI agent on 27 live events
python -m src.main                 # retrain everything from scratch (~3 min)
python -m unittest discover -s tests   # 21/21 tests must pass
```

> All datasets, trained models (17 artifacts) and results ship **inside the repo** —
> the console works immediately after install, no training needed.
>
> **First full retrain** (`python -m src.main`) auto-downloads the two large real
> corpora (CICIDS2017 ~300 MB + Malimg ~1.2 GB) once into gitignored caches;
> without internet it falls back to the documented generators automatically.

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
| Network flows | tabular, 16 features, 5 classes | ~32k real flows | **REAL captured traffic** — CICIDS2017 (Canadian Institute for Cybersecurity): 5 days of PCAP-derived flows mapped to the 16-feature schema (documented in `src/data_collection.py`); a documented synthetic generator remains as the offline fallback |
| SMS Spam | text, ham/spam | 5,572 messages | **REAL public dataset** — UCI ML Repository "SMS Spam Collection" (Almeida & Hidalgo) |
| Malware images | 48×48 grayscale byte-plots, 4 families | ~750 images | **REAL corpus** — Malimg (Nataraj et al., 2011): the families Allaple.A, C2LOP.P, Lolyda.AA2 and Alueron.gen!J resized to 48×48; synthetic fallback included |
| Threat-intel feed | malware domain blocklist | 18,000+ domains | **REAL public feed** downloaded over HTTPS at runtime (web/network programming) — stamparm/blackbook |

> The two large raw corpora (CICIDS2017 parquet files + the Malimg archive)
> are **auto-downloaded on first run** into gitignored cache paths, so the
> repository stays lean while every model is trained on real data. Offline?
> The pipeline transparently falls back to the documented generators and
> records which source was used in `data/raw/dataset_documentation.json`.

## 3. Pipeline (11 stages of the project outline)

```
Stage 1  Problem definition .......... README §1 + report
Stage 2  Data collection ............. src/data_collection.py   (REAL CICIDS2017
                                    flows + REAL Malimg images + UCI SMS + live
                                    threat feed; generated data = offline fallback)
Stage 3  Preprocessing ............... src/preprocessing.py   (duplicates, inf/NaN,
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

- **Real CICIDS2017 flows mapped to the 16-feature schema** → the models are
  trained on genuinely captured traffic (BENIGN/DoS/PortScan/Patator/Bot
  remapped to Benign/DDoS/PortScan/BruteForce/Botnet), with the mapping and
  per-class sampling caps documented in `src/data_collection.py`.
- **Real Malimg byte-plots resized to 48×48** → the families the system models
  (Allaple.A, C2LOP.P, Lolyda.AA2, Alueron.gen!J) come from the corpus the
  byte-plot method was introduced on, so the CNN learns real visual signatures.
- **Median imputation** for missing numeric values → robust to heavy right tails.
- **log1p** on byte/packet counters → compresses skew before scaling.
- **StandardScaler fitted on train only** → no information leakage to the test set.
- **SelectKBest (mutual information, k=12)** → removes noisy features, speeds SVM.
- **Stratified 80/20 split** → preserves the intentional class imbalance.
- **SVM fitted on a capped 10k subsample** → O(n²) training cost (documented trade-off).
- **IsolationForest trained on benign traffic only** → realistic one-class SOC setup that can catch *unseen* attack behaviour.
- **Hand-written Q-learning update** → the learning-from-rewards process is fully explainable, not hidden inside a library.
- **MLP compared against classic ML on identical splits** → fair comparison required by the outline.

## 8. Requirements coverage matrix

| Requirement (from the outline PDF) | Status | Implementation |
|---|---|---|
| 3.1 Data collection | ✅ | `src/data_collection.py` — 4 real sources + fallbacks |
| 3.1 Data preprocessing | ✅ | `src/preprocessing.py` — every step justified |
| 3.1 ≥1 supervised model | ✅ | 6 algorithms + CV + GridSearch (`supervised_model.py`) |
| 3.1 ≥1 unsupervised technique | ✅ | K-Means, DBSCAN, Hierarchical, PCA, IsolationForest (`unsupervised_model.py`) |
| 3.1 Model evaluation | ✅ | accuracy/precision/recall/F1 + confusion matrix + ROC + FP/FN |
| 3.1 Visualization | ✅ | 37 figures in `results/figures/` |
| 3.1 Documentation & presentation | ✅ | 13-section DOCX report + 17-slide PPTX + web console |
| 3.2 Deep Learning | ✅ | MLP (tabular) + LSTM (text) + CNN (images) |
| 3.2 NLP | ✅ | `src/nlp.py` on real UCI SMS data |
| 3.2 Computer Vision | ✅ | `src/computer_vision.py` CNN + HOG/SVM baseline |
| 3.2 Reinforcement Learning | ✅ | hand-implemented Q-Learning (`reinforcement_learning.py`) |
| 3.2 AI Agent | ✅ | SentinelAgent routing + risk fusion + RL decision + report |
| 4. Project structure | ✅ | `data/ src/ models/ results/ tests/ requirements.txt README.md` |
| 5. Stages 1–11 | ✅ | one module per stage + `main.py` orchestrator |
| 6. Project level | ✅ | **Level 3 — Expert** (everything + RL + Agent) |
| 8. Documentation structure (13 sections) | ✅ | `docs/SentinelAI_Report.docx` |
| 9. Grading rubric (100) | ✅ | every component has evidence (see §9 below) |

## 9. Grading-rubric self-assessment

| Component | Marks | Evidence |
|---|---:|---|
| Problem & dataset | 10 | §1 + real CICIDS2017/Malimg/UCI/threat-feed datasets |
| Data collection & preprocessing | 15 | §2 + justified 8-step preprocessing |
| Supervised learning | 15 | 6 models, 5-fold CV, GridSearch, imbalance handling |
| Unsupervised learning | 10 | 5 techniques + supervised-vs-unsupervised discussion |
| Advanced AI technique | 10 | DL + NLP + CV + RL + Agent (all five) |
| Evaluation & visualization | 10 | full metrics + 37 figures |
| Documentation | 10 | 13-section report + README + Arabic guides |
| Presentation & discussion | 10 | 17-slide deck + live console + `docs/Discussion_Q&A_AR.md` |

## 10. Results

All metrics are written to `results/reports/*.json` after each run and
summarized in `results/reports/pipeline_summary.json`; the automated agent
report is `results/reports/security_report.md`. Highlights are listed in
`docs/` (report & presentation) — see `docs/SentinelAI_Report.docx`.

## 11. References

1. Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. (2018) — *Toward
   Generating a New Intrusion Detection Dataset and Intrusion Traffic
   Characterization* (CICIDS2017), Canadian Institute for Cybersecurity.
2. Almeida, T. & Hidalgo, J. — *SMS Spam Collection*, UCI ML Repository.
3. Nataraj, L. et al. (2011) — *Malware Images: Visualization and Automatic
   Classification*, VizSec.
4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12 (2011).
5. Chollet, F. et al. — Keras (TensorFlow), 2015.
6. Sutton, R. & Barto, A. — *Reinforcement Learning: An Introduction* (2018),
   Q-learning chapter.
7. Liu, F. T. et al. (2008) — *Isolation Forest*, ICDM.
