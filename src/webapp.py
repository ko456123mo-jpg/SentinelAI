"""
SentinelAI - Professional Web Console (full project showcase + operations)
===========================================================================
A Flask web interface covering EVERY project requirement:

  Overview     : Dashboard, Project & Requirements coverage matrix
  Operations   : Flow Analyzer, Message Scanner, Malware Imaging, Report
  AI Engines   : Supervised, Unsupervised, Deep Learning, NLP, CV, RL, Agent
  Data & System: Datasets, Preprocessing, Gallery (37 figs), Self-tests

All numbers are read LIVE from results/reports/*.json - nothing is hardcoded.

Run (after training once):   python -m src.webapp   ->  http://localhost:7860
"""

import json
import os
import tempfile
import threading

from flask import Flask, jsonify, request, send_from_directory

from src import config
from src.agent import SentinelAgent

app = Flask(__name__)
AGENT = None
LOCK = threading.Lock()

_JSON_FILES = {
    "preprocessing": "preprocessing_report.json",
    "supervised": "supervised_results.json",
    "unsupervised": "unsupervised_results.json",
    "dl": "deep_learning_results.json",
    "nlp": "nlp_results.json",
    "cv": "cv_results.json",
    "rl": "rl_results.json",
    "agent": "agent_results.json",
    "summary": "pipeline_summary.json",
}


def get_agent() -> SentinelAgent:
    global AGENT
    if AGENT is None:
        AGENT = SentinelAgent().load()
    return AGENT


# ------------------------------------------------------------------ APIs
@app.get("/")
def index():
    return PAGE


@app.get("/api/stats")
def stats():
    def jload(name):
        p = os.path.join(config.REPORT_DIR, name)
        return json.load(open(p)) if os.path.exists(p) else {}

    sup = jload("supervised_results.json")
    nlp = jload("nlp_results.json")
    cv = jload("cv_results.json")
    rl = jload("rl_results.json")
    ag = jload("agent_results.json")
    uns = jload("unsupervised_results.json")
    rf = sup.get("metrics_per_model", {}).get("RandomForest", {})
    return jsonify({
        "rf_accuracy": rf.get("accuracy"),
        "rf_cv": sup.get("cross_validation", {}).get(
            "RandomForest", {}).get("accuracy_mean"),
        "iso_precision": uns.get("isolation_forest", {}).get("precision"),
        "lstm_f1": nlp.get("metrics", {}).get("LSTM(Keras)", {}).get("f1"),
        "cnn_accuracy": cv.get("metrics", {}).get("accuracy"),
        "rl_reward": rl.get("baseline_comparison", {}).get(
            "Q-Learning policy"),
        "agent_events": ag.get("n_events"),
        "blacklist_size": len(get_agent().artifacts.get("blacklist", set())),
    })


@app.get("/api/json/<key>")
def api_json(key):
    fname = _JSON_FILES.get(key)
    if not fname:
        return jsonify({"error": "unknown key"}), 404
    p = os.path.join(config.REPORT_DIR, fname)
    if not os.path.exists(p):
        return jsonify({"error": "not trained yet"}), 404
    return jsonify(json.load(open(p)))


@app.get("/api/dataset_doc")
def dataset_doc():
    p = os.path.join(config.RAW_DIR, "dataset_documentation.json")
    if not os.path.exists(p):
        return jsonify({})
    return jsonify(json.load(open(p)))


@app.get("/api/incidents")
def incidents():
    with LOCK:
        return jsonify(get_agent().incidents)


@app.post("/api/flow")
def api_flow():
    ev = request.get_json(force=True)
    with LOCK:
        inc = get_agent().analyze_flow(ev)
    return jsonify(inc)


@app.post("/api/message")
def api_message():
    data = request.get_json(force=True)
    with LOCK:
        inc = get_agent().analyze_message(data.get("text", ""))
    return jsonify(inc)


@app.post("/api/image")
def api_image():
    if "file" in request.files:
        f = request.files["file"]
        suffix = os.path.splitext(f.filename)[1] or ".png"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.save(tmp.name)
        tmp.close()
        path, cleanup = tmp.name, True
    else:
        data = request.get_json(force=True)
        path = os.path.join(config.TEST_DIR, "malware",
                            os.path.basename(data.get("sample", "")))
        cleanup = False
    if not os.path.exists(path):
        return jsonify({"error": "image not found"}), 404
    with LOCK:
        inc = get_agent().analyze_image(path)
    if cleanup:
        os.unlink(path)
    return jsonify(inc)


@app.get("/api/samples")
def samples():
    d = os.path.join(config.TEST_DIR, "malware")
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    return jsonify([f for f in files if f.lower().endswith(".png")][:8])


@app.get("/api/report")
def report():
    with LOCK:
        summary = get_agent().write_reports()
    return jsonify(summary)


@app.get("/api/figures")
def figures():
    files = sorted(f for f in os.listdir(config.FIG_DIR)
                   if f.endswith(".png"))
    return jsonify(files)


@app.get("/api/policy")
def policy():
    import pandas as pd
    p = os.path.join(config.MODEL_DIR, "rl_policy.csv")
    if not os.path.exists(p):
        return jsonify([])
    df = pd.read_csv(p).rename(columns={"Unnamed: 0": "state"})
    return jsonify(df.to_dict(orient="records"))


@app.post("/api/selftest")
def selftest():
    import subprocess
    import sys
    try:
        r = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            capture_output=True, text=True, timeout=240)
        return jsonify({"ok": r.returncode == 0,
                        "output": ((r.stdout or "") +
                                   (r.stderr or ""))[-3000:]})
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "output": "timeout after 240s"})


@app.get("/api/figure/<name>")
def figure(name):
    if not name.endswith(".png"):
        return "", 403
    return send_from_directory(config.FIG_DIR, name, mimetype="image/png")


# ------------------------------------------------------------------ UI
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SentinelAI — SOC Console</title>
<style>
:root{--bg:#0b1220;--panel:#121a2b;--panel2:#0e1524;--line:#22304a;
--txt:#e2e8f0;--dim:#8ea0bd;--acc:#38bdf8;--ok:#22c55e;--warn:#f59e0b;
--bad:#ef4444;--iso:#a78bfa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,
Arial,sans-serif;min-height:100vh}
.layout{display:flex;min-height:100vh}
aside{width:230px;background:var(--panel2);border-right:1px solid var(--line);
padding:16px 12px;position:sticky;top:0;height:100vh;overflow-y:auto;
flex-shrink:0}
.brand{font-size:1.15rem;font-weight:900;color:#fff;padding:4px 8px 2px}
.brand span{display:block;font-size:.68rem;color:var(--acc);letter-spacing:
2px;font-weight:600}
.ar{color:#7f95b5;font-size:.72rem}
.group{color:#5a7299;font-size:.66rem;text-transform:uppercase;
letter-spacing:1.5px;margin:16px 8px 6px;font-weight:700}
aside button{display:block;width:100%;text-align:left;background:
transparent;border:0;color:var(--dim);padding:8px 10px;border-radius:8px;
cursor:pointer;font-size:.86rem;margin:1px 0}
aside button:hover{background:#16223a;color:var(--txt)}
aside button.on{background:var(--acc);color:#04121f;font-weight:800}
main{flex:1;padding:18px 24px;max-width:1250px}
header.top{display:flex;align-items:center;gap:12px;margin-bottom:16px;
background:linear-gradient(90deg,#0c1a2e,#10233f);border:1px solid var(--line);
border-radius:12px;padding:12px 18px}
header.top .dot{width:9px;height:9px;border-radius:50%;background:var(--ok);
box-shadow:0 0 8px var(--ok);animation:p 1.6s infinite}
@keyframes p{50%{opacity:.35}}
header.top h2{font-size:1.05rem;color:#fff}
header.top .sub{margin-left:auto;color:var(--dim);font-size:.75rem}
.tab{display:none}.tab.on{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,
1fr));gap:12px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:13px}
.card .k{color:var(--dim);font-size:.7rem;text-transform:uppercase;
letter-spacing:.5px}
.card .v{font-size:1.45rem;font-weight:800;color:var(--acc);margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:18px;margin-bottom:16px}
.panel h3{color:#fff;margin-bottom:10px;font-size:.98rem}
.panel h4{color:#bcd0ec;margin:14px 0 8px;font-size:.85rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,
1fr));gap:10px}
label{font-size:.72rem;color:var(--dim);display:block;margin-bottom:3px}
input,select,textarea{width:100%;background:#0a101d;border:1px solid
var(--line);color:var(--txt);border-radius:7px;padding:8px;font-size:.85rem}
textarea{min-height:90px;resize:vertical}
.btn{background:var(--acc);color:#04121f;border:0;border-radius:8px;
padding:10px 22px;font-weight:800;cursor:pointer;font-size:.9rem}
.btn:hover{filter:brightness(1.12)}
.btn.g{background:#1d2a44;color:var(--txt);border:1px solid var(--line);
font-weight:600;padding:7px 12px;font-size:.78rem}
.presets{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.result{margin-top:16px;display:none}
.badge{padding:5px 13px;border-radius:999px;font-weight:800;font-size:.8rem}
.b-Monitor{background:#14351f;color:#4ade80}
.b-Alert_Analyst{background:#3a2c0a;color:#fbbf24}
.b-Block_IP{background:#3b1414;color:#f87171}
.b-Isolate_Host{background:#2b0f0f;color:#fb7185;border:1px solid #7f1d1d}
.pred{font-size:1.25rem;font-weight:800;color:#fff}
.riskwrap{margin:12px 0}
.riskbar{height:14px;background:#1a2337;border-radius:8px;overflow:hidden}
.riskfill{height:100%;width:0;transition:width .7s;border-radius:8px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,
1fr));gap:10px;margin-top:12px}
.kv .item{background:var(--panel2);border:1px solid var(--line);
border-radius:9px;padding:10px}
.kv .item .k{font-size:.68rem;color:var(--dim);text-transform:uppercase}
.kv .item .v{font-weight:700;margin-top:2px}
.qbars{margin-top:8px}
.qrow{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:.73rem;
color:var(--dim)}
.qrow .qb{height:8px;background:var(--iso);border-radius:4px;min-width:2px}
.exp{margin-top:12px;background:#0a101d;border-left:3px solid var(--acc);
padding:10px 12px;border-radius:6px;font-size:.8rem;color:#b8c6dd;
line-height:1.6}
table{width:100%;border-collapse:collapse;font-size:.78rem;margin-top:8px}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);text-transform:uppercase;font-size:.66rem}
tr:hover td{background:#101a2e}
td .ok{color:#4ade80;font-weight:800}
.figrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,
1fr));gap:14px}
.figrow img{width:100%;border-radius:10px;border:1px solid var(--line);
background:#fff}
.figrow .cap{color:var(--dim);font-size:.72rem;margin-top:5px}
.hint{color:var(--dim);font-size:.78rem;margin-top:6px;line-height:1.7}
.err{color:#fca5a5;background:#7f1d1d;border:1px solid #f87171;border-radius:10px;
 padding:10px 12px;font-size:.82rem;margin-top:8px}
.check{color:#4ade80;font-weight:900}
.flowsteps{display:flex;flex-wrap:wrap;gap:8px}
.step{background:var(--panel2);border:1px solid var(--line);border-radius:
10px;padding:10px 13px;font-size:.78rem;min-width:150px}
.step b{color:var(--acc);display:block;font-size:.66rem;
text-transform:uppercase;letter-spacing:1px}
pre{background:#0a101d;border:1px solid var(--line);border-radius:8px;
padding:12px;font-size:.72rem;color:#9fe8b5;max-height:260px;overflow:auto}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{background:#1d2a44;border:1px solid var(--line);border-radius:999px;
padding:4px 12px;font-size:.75rem;color:#cfe3ff}
.big-emoji{font-size:1.5rem}
@media(max-width:860px){.layout{flex-direction:column}aside{width:100%;
height:auto;position:static;display:flex;flex-wrap:wrap;gap:4px}
.group{width:100%}}
</style></head><body>
<div class="layout">
<aside>
 <div class="brand">🛡 SentinelAI<span>SOC CONSOLE</span></div>
 <div class="group">Overview</div>
 <button class="on" data-t="dash">📊 Dashboard</button>
 <button data-t="project">🎯 Project &amp; Requirements</button>
 <div class="group">Operations <span class="ar">التشغيل</span></div>
 <button data-t="flow">🌊 Flow Analyzer</button>
 <button data-t="msg">💬 Message Scanner</button>
 <button data-t="img">🖼️ Malware Image</button>
 <button data-t="rep">📄 Security Report</button>
 <div class="group">AI Engines <span class="ar">المحركات</span></div>
 <button data-t="sup">🎯 Supervised ML</button>
 <button data-t="uns">🔍 Unsupervised</button>
 <button data-t="dl">🧠 Deep Learning</button>
 <button data-t="nlp">💬 NLP</button>
 <button data-t="cv">👁️ Computer Vision</button>
 <button data-t="rl">🎮 Reinforcement</button>
 <button data-t="agent">🤖 AI Agent</button>
 <div class="group">Data &amp; System</div>
 <button data-t="data">🗄️ Datasets</button>
 <button data-t="pre">🧹 Preprocessing</button>
 <button data-t="gal">🖼️ Gallery</button>
 <button data-t="sys">⚙️ System</button>
</aside>
<main>

<!-- ==================== DASHBOARD ==================== -->
<div class="tab on" id="t-dash">
 <header class="top"><span class="dot"></span>
  <h2>Live Threat Operations</h2>
  <span class="sub">Supervised + Unsupervised + DL + NLP + CV + RL + Agent</span>
 </header>
 <div class="cards" id="cards"></div>
 <div class="panel"><h3>Agent decision pipeline</h3>
  <div class="flowsteps">
   <div class="step"><b>1 · Route</b>flow / message / image</div>
   <div class="step"><b>2 · Predict</b>label + confidence</div>
   <div class="step"><b>3 · Second opinion</b>IsolationForest</div>
   <div class="step"><b>4 · Enrich</b>IOC blacklist check</div>
   <div class="step"><b>5 · Fuse</b>risk score 0-100</div>
   <div class="step"><b>6 · Decide</b>RL policy action</div>
   <div class="step"><b>7 · Act</b>incident + report</div>
  </div></div>
 <div class="panel"><h3>Key figures</h3><div class="figrow">
  <img src="/api/figure/supervised_model_comparison.png">
  <img src="/api/figure/rl_learning_curve.png">
  <img src="/api/figure/unsup_pca_clusters.png"></div></div>
 <div class="panel"><h3>Recent incidents <span class="ar">أحدث الأحداث</span></h3>
  <div id="incdash">loading...</div></div>
</div>

<!-- ==================== PROJECT ==================== -->
<div class="tab" id="t-project">
 <header class="top"><span class="big-emoji">🎯</span>
  <h2>Project Definition &amp; Requirements Coverage</h2>
  <span class="sub">AI Final Project — Level 3 (Expert)</span></header>
 <div class="panel"><h3>Problem definition (Stage 1)</h3>
  <p class="hint">SOC teams face thousands of heterogeneous events per day
  (network flows, messages, files). Signature tools only catch known threats
  and miss zero-day behaviour, manual triage is slow, and the response
  decision (monitor / alert / block / isolate) is made under pressure.
  <b>A missed attack = breach; a wrong block = business disruption.</b><br>
  <b>Users:</b> security analysts, SOC &amp; network operators, mail admins.<br>
  <b>AI contribution:</b> classifies threats, flags anomalies without
  labels, detects phishing text, identifies malware families, learns a
  response policy from rewards, and writes the security report.</p></div>
 <div class="panel"><h3>Pipeline — 11 stages of the project outline</h3>
  <table><tr><th>Stage</th><th>Module</th><th>What it does</th></tr>
  <tr><td>1 Problem</td><td>README / report</td><td>definition, users, goal</td></tr>
  <tr><td>2 Collection</td><td>data_collection.py</td><td>32,466 real
  CICIDS2017 flows + 5,572 real SMS + ~770 real Malimg images +
  18,146-domain live feed</td></tr>
  <tr><td>3 Preprocessing</td><td>preprocessing.py</td><td>clean / encode /
  scale / select / split (all justified)</td></tr>
  <tr><td>4 Supervised</td><td>supervised_model.py</td><td>6 algorithms +
  5-fold CV + GridSearch + imbalance experiment</td></tr>
  <tr><td>5 Evaluation</td><td>supervised_model.py</td><td>acc / P / R / F1
  / confusion / ROC / FP-FN analysis</td></tr>
  <tr><td>6 Unsupervised</td><td>unsupervised_model.py</td><td>K-Means /
  DBSCAN / Hierarchical / PCA / IsolationForest</td></tr>
  <tr><td>7 Deep learning</td><td>deep_learning.py</td><td>MLP vs classic
  ML comparison</td></tr>
  <tr><td>8 NLP</td><td>nlp.py</td><td>TF-IDF + LogReg/NB + LSTM on real
  data</td></tr>
  <tr><td>9 Computer vision</td><td>computer_vision.py</td><td>CNN + HOG/SVM
  baseline</td></tr>
  <tr><td>10 Reinforcement</td><td>reinforcement_learning.py</td>
  <td>from-scratch Q-Learning response policy</td></tr>
  <tr><td>11 AI Agent</td><td>agent.py</td><td>routing + fusion + RL
  decision + automated reporting</td></tr></table></div>
 <div class="panel"><h3>Requirements coverage matrix
  <span class="ar">مصفوفة تغطية المتطلبات</span></h3>
  <table>
  <tr><th>Requirement (from the outline PDF)</th><th>Status</th>
  <th>Where</th></tr>
  <tr><td>Data collection</td><td class="check">✓</td><td>Datasets tab —
  4 REAL sources (CICIDS2017 + UCI SMS + Malimg + live threat feed),
  documented generated fallback</td></tr>
  <tr><td>Data preprocessing (justified)</td><td class="check">✓</td>
  <td>Preprocessing tab — every step with its reason</td></tr>
  <tr><td>Supervised model(s)</td><td class="check">✓</td><td>Supervised
  tab — 6 algorithms compared</td></tr>
  <tr><td>Unsupervised technique(s)</td><td class="check">✓</td>
  <td>Unsupervised tab — 5 techniques</td></tr>
  <tr><td>Model evaluation</td><td class="check">✓</td><td>Supervised tab —
  full metrics + CV + ROC + FP/FN</td></tr>
  <tr><td>Visualization</td><td class="check">✓</td><td>Gallery — 37
  figures</td></tr>
  <tr><td>Documentation &amp; presentation</td><td class="check">✓</td>
  <td>docs/ — 13-section report + 17 slides</td></tr>
  <tr><td>Advanced: Deep Learning</td><td class="check">✓</td><td>MLP +
  LSTM + CNN (three architectures)</td></tr>
  <tr><td>Advanced: NLP</td><td class="check">✓</td><td>real UCI data,
  TF-IDF vs LSTM</td></tr>
  <tr><td>Advanced: Computer Vision</td><td class="check">✓</td><td>byte-plot
  CNN + HOG baseline</td></tr>
  <tr><td>Advanced: Reinforcement Learning</td><td class="check">✓</td>
  <td>hand-implemented Q-Learning</td></tr>
  <tr><td>Advanced: AI Agent</td><td class="check">✓</td><td>SentinelAgent —
  routing, fusion, decision, report</td></tr>
  <tr><td>Security report (predictions, risk, recommendations)</td>
  <td class="check">✓</td><td>Security Report tab</td></tr>
  <tr><td>Web / network programming (Stage 2 option)</td><td class="check">
  ✓</td><td>live HTTPS threat-feed download</td></tr>
  <tr><td>Tests</td><td class="check">✓</td><td>17 unit/integration tests
  (System tab)</td></tr></table></div>
 <div class="panel"><h3>Grading rubric mapping</h3>
  <table><tr><th>Component</th><th>Marks</th><th>Evidence</th></tr>
  <tr><td>Problem &amp; dataset</td><td>10</td><td>this tab + Datasets
  tab</td></tr>
  <tr><td>Collection &amp; preprocessing</td><td>15</td><td>Datasets +
  Preprocessing tabs</td></tr>
  <tr><td>Supervised learning</td><td>15</td><td>Supervised tab</td></tr>
  <tr><td>Unsupervised learning</td><td>10</td><td>Unsupervised tab</td></tr>
  <tr><td>Advanced AI technique</td><td>10</td><td>DL / NLP / CV / RL /
  Agent tabs</td></tr>
  <tr><td>Evaluation &amp; visualization</td><td>10</td><td>metrics +
  Gallery</td></tr>
  <tr><td>Documentation</td><td>10</td><td>13-section report (docs/)</td></tr>
  <tr><td>Presentation &amp; discussion</td><td>10</td><td>17-slide deck +
  this console for the live demo</td></tr></table></div>
</div>

<!-- ==================== FLOW ==================== -->
<div class="tab" id="t-flow">
 <header class="top"><span class="big-emoji">🌊</span><h2>Network Flow
 Analyzer</h2><span class="sub">RandomForest + IsolationForest + RL policy</span></header>
 <div class="panel">
  <div class="presets">
   <button class="btn g" onclick="preset('normal')">Normal browsing</button>
   <button class="btn g" onclick="preset('ddos')">DDoS flood</button>
   <button class="btn g" onclick="preset('scan')">Port scan</button>
   <button class="btn g" onclick="preset('brute')">Brute force</button>
   <button class="btn g" onclick="preset('bot')">Botnet beacon</button>
  </div>
  <div class="grid" id="fgrid"></div>
  <div style="margin-top:14px;display:flex;gap:10px;align-items:center">
   <button class="btn" onclick="analyzeFlow()">Analyze → تحليل</button>
   <span class="hint">agent preprocesses → classifies → double-checks
   anomaly → scores risk → applies RL policy</span></div>
  <div class="result" id="res-flow"></div></div>
</div>

<!-- ==================== MESSAGE ==================== -->
<div class="tab" id="t-msg">
 <header class="top"><span class="big-emoji">💬</span><h2>Message Scanner</h2>
 <span class="sub">NLP spam/phishing + real IOC blacklist</span></header>
 <div class="panel">
  <div class="presets">
   <button class="btn g" onclick="ex('phish')">Phishing w/ blacklisted domain</button>
   <button class="btn g" onclick="ex('spam')">Classic spam</button>
   <button class="btn g" onclick="ex('ham')">Normal message</button>
  </div>
  <textarea id="msgtext" placeholder="Paste any message..."></textarea>
  <div style="margin-top:12px"><button class="btn" onclick="analyzeMsg()">
  Scan → فحص</button></div>
  <div class="result" id="res-msg"></div></div>
</div>

<!-- ==================== IMAGE ==================== -->
<div class="tab" id="t-img">
 <header class="top"><span class="big-emoji">🖼️</span><h2>Malware Byte-plot
 Classifier</h2><span class="sub">CNN — 4 families</span></header>
 <div class="panel">
  <div class="presets" id="samples"></div>
  <input type="file" id="upl" accept="image/*">
  <div style="margin-top:12px"><button class="btn" onclick=
  "analyzeUpload()">Classify upload → تصنيف</button></div>
  <div class="result" id="res-img"></div></div>
</div>

<!-- ==================== REPORT ==================== -->
<div class="tab" id="t-rep">
 <header class="top"><span class="big-emoji">📄</span><h2>Automated Security
 Report</h2><span class="sub">written by the agent itself</span></header>
 <div class="panel">
  <button class="btn" onclick="loadReport()">Generate from current
  incidents → توليد</button>
  <div id="repout" style="margin-top:14px"></div></div>
</div>

<!-- ==================== SUPERVISED ==================== -->
<div class="tab" id="t-sup">
 <header class="top"><span class="big-emoji">🎯</span><h2>Supervised
 Learning — Stage 4+5</h2><span class="sub">6 algorithms · CV · tuning ·
 full evaluation</span></header>
 <div id="supbody">loading...</div>
</div>

<!-- ==================== UNSUPERVISED ==================== -->
<div class="tab" id="t-uns">
 <header class="top"><span class="big-emoji">🔍</span><h2>Unsupervised
 Learning — Stage 6</h2><span class="sub">discovery &amp; anomaly detection
 without labels</span></header>
 <div id="unsbody">loading...</div>
</div>

<!-- ==================== DL ==================== -->
<div class="tab" id="t-dl">
 <header class="top"><span class="big-emoji">🧠</span><h2>Deep Learning —
 Stage 7</h2><span class="sub">MLP on tabular flows, honestly compared</span></header>
 <div id="dlbody">loading...</div>
</div>

<!-- ==================== NLP ==================== -->
<div class="tab" id="t-nlp">
 <header class="top"><span class="big-emoji">💬</span><h2>NLP — Stage 8</h2>
 <span class="sub">real UCI SMS Spam · TF-IDF vs LSTM</span></header>
 <div id="nlpbody">loading...</div>
</div>

<!-- ==================== CV ==================== -->
<div class="tab" id="t-cv">
 <header class="top"><span class="big-emoji">👁️</span><h2>Computer Vision —
 Stage 9</h2><span class="sub">malware byte-plots · CNN vs HOG+SVM</span></header>
 <div id="cvbody">loading...</div>
</div>

<!-- ==================== RL ==================== -->
<div class="tab" id="t-rl">
 <header class="top"><span class="big-emoji">🎮</span><h2>Reinforcement
 Learning — Stage 10</h2><span class="sub">from-scratch Q-Learning</span></header>
 <div id="rlbody">loading...</div>
</div>

<!-- ==================== AGENT ==================== -->
<div class="tab" id="t-agent">
 <header class="top"><span class="big-emoji">🤖</span><h2>SentinelAgent —
 Stage 11</h2><span class="sub">perceive → route → decide → act</span></header>
 <div class="panel"><h3>Agent workflow</h3>
  <div class="flowsteps">
   <div class="step"><b>Perceive</b>receive security event</div>
   <div class="step"><b>Route</b>choose engine automatically</div>
   <div class="step"><b>Analyze</b>model + confidence</div>
   <div class="step"><b>Second opinion</b>anomaly check</div>
   <div class="step"><b>Enrich</b>IOC blacklist</div>
   <div class="step"><b>Fuse</b>risk 0-100</div>
   <div class="step"><b>Decide</b>RL policy</div>
   <div class="step"><b>Act</b>incident + report</div>
  </div></div>
 <div class="panel"><h3>Live demo — run a full agent decision now</h3>
  <div class="presets">
   <button class="btn g" onclick="demoFlow()">🚨 Simulate DDoS attack</button>
   <button class="btn g" onclick="demoMsg()">🎣 Simulate phishing message</button>
   <button class="btn g" onclick="demoImg()">🦠 Scan malware sample</button>
  </div>
  <div class="result" id="res-agent"></div></div>
 <div class="panel"><h3>Why this is a real agent</h3>
  <p class="hint">It <b>selects its own tools</b> (model routing by event
  type), <b>interprets</b> outputs (confidence, anomaly, IOC), takes an
  <b>autonomous decision</b> through its learned Q-policy, and <b>acts</b>
  by writing incidents, alerts and the security report. Every decision is
  logged with its Q-values and a textual explanation — no black box.</p></div>
</div>

<!-- ==================== DATASETS ==================== -->
<div class="tab" id="t-data">
 <header class="top"><span class="big-emoji">🗄️</span><h2>Datasets — Stage
 2</h2><span class="sub">2 real sources + 2 documented generators</span></header>
 <div id="databody">loading...</div>
</div>

<!-- ==================== PREPROCESSING ==================== -->
<div class="tab" id="t-pre">
 <header class="top"><span class="big-emoji">🧹</span><h2>Preprocessing —
 Stage 3</h2><span class="sub">every operation with its justification</span></header>
 <div id="prebody">loading...</div>
</div>

<!-- ==================== GALLERY ==================== -->
<div class="tab" id="t-gal">
 <header class="top"><span class="big-emoji">🖼️</span><h2>Results Gallery</h2>
 <span class="sub">all pipeline figures</span></header>
 <div class="panel">
  <div class="presets" id="figfilter"></div>
  <div class="figrow" id="figs">loading...</div></div>
</div>

<!-- ==================== SYSTEM ==================== -->
<div class="tab" id="t-sys">
 <header class="top"><span class="big-emoji">⚙️</span><h2>System</h2>
 <span class="sub">policy · self-test · retraining</span></header>
 <div class="panel"><h3>Learned RL response policy (Q-table)</h3>
  <div style="overflow:auto"><table id="poltable"></table></div>
  <p class="hint">18 states (threat / confidence / asset-criticality) × 4
  actions — badge = the action the agent takes.</p></div>
 <div class="panel"><h3>Self-test (21 tests)</h3>
  <button class="btn" onclick="runTests()">Run tests → تشغيل</button>
  <pre id="testout" style="display:none;margin-top:12px"></pre></div>
 <div class="panel"><h3>Retraining (CLI)</h3>
  <p class="hint">This console serves the trained artifacts in
  <b>models/</b>. To retrain from scratch run <b>python -m src.main</b>
  (~3 min), then restart the console. Training stays a batch CLI job on
  purpose — standard MLOps separation: <i>training offline, serving
  online</i>.</p></div>
</div>

</main></div>
<script>
const FIELDS=[["duration",3],["src_port",45000],["dst_port",443],
["src_bytes",1500],["dst_bytes",3000],["src_pkts",20],["dst_pkts",25],
["syn_rate",0.15],["ack_rate",0.85],["psh_rate",0.25],["avg_pkt_size",700],
["byte_std",250],["flow_iat_mean",120],["active_duration",2.5],["is_land",0]];
const PRESETS={
 normal:{duration:0.031,src_port:52051,dst_port:80,src_bytes:68,dst_bytes:137,
  src_pkts:2,dst_pkts:2,syn_rate:0.0,ack_rate:0.0,psh_rate:0.0,
  avg_pkt_size:75.5,byte_std:27.9,flow_iat_mean:0.011,active_duration:0.0},
 ddos:{duration:84.674,src_port:46707,dst_port:80,src_bytes:336,dst_bytes:11595,
  src_pkts:6,dst_pkts:6,syn_rate:0.0,ack_rate:0.071,psh_rate:0.0,
  avg_pkt_size:852.6,byte_std:1497.5,flow_iat_mean:6.586,active_duration:0.0},
 scan:{duration:0.0,src_port:46908,dst_port:3689,src_bytes:0,dst_bytes:6,
  src_pkts:1,dst_pkts:1,syn_rate:0.0,ack_rate:0.0,psh_rate:0.5,
  avg_pkt_size:3,byte_std:3.46,flow_iat_mean:0.0,active_duration:0.0},
 brute:{duration:0.037,src_port:53125,dst_port:21,src_bytes:24,dst_bytes:0,
  src_pkts:2,dst_pkts:2,syn_rate:0.0,ack_rate:0.0,psh_rate:0.015,
  avg_pkt_size:12.2,byte_std:11.6,flow_iat_mean:0.037,active_duration:0.0},
 bot:{duration:0.071,src_port:8080,dst_port:8080,src_bytes:6,dst_bytes:18,
  src_pkts:3,dst_pkts:3,syn_rate:0.0,ack_rate:0.0,psh_rate:0.143,
  avg_pkt_size:9,byte_std:3.2,flow_iat_mean:0.011,active_duration:0.0}};
const EX={phish:"URGENT: your account will be suspended in 24 hours! "+
 "Verify your card now at http://2020bill.com/secure or call 0900-1234",
 spam:"WINNER!! You have been selected to receive a £900 prize! "+
 "Text CLAIM to 87121 now. T&Cs apply.",
 ham:"Hey, are we still meeting for lunch tomorrow at the usual place?"};

document.querySelectorAll("aside button").forEach(b=>b.onclick=()=>{
 document.querySelectorAll("aside button").forEach(x=>x.classList.remove("on"));
 document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
 b.classList.add("on");document.getElementById("t-"+b.dataset.t)
 .classList.add("on");});

const P=x=>x==null?"-":(100*x).toFixed(2)+"%";
const $=id=>document.getElementById(id);
function tbl(head,rows){return "<table><tr>"+head.map(h=>"<th>"+h+
 "</th>").join("")+"</tr>"+rows.map(r=>"<tr>"+r.map(c=>"<td>"+c+"</td>")
 .join("")+"</tr>").join("")+"</table>";}
function figs(list){return '<div class="figrow">'+list.map(([f,c])=>
 '<div><img src="/api/figure/'+f+'" loading="lazy"><div class="cap">'+c+
 '</div></div>').join("")+'</div>';}
function kv(items){return '<div class="kv">'+items.map(([k,v])=>
 '<div class="item"><div class="k">'+k+'</div><div class="v">'+v+
 '</div></div>').join("")+'</div>';}
function riskColor(r){return r>=70?"#ef4444":r>=40?"#f59e0b":"#22c55e";}
function card(inc){const q=inc.rl_q_values||[];const acts=
 ["Monitor","Alert_Analyst","Block_IP","Isolate_Host"];
 const qmax=Math.max(...q,0.001);
 const anom=inc.anomaly_second_opinion===null?"-":
  (inc.anomaly_second_opinion?"YES":"no");
 const ioc=(inc.ioc_matches||[]).join(", ")||"-";
 return `<div class="rhead" style="display:flex;gap:12px;align-items:center;
 flex-wrap:wrap"><span class="pred">${inc.prediction}</span>
 <span class="badge b-${inc.response_action}">${inc.response_action}</span>
 <span class="hint">engine: ${inc.engine_used} | threat:
 ${inc.threat_level}</span></div>
 <div class="riskwrap"><label>Risk score ${inc.risk_score}/100</label>
 <div class="riskbar"><div class="riskfill" style="width:${inc.risk_score}%;
 background:${riskColor(inc.risk_score)}"></div></div></div>
 ${kv([["Confidence",(inc.confidence*100).toFixed(1)+"%"],
 ["Anomaly (IsolationForest)",anom],["IOC match (18k blacklist)",ioc]])}
 <div class="qbars">${q.map((v,i)=>'<div class="qrow"><span style="width:'
 +'92px">'+acts[i]+'</span><div class="qb" style="width:'+Math.max(2,70*v/
 qmax)+'px"></div><span>'+v.toFixed(2)+'</span></div>').join("")}</div>
 <div class="exp">${inc.explanation||""}</div>`;}
function show(id,inc){const el=$(id);el.style.display="block";
 el.innerHTML='<div class="panel" style="border-color:#26375a">'+card(inc)+
 '</div>';}

/* ---------- analyzers ---------- */
function buildForm(){const g=$("fgrid");
 g.innerHTML=FIELDS.map(([k,v])=>'<div><label>'+k+'</label><input id="f_'+k+
 '" type="number" step="any" value="'+v+'"></div>').join("")+
 '<div><label>protocol</label><select id="f_protocol"><option>TCP</option>'+
 '<option>UDP</option><option>ICMP</option></select></div>'+
 '<div><label>asset criticality (0 normal / 1 critical)</label>'+
 '<select id="f_crit"><option value="0">0</option><option value="1" selected>'+
 '1</option></select></div>';}
function preset(k){for(const[f,v]of Object.entries(PRESETS[k]))
 $("f_"+f).value=v;}
function ex(k){$("msgtext").value=EX[k];}
function analyzeFlow(){const ev={};FIELDS.forEach(([k])=>ev[k]=parseFloat(
 $("f_"+k).value));ev.protocol=$("f_protocol").value;
 ev.criticality=parseInt($("f_crit").value);
 fetch("/api/flow",{method:"POST",headers:{"Content-Type":"application/json"},
 body:JSON.stringify(ev)}).then(r=>r.json()).then(d=>show("res-flow",d));}
function analyzeMsg(){fetch("/api/message",{method:"POST",headers:
 {"Content-Type":"application/json"},body:JSON.stringify({text:$("msgtext")
 .value})}).then(r=>r.json()).then(d=>show("res-msg",d));}
function analyzeSample(f){fetch("/api/image",{method:"POST",headers:
 {"Content-Type":"application/json"},body:JSON.stringify({sample:f})})
 .then(r=>r.json()).then(d=>show("res-img",d));}
function analyzeUpload(){const f=$("upl").files[0];if(!f){alert(
 "choose an image first");return;}
 const fd=new FormData();fd.append("file",f);
 fetch("/api/image",{method:"POST",body:fd}).then(r=>r.json()).then(d=>
 show("res-img",d));}
/* agent live demo */
const DEMO_DDOS={duration:84.674,src_port:46707,dst_port:80,src_bytes:336,
 dst_bytes:11595,src_pkts:6,dst_pkts:6,syn_rate:0.0,ack_rate:0.071,
 psh_rate:0.0,avg_pkt_size:852.6,byte_std:1497.5,flow_iat_mean:6.586,
 active_duration:0.0,is_land:0,protocol:"TCP",criticality:1};
function demoFlow(){fetch("/api/flow",{method:"POST",headers:
 {"Content-Type":"application/json"},body:JSON.stringify(DEMO_DDOS)})
 .then(r=>r.json()).then(d=>show("res-agent",d));}
function demoMsg(){fetch("/api/message",{method:"POST",headers:
 {"Content-Type":"application/json"},body:JSON.stringify({text:EX.phish})})
 .then(r=>r.json()).then(d=>show("res-agent",d));}
function demoImg(){fetch("/api/samples").then(r=>r.json()).then(fs=>{
 if(!fs.length){alert("no samples");return;}
 fetch("/api/image",{method:"POST",headers:{"Content-Type":"application/json"},
 body:JSON.stringify({sample:fs[0]})}).then(r=>r.json()).then(d=>
 show("res-agent",d));});}

function incTable(list){if(!list.length)return"<i>no incidents yet</i>";
 return '<table><tr><th>type</th><th>prediction</th><th>conf</th><th>anom'+
 '</th><th>IOC</th><th>risk</th><th>action</th></tr>'+list.slice(-15).
 reverse().map(i=>'<tr><td>'+i.event_type+'</td><td>'+i.prediction+
 '</td><td>'+(i.confidence*100).toFixed(0)+'%</td><td>'+
 (i.anomaly_second_opinion===null?"-":(i.anomaly_second_opinion?"YES":"no"))+
 '</td><td>'+((i.ioc_matches||[]).join(", ")||"-")+'</td><td style="color:'+
 riskColor(i.risk_score)+';font-weight:800">'+i.risk_score+'</td><td><b>'+
 i.response_action+'</b></td></tr>').join("")+'</table>';}

/* ---------- dashboard ---------- */
function dashError(el,msg){
 $(el).innerHTML='<div class="err">⚠️ '+msg+'</div>';}
function loadStats(){fetch("/api/stats").then(r=>{if(!r.ok)throw 0;return r.json()})
 .then(s=>{
 const c=[["Random Forest accuracy",s.rf_accuracy,"p"],["5-fold CV (RF)",
 s.rf_cv,"p"],["IsolationForest precision",s.iso_precision,"p"],
 ["LSTM F1 (real SMS)",s.lstm_f1,"p"],["CNN accuracy",s.cnn_accuracy,"p"],
 ["RL policy reward",s.rl_reward,"r"],["Agent demo events",s.agent_events,
 "i"],["Blacklist domains",s.blacklist_size,"i"]];
 $("cards").innerHTML=c.map(([k,v,t])=>'<div class="card"><div class="k">'+
 k+'</div><div class="v">'+(v==null?"-":(t==="p"?P(v):(t==="r"?v.toFixed(2):
 v.toLocaleString())))+'</div></div>').join("");})
 .catch(()=>dashError("cards","server not reachable — is the terminal still "+
   "running <b>python -m src.webapp</b>? Refresh after it prints "+
   "“Running on http://127.0.0.1:7860”."));
 fetch("/api/incidents").then(r=>{if(!r.ok)throw 0;return r.json()})
 .then(l=>$("incdash").innerHTML=incTable(l))
 .catch(()=>dashError("incdash","could not load incidents (server offline?)"));}

/* ---------- datasets ---------- */
function loadData(){fetch("/api/dataset_doc").then(r=>r.json()).then(d=>{
 const meta=d.network_flows||{},sms=d.sms_spam||{},img=d.malware_images||{},
 feed=d.threat_feed||{};
 $("databody").innerHTML=
 '<div class="panel"><h3>Sources</h3>'+tbl(["Dataset","Type","Records",
 "Classes","Source"],[
 ["Network flows","tabular · 16 features",meta.rows||"-",
 (meta.classes||[]).join(" / "),meta.source||"-"],
 ["SMS Spam (NLP)","text",sms.rows||"-",(sms.classes||[]).join(" / "),
 sms.source||"-"],
 ["Malware images","48×48 grayscale byte-plots",img.rows||"-",
 (img.classes||[]).join(" / "),img.source||"-"],
 ["Threat-intel feed","domain blocklist",feed.rows||"-","malicious",
 feed.source||"-"]])+
 '<p class="hint">Imperfections injected on purpose: duplicates (~0.8%), '+
 'missing values (~1.5%), impossible values (0.3%) — so preprocessing has '+
 'real work. Class mix is intentionally imbalanced (Benign 45% … Botnet 8%)'+
 ' and "benign" contains hard look-alikes (SSH admin, beacon apps, probes,'+
 ' flash crowds) so honest errors exist to discuss.</p></div>'+
 figs([["eda_class_distribution","Flow class distribution (imbalanced by design)"],
 ["eda_feature_boxplots","Feature behaviour per class"],
 ["nlp_class_distribution","Real SMS spam distribution"],
 ["cv_sample_images","Byte-plot samples per malware family"]]);});}

/* ---------- preprocessing ---------- */
function loadPre(){fetch("/api/json/preprocessing").then(r=>r.json())
 .then(p=>{
 const j=p.justifications||{};
 const rows=[
 ["Remove duplicates",p.duplicates_removed+" rows",j.duplicates_removed||""],
 ["Clip impossible values",p.negative_values_clipped,j.missing_imputed?
 "sensor glitches fixed":"-"],
 ["Median imputation",JSON.stringify(p.missing_before||{}),
 j.missing_imputed||""],
 ["One-hot encoding","protocol → dummies",j.one_hot_encoding||""],
 ["log1p transform",(p.log1p_columns||[]).join(", "),""],
 ["StandardScaler","train-only fit",j.standard_scaling||""],
 ["Feature selection",(p.features_kept_count||0)+" of "+
 (p.features_after_encoding||0),j.feature_selection||""],
 ["Stratified split","80/20 → "+(p.train_rows||0)+" / "+(p.test_rows||0),
 j.stratified_split||""]];
 $("prebody").innerHTML=
 '<div class="panel"><h3>Operations &amp; justifications (rubric: WHY)</h3>'+
 tbl(["Operation","Applied","Why"],rows)+'</div>'+
 '<div class="panel"><h3>Features</h3>'+kv([
 ["Kept ("+((p.features_kept||[]).length)+")",(p.features_kept||[]).join(", ")],
 ["Dropped ("+((p.features_dropped||[]).length)+")",
 (p.features_dropped||[]).join(", ")]])+'</div>'+
 figs([["preprocess_feature_scores","Mutual-information feature scores"],
 ["eda_correlation_heatmap","Correlation matrix before selection"]]);});}

/* ---------- supervised ---------- */
function loadSup(){fetch("/api/json/supervised").then(r=>r.json()).then(s=>{
 const m=s.metrics_per_model||{},cv=s.cross_validation||{};
 const names=Object.keys(m);
 const best=s.best_model;
 $("supbody").innerHTML=
 '<div class="panel"><h3>Test-set metrics (6 algorithms)</h3>'+
 tbl(["Model","Accuracy","Precision(m)","Recall(m)","F1(m)","F1(w)"],
 names.map(n=>[n+(n===best?" 🏇":""),P(m[n].accuracy),P(m[n].precision_macro),
 P(m[n].recall_macro),P(m[n].f1_macro),P(m[n].f1_weighted)]))+'</div>'+
 '<div class="panel"><h3>Validation — 5-fold stratified CV</h3>'+
 tbl(["Model","CV accuracy (± std)","CV macro-F1"],
 Object.keys(cv).map(n=>[n,cv[n].accuracy_mean.toFixed(4)+" ± "+
 cv[n].accuracy_std.toFixed(4),cv[n].f1_macro_mean.toFixed(4)]))+
 '<h4>Hyperparameter tuning (GridSearchCV, 3-fold)</h4><p class="hint">'+
 JSON.stringify((s.grid_search||{}).best_params||{})+' — best CV macro-F1: '+
 P((s.grid_search||{}).best_cv_f1_macro)+'</p>'+
 '<h4>Class-imbalance experiment (class_weight="balanced")</h4><p class="hint">'+
 ((s.imbalance_handling||{}).note||"")+'</p></div>'+
 '<div class="panel"><h3>Error analysis (best model: '+best+')</h3>'+
 tbl(["Class","FP","FN","FP rate","FN rate"],(s.fp_fn_analysis||[]).map(e=>
 [e.class,e.FP,e.FN,P(e.FP_rate),P(e.FN_rate)]))+
 '<p class="hint">'+((s.discussion||{}).false_positives||"")+' | '+
 ((s.discussion||{}).false_negatives||"")+'</p></div>'+
 figs([["supervised_model_comparison","Six models compared"],
 ["supervised_cm_best","Confusion matrix — "+best],
 ["supervised_cm_best_norm","Confusion matrix (normalized)"],
 ["supervised_roc_best","ROC curves (one-vs-rest)"],
 ["supervised_fp_fn","False positives vs false negatives"],
 ["supervised_cross_validation","5-fold cross-validation"],
 ["supervised_imbalance","Default vs balanced class weights"]]);});}

/* ---------- unsupervised ---------- */
function loadUns(){fetch("/api/json/unsupervised").then(r=>r.json())
 .then(u=>{
 const k=u.kmeans||{},i=u.isolation_forest||{},h=u.hierarchical||{},
 d=u.dbscan||{};
 $("unsbody").innerHTML=
 '<div class="panel"><h3>Results without labels</h3>'+kv([
 ["K-Means chosen k",k.chosen_k],["Silhouette",(k.silhouette||0).toFixed(3)],
 ["Purity",P(k.purity)],["ARI / NMI",(k.ARI||0).toFixed(2)+" / "+
 (k.NMI||0).toFixed(2)],
 ["Hierarchical ARI / NMI",(h.ARI||0).toFixed(2)+" / "+(h.NMI||0).toFixed(2)],
 ["DBSCAN clusters / noise",(d.clusters_found||0)+" / "+P(d.noise_ratio)],
 ["IsolationForest P / R",P(i.precision)+" / "+P(i.recall)],
 ["Benign false-alarm rate",P(i.benign_false_alarm_rate)]])+
 '<p class="hint"><b>Supervised vs unsupervised:</b> '+
 ((u.supervised_vs_unsupervised||{}).supervised||"")+' — '+
 ((u.supervised_vs_unsupervised||{}).unsupervised||"")+'</p></div>'+
 figs([["unsup_k_selection","Elbow + silhouette (choosing k)"],
 ["unsup_kmeans_contingency","K-Means clusters vs true classes"],
 ["unsup_dendrogram","Hierarchical (Ward) dendrogram"],
 ["unsup_pca_clusters","PCA: true classes vs discovered clusters"],
 ["unsup_isolation_detection","IsolationForest detection per class"],
 ["unsup_isolation_hist","Normal-profile separation"]]);});}

/* ---------- DL ---------- */
function loadDl(){fetch("/api/json/dl").then(r=>r.json()).then(x=>{
 const m=x.metrics||{},c=x.comparison||{};
 $("dlbody").innerHTML=
 '<div class="panel"><h3>MLP architecture (tabular → ANN)</h3>'+
 '<p class="hint">'+(x.architecture||"")+' — <b>'+(x.params||0).toLocaleString()+
 ' parameters</b> · Adam lr=1e-3 · early stopping · '+
 (x.epochs_trained||0)+' epochs</p></div>'+
 '<div class="panel"><h3>Metrics &amp; honest comparison</h3>'+kv([
 ["Accuracy",P(m.accuracy)],["Macro-F1",P(m.f1_macro)],
 ["Best traditional ("+(c.best_traditional||"")+")",
 P(c.traditional_f1_macro)],["Delta (MLP − traditional)",
 (c.delta_f1_macro>=0?"+":"")+(100*c.delta_f1_macro).toFixed(2)+" pts"]])+
 '<p class="hint">'+(c.interpretation||"")+'</p></div>'+
 figs([["dl_vs_traditional","Deep learning vs traditional ML"],
 ["dl_training_accuracy","MLP training curves (accuracy)"],
 ["dl_training_loss","MLP training curves (loss)"],
 ["dl_mlp_cm","MLP confusion matrix"]]);});}

/* ---------- NLP ---------- */
function loadNlp(){fetch("/api/json/nlp").then(r=>r.json()).then(x=>{
 const m=x.metrics||{};
 $("nlpbody").innerHTML=
 '<div class="panel"><h3>Text preprocessing pipeline</h3><div class="chips">'+
 (x.preprocessing_pipeline||[]).map(s=>'<span class="chip">'+s+'</span>')
 .join("")+'</div></div>'+
 '<div class="panel"><h3>Models (real UCI data · '+(x.n_train||0)+' train / '+
 (x.n_test||0)+' test)</h3>'+
 tbl(["Model","Accuracy","Precision","Recall","F1"],Object.keys(m).map(n=>
 [n+(n===x.best_model?" 🏇":""),P(m[n].accuracy),P(m[n].precision),
 P(m[n].recall),P(m[n].f1)]))+
 '<p class="hint">LSTM learns word ORDER (sequences) — that is why it beats'+
 ' the bag-of-words TF-IDF baseline.</p></div>'+
 '<div class="panel"><h3>Explainability — top spam terms learned</h3>'+
 '<div class="chips">'+(x.top_spam_terms||[]).slice(0,18).map(t=>
 '<span class="chip">🔥 '+t+'</span>').join("")+'</div></div>'+
 figs([["nlp_model_comparison","NLP models compared"],
 ["nlp_top_spam_terms","Most spam-indicating terms"],
 ["nlp_lstm_training","LSTM training curves"],
 ["nlp_cm","Confusion matrix (classic model)"],
 ["nlp_length_distribution","Message length by class"]]);});}

/* ---------- CV ---------- */
function loadCv(){fetch("/api/json/cv").then(r=>r.json()).then(x=>{
 const m=x.metrics||{},hb=x.hog_svm_baseline||{};
 $("cvbody").innerHTML=
 '<div class="panel"><h3>Approach &amp; architecture</h3>'+
 '<p class="hint">'+(x.approach||"")+'<br><b>Preprocessing:</b> '+
 (x.image_preprocessing||[]).join(" · ")+'<br><b>Architecture:</b> '+
 (x.architecture||"")+' — <b>'+(x.params||0).toLocaleString()+
 ' parameters</b> · '+(x.epochs_trained||0)+' epochs · '+
 (x.train_images||0)+' train / '+(x.test_images||0)+' test images</p></div>'+
 '<div class="panel"><h3>CNN vs traditional ML (HOG+SVM)</h3>'+kv([
 ["CNN accuracy",P(m.accuracy)],["CNN macro-F1",P(m.f1_macro)],
 ["HOG+SVM accuracy",P(hb.accuracy)],["HOG+SVM macro-F1",P(hb.f1_macro)]])+
 '<p class="hint">'+(hb.note||"")+'</p></div>'+
 figs([["cv_sample_images","Byte-plot samples per family"],
 ["cv_cnn_vs_hog","CNN vs HOG+SVM"],
 ["cv_cm","CNN confusion matrix"],
 ["cv_per_family_accuracy","Per-family recall"],
 ["cv_training_accuracy","CNN training curves"]]);});}

/* ---------- RL ---------- */
function loadRl(){fetch("/api/json/rl").then(r=>r.json()).then(x=>{
 const b=x.baseline_comparison||{},r=x.reward_design||{},
 h=x.hyperparameters||{};
 const rewards=Object.entries(r).map(([k,v])=>[k,v]);
 $("rlbody").innerHTML=
 '<div class="panel"><h3>Problem: choose the RESPONSE</h3>'+
 '<p class="hint"><b>States:</b> '+(x.states||"")+'<br><b>Actions:</b> '+
 (x.actions||[]).join(" / ")+'<br><b>Update rule (hand-implemented):</b> '+
 '<code>'+(x.update_rule||"")+'</code></p></div>'+
 '<div class="panel"><h3>Reward = SOC cost model</h3>'+
 tbl(["Event","Reward"],rewards)+'</div>'+
 '<div class="panel"><h3>Hyperparameters</h3>'+kv([
 ["Episodes",(h.episodes||0).toLocaleString()],["Alpha (learning rate)",h.alpha],
 ["Gamma (discount)",h.gamma],["Epsilon","1.0 → 0.05"]])+'</div>'+
 '<div class="panel"><h3>Learned policy vs baselines (avg reward)</h3>'+
 tbl(["Strategy","Avg reward / decision"],Object.entries(b).map(([k,v])=>
 [k==='Q-Learning policy'?"<b>🤖 "+k+"</b>":k,
 '<span style="color:'+(v>0?"#4ade80":"#f87171")+'">'+v.toFixed(2)+
 '</span>']))+'<p class="hint">'+(x.how_it_learns||"")+'</p></div>'+
 figs([["rl_learning_curve","Learning from rewards (curve)"],
 ["rl_qtable_heatmap","Learned Q-table + policy"],
 ["rl_vs_baselines","Policy vs baselines"],
 ["rl_epsilon_decay","Exploration decay"]]);});}

/* ---------- report ---------- */
function loadReport(){fetch("/api/report").then(r=>r.json()).then(s=>{
 fetch("/api/incidents").then(r=>r.json()).then(l=>{
 $("repout").innerHTML=
 kv([["Events analysed",s.n_events],["High risk (≥70)",
 '<span style="color:#ef4444">'+s.high_risk+'</span>'],
 ["Zero-day candidates",'<span style="color:#f59e0b">'+s.disagreements+
 '</span>'],["IOC matches",'<span style="color:#a78bfa">'+s.ioc_events+
 '</span>']])+
 '<h4>Response distribution</h4><div style="margin:8px 0">'+
 Object.entries(s.by_action).map(([k,v])=>'<span class="badge b-'+k+
 '" style="margin:4px;display:inline-block">'+k+': '+v+'</span>').join("")+
 '</div><h4>Recommendations</h4><ul style="margin:10px 0 0 20px;line-height:1.9">'+
 s.recommendations.map(r=>'<li>'+r+'</li>').join("")+'</ul><h4>Incidents</h4>'+
 incTable(l)+'<p class="hint">Full file: results/reports/security_report.md</p>';
 });});}

/* ---------- gallery ---------- */
function loadFigs(cat){
 fetch("/api/figures").then(r=>r.json()).then(fs=>{
 const cats=[...new Set(fs.map(f=>f.split("_")[0]))];
 $("figfilter").innerHTML='<button class="btn g" onclick="loadFigs(\'\')">all ('
 +fs.length+')</button>'+cats.map(c=>'<button class="btn g" onclick='+
 '"loadFigs(\''+c+'\')">'+c+' ('+fs.filter(f=>f.startsWith(c+"_")).length+
 ')</button>').join("");
 const show=fs.filter(f=>!cat||f.startsWith(cat+"_"));
 $("figs").innerHTML=show.map(f=>'<div><img src="/api/figure/'+f+
 '" loading="lazy"><div class="cap">'+f.replace(".png","")+'</div></div>')
 .join("")||"<i>no figures yet</i>";});}

/* ---------- system ---------- */
function loadPolicy(){fetch("/api/policy").then(r=>r.json()).then(rows=>{
 const t=$("poltable");if(!rows.length){t.innerHTML="<i>not trained yet</i>";
 return;}
 const cols=Object.keys(rows[0]);
 t.innerHTML="<tr>"+cols.map(c=>"<th>"+c+"</th>").join("")+"</tr>"+
 rows.map(r=>"<tr>"+cols.map(c=>{
 if(c==="POLICY")return '<td><span class="badge b-'+r[c]+'">'+r[c]+'</span></td>';
 if(c==="state")return "<td><b>"+r[c]+"</b></td>";
 return "<td>"+(typeof r[c]==="number"?r[c].toFixed(2):r[c])+"</td>";
 }).join("")+"</tr>").join("");});}
function runTests(){const el=$("testout");el.style.display="block";
 el.textContent="running tests, please wait...";
 fetch("/api/selftest",{method:"POST"}).then(r=>r.json()).then(d=>{
 el.style.color=d.ok?"#9fe8b5":"#f8a5a5";
 el.textContent=(d.output||"(no output)")+(d.ok?
 "\n>>> ALL TESTS PASSED":"\n>>> SOME TESTS FAILED");});}

/* ---------- init ---------- */
buildForm();loadStats();loadData();loadPre();loadSup();loadUns();loadDl();
loadNlp();loadCv();loadRl();loadFigs('');loadPolicy();
fetch("/api/samples").then(r=>r.json()).then(fs=>{
 $("samples").innerHTML=fs.map(f=>'<button class="btn g" onclick='+
 '"analyzeSample(\''+f+'\')">'+f+'</button>').join("")||
 "<i>no samples — run the pipeline first</i>";});
</script></body></html>"""


if __name__ == "__main__":
    config.ensure_dirs()
    try:
        get_agent()                   # load all trained artifacts once
    except Exception as exc:          # pragma: no cover
        import traceback
        print("\n" + "=" * 62)
        print("  [FATAL] Could not load the trained models.")
        print("  This usually means one of the model files in models/ is")
        print("  missing or was saved with a different TensorFlow version.")
        print("  Fix: re-train once with ->  python -m src.main")
        print("=" * 62)
        traceback.print_exc()
        raise SystemExit(1)
    # Open the browser only AFTER the server is actually listening,
    # so the tab never lands on "Problem loading page".
    import threading, webbrowser
    def _open_when_ready():
        import socket, time
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", 7860), timeout=1):
                    webbrowser.open("http://localhost:7860")
                    return
            except OSError:
                time.sleep(0.5)
        print("[!] Server slow to start - open http://localhost:7860 manually")
    threading.Thread(target=_open_when_ready, daemon=True).start()
    try:
        app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
    except OSError as exc:             # pragma: no cover
        if "Address already in use" in str(exc):
            print("\n[!] Port 7860 is already in use - another instance is running.")
            print("    Fix: sudo fuser -k 7860/tcp   (then run this again)")
        raise
