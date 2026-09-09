"""
SentinelAI - Web Console (SOC-style UI)
=======================================
A lightweight Flask web interface on top of the trained SentinelAgent.

Run (after the pipeline has been trained once):
    python src/webapp.py          ->  http://localhost:7860

Features
--------
* Dashboard      - live stats + key figures from results/
* Flow analyzer  - craft a network flow (or use attack presets) -> full decision
* Message scanner- paste text -> spam/phishing + real IOC blacklist check
* Malware imaging- upload a byte-plot PNG (or use samples) -> CNN family
* Report         - incidents table + response distribution + recommendations
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


def get_agent() -> SentinelAgent:
    global AGENT
    if AGENT is None:
        AGENT = SentinelAgent().load()
    return AGENT


# ------------------------------------------------------------------ API
@app.get("/")
def index():
    return PAGE


@app.get("/api/stats")
def stats():
    def jload(name):
        p = os.path.join(config.REPORT_DIR, name)
        return json.load(open(p)) if os.path.exists(p) else {}

    sup = jload("supervised_results.json")
    uns = jload("unsupervised_results.json")
    nlp = jload("nlp_results.json")
    cv = jload("cv_results.json")
    rl = jload("rl_results.json")
    ag = jload("agent_results.json")
    rf = sup.get("metrics_per_model", {}).get("RandomForest", {})
    cards = {
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
    }
    return jsonify(cards)


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
        path = tmp.name
        cleanup = True
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
    return jsonify([f for f in files if f.lower().ends(".png")][:8])


@app.get("/api/report")
def report():
    with LOCK:
        summary = get_agent().write_reports()
    return jsonify(summary)


@app.get("/api/figure/<name>")
def figure(name):
    if not name.endswith(".png"):
        return "", 403
    return send_from_directory(config.FIG_DIR, name, mimetype="image/png")


# ------------------------------------------------------------------ UI
PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SentinelAI - SOC Console</title>
<style>
:root{--bg:#0b1220;--panel:#121a2b;--panel2:#0e1524;--line:#22304a;
--txt:#e2e8f0;--dim:#8ea0bd;--acc:#38bdf8;--ok:#22c55e;--warn:#f59e0b;
--bad:#ef4444;--iso:#a78bfa}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,
Arial,sans-serif;min-height:100vh}
header{background:linear-gradient(90deg,#0c1a2e,#10233f);border-bottom:
1px solid var(--line);padding:14px 22px;display:flex;align-items:center;
gap:14px;position:sticky;top:0;z-index:9}
header h1{font-size:1.25rem;color:#fff;letter-spacing:.5px}
header .dot{width:10px;height:10px;border-radius:50%;background:var(--ok);
box-shadow:0 0 8px var(--ok);animation:p 1.6s infinite}
@keyframes p{50%{opacity:.35}}
header .sub{color:var(--dim);font-size:.8rem;margin-left:auto}
nav{display:flex;gap:6px;padding:10px 22px;background:var(--panel2);
border-bottom:1px solid var(--line);flex-wrap:wrap}
nav button{background:transparent;border:1px solid var(--line);color:var(--dim);
padding:8px 16px;border-radius:8px;cursor:pointer;font-size:.9rem}
nav button.on{background:var(--acc);color:#04121f;border-color:var(--acc);
font-weight:700}
main{padding:20px 22px;max-width:1200px;margin:0 auto}
.tab{display:none}.tab.on{display:block}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,
1fr));gap:12px;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:14px}
.card .k{color:var(--dim);font-size:.75rem;text-transform:uppercase;
letter-spacing:.5px}
.card .v{font-size:1.55rem;font-weight:800;color:var(--acc);margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:
12px;padding:18px;margin-bottom:16px}
.panel h3{color:#fff;margin-bottom:10px;font-size:1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,
1fr));gap:10px}
label{font-size:.75rem;color:var(--dim);display:block;margin-bottom:3px}
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
.rhead{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.badge{padding:6px 14px;border-radius:999px;font-weight:800;font-size:.85rem}
.b-Monitor{background:#14351f;color:#4ade80}.b-Alert_Analyst{background:
#3a2c0a;color:#fbbf24}.b-Block_IP{background:#3b1414;color:#f87171}
.b-Isolate_Host{background:#2b0f0f;color:#fb7185;border:1px solid #7f1d1d}
.pred{font-size:1.3rem;font-weight:800;color:#fff}
.riskwrap{margin:14px 0}.riskbar{height:16px;background:#1a2337;border:
radius:8px;position:relative;overflow:hidden;border-radius:8px}
.riskfill{height:100%;width:0;transition:width .7s;border-radius:8px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,
1fr));gap:10px;margin-top:12px}
.kv .item{background:var(--panel2);border:1px solid var(--line);
border-radius:9px;padding:10px}
.kv .item .k{font-size:.7rem;color:var(--dim);text-transform:uppercase}
.kv .item .v{font-weight:700;margin-top:2px}
.qbars{margin-top:8px}.qrow{display:flex;align-items:center;gap:8px;
margin:4px 0;font-size:.75rem;color:var(--dim)}
.qrow .qb{height:8px;background:var(--iso);border-radius:4px;min-width:2px}
.exp{margin-top:12px;background:#0a101d;border-left:3px solid var(--acc);
padding:10px 12px;border-radius:6px;font-size:.8rem;color:#b8c6dd;
line-height:1.6}
table{width:100%;border-collapse:collapse;font-size:.78rem;margin-top:10px}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--dim);text-transform:uppercase;font-size:.68rem}
tr:hover td{background:#101a2e}
.figrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,
1fr));gap:14px}
.figrow img{width:100%;border-radius:10px;border:1px solid var(--line);
background:#fff}
.hint{color:var(--dim);font-size:.78rem;margin-top:6px}
.ar{color:#7f95b5;font-size:.72rem}
</style></head><body>
<header>
 <div style="font-size:1.6rem">&#128737;&#65039;</div>
 <div><h1>SentinelAI &mdash; SOC Console</h1>
 <div class="ar">لوحة تحكم مساعد الأمن السيبراني</div></div>
 <span class="dot"></span><span style="color:#4ade80;font-size:.8rem">
 AGENT ONLINE</span>
 <span class="sub">Supervised + Unsupervised + DL + NLP + CV + RL</span>
</header>
<nav>
 <button class="on" data-t="dash">Dashboard <span class="ar">اللوحة</span></button>
 <button data-t="flow">Flow Analyzer <span class="ar">تحليل تدفق</span></button>
 <button data-t="msg">Message Scanner <span class="ar">فحص رسالة</span></button>
 <button data-t="img">Malware Image <span class="ar">صورة برمجية</span></button>
 <button data-t="rep">Report <span class="ar">التقرير</span></button>
</nav>
<main>

<!-- ============ DASHBOARD ============ -->
<div class="tab on" id="t-dash">
 <div class="cards" id="cards"></div>
 <div class="panel"><h3>Key figures from the trained pipeline</h3>
  <div class="figrow">
   <img src="/api/figure/supervised_model_comparison.png">
   <img src="/api/figure/rl_learning_curve.png">
   <img src="/api/figure/unsup_pca_clusters.png">
  </div></div>
 <div class="panel"><h3>Recent incidents <span class="ar">أحدث الأحداث</span></h3>
  <div id="incdash">loading...</div></div>
</div>

<!-- ============ FLOW ============ -->
<div class="tab" id="t-flow">
 <div class="panel"><h3>Network Flow Analyzer <span class="ar">حلل تدفق شبكي</span></h3>
  <div class="presets">
   <button class="btn g" onclick="preset('normal')">Normal browsing</button>
   <button class="btn g" onclick="preset('ddos')">DDoS flood</button>
   <button class="btn g" onclick="preset('scan')">Port scan</button>
   <button class="btn g" onclick="preset('brute')">Brute force</button>
   <button class="btn g" onclick="preset('bot')">Botnet beacon</button>
  </div>
  <div class="grid" id="fgrid"></div>
  <div style="margin-top:14px;display:flex;gap:10px;align-items:center">
   <button class="btn" onclick="analyzeFlow()">Analyze &rarr; تحليل</button>
   <span class="hint">The agent preprocesses, classifies, double-checks
   with IsolationForest, scores risk and applies the RL policy.</span>
  </div>
  <div class="result" id="res-flow"></div>
 </div>
</div>

<!-- ============ MESSAGE ============ -->
<div class="tab" id="t-msg">
 <div class="panel"><h3>Message Scanner (spam / phishing + IOC)
  <span class="ar">فاحص الرسائل</span></h3>
  <div class="presets">
   <button class="btn g" onclick="ex('phish')">Phishing w/ blacklisted
   domain</button>
   <button class="btn g" onclick="ex('spam')">Classic spam</button>
   <button class="btn g" onclick="ex('ham')">Normal message</button>
  </div>
  <textarea id="msgtext" placeholder="Paste any message..."></textarea>
  <div style="margin-top:12px"><button class="btn" onclick=
  "analyzeMsg()">Scan &rarr; فحص</button></div>
  <div class="result" id="res-msg"></div>
 </div>
</div>

<!-- ============ IMAGE ============ -->
<div class="tab" id="t-img">
 <div class="panel"><h3>Malware Byte-plot Classifier (CNN)
  <span class="ar">مصنف الصور</span></h3>
  <div class="presets" id="samples"></div>
  <input type="file" id="upl" accept="image/*">
  <div style="margin-top:12px"><button class="btn" onclick=
  "analyzeUpload()">Classify upload &rarr; تصنيف</button></div>
  <div class="result" id="res-img"></div>
 </div>
</div>

<!-- ============ REPORT ============ -->
<div class="tab" id="t-rep">
 <div class="panel"><h3>Automated Security Report
  <span class="ar">التقرير الأمني التلقائي</span></h3>
  <button class="btn" onclick="loadReport()">Generate from current
  incidents &rarr; توليد</button>
  <div id="repout" style="margin-top:14px"></div>
 </div>
</div>

</main>
<script>
const FIELDS=[["duration",3],["src_port",45000],["dst_port",443],
["src_bytes",1500],["dst_bytes",3000],["src_pkts",20],["dst_pkts",25],
["syn_rate",0.15],["ack_rate",0.85],["psh_rate",0.25],["avg_pkt_size",700],
["byte_std",250],["flow_iat_mean",120],["active_duration",2.5],["is_land",0]];
const PRESETS={
 normal:{duration:3,src_port:45000,dst_port:443,src_bytes:1500,dst_bytes:3000,
  src_pkts:20,dst_pkts:25,syn_rate:0.15,ack_rate:0.85,psh_rate:0.25,
  avg_pkt_size:700,byte_std:250,flow_iat_mean:120,active_duration:2.5},
 ddos:{duration:5,src_port:50000,dst_port:80,src_bytes:60,dst_bytes:0,
  src_pkts:3000,dst_pkts:0,syn_rate:0.99,ack_rate:0.01,psh_rate:0.02,
  avg_pkt_size:60,byte_std:8,flow_iat_mean:0.5,active_duration:4},
 scan:{duration:0.02,src_port:52000,dst_port:31337,src_bytes:40,dst_bytes:0,
  src_pkts:2,dst_pkts:0,syn_rate:0.9,ack_rate:0.05,psh_rate:0,
  avg_pkt_size:40,byte_std:5,flow_iat_mean:0.05,active_duration:0.02},
 brute:{duration:4,src_port:41000,dst_port:22,src_bytes:200,dst_bytes:800,
  src_pkts:30,dst_pkts:25,syn_rate:0.4,ack_rate:0.55,psh_rate:0.5,
  avg_pkt_size:250,byte_std:60,flow_iat_mean:200,active_duration:3},
 bot:{duration:1,src_port:39000,dst_port:6667,src_bytes:120,dst_bytes:200,
  src_pkts:8,dst_pkts:8,syn_rate:0.15,ack_rate:0.7,psh_rate:0.2,
  avg_pkt_size:100,byte_std:30,flow_iat_mean:30000,active_duration:1}};
const EX={phish:"URGENT: your account will be suspended in 24 hours! "+
 "Verify your card now at http://2020bill.com/secure or call 0900-1234",
 spam:"WINNER!! You have been selected to receive a £900 prize! "+
 "Text CLAIM to 87121 now. T&Cs apply.",
 ham:"Hey, are we still meeting for lunch tomorrow at the usual place?"};

document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{
 document.querySelectorAll("nav button").forEach(x=>x.classList.remove("on"));
 document.querySelectorAll(".tab").forEach(x=>x.classList.remove("on"));
 b.classList.add("on");document.getElementById("t-"+b.dataset.t)
 .classList.add("on");});

function buildForm(){const g=document.getElementById("fgrid");
 g.innerHTML=FIELDS.map(([k,v])=>`<div><label>${k}</label>
  <input id="f_${k}" type="number" step="any" value="${v}"></div>`).join("")+
 `<div><label>protocol</label><select id="f_protocol">
  <option>TCP</option><option>UDP</option><option>ICMP</option></select></div>
 <div><label>asset criticality (0=normal 1=critical)</label>
  <select id="f_crit"><option value="0">0</option>
  <option value="1" selected>1</option></select></div>`;}
function preset(k){const p=PRESETS[k];for(const[f,v]of Object.entries(p))
 document.getElementById("f_"+f).value=v;}
function ex(k){document.getElementById("msgtext").value=EX[k];}

function riskColor(r){return r>=70?"#ef4444":r>=40?"#f59e0b":"#22c55e";}
function card(inc){const q=inc.rl_q_values||[];const acts=
 ["Monitor","Alert_Analyst","Block_IP","Isolate_Host"];
 const qmax=Math.max(...q,0.001);
 const anom=inc.anomaly_second_opinion===null?"-":
  (inc.anomaly_second_opinion?"YES":"no");
 const ioc=(inc.ioc_matches||[]).join(", ")||"-";
 return `<div class="rhead"><span class="pred">${inc.prediction}</span>
 <span class="badge b-${inc.response_action}">${inc.response_action}</span>
 <span class="hint">engine: ${inc.engine_used} | threat:
 ${inc.threat_level}</span></div>
 <div class="riskwrap"><label>Risk score ${inc.risk_score}/100</label>
 <div class="riskbar"><div class="riskfill" style=
 "width:${inc.risk_score}%;background:${riskColor(inc.risk_score)}">
 </div></div></div>
 <div class="kv">
  <div class="item"><div class="k">Confidence</div><div class="v">
  ${(inc.confidence*100).toFixed(1)}%</div></div>
  <div class="item"><div class="k">Anomaly (IsolationForest)</div>
  <div class="v">${anom}</div></div>
  <div class="item"><div class="k">IOC match (18k blacklist)</div>
  <div class="v">${ioc}</div></div></div>
 <div class="qbars">${q.map((v,i)=>`<div class="qrow"><span style=
 "width:92px">${acts[i]}</span><div class="qb" style="width:${Math.max(2,
 70*v/qmax)}px"></div><span>${v.toFixed(2)}</span></div>`).join("")}</div>
 <div class="exp">${inc.explanation||""}</div>`;}
function show(id,inc){const el=document.getElementById(id);
 el.style.display="block";el.innerHTML=`<div class="panel" style=
 "border-color:#26375a">${card(inc)}</div>`;}

function analyzeFlow(){const ev={};FIELDS.forEach(([k])=>ev[k]=parseFloat(
 document.getElementById("f_"+k).value));
 ev.protocol=document.getElementById("f_protocol").value;
 ev.criticality=parseInt(document.getElementById("f_crit").value);
 fetch("/api/flow",{method:"POST",headers:{"Content-Type":"application/json"},
  body:JSON.stringify(ev)}).then(r=>r.json()).then(d=>show("res-flow",d));}
function analyzeMsg(){fetch("/api/message",{method:"POST",headers:
 {"Content-Type":"application/json"},body:JSON.stringify({text:document.
  getElementById("msgtext").value})}).then(r=>r.json()).then(d=>show(
 "res-msg",d));}
function analyzeSample(f){fetch("/api/image",{method:"POST",headers:
 {"Content-Type":"application/json"},body:JSON.stringify({sample:f})})
 .then(r=>r.json()).then(d=>show("res-img",d));}
function analyzeUpload(){const f=document.getElementById("upl").files[0];
 if(!f){alert("choose an image first");return;}
 const fd=new FormData();fd.append("file",f);
 fetch("/api/image",{method:"POST",body:fd}).then(r=>r.json()).then(d=>
 show("res-img",d));}

function incTable(list){if(!list.length)return"<i>no incidents yet</i>";
 return `<table><tr><th>type</th><th>prediction</th><th>conf</th><th>anom
 </th><th>IOC</th><th>risk</th><th>action</th></tr>`+list.slice(-15).
 reverse().map(i=>`<tr><td>${i.event_type}</td><td>${i.prediction}</td>
 <td>${(i.confidence*100).toFixed(0)}%</td><td>${i.anomaly_second_opinion===
 null?"-":(i.anomaly_second_opinion?"YES":"no")}</td><td>${(i.ioc_matches||
 []).join(", ")||"-"}</td><td style="color:${riskColor(i.risk_score)};
 font-weight:800">${i.risk_score}</td><td><b>${i.response_action}</b>
 </td></tr>`).join("")+"</table>";}

function loadStats(){fetch("/api/stats").then(r=>r.json()).then(s=>{
 const c=[["Random Forest accuracy",s.rf_accuracy,"percent"],
 ["5-fold CV (RF)",s.rf_cv,"percent"],["IsolationForest precision",
 s.iso_precision,"percent"],["LSTM F1 (real SMS)",s.lstm_f1,"percent"],
 ["CNN accuracy",s.cnn_accuracy,"percent"],["RL policy reward",
 s.rl_reward,"raw"],["Agent demo events",s.agent_events,"int"],
 ["Blacklist domains",s.blacklist_size,"int"]];
 document.getElementById("cards").innerHTML=c.map(([k,v,t])=>
 `<div class="card"><div class="k">${k}</div><div class="v">${
 v==null?"-":(t==="percent"?(100*v).toFixed(1)+"%":v.toLocaleString())}
 </div></div>`).join("");});
 fetch("/api/incidents").then(r=>r.json()).then(l=>
 document.getElementById("incdash").innerHTML=incTable(l));}

function loadReport(){fetch("/api/report").then(r=>r.json()).then(s=>{
 fetch("/api/incidents").then(r=>r.json()).then(l=>{
 document.getElementById("repout").innerHTML=
 `<div class="kv"><div class="item"><div class="k">Events analysed</div>
 <div class="v">${s.n_events}</div></div><div class="item"><div class="k">
 High risk (>=70)</div><div class="v" style="color:#ef4444">${s.high_risk}
 </div></div><div class="item"><div class="k">Zero-day candidates</div>
 <div class="v" style="color:#f59e0b">${s.disagreements}</div></div>
 <div class="item"><div class="k">IOC matches</div><div class="v" style=
 "color:#a78bfa">${s.ioc_events}</div></div></div>
 <h3 style="margin-top:14px">Response distribution</h3><div>${
 Object.entries(s.by_action).map(([k,v])=>`<span class="badge b-${k}"
 style="margin:4px">${k}: ${v}</span>`).join("")}</div>
 <h3 style="margin-top:14px">Recommendations</h3><ul style="margin:10px 0 0
 20px;line-height:1.9">${s.recommendations.map(r=>`<li>${r}</li>`).join("")}
 </ul><h3 style="margin-top:14px">Incidents</h3>${incTable(l)}
 <p class="hint">Full file: results/reports/security_report.md</p>`;});});}

buildForm();loadStats();
fetch("/api/samples").then(r=>r.json()).then(fs=>{
 document.getElementById("samples").innerHTML=fs.map(f=>
 `<button class="btn g" onclick="analyzeSample('${f}')">${f}</button>`
 ).join("")||"<i>no samples - run the pipeline first</i>";});
</script></body></html>"""


if __name__ == "__main__":
    config.ensure_dirs()
    get_agent()                       # load all 17 artifacts once
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)
