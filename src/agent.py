"""
SentinelAI - Stage 11: The AI AGENT (SentinelAgent)
===================================================
The "brain" of the system. Its workflow (fully explainable, as required):

    1. RECEIVE     an incoming security event (flow / message / file image)
    2. ROUTE       pick the right analysis engine for the event type
                   - flow    -> supervised classifier (RandomForest/...)
                                + IsolationForest second opinion
                   - message -> NLP spam/phishing engine
                   - image   -> CNN malware-family engine
    3. ANALYZE     run the model -> label + confidence
    4. SECOND      label-free anomaly check (IsolationForest) - catches
       OPINION     novel/zero-day behaviour the classifier cannot name
    5. SCORE       fuse everything into a 0-100 risk score
    6. DECIDE      the RL policy (learned Q-table) chooses the response:
                   Monitor / Alert_Analyst / Block_IP / Isolate_Host
    7. REPORT      write an incident record + aggregate security report
                   with recommendations (results/reports/)

Why it qualifies as an AGENT: it perceives the event, SELECTS the tool
to invoke (model routing), interprets the outputs, takes an autonomous
decision through its learned policy and acts by writing alerts/report.
"""

import json
import os
import re
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src import config
from src.config import StageTimer, save_json

# threat weight per predicted flow class (used for risk score + RL state)
_CLASS_THREAT = {"Benign": 0, "PortScan": 1, "BruteForce": 2,
                 "DDoS": 2, "Botnet": 2}
_CLASS_BASE_RISK = {"Benign": 6, "PortScan": 35, "BruteForce": 60,
                    "DDoS": 75, "Botnet": 85}

_DOMAIN_RE = re.compile(r"(?:[a-z0-9-]+\.)+[a-z]{2,}", re.I)


class SentinelAgent:
    """Model-routing + decision-making agent."""

    def __init__(self):
        self.artifacts = {}
        self.incidents = []

    # ------------------------------------------------------------------
    def load(self):
        """Load every trained artifact needed at inference time."""
        M = config.MODEL_DIR
        self.artifacts["pipeline"] = joblib.load(
            os.path.join(M, "preprocess_pipeline.joblib"))
        with open(os.path.join(config.REPORT_DIR,
                               "supervised_results.json")) as fh:
            sup = json.load(fh)
        best = sup["best_model"]
        self.artifacts["flow_model"] = joblib.load(
            os.path.join(M, f"sup_{best}.joblib"))
        self.artifacts["flow_model_name"] = best
        iso = joblib.load(os.path.join(M, "unsup_isolation_forest.joblib"))
        self.artifacts["iso"], self.artifacts["iso_thr"] = (
            iso["model"], iso["threshold"])
        nlp = joblib.load(os.path.join(M, "nlp_spam.joblib"))
        self.artifacts.update({"tfidf": nlp["tfidf"], "nlp_model": nlp["model"],
                               "nlp_name": nlp["model_name"]})
        rl = joblib.load(os.path.join(M, "rl_qtable.joblib"))
        self.artifacts["Q"], self.artifacts["actions"] = rl["Q"], rl["actions"]

        # threat-intelligence feed (real public blocklist, IOC matching)
        feed_path = os.path.join(config.RAW_DIR, "threat_feed_domains.csv")
        if os.path.exists(feed_path):
            self.artifacts["blacklist"] = set(
                pd.read_csv(feed_path)["domain"].str.lower())

        # load lazily to keep memory small
        import tensorflow as tf
        self.artifacts["cnn"] = tf.keras.models.load_model(
            os.path.join(M, "cv_malware_cnn.keras"))
        self.artifacts["families"] = joblib.load(
            os.path.join(M, "cv_label_map.joblib"))["families"]
        return self

    def ioc_check(self, text: str):
        """Return the blacklisted domains found inside a text (IOCs)."""
        bl = self.artifacts.get("blacklist", set())
        if not bl:
            return []
        found = {d.lower() for d in _DOMAIN_RE.findall(text or "")}
        return sorted(found & bl)

    # ------------------------------------------------------------------
    # STEP 2: routing
    # ------------------------------------------------------------------
    def route(self, event: dict) -> str:
        if "dst_port" in event or "src_bytes" in event:
            return "flow"
        if "text" in event:
            return "message"
        if "image_path" in event:
            return "image"
        raise ValueError(f"cannot route event: {list(event)[:5]}")

    # ------------------------------------------------------------------
    # STEP 6: RL decision
    # ------------------------------------------------------------------
    def rl_decide(self, threat: int, confidence: float, criticality: int):
        conf = 0 if confidence < 0.70 else (1 if confidence < 0.90 else 2)
        s = threat * 6 + conf * 2 + criticality
        q = self.artifacts["Q"][s]
        action = self.artifacts["actions"][int(np.argmax(q))]
        return action, s, q.tolist()

    def _finish(self, event, etype, engine, label, confidence, anomaly,
                threat, risk, notes, iocs=None):
        action, s_idx, qvals = self.rl_decide(
            threat, confidence, int(event.get("criticality", 1)))
        incident = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event_type": etype,
            "engine_used": engine,
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "anomaly_second_opinion": bool(anomaly),
            "threat_level": config.THREAT_LEVELS[threat],
            "risk_score": int(risk),
            "response_action": action,
            "rl_state_index": s_idx,
            "rl_q_values": [round(v, 2) for v in qvals],
            "ioc_matches": iocs or [],
            "explanation": notes,
        }
        self.incidents.append(incident)
        return incident

    # ------------------------------------------------------------------
    # engine 1: network flow
    # ------------------------------------------------------------------
    def analyze_flow(self, features: dict) -> dict:
        pl = self.artifacts["pipeline"]
        row = pd.DataFrame([features])
        # --- same preprocessing as training (no leakage of unseen values) --
        row["avg_pkt_size"] = row.get("avg_pkt_size", 0).clip(lower=0)
        for col, med in pl["medians"].items():
            if col in row.columns:
                row[col] = row[col].fillna(med)
        row = row.fillna(0)
        for col in pl["log_columns"]:
            if col in row.columns:
                row[col] = np.log1p(row[col].clip(lower=0))
        row = pd.get_dummies(row, columns=["protocol"], prefix="proto",
                             drop_first=True)
        row = row.reindex(columns=pl["columns"], fill_value=0)
        Xs = pl["scaler"].transform(row.values.astype(float))
        Xf = pl["selector"].transform(Xs)

        model = self.artifacts["flow_model"]
        proba = model.predict_proba(Xf)[0]
        idx = int(np.argmax(proba))
        label = pl["label_encoder"].inverse_transform([idx])[0]
        confidence = proba[idx]

        # second opinion (label-free)
        score = float(self.artifacts["iso"].decision_function(Xf)[0])
        anomaly = score < self.artifacts["iso_thr"]

        threat = _CLASS_THREAT[label]
        risk = _CLASS_BASE_RISK[label]
        risk += (confidence - 0.5) * 30
        if anomaly and label == "Benign":
            risk += 20          # classifier says ok, anomaly says NO
        risk = float(np.clip(risk, 1, 99))

        notes = (f"Routed to tabular engine ({self.artifacts
                 ['flow_model_name']}) because the event is a network "
                 f"flow. Classified as {label} (p={confidence:.2f}). "
                 f"IsolationForest score {score:.3f} vs threshold "
                 f"{self.artifacts['iso_thr']:.3f} -> "
                 f"{'ANOMALOUS' if anomaly else 'normal'}.")
        return self._finish(features, "network_flow",
                            f"supervised:{self.artifacts['flow_model_name']}",
                            label, confidence, anomaly, threat, risk, notes)

    # ------------------------------------------------------------------
    # engine 2: message text
    # ------------------------------------------------------------------
    def analyze_message(self, text: str) -> dict:
        from src.nlp import clean_text
        clean = clean_text(text)
        X = self.artifacts["tfidf"].transform([clean])
        proba = self.artifacts["nlp_model"].predict_proba(X)[0]
        spam_p = float(proba[1])
        label = "spam/phishing" if spam_p >= 0.5 else "ham"
        confidence = spam_p if spam_p >= 0.5 else 1 - spam_p

        threat = 2 if spam_p >= 0.5 else 0
        risk = float(np.clip(spam_p * 90 + 5, 1, 99))

        # threat-intelligence enrichment: known-bad domain in the text?
        iocs = self.ioc_check(text)
        if iocs:
            threat = 2
            risk = float(np.clip(risk + 30, 1, 99))
            label = f"{label} + IOC"

        action_hint = {
            "Monitor": "deliver the message normally",
            "Alert_Analyst": "queue for analyst review",
            "Block_IP": "reject sender & purge queued copies",
            "Isolate_Host": "quarantine mailbox & reset credentials",
        }
        notes = (f"Routed to NLP engine (TF-IDF + "
                 f"{self.artifacts['nlp_name']}) because the event is text. "
                 f"P(spam)={spam_p:.3f}.")
        if iocs:
            notes += (f" Threat-intel IOC match: {', '.join(iocs)} "
                      f"(domain present in the public malware blocklist) "
                      f"-> risk raised.")
        notes += (f" Suggested handling: "
                  f"{action_hint[self.rl_decide(threat, confidence, 1)[0]]}.")
        return self._finish({"text": text[:80], "criticality": 1},
                            "message", f"nlp:{self.artifacts['nlp_name']}",
                            label, confidence, anomaly=None, threat=threat,
                            risk=risk, notes=notes, iocs=iocs)

    # ------------------------------------------------------------------
    # engine 3: malware image
    # ------------------------------------------------------------------
    def analyze_image(self, path: str) -> dict:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("L").resize(
            (config.MALWARE_IMG_SIZE, config.MALWARE_IMG_SIZE)),
            dtype=np.float32)[..., None] / 255.0
        proba = self.artifacts["cnn"].predict(img[None, ...], verbose=0)[0]
        idx = int(np.argmax(proba))
        label = self.artifacts["families"][idx]
        confidence = float(proba[idx])
        risk = float(np.clip(confidence * 95 + 4, 1, 99))
        notes = (f"Routed to CNN vision engine because the event is a file "
                 f"rendered as a byte-plot. Malware family: {label} "
                 f"(p={confidence:.2f}).")
        return self._finish({"image_path": os.path.basename(path),
                             "criticality": 1},
                            "malware_image", "cnn:tensorflow",
                            f"malware:{label}", confidence, anomaly=None,
                            threat=2, risk=risk, notes=notes)

    # ------------------------------------------------------------------
    def analyze(self, event: dict) -> dict:
        """Perceive -> route -> invoke the right tool -> decide."""
        etype = self.route(event)
        if etype == "flow":
            return self.analyze_flow(event)
        if etype == "message":
            return self.analyze_message(event["text"])
        return self.analyze_image(event["image_path"])

    # ------------------------------------------------------------------
    # STEP 7: reporting
    # ------------------------------------------------------------------
    def write_reports(self, out_prefix: str = "security_report") -> dict:
        df = pd.DataFrame(self.incidents)
        os.makedirs(config.PRED_DIR, exist_ok=True)
        os.makedirs(config.REPORT_DIR, exist_ok=True)

        df.to_csv(os.path.join(config.PRED_DIR,
                               "agent_decisions.csv"), index=False)
        with open(os.path.join(config.PRED_DIR, "incidents.jsonl"),
                  "w", encoding="utf-8") as fh:
            for inc in self.incidents:
                fh.write(json.dumps(inc, ensure_ascii=False) + "\n")

        n = len(df)
        by_action = df.response_action.value_counts().to_dict()
        by_pred = df.prediction.value_counts().to_dict()
        high = df[df.risk_score >= 70]
        disagreements = df[(df.anomaly_second_opinion.isin([True])) &
                           (df.threat_level == "Benign")]
        ioc_events = df[df.ioc_matches.apply(len) > 0]
        all_iocs = sorted({d for lst in df.ioc_matches for d in lst})

        recs = []
        if len(high):
            recs.append(f"Immediate containment: {len(high)} events scored "
                        f">=70 - apply the recommended Block/Isolate "
                        f"actions now.")
        if len(disagreements):
            recs.append(f"{len(disagreements)} events look benign to the "
                        f"classifier but anomalous to IsolationForest - "
                        f"possible ZERO-DAY behaviour, escalate to manual "
                        f"forensics.")
        if by_pred.get("DDoS", 0) + by_pred.get("Botnet", 0) > 0:
            recs.append("DDoS/Botnet activity detected - enable upstream "
                        "rate-limiting and review border firewall rules.")
        if by_pred.get("BruteForce", 0) > 0:
            recs.append("Brute-force attempts seen - enforce account "
                        "lockout + MFA on exposed services (SSH/RDP).")
        if "spam/phishing" in str(by_pred):
            recs.append("Phishing messages detected - purge from mailboxes "
                        "and run a user-awareness reminder.")
        if len(all_iocs):
            recs.append(f"Blocklist IOCs matched ({len(ioc_events)} events): "
                        f"{', '.join(all_iocs[:5])} - add to DNS/proxy "
                        f"sinkhole immediately.")
        if any(str(p).startswith("malware:") for p in by_pred):
            recs.append("Malware family identified by CNN - push the file "
                        "hash to endpoint detection (EDR) blocklists.")
        if not recs:
            recs.append("No significant threats in this batch - keep "
                        "monitoring.")

        lines = [
            "# SentinelAI - Automated Security Report",
            f"*Generated by the AI agent on {datetime.now():%Y-%m-%d %H:%M}*",
            "",
            f"**Events analysed:** {n}",
            "",
            "## 1. Decisions taken by the agent",
            "",
            "| # | type | engine | prediction | conf | anomaly? | truth | IOC | risk |",
            "|---|------|--------|-----------|------|----------|------|-----|",
        ]
        for i, inc in enumerate(self.incidents, 1):
            anom = "-" if inc["anomaly_second_opinion"] is None else \
                ("YES" if inc["anomaly_second_opinion"] else "no")
            truth = (f"{inc.get('ground_truth')} "
                     f"{'OK' if inc.get('agent_correct') else 'MISS'}"
                     if "ground_truth" in inc else "-")
            ioc = ", ".join(inc.get("ioc_matches") or []) or "-"
            lines.append(
                f"| {i} | {inc['event_type']} | {inc['engine_used']} | "
                f"{inc['prediction']} | {inc['confidence']:.2f} | {anom} | "
                f"{truth} | {ioc} | {inc['risk_score']} -> "
                f"**{inc['response_action']}** |")
        lines += ["", "## 2. Response distribution", ""]
        for k, v in by_action.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## 3. Risk analysis", "",
                  f"- High-risk events (>=70): **{len(high)}**",
                  f"- Classifier/anomaly disagreements (possible zero-day):"
                  f" **{len(disagreements)}**",
                  f"- Threat-intelligence IOC matches: **{len(ioc_events)}**",
                  "", "## 4. Recommendations", ""]
        lines += [f"- {r}" for r in recs]
        lines += ["", "---", "*Pipeline: supervised classification + "
                  "unsupervised anomaly detection + NLP + CNN + "
                  "RL response policy, orchestrated by SentinelAgent.*"]

        md_path = os.path.join(config.REPORT_DIR, f"{out_prefix}.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        return {"report_md": md_path, "n_events": n,
                "high_risk": int(len(high)),
                "disagreements": int(len(disagreements)),
                "ioc_events": int(len(ioc_events)),
                "by_action": by_action, "recommendations": recs}


# ======================================================================
# Demo on held-out data (data/test/)
# ======================================================================
def run() -> dict:
    with StageTimer(11, "AI Agent - orchestration demo on test events"):
        config.ensure_dirs()
        agent = SentinelAgent().load()

        # 1) network flows - RAW rows from data/raw (true label kept for
        #    agent-level evaluation), the agent preprocesses them itself
        raw_flows = pd.read_csv(os.path.join(config.RAW_DIR,
                                             "network_flows.csv"))
        crit_rng = np.random.default_rng(3)
        sample = raw_flows.sample(14, random_state=config.SEED)
        correct = 0
        for _, row in sample.iterrows():
            truth = row["label"]
            ev = row.drop(labels=["label"]).to_dict()
            ev["criticality"] = int(crit_rng.integers(0, 2))
            inc = agent.analyze_flow(ev)
            ok = (inc["prediction"] == truth)
            correct += int(ok)
            inc["ground_truth"] = truth
            inc["agent_correct"] = bool(ok)

        # 2) messages
        msgs = pd.read_csv(os.path.join(config.TEST_DIR,
                                        "messages_test_sample.csv"))
        for txt in msgs.sample(8, random_state=config.SEED)["text"]:
            agent.analyze_message(txt)

        # 2b) live IOC demonstration: a crafted message containing a REAL
        #     domain pulled from the downloaded public malware blocklist
        feed_path = os.path.join(config.RAW_DIR, "threat_feed_domains.csv")
        if os.path.exists(feed_path):
            demo_domain = pd.read_csv(feed_path)["domain"].iloc[100]
            agent.analyze_message(
                f"URGENT: verify your account now at http://{demo_domain} "
                f"or your card will be blocked today!")

        # 3) malware images
        img_dir = os.path.join(config.TEST_DIR, "malware")
        for f in sorted(os.listdir(img_dir))[:4]:
            agent.analyze_image(os.path.join(img_dir, f))

        summary = agent.write_reports()

        n_flows = sum(1 for i in agent.incidents if "agent_correct" in i)
        acc = correct / max(n_flows, 1)
        print(f"    agent handled {summary['n_events']} events across 3 "
              f"engines (flow accuracy {acc:.0%}) -> "
              f"{os.path.relpath(summary['report_md'], config.ROOT)}")
        for inc in agent.incidents[:8]:
            print(f"      [{inc['event_type']:<13}] {inc['prediction']:<22} "
                  f"risk={inc['risk_score']:>2} -> {inc['response_action']}")

        save_json(os.path.join(config.REPORT_DIR, "agent_results.json"),
                  {"n_events": summary["n_events"],
                   "high_risk": summary["high_risk"],
                   "disagreements": summary["disagreements"],
                   "ioc_events": summary["ioc_events"],
                   "by_action": summary["by_action"],
                   "flow_accuracy": round(acc, 4),
                   "workflow": ["receive event", "route to engine",
                                "model prediction + confidence",
                                "IsolationForest second opinion",
                                "threat-intel IOC enrichment",
                                "risk score fusion",
                                "RL policy response",
                                "incident + report writing"]})
    return summary


if __name__ == "__main__":
    run()
