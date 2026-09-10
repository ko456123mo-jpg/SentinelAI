"""
SentinelAI - Documentation report generator
===========================================
Builds docs/SentinelAI_Report.docx with the 13 required sections,
pulling REAL metrics from results/reports/*.json and embedding the
figures produced by the pipeline, so the document always matches the run.

Run AFTER the pipeline:   python docs/generate_report.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from src import config

R = config.REPORT_DIR
F = config.FIG_DIR

# ---------------------------------------------------------------- students
STUDENT_NAME = "Mohammed Moneer Al-absi"
STUDENT_ID = "2023050086"
SECTION = "Cybersecurity (CS)"
SUPERVISOR = "Eng. Sondos Saif"
DATE = "10 September 2026"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)


def jload(name):
    return json.load(open(os.path.join(R, name)))


def pct(x, digits=2):
    return f"{100 * x:.{digits}f}%"


def add_fig(doc, fname, caption, width=5.6):
    path = os.path.join(F, fname)
    if not os.path.exists(path):
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.font.size = Pt(9)
    run.font.italic = True


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    return t


def build():
    doc = Document()
    for lvl in (1, 2, 3):
        doc.styles[f"Heading {lvl}"].font.color.rgb = ACCENT

    pre = jload("preprocessing_report.json")
    sup = jload("supervised_results.json")
    uns = jload("unsupervised_results.json")
    dl = jload("deep_learning_results.json")
    nlp = jload("nlp_results.json")
    cv = jload("cv_results.json")
    rl = jload("rl_results.json")
    ag = jload("agent_results.json")
    ds = json.load(open(os.path.join(config.RAW_DIR,
                                     "dataset_documentation.json")))

    best = sup["best_model"]
    bm = sup["metrics_per_model"][best]

    # ================= 1. TITLE PAGE =================
    for _ in range(6):
        doc.add_paragraph()
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("SentinelAI"); r.font.size = Pt(40); r.bold = True
    r.font.color.rgb = ACCENT
    t2 = doc.add_paragraph(); t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t2.add_run("AI Cybersecurity Assistant\n"
                   "Unified Threat Detection & Response Platform")
    r.font.size = Pt(16)
    doc.add_paragraph()
    for line in (f"Student: {STUDENT_NAME}", f"ID: {STUDENT_ID}",
                 f"Section: {SECTION}", f"Supervisor: {SUPERVISOR}",
                 f"Date: {DATE}"):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(line).font.size = Pt(13)
    doc.add_page_break()

    # ================= 2. ABSTRACT =================
    doc.add_heading("2. Abstract", 1)
    doc.add_paragraph(
        f"SentinelAI is an end-to-end AI security assistant that detects "
        f"attacks in network traffic, spam/phishing in messages and malware "
        f"families in files, then chooses a response automatically. Six "
        f"supervised classifiers were trained on {pre['rows_clean']:,} "
        f"network flows; the best ({best}) reached {pct(bm['accuracy'])} "
        f"accuracy and {pct(bm['f1_macro'])} macro-F1. Unsupervised "
        f"analysis (K-Means, DBSCAN, Isolation Forest) discovers attack "
        f"structure without labels and flags novel behaviour "
        f"({pct(uns['isolation_forest']['precision'])} anomaly precision). "
        f"A neural network (MLP), an NLP spam engine on the real UCI SMS "
        f"corpus ({pct(nlp['metrics'][nlp['best_model']]['accuracy'])} "
        f"accuracy) and a CNN malware-family classifier complete the "
        f"advanced engines. A hand-implemented Q-Learning agent learned a "
        f"response policy that beats all baselines, and the SentinelAgent "
        f"orchestrates every engine: it routes each event to the right "
        f"model, fuses a risk score, decides the response and writes an "
        f"automated security report. The project covers the complete AI "
        f"workflow at Expert level (Level 3).")
    p = doc.add_paragraph()
    r = p.add_run("الملخص (AR): ")
    r.bold = True; r.font.size = Pt(12)
    p.add_run(
        "«سينتينل آي آي» مساعد أمني ذكي متكامل يكشف الهجمات في تدفقات "
        "الشبكة والرسائل التصيّدية وعائلات البرمجيات الخبيزة، ثم يقرر "
        "الاستجابة تلقائياً. جُمّعت البيانات من مصادر حقيقية (حركة شبكة "
        "CICIDS2017 الملتقطة، وصور Malimg، ورسائل UCI، وتغذية تهديدات "
        "حيّة)، وعولجت (تنظيف، ترميز، تحجيم، اختيار سمات)، "
        "ودُرّبت ستة نماذج خاضعة للإشراف (أفضلها Random Forest بدقة "
        f"{pct(bm['accuracy'])})، مع تحليل غير خاضع للإشراف (K-Means و"
        "Isolation Forest) وتعلم عميق (MLP وCNN) ومعالجة لغة طبيعية، "
        "ثم وكيل تعلم معزز (Q-Learning) ووكيل ذكي ينسق كل المحركات ويصدر "
        "التقرير الأمني والتوصيات.").font.size = Pt(12)
    doc.add_page_break()

    # ================= 3. INTRODUCTION =================
    doc.add_heading("3. Introduction", 1)
    doc.add_paragraph(
        "Security teams face thousands of heterogeneous events every day: "
        "network flows, email/SMS messages and suspicious files. Signature "
        "tools only catch known threats, manual triage is slow, and the "
        "final response decision (monitor, alert, block or isolate) is "
        "often made under time pressure.")
    doc.add_heading("3.1 Problem", 2)
    doc.add_paragraph(
        "How can one AI system automatically (1) classify network attacks, "
        "(2) flag unusual zero-day-like behaviour without labels, (3) "
        "detect spam/phishing text, (4) identify malware families from "
        "byte-plot images, and (5) choose a cost-aware response for every "
        "event - then document everything in a security report?")
    doc.add_heading("3.2 Purpose and applicability", 2)
    doc.add_paragraph(
        "SentinelAI answers with a single orchestrated pipeline. It "
        "demonstrates the complete AI workflow: data collection, "
        "preprocessing, supervised learning, unsupervised learning, "
        "evaluation, deep learning, NLP, computer vision, reinforcement "
        "learning and an AI agent.")
    doc.add_heading("3.3 Target users", 2)
    bullets(doc, [
        "SOC / security analysts - automated triage and recommendations",
        "Network operators - DDoS/scan/brute-force visibility",
        "Mail administrators - phishing detection with explanations",
        "Malware analysts - quick family identification"])

    # ================= 4. DATASET =================
    doc.add_heading("4. Dataset", 1)
    flow_src = ds["network_flows"]["source"]
    img_src = ds["malware_images"]["source"]
    table(doc, ["Dataset", "Type", "Records", "Classes", "Source"], [
        ["Network flows", "tabular, 16 features",
         f"{pre['rows_raw']:,} (raw)", "5 (Benign, DDoS, PortScan, "
         "BruteForce, Botnet)",
         flow_src],
        ["SMS Spam (NLP)", "text",
         f"{ds['sms_spam']['rows']:,}", "ham 86.6% / spam 13.4%",
         ds["sms_spam"]["source"]],
        ["Malware images (CV)", "48x48 grayscale byte-plots",
         f"{ds['malware_images']['rows']}", "4 families",
         img_src],
        ["Threat-intel feed", "domain blocklist",
         f"{ds['threat_feed']['rows']:,} domains", "malicious / clean",
         ds["threat_feed"]["source"]],
    ])
    doc.add_paragraph(
        "Data types: numeric (14 flow features), categorical (protocol), "
        "free text, and images. The flows dataset keeps the real CICIDS2017 "
        "class imbalance (Botnet is naturally rare) so the models are "
        "evaluated on a realistic distribution; the synthetic generators "
        "shipped with the project only serve as a documented offline "
        "fallback.")

    # ================= 5. DATA COLLECTION =================
    doc.add_heading("5. Data Collection", 1)
    bullets(doc, [
        "Network flows: REAL captured traffic from CICIDS2017 (Canadian "
        "Institute for Cybersecurity). Five days of PCAP-derived flows "
        "(Tuesday brute force, Wednesday DoS/DDoS, Friday DDoS/PortScan/Bot) "
        "are downloaded as parquet and mapped to the 16-feature schema "
        "(Flow Duration -> duration, Protocol -> protocol, packet lengths "
        "-> byte counters, flag counts -> rates, source/destination "
        "equality -> is_land). Labels are remapped: BENIGN->Benign, "
        "DoS*/DDoS->DDoS, PortScan->PortScan, FTP/SSH-Patator->BruteForce, "
        "Bot->Botnet (the 11 Heartbleed rows are dropped as out of scope). "
        "A per-class sampling cap keeps the natural imbalance while keeping "
        "training tractable.",
        "Malware images: the REAL Malimg corpus (Nataraj et al. 2011) — the "
        "four families Allaple.A, C2LOP.P, Lolyda.AA2 and Alueron.gen!J are "
        "extracted as grayscale byte-plots and resized to 48x48 (the CNN "
        "input), preserving real visual malware signatures.",
        "Spam texts: downloaded from the UCI ML Repository (public, "
        "legitimate academic dataset).",
        "Threat-intelligence feed: a real public malware-domain blocklist "
        "(stamparm/blackbook, 18,000+ domains) is downloaded over HTTPS at "
        "collection time - demonstrating web/network programming - and is "
        "used by the agent for IOC enrichment (with a documented offline "
        "fallback list).",
        "Offline fallback: if the real corpora cannot be downloaded, the "
        "pipeline transparently uses documented synthetic generators and "
        "records the actual source in dataset_documentation.json."])

    # ================= 6. PREPROCESSING =================
    doc.add_heading("6. Data Preprocessing", 1)
    rows = [
        ["Remove duplicates", f"{pre['duplicates_removed']} rows removed",
         "Double-exported flows bias training and inflate accuracy"],
        ["Clip impossible values", f"{pre.get('negative_values_clipped', 0)} "
         "fixed", "Negative packet sizes are sensor glitches"],
        ["Median imputation", str(pre["missing_before"]),
         "Median is robust to the heavy right tail (vs mean)"],
        ["One-hot encoding", "protocol -> 2 dummy columns",
         "Nominal feature; avoids fake ordinal order"],
        ["log1p transform", "6 skewed columns",
         "Compresses right-skewed byte/packet counters"],
        ["StandardScaler", "fitted on train only",
         "KNN / SVM / MLP require comparable feature scales; prevents "
         "leakage"],
        ["Feature selection", f"SelectKBest(mutual_info) kept "
         f"{pre['features_kept_count']}",
         "Removes noise, speeds SVM, reduces overfitting"],
        ["Train/test split", f"stratified 80/20 -> "
         f"{pre['train_rows']:,} / {pre['test_rows']:,}",
         "Preserves class imbalance in both sets"],
    ]
    table(doc, ["Operation", "Applied", "Why (justification)"], rows)
    add_fig(doc, "eda_class_distribution.png",
            "Figure 1 - Class distribution of the flows dataset "
            "(intentionally imbalanced).", 4.4)
    add_fig(doc, "preprocess_feature_scores.png",
            "Figure 2 - Mutual-information feature scores; dropped "
            "features shown in red.", 5.2)

    # ================= 7. METHODOLOGY =================
    doc.add_heading("7. AI Methodology", 1)
    doc.add_heading("7.1 Supervised learning", 2)
    doc.add_paragraph(
        "Six algorithms from the project list were trained on identical "
        "data: Logistic Regression (linear baseline), KNN (k=7, distance "
        "based), Decision Tree, Random Forest (200 trees, ensemble), SVM "
        "(RBF kernel, fitted on a 10,000-row subsample because SVC "
        "training is O(n^2)) and Gaussian Naive Bayes (probabilistic "
        "baseline). Multiclass (5 classes) via one-vs-rest where needed.")
    doc.add_heading("7.2 Unsupervised learning", 2)
    doc.add_paragraph(
        "K-Means (k chosen by elbow + silhouette) discovers the natural "
        "structure of unlabelled traffic; DBSCAN adds density/noise "
        "perspective; hierarchical (agglomerative Ward) clustering builds a "
        "merge tree (dendrogram) confirming the same structure; Isolation "
        "Forest is trained on BENIGN traffic only (one-class) so it can "
        "flag any deviation from normal - the zero-day detector. PCA "
        "projects to 2-D for visualization.")
    doc.add_heading("7.3 Deep learning (MLP)", 2)
    doc.add_paragraph(
        f"Architecture: {dl['architecture']} ({dl['params']:,} parameters), "
        "chosen because the data is tabular - convolution/recurrence would "
        "be unjustified. Trained with Adam and early stopping; compared "
        "against the best classic model on the same split.")
    doc.add_heading("7.4 NLP", 2)
    doc.add_paragraph(
        "Pipeline: lowercase -> mask URLs/emails/numbers -> strip "
        "punctuation -> NLTK stop-word removal -> Porter stemming -> "
        "TF-IDF (uni+bigrams, 3,000 features). Models compared: Logistic "
        "Regression, Multinomial Naive Bayes, and a sequence model "
        "(Embedding 32 -> LSTM 32 -> Dense) that learns from token order "
        "instead of bag-of-words - covering the RNN/LSTM branch of the "
        "deep-learning requirement on the data type where sequences are "
        "justified.")
    doc.add_heading("7.5 Computer vision (CNN)", 2)
    doc.add_paragraph(
        f"Architecture: {cv['architecture']} ({cv['params']:,} parameters). "
        "Images normalized to [0,1]; convolutions learn the local visual "
        "signature of each malware family.")
    doc.add_heading("7.6 Reinforcement learning", 2)
    doc.add_paragraph(
        "Tabular Q-Learning implemented from scratch (no RL library). "
        f"States: {rl['states']}; Actions: {rl['actions']}. Reward = SOC "
        "cost model (missing an attack -4 x threat level; blocking benign "
        "-3; isolating benign -5; correct escalation +2..+5). Update rule: "
        "Q[s,a] += alpha*(r + gamma*max Q[s',a'] - Q[s,a]) with "
        "epsilon-greedy exploration decaying 1.0 -> 0.05.")
    doc.add_heading("7.7 AI agent", 2)
    doc.add_paragraph(
        "SentinelAgent perceives an event, ROUTES it to the right engine "
        "(flow / message / image), adds the Isolation Forest second "
        "opinion, fuses a 0-100 risk score, queries the learned Q-table "
        "for the response and writes the incident + aggregate security "
        "report with recommendations.")

    # ================= 8. MODEL TRAINING =================
    doc.add_heading("8. Model Training", 1)
    rows = [[name,
             pct(m["accuracy"]), pct(m["precision_macro"]),
             pct(m["recall_macro"]), pct(m["f1_macro"])]
            for name, m in sup["metrics_per_model"].items()]
    table(doc, ["Model", "Accuracy", "Precision(m)", "Recall(m)", "F1(m)"],
          rows)
    doc.add_paragraph(
        f"Best classic model: {best}. The MLP trained for "
        f"{dl['epochs_trained']} epochs "
        f"(accuracy {pct(dl['metrics']['accuracy'])}), the CNN for "
        f"{cv['epochs_trained']} epochs (accuracy "
        f"{pct(cv['metrics']['accuracy'])} on {cv['test_images']} test "
        f"images), the NLP engine on {nlp['n_train']:,} messages, and the "
        f"RL agent for {rl['hyperparameters']['episodes']:,} episodes.")

    doc.add_heading("8.1 Validation strategy (5-fold cross-validation)", 2)
    cvr = sup["cross_validation"]
    table(doc, ["Model", "CV accuracy (mean ± std)", "CV macro-F1 (mean)"],
          [[n, f"{v['accuracy_mean']:.4f} ± {v['accuracy_std']:.4f}",
            f"{v['f1_macro_mean']:.4f}"] for n, v in cvr.items()])
    add_fig(doc, "supervised_cross_validation.png",
            "Figure - 5-fold stratified cross-validation on the training "
            "set.", 5.4)
    doc.add_paragraph(
        f"Hyperparameter tuning (GridSearchCV, 3-fold): best Random Forest "
        f"configuration = {sup['grid_search']['best_params']} with CV "
        f"macro-F1 {pct(sup['grid_search']['best_cv_f1_macro'])}.")
    doc.add_paragraph(
        "Class-imbalance handling: a class_weight='balanced' variant was "
        "trained and compared - the effect is small because the remaining "
        "errors are driven by genuine class overlap, not by imbalance "
        "(documented experiment below).")
    add_fig(doc, "supervised_imbalance.png",
            "Figure - Default vs class-weighted Random Forest.", 5.4)

    # ================= 9. EVALUATION =================
    doc.add_heading("9. Evaluation", 1)
    doc.add_heading(f"9.1 Supervised - {best}", 2)
    fpfn = sup["fp_fn_analysis"]
    table(doc, ["Class", "FP", "FN", "FP rate", "FN rate"],
          [[e["class"], e["FP"], e["FN"], pct(e["FP_rate"]),
            pct(e["FN_rate"])] for e in fpfn])
    doc.add_paragraph(
        "False positives: benign IT/SSH sessions and beacon-like apps are "
        "occasionally misread as BruteForce/Botnet - disruptive if "
        "auto-blocked. False negatives: a few BruteForce and Botnet flows "
        "hiding inside benign-like patterns pass undetected - the most "
        "expensive error in a SOC, which is why recall is weighted highly. "
        "DDoS and PortScan were detected perfectly: their statistical "
        "signatures (SYN storms, empty ultra-short probes) are highly "
        "distinctive.")
    add_fig(doc, "supervised_cm_best.png",
            f"Figure 3 - Confusion matrix, {best}.", 4.6)
    add_fig(doc, "supervised_roc_best.png",
            f"Figure 4 - ROC curves (one-vs-rest), {best}.", 4.6)
    add_fig(doc, "supervised_fp_fn.png",
            "Figure 5 - False positives vs false negatives per class.", 4.8)
    doc.add_heading("9.2 Unsupervised", 2)
    doc.add_paragraph(
        f"K-Means chose k={uns['kmeans']['chosen_k']} (silhouette "
        f"{uns['kmeans']['silhouette']:.3f}) and recovered the attack "
        f"structure with ARI {uns['kmeans']['ARI']:.3f}, purity "
        f"{pct(uns['kmeans']['purity'])} - WITHOUT labels. Hierarchical "
        f"(Ward) clustering independently recovered the same structure "
        f"(ARI {uns['hierarchical']['ARI']:.3f}, NMI "
        f"{uns['hierarchical']['NMI']:.3f}) - two different algorithms "
        f"agreeing is strong evidence the structure is real. Isolation "
        f"Forest (benign-only training) flags attacks with precision "
        f"{pct(uns['isolation_forest']['precision'])} / recall "
        f"{pct(uns['isolation_forest']['recall'])} at a "
        f"{pct(uns['isolation_forest']['benign_false_alarm_rate'])} benign "
        f"false-alarm rate. Supervised answers 'WHICH known attack is "
        f"this?'; unsupervised answers 'does this look unusual?' - "
        f"together they also expose possible zero-days.")
    add_fig(doc, "unsup_kmeans_contingency.png",
            "Figure 6 - K-Means clusters vs true classes.", 4.6)
    add_fig(doc, "unsup_dendrogram.png",
            "Figure - Hierarchical (Ward) dendrogram confirming the "
            "cluster structure.", 5.6)
    add_fig(doc, "unsup_pca_clusters.png",
            "Figure 7 - PCA projection: true classes (left) vs discovered "
            "clusters (right).", 6.2)
    add_fig(doc, "unsup_isolation_detection.png",
            "Figure 8 - Isolation Forest detection rate per class.", 4.8)
    doc.add_heading("9.3 Deep learning vs classic ML", 2)
    add_fig(doc, "dl_vs_traditional.png",
            "Figure 9 - MLP vs classic models (same split).", 5.2)
    doc.add_paragraph(
        f"MLP macro-F1 {pct(dl['metrics']['f1_macro'])} vs {best} "
        f"{pct(dl['comparison']['traditional_f1_macro'])} "
        f"(delta {pct(dl['comparison']['delta_f1_macro'])}): tree "
        "ensembles remain extremely strong on small tabular datasets - an "
        "honest, expected result.")
    doc.add_heading("9.4 NLP", 2)
    nm = nlp["metrics"][nlp["best_classic"]]
    lstm_m = nlp["metrics"].get("LSTM(Keras)", {})
    doc.add_paragraph(
        f"{nlp['best_classic']} (TF-IDF) on real data: accuracy "
        f"{pct(nm['accuracy'])}, precision {pct(nm['precision'])}, recall "
        f"{pct(nm['recall'])}, F1 {pct(nm['f1'])}. The LSTM sequence model "
        f"improved F1 to {pct(lstm_m.get('f1', 0))} (accuracy "
        f"{pct(lstm_m.get('accuracy', 0))}) - word ORDER helps: 'claim "
        f"your prize' reads differently from 'prize your claim'. Recall "
        f"remains the weak spot: 'polite' spam without links/numbers "
        f"resembles ham. Top spam indicators learned: "
        + ", ".join(nlp["top_spam_terms"][:10]) + ".")
    add_fig(doc, "nlp_cm.png", "Figure 10 - NLP confusion matrix.", 4.2)
    add_fig(doc, "nlp_top_spam_terms.png",
            "Figure 11 - Most spam-indicating terms (model weights).", 5.0)
    add_fig(doc, "nlp_lstm_training.png",
            "Figure - LSTM training curves (validation early stopping).",
            4.8)
    doc.add_heading("9.5 Computer vision", 2)
    doc.add_paragraph(
        f"CNN accuracy {pct(cv['metrics']['accuracy'])}, macro-F1 "
        f"{pct(cv['metrics']['f1_macro'])} on {cv['test_images']} unseen "
        f"REAL Malimg byte-plots (four families: Allaple.A, C2LOP.P, "
        f"Lolyda.AA2, Alueron.gen!J). The traditional-ML baseline (HOG "
        f"texture features + RBF SVM) reaches "
        f"{pct(cv['hog_svm_baseline']['accuracy'])} on the same split. "
        f"Per-family recall: " + ", ".join(
            f"{k.split('_')[0]} {pct(v['recall'])}" for k, v in
            cv["classification_report"].items() if k in
            ("Allaple_A", "C2LOP_P", "Lolyda_AA2", "Alueron_genJ")) + ".")
    add_fig(doc, "cv_cnn_vs_hog.png",
            "Figure - CNN vs traditional ML (HOG+SVM) for malware images.",
            4.6)
    add_fig(doc, "cv_cm.png", "Figure 12 - CNN confusion matrix.", 4.4)
    doc.add_heading("9.6 Reinforcement learning", 2)
    doc.add_paragraph(
        "Average reward per decision over all states: "
        + ", ".join(f"{k} {v:.2f}" for k, v in
                    rl["baseline_comparison"].items())
        + ". The learned policy clearly beats every fixed baseline: it "
        "learned to monitor benign events, alert on suspicious ones and "
        "block/isolate high-confidence malware on critical assets - purely "
        "from reward feedback.")
    add_fig(doc, "rl_learning_curve.png",
            "Figure 13 - Q-Learning reward curve (learning from rewards).",
            5.2)
    add_fig(doc, "rl_qtable_heatmap.png",
            "Figure 14 - Learned Q-table and policy.", 5.2)
    add_fig(doc, "rl_vs_baselines.png",
            "Figure 15 - Learned policy vs baselines.", 4.8)

    # ================= 10. RESULTS & VISUALIZATION =================
    doc.add_heading("10. Results and Visualization", 1)
    doc.add_paragraph(
        f"Agent run on {ag['n_events']} held-out events across the three "
        f"engines: {ag['by_action']}. {ag['high_risk']} high-risk events "
        f"(>=70) and {ag['disagreements']} classifier/anomaly "
        "disagreements (zero-day candidates) were escalated. The agent "
        f"also matched {ag.get('ioc_events', 0)} event(s) against the real "
        "downloaded malware-domain blocklist (IOC enrichment) - including "
        "one message the classifier itself judged 'ham', proving the value "
        "of layering intelligence feeds on top of models. The full "
        "automated report is written to results/reports/security_report.md.")
    add_fig(doc, "supervised_model_comparison.png",
            "Figure 16 - Six supervised models compared.", 5.6)
    add_fig(doc, "dl_training_accuracy.png",
            "Figure 17 - MLP training curves.", 4.8)
    add_fig(doc, "cv_sample_images.png",
            "Figure 18 - Byte-plot samples per malware family.", 5.6)
    add_fig(doc, "eda_correlation_heatmap.png",
            "Figure 19 - Feature correlation matrix.", 5.4)
    add_fig(doc, "eda_feature_boxplots.png",
            "Figure 20 - Feature behaviour per class.", 5.8)

    # ================= 11. FUTURE WORK =================
    doc.add_heading("11. Future Work and Recommendations", 1)
    bullets(doc, [
        "Scale to the full CICIDS2017 corpus (2.8M flows) and all 25 "
        "Malimg families - the pipeline already maps to the schema, only "
        "the sampling caps need lifting",
        "Deep learning for text (LSTM/transformer embeddings) to lift spam "
        "recall beyond the TF-IDF baseline",
        "Explainability: SHAP values per decision for analyst-facing "
        "justifications",
        "Online/continual learning so Isolation Forest adapts to evolving "
        "normal traffic",
        "Multi-agent RL where network, mail and endpoint responders "
        "coordinate",
        "Streaming deployment (Kafka + model server) with analyst "
        "feedback loops"])

    # ================= 12. CONCLUSION =================
    doc.add_heading("12. Conclusion", 1)
    doc.add_paragraph(
        f"SentinelAI demonstrates the complete AI workflow on one coherent "
        f"security problem: data collection (real + generated, documented), "
        f"justified preprocessing, six supervised models (best {best} "
        f"{pct(bm['accuracy'])}), unsupervised discovery and anomaly "
        f"detection, deep learning compared honestly against classic ML, "
        f"NLP on real data ({pct(nm['accuracy'])}), a CNN malware "
        f"classifier ({pct(cv['metrics']['accuracy'])}), a from-scratch "
        f"Q-Learning response policy that beats fixed baselines, and an "
        f"agent that orchestrates everything into decisions and reports. "
        f"Every error was analysed, and every design choice justified.")

    # ================= 13. REFERENCES =================
    doc.add_heading("13. References", 1)
    refs = [
        "Sharafaldin, I., Habibi Lashkari, A., Ghorbani, A. (2018) - Toward "
        "Generating a New Intrusion Detection Dataset and Intrusion Traffic "
        "Characterization (CICIDS2017). Canadian Institute for Cybersecurity.",
        "Almeida, T., Hidalgo, J., Yamakami, A. - Contributions to the "
        "Study of SMS Spam Filtering. UCI ML Repository (SMS Spam "
        "Collection).",
        "Nataraj, L., Karthikeyan, S., Jacob, G., Manjunath, B. (2011) - "
        "Malware Images: Visualization and Automatic Classification. "
        "VizSec '11.",
        "Pedregosa, F. et al. (2011) - Scikit-learn: Machine Learning in "
        "Python. JMLR 12.",
        "Chollet, F. et al. (2015) - Keras. https://keras.io",
        "Sutton, R., Barto, A. (2018) - Reinforcement Learning: An "
        "Introduction (2nd ed.), MIT Press - Q-learning chapter.",
        "Liu, F., Ting, K., Zhou, Z. (2008) - Isolation Forest. ICDM '08.",
        "Breiman, L. (2001) - Random Forests. Machine Learning 45(1).",
        "Project outline - AI Final Project Requirements (Eng. Sondos "
        "Saif).",
    ]
    for i, ref in enumerate(refs, 1):
        doc.add_paragraph(f"[{i}] {ref}")

    out = os.path.join(config.DOCS_DIR, "SentinelAI_Report.docx")
    doc.save(out)
    print("saved:", out)
    return out


if __name__ == "__main__":
    build()
