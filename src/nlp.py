"""
SentinelAI - Stage 8: NLP - Spam / Phishing Text Detection
==========================================================
Real dataset: UCI SMS Spam Collection (5,574 real messages).

Text preprocessing pipeline (documented as required):
    1. lowercase                     models treat "FREE" and "free" equally
    2. URL/email/number masking      spam uses many links & phone numbers -
                                     replacing them with tokens (URL, EMAIL,
                                     NUM) keeps the SIGNAL without the noise
    3. punctuation stripping         symbols carry little class information
    4. stop-word removal (NLTK)      'the/is/at' do not discriminate
    5. Porter stemming               'winner/winners/winning' -> 'win'
    6. TF-IDF vectorization          unigrams + bigrams, max 3,000 features
                                     (bigrams capture "claim now", "txt me")

Models compared: Logistic Regression vs Multinomial Naive Bayes (classic
baseline for text). Evaluation: accuracy / precision / recall / F1 +
confusion matrix + the most "spammy" terms (model explainability) +
misclassified examples for discussion.
"""

import os
import re
import string

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src import config
from src import visualization as viz
from src.config import StageTimer, save_json


def get_stopwords():
    """NLTK stopwords with a safe offline fallback."""
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
    except Exception:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)


_STOP = None


def clean_text(text: str) -> str:
    """Steps 1-5 of the documented text-preprocessing pipeline."""
    global _STOP
    if _STOP is None:
        _STOP = get_stopwords()
    from nltk.stem.porter import PorterStemmer

    t = text.lower()                                  # 1. lowercase
    t = re.sub(r"http\S+|www\.\S+", " URL ", t)       # 2. mask urls
    t = re.sub(r"\S+@\S+", " EMAIL ", t)              #    mask emails
    t = re.sub(r"\b\d[\d.,:;-]*\b", " NUM ", t)       #    mask numbers
    t = t.translate(str.maketrans("", "", string.punctuation))  # 3.
    words = [w for w in t.split() if w not in _STOP]  # 4. stop words
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(w) for w in words)   # 5. stemming


def train_lstm(X_tr_series, y_tr, X_te_series, y_te):
    """Sequence model (Embedding -> LSTM) on RAW cleaned texts.

    Demonstrates the RNN/LSTM branch of the deep-learning requirement on
    TEXT data (the data type where sequences are justified), and is
    compared against the TF-IDF classic models on the SAME split.
    """
    import tensorflow as tf
    tf.random.set_seed(config.SEED)

    vocab, maxlen = 5000, 40
    vec = tf.keras.layers.TextVectorization(
        max_tokens=vocab, output_mode="int", output_sequence_length=maxlen)
    tr_texts = tf.constant(X_tr_series.to_numpy().tolist())   # string tensor
    te_texts = tf.constant(X_te_series.to_numpy().tolist())
    vec.adapt(tr_texts)                              # fitted on train only

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        vec,
        tf.keras.layers.Embedding(vocab, 32, mask_zero=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="SentinelAI_LSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(tr_texts, np.array(y_tr),
                     validation_split=0.15, epochs=8, batch_size=32,
                     verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(
                         patience=3, restore_best_weights=True,
                         monitor="val_loss")])
    y_prob = model.predict(te_texts, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    model.save(os.path.join(config.MODEL_DIR, "nlp_lstm.keras"))
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
    }, hist, "LSTM(Keras)"


def run() -> dict:
    with StageTimer(8, "NLP - spam/phishing text detection (real UCI data)"):
        config.ensure_dirs()
        from src.data_collection import load_sms_dataset
        df = load_sms_dataset()          # always rebuild from the raw source

        # class distribution figure
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        vc = df["label"].value_counts()
        ax.bar(vc.index, vc.values, color=[viz.PALETTE[0], viz.PALETTE[1]])
        ax.bar_label(ax.containers[0], fmt="%d")
        ax.set_title("SMS Spam dataset - class distribution")
        viz.save_fig(fig, "nlp_class_distribution.png")

        # text length by class
        df["n_chars"] = df["text"].str.len()
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for lab, c in zip(["ham", "spam"], [viz.PALETTE[0], viz.PALETTE[1]]):
            ax.hist(df.loc[df.label == lab, "n_chars"], bins=60, alpha=.65,
                    label=lab, color=c)
        ax.set_xlabel("message characters"); ax.legend()
        ax.set_title("Message length by class (spam is longer)")
        viz.save_fig(fig, "nlp_length_distribution.png")

        # ---- preprocessing -------------------------------------------
        df["clean"] = df["text"].apply(clean_text)
        df["y"] = (df["label"] == "spam").astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(
            df["clean"], df["y"], test_size=config.TEST_SIZE,
            stratify=df["y"], random_state=config.SEED)

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000,
                                min_df=2, sublinear_tf=True)
        Xv_tr = tfidf.fit_transform(X_tr)             # fitted on train only
        Xv_te = tfidf.transform(X_te)

        models = {
            "LogisticRegression": LogisticRegression(max_iter=2000,
                                                     random_state=config.SEED),
            "NaiveBayes": MultinomialNB(),
        }
        results = {}
        for name, m in models.items():
            m.fit(Xv_tr, y_tr)
            y_pred = m.predict(Xv_te)
            results[name] = {
                "accuracy": accuracy_score(y_te, y_pred),
                "precision": precision_score(y_te, y_pred),
                "recall": recall_score(y_te, y_pred),
                "f1": f1_score(y_te, y_pred),
            }
            print(f"    {name:<19} acc={results[name]['accuracy']:.4f} "
                  f"f1={results[name]['f1']:.4f}")

        best_name = max(results, key=lambda n: results[n]["f1"])
        best = models[best_name]
        y_pred = best.predict(Xv_te)

        # ---- sequence model: Embedding -> LSTM (advanced comparison) -----
        lstm_metrics, hist, lstm_name = train_lstm(X_tr, y_tr, X_te, y_te)
        results[lstm_name] = lstm_metrics
        print(f"    {lstm_name:<19} acc={lstm_metrics['accuracy']:.4f} "
              f"f1={lstm_metrics['f1']:.4f}")
        fig = viz.line_chart(
            range(1, len(hist.history["loss"]) + 1),
            [hist.history["loss"], hist.history["val_loss"]],
            ["train loss", "validation loss"],
            "LSTM training - binary cross-entropy", "epoch", "loss")
        viz.save_fig(fig, "nlp_lstm_training.png")
        best_classic_name = best_name                     # sklearn model
        best_overall = max(results, key=lambda n: results[n]["f1"])

        fig = viz.grouped_bar(results, ["accuracy", "precision", "recall",
                                        "f1"],
                              "NLP model comparison - spam detection",
                              "score", figsize=(7.4, 4.4))
        viz.save_fig(fig, "nlp_model_comparison.png")

        cm = confusion_matrix(y_te, y_pred)
        viz.save_fig(viz.confusion_matrix_heatmap(
            cm, ["ham", "spam"],
            f"Confusion matrix - {best_classic_name} (classic)"), "nlp_cm.png")

        # ---- explainability: most spam-indicating terms ----------------
        if hasattr(best, "coef_"):
            imp = best.coef_[0]
        else:                                          # MultinomialNB
            imp = np.log((best.feature_log_prob_[1] /
                          best.feature_log_prob_[0]))
        terms = np.array(tfidf.get_feature_names_out())
        order = np.argsort(imp)
        top_spam = terms[order[-18:]][::-1]
        top_spam_v = imp[order[-18:]][::-1]
        top_ham = terms[order[:18]]
        top_ham_v = imp[order[:18]]

        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        ax.barh(top_spam[::-1], top_spam_v[::-1], color=viz.PALETTE[1])
        ax.set_title(f"Top spam-indicating terms learned by {best_classic_name}")
        ax.set_xlabel("model weight (log-odds)")
        viz.save_fig(fig, "nlp_top_spam_terms.png")

        # ---- misclassified examples (for the discussion) ---------------
        mis = np.where(y_pred != y_te.values)[0]
        pd.DataFrame({
            "true": ["spam" if t else "ham" for t in y_te.values[mis[:40]]],
            "pred": ["spam" if p else "ham" for p in y_pred[mis[:40]]],
            "text": X_te.values[mis[:40]],
        }).to_csv(os.path.join(config.PRED_DIR,
                               "nlp_misclassified.csv"), index=False)

        # persist artifacts for the agent --------------------------------
        joblib.dump({"tfidf": tfidf, "model": best,
                     "model_name": best_classic_name},
                    os.path.join(config.MODEL_DIR, "nlp_spam.joblib"))
        # keep RAW texts so the agent demo handles realistic input
        pd.DataFrame({"text": df.loc[X_te.index, "text"],
                      "label": y_te.values}).to_csv(
            os.path.join(config.TEST_DIR, "messages_test_sample.csv"),
            index=False)

        summary = {
            "dataset": "UCI SMS Spam Collection (real, 5,574 messages)",
            "preprocessing_pipeline": ["lowercase", "URL/EMAIL/NUM masking",
                                       "punctuation removal",
                                       "NLTK stop-word removal",
                                       "Porter stemming",
                                       "TF-IDF (1-2 grams, 3000 features)"],
            "metrics": results,
            "best_model": best_overall,
            "best_classic": best_classic_name,
            "classification_report": classification_report(
                y_te, y_pred, target_names=["ham", "spam"], output_dict=True),
            "top_spam_terms": top_spam.tolist(),
            "top_ham_terms": top_ham.tolist(),
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        }
        save_json(os.path.join(config.REPORT_DIR, "nlp_results.json"),
                  summary)
    return summary


if __name__ == "__main__":
    run()
