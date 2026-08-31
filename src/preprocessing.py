"""
SentinelAI - Stage 3: Data Preprocessing (network-flow tabular data)
====================================================================
Applied operations  (and WHY - as required by the project rubric):

1. REMOVE DUPLICATES      double-exported flows would bias training toward
                           duplicated behaviour and inflate test accuracy.
2. FIX IMPOSSIBLE VALUES   negative packet sizes are sensor glitches ->
                           clipped to 0 (physical lower bound).
3. HANDLE MISSING VALUES   byte_std / flow_iat_mean have ~1.5% holes.
                           Median imputation (robust to the heavy right
                           tail, unlike the mean).
4. ENCODE CATEGORICALS     'protocol' is nominal -> One-Hot encoding
                           (no artificial order between TCP/UDP/ICMP);
                           target 'label' -> integer Label Encoding.
5. LOG TRANSFORM           byte/packet counts are heavily right-skewed ->
                           log1p compresses the tail, makes scaling useful.
6. SCALING                 StandardScaler (mean 0 / std 1) required by
                           KNN, SVM and the neural network (distance and
                           gradient based models). Fitted on TRAIN only
                           to avoid information leakage.
7. FEATURE SELECTION       SelectKBest(mutual_info) keeps the 12 most
                           informative features -> removes noise, speeds
                           SVM, reduces overfitting risk.
8. TRAIN/TEST SPLIT        stratified 80/20 - keeps class ratios in both
                           sets (important with an imbalanced dataset).

Outputs -> data/processed/*.csv + models/preprocess_pipeline.joblib
           + results/figures/eda_*.png + preprocessing_report.json
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config
from src import visualization as viz
from src.config import StageTimer, save_json

RAW_FLOWS = os.path.join(config.RAW_DIR, "network_flows.csv")
JUSTIFICATIONS = {
    "duplicates_removed": "Duplicated flow records (double export) would "
                          "bias the trained models and inflate accuracy.",
    "missing_imputed": "Median imputation for byte_std / flow_iat_mean - "
                       "median is robust to the heavy right tail.",
    "one_hot_encoding": "'protocol' is nominal (no order) -> one-hot "
                        "avoids inventing a fake ordinal relationship.",
    "log1p_transform": "Byte/packet counts are right-skewed; log1p "
                       "compresses extreme values toward normality.",
    "standard_scaling": "KNN, SVM and MLP need features on the same scale; "
                        "fitted on train only (no leakage).",
    "feature_selection": "Mutual-information SelectKBest keeps the 12 most "
                         "informative features (less noise, faster SVM).",
    "stratified_split": "80/20 stratified split preserves the class "
                        "imbalance in both train and test sets.",
}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_FLOWS)


def clean(data: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Steps 1-3: duplicates, impossible values, missing values."""
    n0 = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(n0 - len(data))

    # clip impossible negative packet sizes to 0
    neg = (data["avg_pkt_size"] < 0).sum()
    data["avg_pkt_size"] = data["avg_pkt_size"].clip(lower=0)
    report["negative_values_clipped"] = int(neg)

    # missing values -> median imputation
    miss_before = data.isna().sum()
    report["missing_before"] = {k: int(v) for k, v in miss_before.items()
                                if v > 0}
    medians = data.median(numeric_only=True)
    data = data.fillna(medians)
    report["missing_after"] = int(data.isna().sum().sum())
    report["medians"] = medians.to_dict()          # reused by the agent
    return data


def encode_and_transform(data: pd.DataFrame, report: dict):
    """Steps 4-5: one-hot protocol, label-encode target, log1p skewed cols."""
    y = LabelEncoder().fit(data["label"])
    labels = y.transform(data["label"])
    report["label_mapping"] = dict(zip(y.classes_,
                                       y.transform(y.classes_).tolist()))

    X = pd.get_dummies(data.drop(columns=["label"]), columns=["protocol"],
                       prefix="proto", drop_first=True)

    skewed = ["src_bytes", "dst_bytes", "byte_std", "flow_iat_mean",
              "duration", "active_duration"]
    X[skewed] = np.log1p(X[skewed].clip(lower=0))
    report["log1p_columns"] = skewed
    return X, labels, y


def eda_figures(raw: pd.DataFrame):
    """Exploratory-visualizations used in the report."""
    # class distribution
    fig, ax = viz.plt.subplots(figsize=(6.4, 4))
    counts = raw["label"].value_counts()
    ax.bar(counts.index, counts.values,
           color=viz.PALETTE[:len(counts)])
    ax.bar_label(ax.containers[0], fmt="%d")
    ax.set_title("Class distribution - network flows (imbalanced by design)")
    ax.set_ylabel("flows")
    viz.save_fig(fig, "eda_class_distribution.png")

    # correlation heatmap
    num = raw.select_dtypes("number")
    fig, ax = viz.plt.subplots(figsize=(8.6, 7))
    im = ax.imshow(num.corr(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num.columns)))
    ax.set_yticks(range(len(num.columns)))
    ax.set_xticklabels(num.columns, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(num.columns, fontsize=8)
    ax.set_title("Feature correlation matrix (before selection)")
    ax.grid(False)
    fig.colorbar(im, fraction=0.046)
    viz.save_fig(fig, "eda_correlation_heatmap.png")

    # feature behaviour per class (2 informative features)
    fig, axes = viz.plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, scale in zip(axes, ["syn_rate", "avg_pkt_size"],
                              [False, True]):
        groups = [raw.loc[raw.label == c, col] for c in config.FLOW_CLASSES]
        ax.boxplot(groups, labels=config.FLOW_CLASSES, showfliers=False)
        ax.set_title(f"{col} by class")
        if scale:
            ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=35)
    viz.save_fig(fig, "eda_feature_boxplots.png")


def run() -> dict:
    with StageTimer(3, "Data Preprocessing (tabular flows)"):
        config.ensure_dirs()
        raw = load_raw()
        report = {"rows_raw": int(len(raw)),
                  "justifications": JUSTIFICATIONS}

        eda_figures(raw)

        data = clean(raw, report)
        X, labels, label_encoder = encode_and_transform(data, report)

        # 8. stratified split FIRST, then fit scaler/selector on train only
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, labels, test_size=config.TEST_SIZE, stratify=labels,
            random_state=config.SEED)

        # 6. scaling
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # 7. feature selection on scaled train data
        selector = SelectKBest(mutual_info_classif,
                               k=min(config.SELECT_K_FEATURES, X.shape[1]))
        selector.fit(X_tr_s, y_tr)
        kept = X.columns[selector.get_support()].tolist()
        dropped = [c for c in X.columns if c not in kept]
        report["features_kept"] = kept
        report["features_dropped"] = dropped

        X_tr_f = selector.transform(X_tr_s)
        X_te_f = selector.transform(X_te_s)

        # save processed datasets
        cols = kept
        pd.DataFrame(X_tr_f, columns=cols).assign(
            label=y_tr).to_csv(
            os.path.join(config.PROC_DIR, "flows_train.csv"), index=False)
        pd.DataFrame(X_te_f, columns=cols).assign(
            label=y_te).to_csv(
            os.path.join(config.PROC_DIR, "flows_test.csv"), index=False)

        # save fitted pipeline for the agent (train-only fitted!)
        joblib.dump({"scaler": scaler, "selector": selector,
                     "columns": list(X.columns), "kept_columns": cols,
                     "medians": report["medians"],
                     "log_columns": report["log1p_columns"],
                     "label_encoder": label_encoder},
                    os.path.join(config.MODEL_DIR,
                                 "preprocess_pipeline.joblib"))

        # feature-score chart
        scores = pd.Series(selector.scores_, index=X.columns).sort_values()
        fig, ax = viz.plt.subplots(figsize=(8.4, 5.4))
        colors = [viz.PALETTE[1] if c in dropped else viz.PALETTE[0]
                  for c in scores.index]
        ax.barh(scores.index, scores.values, color=colors)
        ax.set_title("Mutual-information feature scores\n"
                     "(red = dropped by SelectKBest)")
        viz.save_fig(fig, "preprocess_feature_scores.png")

        report.update({
            "rows_clean": int(len(data)),
            "features_after_encoding": int(X.shape[1]),
            "features_kept_count": len(kept),
            "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
            "train_class_distribution": {c: int(v) for c, v in
                                         zip(*np.unique(
                                             label_encoder.inverse_transform(
                                                 y_tr),
                                             return_counts=True))},
            "test_class_distribution": {c: int(v) for c, v in
                                        zip(*np.unique(
                                            label_encoder.inverse_transform(
                                                y_te),
                                            return_counts=True))},
        })
        save_json(os.path.join(config.REPORT_DIR,
                               "preprocessing_report.json"), report)
    return report


if __name__ == "__main__":
    run()
