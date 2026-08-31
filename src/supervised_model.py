"""
SentinelAI - Stage 4+5: Supervised Learning & Model Evaluation
==============================================================
Six classic algorithms (all listed in the project requirements) are trained
on the SAME processed training set and compared on the SAME test set:

    Logistic Regression, K-Nearest Neighbours, Decision Tree,
    Random Forest, Support Vector Machine (RBF), Naive Bayes.

Evaluation (Stage 5): accuracy, precision, recall, F1 (macro + weighted),
confusion matrix, ROC curves, and an explicit FALSE-POSITIVE / FALSE-
NEGATIVE discussion for the best model. Predictions are exported to
results/predictions/ for the report.

SVM note: SVC training is O(n^2); it is fitted on a capped 10,000-row
subsample (standard practice) to keep runtime reasonable - documented
decision, evaluated on the full test set.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize

from src import config
from src import visualization as viz
from src.config import StageTimer, save_json

TRAIN = os.path.join(config.PROC_DIR, "flows_train.csv")
TEST = os.path.join(config.PROC_DIR, "flows_test.csv")


def _load():
    tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
    return (tr.drop(columns=["label"]), tr["label"].values,
            te.drop(columns=["label"]), te["label"].values)


def build_models(random_state=config.SEED):
    """The six candidate algorithms."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB

    return {
        "LogisticRegression": LogisticRegression(max_iter=2000,
                                                 random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=200,
                                               random_state=random_state,
                                               n_jobs=-1),
        "SVM": SVC(kernel="rbf", probability=True,
                   random_state=random_state),      # capped subsample fit
        "NaiveBayes": GaussianNB(),
    }


def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    return {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision_macro": precision_score(y_te, y_pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_te, y_pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_te, y_pred, average="macro"),
        "f1_weighted": f1_score(y_te, y_pred, average="weighted"),
    }, y_pred


def cross_validate_models(X_tr, y_tr):
    """5-fold stratified cross-validation on the TRAINING set (validation
    strategy required by Stage 5/8 of the outline)."""
    from sklearn.model_selection import StratifiedKFold, cross_validate

    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.SEED)
    out = {}
    for name, model in build_models().items():
        fit_X, y_fit = X_tr, y_tr
        if name == "SVM" and len(X_tr) > config.SVM_TRAIN_CAP:
            idx = np.random.RandomState(config.SEED).choice(
                len(X_tr), config.SVM_TRAIN_CAP, replace=False)
            fit_X, y_fit = X_tr.iloc[idx], y_tr[idx]
        scores = cross_validate(model, fit_X, y_fit, cv=skf,
                                scoring=("accuracy", "f1_macro"),
                                n_jobs=-1)
        out[name] = {
            "accuracy_mean": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "f1_macro_mean": float(scores["test_f1_macro"].mean()),
            "f1_macro_std": float(scores["test_f1_macro"].std()),
            "folds": config.CV_FOLDS,
        }
        print(f"    CV {name:<19} acc={out[name]['accuracy_mean']:.4f}"
              f"±{out[name]['accuracy_std']:.4f}")
    return out


def grid_search_random_forest(X_tr, y_tr):
    """Hyperparameter tuning for the best model family (documented,
    small grid to keep runtime reasonable)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    grid = {"n_estimators": [100, 200, 400],
            "max_depth": [None, 14, 22]}
    gs = GridSearchCV(RandomForestClassifier(random_state=config.SEED,
                                             n_jobs=-1),
                      grid, cv=3, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return {"best_params": gs.best_params_,
            "best_cv_f1_macro": float(gs.best_score_),
            "grid": grid}


def imbalance_experiment(X_tr, y_tr, X_te, y_te, le):
    """Compare default vs class_weight='balanced' RandomForest on the
    minority classes (the dataset is intentionally imbalanced)."""
    from sklearn.ensemble import RandomForestClassifier

    classes = list(le.classes_)
    results = {}
    for tag, kwargs in {"default": {},
                        "balanced": {"class_weight": "balanced"}}.items():
        m = RandomForestClassifier(n_estimators=200, random_state=config.SEED,
                                   n_jobs=-1, **kwargs).fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        results[tag] = {
            "f1_macro": f1_score(y_te, y_pred, average="macro"),
            "recall_Botnet": float(recall_score(
                y_te, y_pred, labels=[classes.index("Botnet")],
                average=None)[0]),
            "recall_BruteForce": float(recall_score(
                y_te, y_pred, labels=[classes.index("BruteForce")],
                average=None)[0]),
        }
    return results


def run() -> dict:
    with StageTimer(4, "Supervised Learning (6 models) + Evaluation"):
        config.ensure_dirs()
        X_tr, y_tr, X_te, y_te = _load()
        le = joblib.load(os.path.join(
            config.MODEL_DIR, "preprocess_pipeline.joblib"))["label_encoder"]
        classes = list(le.classes_)

        results, preds = {}, {}
        for name, model in build_models().items():
            fit_X = X_tr
            if name == "SVM" and len(X_tr) > config.SVM_TRAIN_CAP:
                idx = np.random.RandomState(config.SEED).choice(
                    len(X_tr), config.SVM_TRAIN_CAP, replace=False)
                fit_X = X_tr.iloc[idx]
                y_fit = y_tr[idx]
            else:
                y_fit = y_tr
            model.fit(fit_X, y_fit)
            metrics, y_pred = evaluate(model, X_te, y_te)
            results[name] = metrics
            preds[name] = y_pred
            joblib.dump(model, os.path.join(
                config.MODEL_DIR, f"sup_{name}.joblib"))
            print(f"    {name:<19} acc={metrics['accuracy']:.4f} "
                  f"f1m={metrics['f1_macro']:.4f}")

        # -------- model comparison chart --------------------------------
        fig = viz.grouped_bar(results,
                              ["accuracy", "precision_macro", "recall_macro",
                               "f1_macro"],
                              "Supervised model comparison - network flows",
                              "score")
        viz.save_fig(fig, "supervised_model_comparison.png")

        # -------- 5-fold cross-validation (validation strategy) -----------
        cv_results = cross_validate_models(X_tr, y_tr)
        names = list(cv_results.keys())
        accs = [cv_results[n]["accuracy_mean"] for n in names]
        stds = [cv_results[n]["accuracy_std"] for n in names]
        fig, ax = plt.subplots(figsize=(9.6, 4.6))
        ax.bar(names, accs, yerr=stds, capsize=4,
               color=viz.PALETTE[:len(names)], alpha=.9)
        ax.set_ylim(0.85, 1.01)
        ax.set_ylabel("mean CV accuracy (± std)")
        ax.set_title(f"{config.CV_FOLDS}-fold stratified cross-validation "
                     "(training set)")
        ax.tick_params(axis="x", rotation=20)
        viz.save_fig(fig, "supervised_cross_validation.png")

        # -------- hyperparameter tuning (GridSearchCV) --------------------
        gs = grid_search_random_forest(X_tr, y_tr)
        print(f"    GridSearch RF best: {gs['best_params']} "
              f"(cv f1_macro={gs['best_cv_f1_macro']:.4f})")

        # -------- class-imbalance handling experiment ---------------------
        imb = imbalance_experiment(X_tr, y_tr, X_te, y_te, le)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        labels = ["default", "balanced"]
        ax[0].bar(labels, [imb[l]["f1_macro"] for l in labels],
                  color=[viz.PALETTE[0], viz.PALETTE[2]])
        ax[0].bar_label(ax[0].containers[0], fmt="%.4f")
        ax[0].set_ylim(0.98, 1.0)
        ax[0].set_title("macro-F1")
        x = np.arange(2); w = .38
        ax[1].bar(x - w/2, [imb[l]["recall_Botnet"] for l in labels], w,
                  label="Botnet recall", color=viz.PALETTE[1])
        ax[1].bar(x + w/2, [imb[l]["recall_BruteForce"] for l in labels], w,
                  label="BruteForce recall", color=viz.PALETTE[3])
        ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
        ax[1].set_ylim(0.9, 1.01); ax[1].legend()
        ax[1].set_title("minority-class recall")
        fig.suptitle("Class-imbalance handling: class_weight='balanced'",
                     fontweight="bold")
        viz.save_fig(fig, "supervised_imbalance.png")

        best_name = max(results, key=lambda n: results[n]["f1_macro"])
        best_model = joblib.load(os.path.join(
            config.MODEL_DIR, f"sup_{best_name}.joblib"))
        y_best = preds[best_name]

        # -------- confusion matrices (counts + normalized) ---------------
        cm = confusion_matrix(y_te, y_best, labels=range(len(classes)))
        viz.save_fig(viz.confusion_matrix_heatmap(
            cm, classes,
            f"Confusion matrix - {best_name}"), "supervised_cm_best.png")
        viz.save_fig(viz.confusion_matrix_heatmap(
            cm, classes,
            f"Confusion matrix (normalized) - {best_name}", normalize=True),
            "supervised_cm_best_norm.png")

        # -------- ROC curves (one-vs-rest micro/macro) -------------------
        y_bin = label_binarize(y_te, classes=range(len(classes)))
        if hasattr(best_model, "predict_proba"):
            proba = best_model.predict_proba(X_te)
        else:
            proba = best_model.decision_function(X_te)
            proba = np.exp(proba) / np.exp(proba).sum(1, keepdims=True)
        fpr, tpr = {}, {}
        fig, ax = plt.subplots(figsize=(6.6, 5.4))
        for i, cls in enumerate(classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], proba[:, i])
            ax.plot(fpr[i], tpr[i], label=f"{cls} (AUC={auc(fpr[i], tpr[i]):.3f})",
                    color=viz.PALETTE[i])
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
        ax.set_title(f"ROC curves (one-vs-rest) - {best_name}")
        ax.legend(fontsize=8)
        viz.save_fig(fig, "supervised_roc_best.png")

        # -------- FP / FN analysis (Stage 5 requirement) ------------------
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        err = pd.DataFrame({"class": classes, "FP": fp, "FN": fn})
        err["FP_rate"] = (err.FP / cm.sum(axis=0)).round(4)
        err["FN_rate"] = (err.FN / cm.sum(axis=1)).round(4)
        err.to_csv(os.path.join(config.PRED_DIR,
                                "supervised_fp_fn_analysis.csv"),
                   index=False)

        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        x = np.arange(len(classes)); w = 0.38
        ax.bar(x - w / 2, err.FP, w, label="False Positives", color="#e63946")
        ax.bar(x + w / 2, err.FN, w, label="False Negatives", color="#2e86ab")
        ax.set_xticks(x); ax.set_xticklabels(classes)
        ax.set_title(f"Error analysis - {best_name}")
        ax.legend()
        viz.save_fig(fig, "supervised_fp_fn.png")

        # misclassified examples for the report discussion
        mis_idx = np.where(y_best != y_te)[0][:60]
        pd.DataFrame({"true": le.inverse_transform(y_te[mis_idx]),
                      "pred": le.inverse_transform(y_best[mis_idx])}
                     ).to_csv(os.path.join(
                         config.PRED_DIR,
                         "supervised_misclassified.csv"), index=False)

        pd.DataFrame({"y_true": le.inverse_transform(y_te),
                      "y_pred": le.inverse_transform(y_best)}
                     ).to_csv(os.path.join(
                         config.PRED_DIR, "supervised_predictions.csv"),
                         index=False)

        report = classification_report(y_te, y_best,
                                       target_names=classes, output_dict=True)

        summary = {
            "metrics_per_model": results,
            "best_model": best_name,
            "classification_report_best": report,
            "fp_fn_analysis": err.to_dict(orient="records"),
            "cross_validation": cv_results,
            "grid_search": gs,
            "imbalance_handling": {
                **imb,
                "note": "class_weight='balanced' re-weights minority "
                        "classes in the loss; effect on this dataset is "
                        "small because the overlap (not the imbalance) "
                        "drives the remaining errors.",
            },
            "svm_fit_subsample": (f"SVC fitted on a capped sample of "
                                  f"{config.SVM_TRAIN_CAP} rows (O(n^2) "
                                  f"cost), evaluated on the full test set"),
            "discussion": {
                "false_positives": "A FP flags legitimate traffic as an "
                                   "attack -> analyst time wasted and "
                                   "possible service disruption.",
                "false_negatives": "A FN lets real attack traffic pass "
                                   "unnoticed -> the most expensive error "
                                   "in a SOC, hence recall matters most.",
            },
        }
        save_json(os.path.join(config.REPORT_DIR,
                               "supervised_results.json"), summary)
    return summary


if __name__ == "__main__":
    run()
