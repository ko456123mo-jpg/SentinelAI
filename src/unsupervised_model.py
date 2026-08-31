"""
SentinelAI - Stage 6: Unsupervised Learning
===========================================
Three complementary techniques, all trained WITHOUT labels:

1. K-MEANS clustering        - discovers the natural attack structure of
                               the traffic. k is chosen with the elbow
                               method + silhouette score; clusters are then
                               compared to true labels ONLY to judge what
                               was discovered (ARI, purity).
2. DBSCAN                    - density-based clustering on PCA-reduced
                               data; finds noise/outliers without choosing k.
3. ISOLATION FOREST          - one-class anomaly detection: trained on
                               BENIGN traffic only (a realistic SOC setup:
                               "learn what normal looks like, flag the
                               rest"), then used to flag anomalies on the
                               mixed test set. Catches attacks (incl.
                               zero-day-like deviations) it never saw.

PCA is used for 2-D visualization of the clusters.

Supervised-vs-unsupervised comparison is written into the stage report
(required by the project outline).
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (adjusted_rand_score, confusion_matrix,
                             normalized_mutual_info_score, precision_score,
                             recall_score, f1_score, silhouette_score)

from src import config
from src import visualization as viz
from src.config import StageTimer, save_json

TRAIN = os.path.join(config.PROC_DIR, "flows_train.csv")
TEST = os.path.join(config.PROC_DIR, "flows_test.csv")


def _load():
    tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
    return (tr.drop(columns=["label"]), tr["label"].values,
            te.drop(columns=["label"]), te["label"].values)


def choose_k(X, sample=None):
    """Elbow (inertia) + silhouette for every candidate k."""
    Xs = X if sample is None else X[sample]
    inertias, sils = [], []
    for k in config.K_RANGE:
        km = KMeans(n_clusters=k, random_state=config.SEED, n_init=10)
        lab = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, lab))
    return list(config.K_RANGE), inertias, sils


def run() -> dict:
    with StageTimer(6, "Unsupervised Learning (K-Means / DBSCAN / IsolationForest)"):
        config.ensure_dirs()
        X_tr, y_tr, X_te, y_te = _load()
        le = joblib.load(os.path.join(
            config.MODEL_DIR, "preprocess_pipeline.joblib"))["label_encoder"]
        classes = list(le.classes_)

        silhouette_n = min(6000, len(X_tr))
        sil_idx = np.random.RandomState(config.SEED).choice(
            len(X_tr), silhouette_n, replace=False)

        # ---------------- 1. K-MEANS ------------------------------------
        ks, inertias, sils = choose_k(X_tr.values, sil_idx)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].plot(ks, inertias, "o-", color=viz.PALETTE[0])
        ax[0].set_title("Elbow method (inertia)"); ax[0].set_xlabel("k")
        ax[1].plot(ks, sils, "o-", color=viz.PALETTE[2])
        ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
        viz.save_fig(fig, "unsup_k_selection.png")

        best_k = ks[int(np.argmax(sils))]
        km = KMeans(n_clusters=best_k, random_state=config.SEED,
                    n_init=10).fit(X_tr)
        cluster_labels = km.predict(X_te)
        ari = adjusted_rand_score(y_te, cluster_labels)
        nmi = normalized_mutual_info_score(y_te, cluster_labels)

        # contingency matrix: which true class lands in which cluster?
        cont = confusion_matrix(y_te, cluster_labels)
        purity = cont.max(axis=0).sum() / cont.sum()
        viz.save_fig(_cont_fig(cont, classes, best_k),
                     "unsup_kmeans_contingency.png")
        joblib.dump(km, os.path.join(config.MODEL_DIR, "unsup_kmeans.joblib"))

        # ---------------- 2. PCA (visualization + DBSCAN input) ---------
        pca2 = PCA(n_components=2, random_state=config.SEED).fit(X_tr)
        te_2d = pca2.transform(X_te)

        fig, ax = plt.subplots(1, 2, figsize=(12.4, 5))
        for i, cls in enumerate(classes):
            m = y_te == i
            ax[0].scatter(te_2d[m, 0], te_2d[m, 1], s=5, alpha=.55,
                          color=viz.PALETTE[i], label=cls)
        ax[0].set_title("PCA projection - TRUE classes")
        ax[1].scatter(te_2d[:, 0], te_2d[:, 1], s=5, alpha=.55,
                      c=cluster_labels, cmap="viridis")
        ax[1].set_title(f"PCA projection - K-Means clusters (k={best_k})")
        for a in ax:
            a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.legend(
                fontsize=7) if a is ax[0] else None
        viz.save_fig(fig, "unsup_pca_clusters.png")

        # ---------------- 3. DBSCAN on PCA-3D sample ---------------------
        pca3 = PCA(n_components=3, random_state=config.SEED).fit(X_tr)
        db_idx = np.random.RandomState(config.SEED).choice(
            len(X_tr), min(4000, len(X_tr)), replace=False)
        db = DBSCAN(eps=1.2, min_samples=10).fit(
            pca3.transform(X_tr.iloc[db_idx]))
        n_clusters_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        noise_ratio = float((db.labels_ == -1).mean())

        # ---------------- 3b. HIERARCHICAL (Agglomerative/Ward) ---------
        from scipy.cluster.hierarchy import dendrogram, linkage
        from sklearn.cluster import AgglomerativeClustering

        h_idx = np.random.RandomState(config.SEED).choice(
            len(X_tr), min(2500, len(X_tr)), replace=False)
        pca_h = PCA(n_components=8, random_state=config.SEED).fit(X_tr)
        H = pca_h.transform(X_tr.iloc[h_idx])
        Z = linkage(H, method="ward")
        fig, ax = plt.subplots(figsize=(10, 4.6))
        dendrogram(Z, truncate_mode="lastp", p=12, ax=ax, leaf_rotation=45)
        ax.set_title("Hierarchical clustering dendrogram (Ward linkage)\n"
                     "on an 8-D PCA subsample of 2,500 flows")
        ax.set_ylabel("merge distance (Ward)")
        viz.save_fig(fig, "unsup_dendrogram.png")

        agg = AgglomerativeClustering(
            n_clusters=best_k, linkage="ward").fit(H)
        h_ari = adjusted_rand_score(y_tr[h_idx], agg.labels_)
        h_nmi = normalized_mutual_info_score(y_tr[h_idx], agg.labels_)

        # ---------------- 4. ISOLATION FOREST (benign-only training) ----
        benign_mask = y_tr == list(classes).index("Benign")
        iso = IsolationForest(n_estimators=200, contamination=0.03,
                              random_state=config.SEED).fit(
            X_tr.iloc[benign_mask])
        scores = iso.decision_function(X_te)          # higher = more normal
        thr = np.quantile(iso.decision_function(
            X_tr.iloc[benign_mask]), 0.05)            # 5% benign FP budget
        anomaly = scores < thr

        # attack = positive class, benign = negative
        y_attack = (y_te != list(classes).index("Benign")).astype(int)
        y_anom = anomaly.astype(int)
        iso_metrics = {
            "precision": precision_score(y_attack, y_anom),
            "recall": recall_score(y_attack, y_anom),
            "f1": f1_score(y_attack, y_anom),
            "benign_false_alarm_rate": float(
                (y_anom[y_attack == 0] == 1).mean()),
        }
        # per-class detection rates
        det = {cls: float(y_anom[y_te == i].mean())
               for i, cls in enumerate(classes)}
        joblib.dump({"model": iso, "threshold": float(thr)},
                    os.path.join(config.MODEL_DIR, "unsup_isolation_forest.joblib"))

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        names, vals = zip(*det.items())
        ax.bar(names, vals, color=viz.PALETTE[:len(vals)])
        ax.bar_label(ax.containers[0], fmt="%.2f")
        ax.set_ylabel("fraction flagged anomalous")
        ax.set_title("IsolationForest detection rate per class\n"
                     "(trained on BENIGN traffic only)")
        ax.tick_params(axis="x", rotation=20)
        viz.save_fig(fig, "unsup_isolation_detection.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for lab, arr, c in [("Benign", scores[y_attack == 0], viz.PALETTE[0]),
                            ("Attacks", scores[y_attack == 1], viz.PALETTE[1])]:
            ax.hist(arr, bins=60, alpha=.65, label=lab, color=c)
        ax.axvline(thr, color="k", ls="--", label="anomaly threshold")
        ax.set_xlabel("IsolationForest decision score (low = anomalous)")
        ax.set_title("Normal-profile separation on the test set")
        ax.legend()
        viz.save_fig(fig, "unsup_isolation_hist.png")

        summary = {
            "kmeans": {"chosen_k": int(best_k), "ARI": ari, "NMI": nmi,
                       "purity": purity,
                       "silhouette": float(max(sils))},
            "dbscan": {"clusters_found": int(n_clusters_db),
                       "noise_ratio": noise_ratio,
                       "eps": 1.2, "min_samples": 10,
                       "note": "run on a 3-D PCA subsample (density "
                               "estimation in 12-D is expensive)"},
            "hierarchical": {
                "method": "Ward linkage, AgglomerativeClustering, 8-D PCA "
                          "subsample (2,500 flows)",
                "n_clusters": int(best_k),
                "ARI": float(h_ari), "NMI": float(h_nmi),
                "dendrogram": "results/figures/unsup_dendrogram.png",
            },
            "isolation_forest": {**iso_metrics,
                                 "threshold": float(thr),
                                 "training": "benign flows only (one-class)",
                                 "per_class_detection": det},
            "supervised_vs_unsupervised": {
                "supervised": "Learns a mapping features->label from "
                              "labelled examples; answers 'WHICH known "
                              "attack is this?'. Needs labels, very "
                              "accurate for seen classes.",
                "unsupervised": "Learns the structure of the data without "
                                "labels; answers 'does this look unusual / "
                                "which flows group together?'. Can flag "
                                "novel (zero-day) behaviour but cannot "
                                "name the attack family.",
                "complementarity": "SentinelAgent uses BOTH: the "
                                   "classifier names the threat, the "
                                   "anomaly detector gives a second, "
                                   "label-free opinion.",
            },
        }
        save_json(os.path.join(config.REPORT_DIR,
                               "unsupervised_results.json"), summary)
    return summary


def _cont_fig(cont, classes, k):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(cont.T, cmap="Purples")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticks(range(cont.shape[1]))
    ax.set_yticklabels([f"cluster {i}" for i in range(cont.shape[1])],
                       fontsize=8)
    ax.set_xlabel("true class"); ax.set_ylabel("K-Means cluster")
    ax.set_title(f"Cluster-to-class contingency (k={k})")
    ax.grid(False)
    for i in range(cont.shape[0]):
        for j in range(cont.shape[1]):
            ax.text(j, i, int(cont[i, j]), ha="center", va="center",
                    fontsize=7,
                    color="white" if cont[i, j] > cont.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return fig


if __name__ == "__main__":
    run()
