"""
SentinelAI - Stage 9: Computer Vision - Malware Image Classification
====================================================================
Malware-visualization approach (Nataraj et al., 2011): each malware
binary is rendered as a grayscale "byte-plot" image; a CNN then learns
the visual signature of each malware family.

Image preprocessing (documented as required):
    * images loaded as 48x48 grayscale PNGs
    * pixel values normalized to [0,1] (helps gradient descent)
    * stratified 80/20 train/test split by family
    * labels one-hot encoded for the softmax output layer

CNN architecture (justified by IMAGE data - convolution learns local
visual patterns, pooling adds small translation robustness):
    Conv2D(32,3x3,ReLU) -> MaxPool(2) ->
    Conv2D(64,3x3,ReLU) -> MaxPool(2) -> Flatten ->
    Dense(128,ReLU) -> Dropout(0.35) -> Dense(4,Softmax)
"""

import os

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

from src import config
from src import visualization as viz
from src.config import StageTimer, save_json

tf.random.set_seed(config.SEED)
KERAS_PATH = os.path.join(config.MODEL_DIR, "cv_malware_cnn.keras")
IMG = config.MALWARE_IMG_SIZE


def load_images():
    """Load every PNG byte-plot + its family label.

    Real Malimg images arrive at their native size (e.g. 256x264 or
    768x683), so each is converted to grayscale and RESIZED to 48x48 to
    match the CNN input - the standard fixed-size normalization of the
    byte-plot approach (Nataraj et al. 2011)."""
    X, y, paths = [], [], []
    for idx, fam in enumerate(config.MALWARE_FAMILIES):
        fam_dir = os.path.join(config.MALWARE_IMG_DIR, fam)
        for fname in sorted(os.listdir(fam_dir)):
            p = os.path.join(fam_dir, fname)
            im = Image.open(p).convert("L").resize((IMG, IMG))
            X.append(np.asarray(im, dtype=np.float32) / 255.0)  # -> [0,1]
            y.append(idx)
            paths.append(p)
    return np.stack(X)[..., None], np.array(y), paths


def build_cnn(n_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG, IMG, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="SentinelAI_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def run() -> dict:
    with StageTimer(9, "Computer Vision - malware family CNN"):
        config.ensure_dirs()
        X, y, paths = load_images()
        families = config.MALWARE_FAMILIES

        # ---- EDA: sample images per family -----------------------------
        fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
        for col, fam_idx in enumerate(np.random.RandomState(config.SEED)
                                      .choice(len(families), 4, False)):
            for row in range(2):
                sample = X[y == fam_idx][row * 37 + col]
                axes[row, col].imshow(sample[:, :, 0], cmap="gray")
                axes[row, col].set_title(
                    families[fam_idx] if row == 0 else "", fontsize=9)
                axes[row, col].axis("off")
        fig.suptitle("Byte-plot samples per malware family")
        viz.save_fig(fig, "cv_sample_images.png")

        # ---- split ------------------------------------------------------
        X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
            X, y, paths, test_size=config.TEST_SIZE, stratify=y,
            random_state=config.SEED)

        model = build_cnn(len(families))
        hist = model.fit(
            X_tr, y_tr, validation_split=0.15, epochs=14, batch_size=32,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True,
                monitor="val_loss")],
        )
        model.save(KERAS_PATH)

        fig = viz.line_chart(
            range(1, len(hist.history["loss"]) + 1),
            [hist.history["loss"], hist.history["val_loss"]],
            ["train loss", "validation loss"],
            "CNN training - cross-entropy loss", "epoch", "loss")
        viz.save_fig(fig, "cv_training_loss.png")

        fig = viz.line_chart(
            range(1, len(hist.history["accuracy"]) + 1),
            [hist.history["accuracy"], hist.history["val_accuracy"]],
            ["train accuracy", "validation accuracy"],
            "CNN training - accuracy", "epoch", "accuracy")
        viz.save_fig(fig, "cv_training_accuracy.png")

        # ---- evaluation --------------------------------------------------
        proba = model.predict(X_te, verbose=0, batch_size=64)
        y_pred = proba.argmax(1)
        metrics = {
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        }

        # ---- traditional-ML baseline: HOG features + SVM -----------------
        # (comparison required: deep learning vs classic approach)
        from skimage.feature import hog
        from sklearn.svm import SVC

        def hog_feats(arr):
            return np.array([
                hog(im, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm="L2-Hys",
                    feature_vector=True) for im in arr[..., 0]])

        hog_tr, hog_te = hog_feats(X_tr), hog_feats(X_te)
        svm_base = SVC(kernel="rbf", C=10, random_state=config.SEED)
        svm_base.fit(hog_tr, y_tr)
        y_hog = svm_base.predict(hog_te)
        hog_metrics = {
            "accuracy": float(accuracy_score(y_te, y_hog)),
            "f1_macro": float(f1_score(y_te, y_hog, average="macro")),
        }
        joblib.dump(svm_base, os.path.join(config.MODEL_DIR,
                                           "cv_hog_svm.joblib"))
        print(f"    HOG+SVM baseline: acc={hog_metrics['accuracy']:.4f} "
              f"f1={hog_metrics['f1_macro']:.4f} "
              f"(CNN: acc={metrics['accuracy']:.4f})")

        fig = viz.grouped_bar(
            {"HOG + SVM (classic)": hog_metrics, "CNN (deep)": metrics},
            ["accuracy", "f1_macro"],
            "Computer vision: CNN vs traditional ML (HOG+SVM)", "score",
            figsize=(7.0, 4.4))
        viz.save_fig(fig, "cv_cnn_vs_hog.png")

        cm = confusion_matrix(y_te, y_pred)
        viz.save_fig(viz.confusion_matrix_heatmap(
            cm, families,
            f"CNN confusion matrix - malware families "
            f"(acc={metrics['accuracy']:.3f})"), "cv_cm.png")
        rep = classification_report(y_te, y_pred, target_names=families,
                                    output_dict=True)

        # per-family accuracy bar chart
        per_fam = {f: rep[f]["recall"] for f in families}
        fig, ax = plt.subplots(figsize=(6.6, 4))
        ax.bar(per_fam.keys(), per_fam.values(),
               color=viz.PALETTE[:len(per_fam)])
        ax.bar_label(ax.containers[0], fmt="%.3f")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("recall (per-family accuracy)")
        ax.set_title("CNN recall per malware family")
        ax.tick_params(axis="x", rotation=15)
        viz.save_fig(fig, "cv_per_family_accuracy.png")

        # persist artifacts for the agent ---------------------------------
        joblib.dump({"families": families},
                    os.path.join(config.MODEL_DIR, "cv_label_map.joblib"))
        test_mal_dir = os.path.join(config.TEST_DIR, "malware")
        os.makedirs(test_mal_dir, exist_ok=True)
        for old in os.listdir(test_mal_dir):         # clear stale held-outs
            os.remove(os.path.join(test_mal_dir, old))
        chosen = np.random.RandomState(config.SEED).choice(
            len(p_te), 8, replace=False)
        for i in chosen:
            os.replace(p_te[i], os.path.join(
                test_mal_dir, os.path.basename(p_te[i])))

        summary = {
            "approach": "byte-plot malware visualization + CNN "
                        "(Nataraj et al. 2011)",
            "image_preprocessing": ["grayscale conversion",
                                    "resize to 48x48 (native Malimg sizes "
                                    "vary, e.g. 256x264 - 768x683)",
                                    "normalize pixels to [0,1]",
                                    "stratified 80/20 split",
                                    "sparse integer labels"],
            "architecture": "Conv2D(32,3x3)+MaxPool -> Conv2D(64,3x3)+"
                            "MaxPool -> Dense128 + Dropout0.35 -> Softmax4",
            "params": int(model.count_params()),
            "epochs_trained": len(hist.history["loss"]),
            "train_images": int(len(y_tr)), "test_images": int(len(y_te)),
            "metrics": metrics,
            "hog_svm_baseline": {
                **hog_metrics,
                "features": "HOG (9 orientations, 8x8 cells, 2x2 blocks)",
                "note": "Classic pipeline: hand-crafted texture features + "
                        "RBF SVM, compared against the CNN on the same "
                        "split.",
            },
            "classification_report": rep,
        }
        save_json(os.path.join(config.REPORT_DIR, "cv_results.json"),
                  summary)
    return summary


if __name__ == "__main__":
    run()
