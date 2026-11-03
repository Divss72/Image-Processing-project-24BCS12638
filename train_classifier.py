"""
Traffic Sign Recognition - Model Training & Evaluation Module
==============================================================
Trains a machine learning classifier on the preprocessed traffic sign dataset
(5,678 training images across 52 classes) and evaluates it on unseen test images
(432 test images across 52 classes).

Outputs:
  - models/traffic_sign_classifier.pkl : Trained model artifact
  - models/model_metadata.json         : Class names, accuracy, and hyperparameters
  - outputs/confusion_matrix.png       : Test evaluation confusion matrix plot
"""

import os
import glob
import time
import json
import pickle
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    classification_report,
    confusion_matrix,
)


class FeatureExtractor:
    """Extracts invariant shape and colour features with fine-grained digit stroke resolution."""

    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size
        # 4x4 cells provide 4x finer spatial resolution for digit strokes (e.g. 30 vs 70/80)
        self.hog = cv2.HOGDescriptor(
            _winSize=target_size,
            _blockSize=(8, 8),
            _blockStride=(4, 4),
            _cellSize=(4, 4),
            _nbins=9,
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

    def extract(self, img_bgr_or_rgb: np.ndarray, is_rgb: bool = True) -> np.ndarray:
        if is_rgb:
            img_rgb = img_bgr_or_rgb
        else:
            img_rgb = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2RGB)

        # Preprocessing: Gaussian smoothing + boundary sharpening
        smooth = cv2.GaussianBlur(img_rgb, (3, 3), 0.8)
        resized = cv2.resize(smooth, self.target_size, interpolation=cv2.INTER_AREA)

        # 1. Structural features via fine-grained HOG on CLAHE enhanced Grayscale plane
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gray_enhanced = self.clahe.apply(gray)
        hog_feat = self.hog.compute(gray_enhanced).flatten()

        # 2. Chromatic features via HSV Hue & Saturation histograms
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()

        h_norm = h_hist / (np.sum(h_hist) + 1e-6)
        s_norm = s_hist / (np.sum(s_hist) + 1e-6)

        return np.concatenate([hog_feat, h_norm, s_norm]).astype(np.float32)

    def extract_from_path(self, path: str) -> Optional[np.ndarray]:
        img = cv2.imread(path)
        if img is None:
            return None
        return self.extract(img, is_rgb=False)


def load_dataset(data_dir: str, extractor: FeatureExtractor) -> Tuple[np.ndarray, np.ndarray, list]:
    X, y, paths = [], [], []
    class_folders = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    for cid_str in class_folders:
        cid = int(cid_str)
        folder = os.path.join(data_dir, cid_str)
        img_paths = glob.glob(os.path.join(folder, "*.png"))
        for p in img_paths:
            feat = extractor.extract_from_path(p)
            if feat is not None:
                X.append(feat)
                y.append(cid)
                paths.append(p)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), paths


def train_and_evaluate(
    data_dir: str = "Traffic data/DATA",
    test_dir: str = "Traffic data/TEST",
    labels_csv: str = "Traffic data/labels.csv",
    model_output_dir: str = "models",
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    print("=" * 70)
    print("TRAFFIC SIGN CLASSIFIER - TRAINING & TEST BENCHMARK")
    print("=" * 70)

    # 1. Load Labels
    labels_df = pd.read_csv(labels_csv)
    label_map = dict(zip(labels_df["ClassId"], labels_df["Name"]))
    num_classes = len(label_map)
    print(f"[+] Loaded {num_classes} classes from {labels_csv}")

    extractor = FeatureExtractor()

    # 2. Load Train Dataset
    print(f"\n[*] Extracting preprocessed features from Training Set ({data_dir})...")
    t0 = time.time()
    X_train, y_train, train_paths = load_dataset(data_dir, extractor)
    t_train_load = time.time() - t0
    print(f"[+] Extracted {len(X_train)} training feature vectors in {t_train_load:.1f}s (Shape: {X_train.shape})")

    # 3. Load Test Dataset
    print(f"\n[*] Extracting preprocessed features from Testing Set ({test_dir})...")
    t1 = time.time()
    X_test, y_test, test_paths = load_dataset(test_dir, extractor)
    t_test_load = time.time() - t1
    print(f"[+] Extracted {len(X_test)} test feature vectors in {t_test_load:.1f}s (Shape: {X_test.shape})")

    # 4. Train Model
    print("\n[*] Training Random Forest Classifier on preprocessed features...")
    t2 = time.time()
    clf = RandomForestClassifier(
        n_estimators=160,
        max_depth=26,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    t_train_fit = time.time() - t2
    print(f"[+] Model trained in {t_train_fit:.1f}s!")

    # 5. Evaluate on Test Set
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS ON UNSEEN TEST SET")
    print("=" * 70)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # All classes present in training vs testing
    classes_trained = clf.classes_
    top1_acc = accuracy_score(y_test, y_pred)
    top3_acc = top_k_accuracy_score(y_test, y_proba, k=3, labels=classes_trained)

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    print(f"Top-1 Test Accuracy : {top1_acc * 100:.2f}%")
    print(f"Top-3 Test Accuracy : {top3_acc * 100:.2f}%")
    print(f"Macro F1-Score      : {macro_f1:.4f}")
    print(f"Weighted F1-Score   : {weighted_f1:.4f}")

    # 6. Save Model Artifacts
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(model_output_dir, "traffic_sign_classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n[+] Saved trained model artifact to: {model_path}")

    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 160,
        "max_depth": 26,
        "feature_dim": int(X_train.shape[1]),
        "num_classes": num_classes,
        "classes": [int(c) for c in classes_trained],
        "label_map": {str(k): v for k, v in label_map.items()},
        "metrics": {
            "top1_test_accuracy": round(float(top1_acc) * 100.0, 2),
            "top3_test_accuracy": round(float(top3_acc) * 100.0, 2),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "total_train_samples": int(len(X_train)),
            "total_test_samples": int(len(X_test)),
        },
        "training_time_sec": round(float(t_train_fit), 2),
    }

    meta_path = os.path.join(model_output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved model metadata to: {meta_path}")

    # 7. Generate & Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=classes_trained)
    plt.figure(figsize=(14, 12), facecolor="#0F172A")
    ax = plt.subplot(111)
    ax.set_facecolor("#1E293B")
    cax = ax.matshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    plt.title(
        f"Traffic Sign Recognition - Confusion Matrix on Unseen Test Set (Accuracy: {top1_acc*100:.2f}%)",
        color="#F8FAFC",
        fontsize=13,
        pad=16,
        fontweight="bold"
    )
    plt.xlabel("Predicted Class ID", color="#94A3B8", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Class ID", color="#94A3B8", fontsize=11, labelpad=10)
    plt.tick_params(colors="#94A3B8", labelsize=8)

    cm_path = os.path.join(output_dir, "evaluation_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[+] Saved confusion matrix visualization to: {cm_path}")

    print("\n" + "=" * 70)
    print("[SUCCESS] Model training and evaluation finished successfully!")
    print("=" * 70)
    return metadata


if __name__ == "__main__":
    train_and_evaluate()

