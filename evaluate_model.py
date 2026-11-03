"""
Traffic Sign Recognition - Comprehensive Model Evaluation Tool
==============================================================
Computes:
  - Confusion Matrix (raw and normalized)
  - Precision, Recall, F1-Score (Macro and Weighted)
  - Top-1 & Top-3 Accuracies
  - Cohen's Kappa & Matthews Correlation Coefficient (MCC)
  - Confidence Calibration Analysis (Confidence on Correct vs Incorrect Predictions)
  - Per-Class Metrics across all 52 traffic sign classes

Outputs:
  - outputs/evaluation_report.json
  - outputs/per_class_metrics.csv
  - outputs/confusion_matrix_annotated.png
  - outputs/confidence_calibration.png
"""

import os
import glob
import json
import pickle
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
from train_classifier import FeatureExtractor


def run_full_evaluation(
    model_path: str = "models/traffic_sign_classifier.pkl",
    metadata_path: str = "models/model_metadata.json",
    test_dir: str = "Traffic data/TEST",
    labels_csv: str = "Traffic data/labels.csv",
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    print("=" * 75)
    print("TRAFFIC SIGN MODEL EVALUATION - FULL METRICS & CONFUSION MATRIX")
    print("=" * 75)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train_classifier.py first.")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    # Load label mapping
    labels_df = pd.read_csv(labels_csv)
    label_map = dict(zip(labels_df["ClassId"], labels_df["Name"]))

    extractor = FeatureExtractor()

    # Load test set
    test_images, y_true = [], []
    class_folders = sorted(
        [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))],
        key=lambda x: int(x) if x.isdigit() else x
    )

    print(f"[*] Loading test images from '{test_dir}'...")
    for cid_str in class_folders:
        cid = int(cid_str)
        folder = os.path.join(test_dir, cid_str)
        for p in glob.glob(os.path.join(folder, "*.png")):
            test_images.append(p)
            y_true.append(cid)

    X_test = []
    for p in test_images:
        X_test.append(extractor.extract_from_path(p))
    X_test = np.array(X_test, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    total_test = len(y_true)
    print(f"[+] Loaded {total_test} test images across {len(class_folders)} classes.")

    # 1. Predictions & Probabilities
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    classes = clf.classes_

    # 2. Global Accuracy Metrics
    top1_acc = accuracy_score(y_true, y_pred)
    top3_acc = top_k_accuracy_score(y_true, y_proba, k=3, labels=classes)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # 3. Precision, Recall, F1 Scores
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 4. Correlation & Agreement
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # 5. Confidence Analysis (Confidence Matrix / Calibration)
    confidences = np.max(y_proba, axis=1) * 100.0
    correct_mask = (y_true == y_pred)
    incorrect_mask = ~correct_mask

    mean_conf_all = float(np.mean(confidences))
    median_conf_all = float(np.median(confidences))
    mean_conf_correct = float(np.mean(confidences[correct_mask]))
    mean_conf_incorrect = float(np.mean(confidences[incorrect_mask])) if np.any(incorrect_mask) else 0.0

    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    # Print Global Results
    print("\n" + "-" * 75)
    print("GLOBAL PERFORMANCE METRICS")
    print("-" * 75)
    print(f"Total Test Samples          : {total_test}")
    print(f"Top-1 Accuracy              : {top1_acc * 100:.2f}% ({np.sum(correct_mask)} / {total_test} correct)")
    print(f"Top-3 Accuracy              : {top3_acc * 100:.2f}%")
    print(f"Balanced Accuracy           : {bal_acc * 100:.2f}%")
    print(f"Macro Precision             : {macro_p:.4f} ({macro_p * 100:.2f}%)")
    print(f"Macro Recall                : {macro_r:.4f} ({macro_r * 100:.2f}%)")
    print(f"Macro F1-Score              : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print(f"Weighted Precision          : {weighted_p:.4f} ({weighted_p * 100:.2f}%)")
    print(f"Weighted Recall             : {weighted_r:.4f} ({weighted_r * 100:.2f}%)")
    print(f"Weighted F1-Score           : {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")
    print(f"Cohen's Kappa (k)           : {kappa:.4f} (Near-perfect reliability > 0.80)")
    print(f"Matthews Corr. Coeff. (MCC) : {mcc:.4f}")
    print("-" * 75)
    print("CONFIDENCE CALIBRATION METRICS")
    print("-" * 75)
    print(f"Mean Confidence (Overall)   : {mean_conf_all:.2f}% (Median: {median_conf_all:.2f}%)")
    print(f"Mean Confidence (Correct)   : {mean_conf_correct:.2f}%")
    print(f"Mean Confidence (Incorrect) : {mean_conf_incorrect:.2f}%")
    print(f"Confidence Separation Ratio : {mean_conf_correct / (mean_conf_incorrect + 1e-6):.2f}x higher for correct predictions")

    # 7. Per-Class Metrics DataFrame
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    per_class_rows = []
    for idx, cid in enumerate(classes):
        mask_c = (y_true == cid)
        conf_c = float(np.mean(confidences[mask_c])) if np.any(mask_c) else 0.0
        acc_c = float(np.mean(correct_mask[mask_c])) * 100.0 if np.any(mask_c) else 0.0
        per_class_rows.append({
            "ClassId": int(cid),
            "ClassName": label_map.get(cid, f"Class {cid}"),
            "Support": int(per_sup[idx]),
            "Accuracy_Percent": round(acc_c, 2),
            "Precision": round(float(per_p[idx]), 4),
            "Recall": round(float(per_r[idx]), 4),
            "F1_Score": round(float(per_f1[idx]), 4),
            "Mean_Confidence": round(conf_c, 2),
        })

    df_per_class = pd.DataFrame(per_class_rows)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "per_class_metrics.csv")
    df_per_class.to_csv(csv_path, index=False)
    print(f"\n[+] Saved detailed per-class CSV to: {csv_path}")

    # Preview sample per-class table
    print("\n" + "-" * 75)
    print("SAMPLE PER-CLASS METRICS TABLE (First 15 Classes):")
    print("-" * 75)
    print(f"{'ID':>3} | {'Class Name':<28} | {'Supp':>4} | {'Acc %':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'Conf':>6}")
    print("-" * 75)
    for _, row in df_per_class.head(15).iterrows():
        print(f"{row['ClassId']:3d} | {row['ClassName'][:28]:<28} | {row['Support']:4d} | {row['Accuracy_Percent']:6.1f} | {row['Precision']:6.2f} | {row['Recall']:6.2f} | {row['F1_Score']:6.2f} | {row['Mean_Confidence']:5.1f}%")

    # 8. Export Plots
    # Plot 1: Confusion Matrix Heatmap
    plt.figure(figsize=(14, 12), facecolor="#0F172A")
    ax = plt.subplot(111)
    ax.set_facecolor("#1E293B")
    cax = ax.matshow(cm, cmap="Blues", interpolation="nearest")
    cbar = plt.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94A3B8")

    plt.title(
        f"Traffic Sign Recognition - Confusion Matrix ({total_test} Test Samples, Top-1 Acc: {top1_acc*100:.2f}%)",
        color="#F8FAFC",
        fontsize=13,
        pad=16,
        fontweight="bold"
    )
    plt.xlabel("Predicted Class ID", color="#94A3B8", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Class ID", color="#94A3B8", fontsize=11, labelpad=10)
    plt.tick_params(colors="#94A3B8", labelsize=8)

    cm_path = os.path.join(output_dir, "confusion_matrix_annotated.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[+] Saved confusion matrix heatmap to: {cm_path}")

    # Plot 2: Confidence Calibration Distribution
    plt.figure(figsize=(10, 4.5), facecolor="#0F172A")
    ax = plt.subplot(111)
    ax.set_facecolor("#1E293B")

    plt.hist(
        confidences[correct_mask],
        bins=20,
        range=(0, 100),
        alpha=0.75,
        color="#10B981",
        label=f"Correct Predictions (Mean: {mean_conf_correct:.1f}%)"
    )
    if np.any(incorrect_mask):
        plt.hist(
            confidences[incorrect_mask],
            bins=20,
            range=(0, 100),
            alpha=0.75,
            color="#F87171",
            label=f"Incorrect Predictions (Mean: {mean_conf_incorrect:.1f}%)"
        )

    plt.title("Model Confidence Calibration (Correct vs Incorrect Classifications)", color="#F8FAFC", fontsize=12, fontweight="bold")
    plt.xlabel("Prediction Confidence (%)", color="#94A3B8", fontsize=10)
    plt.ylabel("Frequency (Sample Count)", color="#94A3B8", fontsize=10)
    plt.tick_params(colors="#94A3B8")
    plt.grid(True, alpha=0.25, color="#334155")
    plt.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    calib_path = os.path.join(output_dir, "confidence_calibration.png")
    plt.savefig(calib_path, dpi=200, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[+] Saved confidence calibration plot to: {calib_path}")

    # 9. Save JSON Summary
    full_report = {
        "dataset_summary": {
            "total_test_samples": total_test,
            "total_classes": len(classes),
            "total_correct": int(np.sum(correct_mask)),
            "total_incorrect": int(np.sum(incorrect_mask)),
        },
        "performance_metrics": {
            "top1_accuracy_percent": round(top1_acc * 100.0, 2),
            "top3_accuracy_percent": round(top3_acc * 100.0, 2),
            "balanced_accuracy_percent": round(bal_acc * 100.0, 2),
            "macro_precision": round(macro_p, 4),
            "macro_recall": round(macro_r, 4),
            "macro_f1_score": round(macro_f1, 4),
            "weighted_precision": round(weighted_p, 4),
            "weighted_recall": round(weighted_r, 4),
            "weighted_f1_score": round(weighted_f1, 4),
            "cohen_kappa": round(kappa, 4),
            "matthews_corrcoef": round(mcc, 4),
        },
        "confidence_calibration": {
            "mean_confidence_overall": round(mean_conf_all, 2),
            "median_confidence_overall": round(median_conf_all, 2),
            "mean_confidence_correct": round(mean_conf_correct, 2),
            "mean_confidence_incorrect": round(mean_conf_incorrect, 2),
            "confidence_separation_ratio": round(mean_conf_correct / (mean_conf_incorrect + 1e-6), 2),
        },
        "artifacts": {
            "per_class_csv": csv_path,
            "confusion_matrix_png": cm_path,
            "confidence_calibration_png": calib_path,
        }
    }

    json_report_path = os.path.join(output_dir, "full_evaluation_report.json")
    with open(json_report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"[+] Saved full evaluation JSON report to: {json_report_path}")

    print("=" * 75)
    return full_report


if __name__ == "__main__":
    run_full_evaluation()
