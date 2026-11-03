"""
Traffic Sign Recognition - Prediction & Inference Tool
======================================================
Predicts traffic sign class and confidence for single images or test sets.

Usage:
  python predict.py --image "Traffic data/TEST/0/000_0001.png"
  python predict.py --test-all
"""

import os
import sys
import json
import glob
import pickle
import argparse
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from train_classifier import FeatureExtractor


class TrafficSignPredictor:
    """Inference engine for trained Traffic Sign Recognition Model."""

    def __init__(
        self,
        model_path: str = "models/traffic_sign_classifier.pkl",
        metadata_path: str = "models/model_metadata.json",
        labels_csv: str = "Traffic data/labels.csv",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python train_classifier.py' first."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.label_map: Dict[int, str] = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                self.label_map = {int(k): v for k, v in meta.get("label_map", {}).items()}

        if not self.label_map and os.path.exists(labels_csv):
            df = pd.read_csv(labels_csv)
            self.label_map = dict(zip(df["ClassId"], df["Name"]))

        self.extractor = FeatureExtractor()

    def predict_image(self, img_bgr_or_rgb: np.ndarray, is_rgb: bool = True, top_k: int = 3) -> Dict[str, Any]:
        feat = self.extractor.extract(img_bgr_or_rgb, is_rgb=is_rgb)
        proba = self.model.predict_proba([feat])[0]
        classes = self.model.classes_

        top_indices = np.argsort(proba)[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            cid = int(classes[idx])
            prob_percent = round(float(proba[idx]) * 100.0, 2)
            cname = self.label_map.get(cid, f"Class {cid}")
            candidates.append({
                "class_id": cid,
                "class_name": cname,
                "confidence_percent": prob_percent
            })

        best = candidates[0]
        return {
            "predicted_class_id": best["class_id"],
            "predicted_class_name": best["class_name"],
            "confidence_percent": best["confidence_percent"],
            "top_candidates": candidates
        }

    def predict_path(self, image_path: str, top_k: int = 3) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not decode image at {image_path}")

        result = self.predict_image(img, is_rgb=False, top_k=top_k)
        result["image_path"] = image_path
        return result


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Inference & Testing")
    parser.add_argument("--image", "-i", type=str, help="Path to single traffic sign image to classify")
    parser.add_argument("--test-all", "-t", action="store_true", help="Evaluate model on all test images")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of top candidates to report")
    args = parser.parse_args()

    predictor = TrafficSignPredictor()

    if args.image:
        res = predictor.predict_path(args.image, top_k=args.top_k)
        print("\n" + "=" * 60)
        print("TRAFFIC SIGN RECOGNITION - PREDICTION RESULT")
        print("=" * 60)
        print(f"Image Path        : {res['image_path']}")
        print(f"Predicted Sign    : {res['predicted_class_name']} (Class ID: {res['predicted_class_id']})")
        print(f"Confidence        : {res['confidence_percent']}%")
        print("-" * 60)
        print("Top Candidates:")
        for idx, c in enumerate(res["top_candidates"], 1):
            bar = "#" * int(c["confidence_percent"] / 5)
            print(f"  #{idx}: [{bar:<20}] {c['confidence_percent']:5.1f}% - {c['class_name']} (ID: {c['class_id']})")
        print("=" * 60)

    elif args.test_all:
        test_images = glob.glob("Traffic data/TEST/*/*.png")
        print(f"\n[*] Evaluating on {len(test_images)} test images...")
        correct = 0
        for p in test_images:
            # Ground truth from parent directory name
            true_cid = int(os.path.basename(os.path.dirname(p)))
            res = predictor.predict_path(p, top_k=1)
            if res["predicted_class_id"] == true_cid:
                correct += 1

        acc = (correct / len(test_images)) * 100.0 if test_images else 0.0
        print(f"\n[+] Evaluated {len(test_images)} images.")
        print(f"[+] Total Correct: {correct} / {len(test_images)}")
        print(f"[+] Test Accuracy : {acc:.2f}%\n")
    else:
        # Default test on 3 sample images from test set
        test_imgs = glob.glob("Traffic data/TEST/*/*.png")[:3]
        if not test_imgs:
            test_imgs = glob.glob("Traffic data/DATA/*/*.png")[:3]
        print("\nRunning sample predictions...")
        for s in test_imgs:
            res = predictor.predict_path(s)
            print(f"[*] {s} -> {res['predicted_class_name']} ({res['confidence_percent']}%)")


if __name__ == "__main__":
    main()
