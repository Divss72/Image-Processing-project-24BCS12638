import os
import cv2
import glob
import time
import pickle
import numpy as np
from test_enhanced_extractor import EnhancedFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_eval():
    ext = EnhancedFeatureExtractor()
    train_dir = "Traffic data/DATA"
    test_dir = "Traffic data/TEST"

    print("[*] Extracting features from train set...")
    t0 = time.time()
    X_train, y_train = [], []
    for class_folder in sorted(os.listdir(train_dir), key=lambda x: int(x) if x.isdigit() else 999):
        c_path = os.path.join(train_dir, class_folder)
        if not os.path.isdir(c_path):
            continue
        c_id = int(class_folder)
        for img_p in glob.glob(os.path.join(c_path, "*.png")):
            im = cv2.imread(img_p)
            if im is not None:
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                X_train.append(ext.extract(im_rgb))
                y_train.append(c_id)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"[+] Extracted {len(X_train)} train samples in {time.time()-t0:.1f}s.")

    print("[*] Extracting features from test set...")
    X_test, y_test = [], []
    for class_folder in sorted(os.listdir(test_dir), key=lambda x: int(x) if x.isdigit() else 999):
        c_path = os.path.join(test_dir, class_folder)
        if not os.path.isdir(c_path):
            continue
        c_id = int(class_folder)
        for img_p in glob.glob(os.path.join(c_path, "*.png")):
            im = cv2.imread(img_p)
            if im is not None:
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                X_test.append(ext.extract(im_rgb))
                y_test.append(c_id)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    print(f"[+] Extracted {len(X_test)} test samples.")

    print("[*] Training enhanced Random Forest...")
    clf = RandomForestClassifier(n_estimators=180, max_depth=28, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[+] Test Accuracy with Enhanced Extractor: {acc*100:.2f}% (was 91.20%)")

    # Evaluate on speed limit 30 crops
    for fname in ["outputs/crop_tight_30.png", "outputs/crop_restored_30.png", "outputs/crop_copy_30.png"]:
        if os.path.exists(fname):
            im = cv2.imread(fname)
            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            feat = ext.extract(im_rgb)
            probs = clf.predict_proba([feat])[0]
            top_3 = np.argsort(probs)[::-1][:3]
            print(f"\n{fname}:")
            for idx in top_3:
                cid = clf.classes_[idx]
                print(f"  Class {cid}: {probs[idx]*100:.1f}%")

    # Save model if better
    if acc >= 0.91:
        with open("models/traffic_sign_classifier.pkl", "wb") as f:
            pickle.dump(clf, f)
        print("[+] Saved improved model to models/traffic_sign_classifier.pkl")

if __name__ == "__main__":
    train_and_eval()

