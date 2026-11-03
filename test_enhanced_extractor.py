import os
import cv2
import numpy as np
import pickle
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class EnhancedFeatureExtractor:
    def __init__(self, target_size=(32, 32)):
        self.target_size = target_size
        # 4x4 cells provide 4x finer spatial resolution for digit strokes
        self.hog = cv2.HOGDescriptor(
            _winSize=target_size,
            _blockSize=(8, 8),
            _blockStride=(4, 4),
            _cellSize=(4, 4),
            _nbins=9,
        )
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))

    def extract(self, img_rgb: np.ndarray) -> np.ndarray:
        resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        # 1. CLAHE enhanced grayscale for digit resolution
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gray_enhanced = self.clahe.apply(gray)
        hog_feat = self.hog.compute(gray_enhanced).flatten()

        # 2. Invariant Color histograms
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()

        # Normalize
        h_norm = h_hist / (np.sum(h_hist) + 1e-6)
        s_norm = s_hist / (np.sum(s_hist) + 1e-6)

        return np.concatenate([hog_feat, h_norm, s_norm]).astype(np.float32)

print("EnhancedFeatureExtractor ready!")

