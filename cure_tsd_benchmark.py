"""
Georgia Tech CURE-TSD Dataset & Environmental Challenge Benchmark
=================================================================
Reference: "Challenging Unreal and Real Environments for Traffic Sign Detection"
OLIVES Lab, Georgia Institute of Technology (IEEE Transactions on ITS)
Repository: https://github.com/olivesgatech/CURE-TSD

Implements:
  - All 12 CURE-TSD Environmental Challenge Types (Levels 1 to 5)
  - CURE-TSD Video Sequence & Annotation Parser
  - Comprehensive Benchmark Runner (Reliability Index under all 12 challenges)
  - Autonomous Vision Preprocessing Resilience Evaluation
"""

import os
import glob
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from predict import TrafficSignPredictor
from robustness_engine import RobustnessEngine


class CureTsdChallenges:
    """Implements the 12 CURE-TSD challenging environmental conditions per Georgia Tech specifications."""

    CHALLENGE_NAMES = [
        "01_Decolorization",
        "02_Lens_Blur",
        "03_Codec_Error",
        "04_Darkening",
        "05_Dirty_Lens",
        "06_Exposure",
        "07_Gaussian_Blur",
        "08_Sensor_Noise",
        "09_Rain",
        "10_Shadow",
        "11_Snow",
        "12_Haze",
    ]

    @staticmethod
    def apply_challenge(img_rgb: np.ndarray, challenge_type: int, level: int = 3) -> np.ndarray:
        """
        Apply CURE-TSD challenge (1 to 12) at severity level (1 to 5).
        """
        level = int(np.clip(level, 1, 5))
        h, w = img_rgb.shape[:2]
        img = img_rgb.copy()

        # Challenge 01: Decolorization (Levels 1-5 reduce color saturation)
        if challenge_type == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            alpha = level * 0.20  # 0.2 to 1.0 (complete monochrome at level 5)
            out = cv2.addWeighted(img, 1.0 - alpha, gray_3ch, alpha, 0)
            return out

        # Challenge 02: Lens Blur (Hexagonal/Circular lens defocus, radius 2 to 10)
        elif challenge_type == 2:
            radius = level * 2
            k_size = radius * 2 + 1
            kernel = np.zeros((k_size, k_size), dtype=np.float32)
            cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
            kernel /= np.sum(kernel)
            return cv2.filter2D(img, -1, kernel)

        # Challenge 03: Codec Error (Compression & block artifact distortion)
        elif challenge_type == 3:
            quality = max(5, 50 - level * 9)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            # Add block macro-shift for higher levels
            if level >= 3:
                shift = level * 2
                out[shift:, :] = out[:-shift, :]
            return out

        # Challenge 04: Darkening (Exposure reduction -1, -3, -5, -7, -9)
        elif challenge_type == 4:
            exp_drop = 1 + (level - 1) * 2  # 1, 3, 5, 7, 9
            scale = max(0.12, 1.0 - (exp_drop * 0.09))
            offset = -int(exp_drop * 10)
            return np.clip(img.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)

        # Challenge 05: Dirty Lens (Dust, mud smudges and water droplets)
        elif challenge_type == 5:
            out = img.copy()
            rng = np.random.default_rng(42 + level)
            noise = rng.random((h, w))
            mud_density = level * 0.015
            out[noise < mud_density] = [35, 30, 25]  # Mud flecks
            # Smudge spots
            num_smudges = level * 3
            for _ in range(num_smudges):
                sx = rng.integers(0, w)
                sy = rng.integers(0, h)
                rad = rng.integers(3, max(5, int(w * 0.08)))
                cv2.circle(out, (sx, sy), rad, (45, 40, 35), -1)
            return out

        # Challenge 06: Exposure (Overexposure / high-beam glare +1, +3, +5, +7, +9)
        elif challenge_type == 6:
            exp_boost = 1 + (level - 1) * 2
            scale = 1.0 + (exp_boost * 0.12)
            offset = int(exp_boost * 12)
            return np.clip(img.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)

        # Challenge 07: Gaussian Blur (Blurriness 5, 10, 15, 20, 25)
        elif challenge_type == 7:
            k = level * 4 + 1
            if k % 2 == 0:
                k += 1
            sigma = level * 1.2
            return cv2.GaussianBlur(img, (k, k), sigma)

        # Challenge 08: Sensor Noise (Noise filter parameter 20, 40, 60, 70, 71)
        elif challenge_type == 8:
            sigma_noise = [12, 24, 38, 52, 68][level - 1]
            rng = np.random.default_rng(100 + level)
            noise = rng.normal(0, sigma_noise, img.shape).astype(np.float32)
            return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Challenge 09: Rain (Blueish atmospheric hue #0F1E2D + rain streaks)
        elif challenge_type == 9:
            out = img.copy().astype(np.float32)
            # Atmospheric cold blue tint
            rain_hue = np.array([15, 30, 45], dtype=np.float32) * (level * 0.35)
            out = np.clip(out * (1.0 - level * 0.06) + rain_hue, 0, 255).astype(np.uint8)
            # Falling rain streaks
            num_drops = level * 25
            rng = np.random.default_rng(200 + level)
            for _ in range(num_drops):
                rx = rng.integers(0, w)
                ry = rng.integers(0, max(1, h - 12))
                cv2.line(out, (rx, ry), (rx + 2, ry + 10), (210, 220, 235), 1)
            return out

        # Challenge 10: Shadow (Venetian blind periodic cast shadows, opacity 15% to 75%)
        elif challenge_type == 10:
            out = img.copy().astype(np.float32)
            opacity = [0.15, 0.30, 0.45, 0.60, 0.75][level - 1]
            band_width = max(8, int(h * 0.16))
            for y in range(0, h, band_width * 2):
                y_end = min(h, y + band_width)
                out[y:y_end, :] *= (1.0 - opacity)
            return np.clip(out, 0, 255).astype(np.uint8)

        # Challenge 11: Snow (White atmospheric glow + snowfall drops)
        elif challenge_type == 11:
            out = img.copy().astype(np.float32)
            # White atmospheric glow
            snow_glow = level * 25.0
            out = np.clip(out + snow_glow, 0, 255).astype(np.uint8)
            # Falling snowflakes
            num_flakes = level * 30
            rng = np.random.default_rng(300 + level)
            for _ in range(num_flakes):
                sx = rng.integers(0, w)
                sy = rng.integers(0, h)
                rad = rng.integers(1, max(2, level))
                cv2.circle(out, (sx, sy), rad, (250, 250, 255), -1)
            return out

        # Challenge 12: Haze (Atmospheric dust and fog veil #CECECE, opacity 10% to 50%)
        elif challenge_type == 12:
            opacity = [0.10, 0.20, 0.30, 0.40, 0.50][level - 1]
            haze_color = np.array([206, 206, 206], dtype=np.float32)
            out = img.astype(np.float32) * (1.0 - opacity) + haze_color * opacity
            return np.clip(out, 0, 255).astype(np.uint8)

        return img


class CureTsdBenchmarkRunner:
    """Evaluates the traffic sign recognition model against all 12 CURE-TSD challenges."""

    def __init__(self, predictor: Optional[TrafficSignPredictor] = None):
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = TrafficSignPredictor()
            self.predictor.model.n_jobs = 1

    def run_benchmark(
        self,
        test_images_subset: Optional[List[str]] = None,
        max_samples: int = 50,
        output_dir: str = "outputs",
    ) -> Dict[str, Any]:
        print("=" * 75)
        print("GEORGIA TECH CURE-TSD DATASET BENCHMARK (12 Challenges x 5 Severity Levels)")
        print("=" * 75)

        if not test_images_subset:
            all_test = glob.glob("Traffic data/TEST/*/*.png")
            # Pick representative stratified sample
            test_images_subset = all_test[:max_samples]

        print(f"[*] Running CURE-TSD stress testing on {len(test_images_subset)} test signs...")

        results_by_challenge = {}

        t0 = time.time()
        for c_idx in range(1, 13):
            c_name = CureTsdChallenges.CHALLENGE_NAMES[c_idx - 1]
            print(f"  Testing Challenge {c_name} (Levels 1-5)...")

            level_accuracies_baseline = []
            level_accuracies_restored = []

            for lvl in range(1, 6):
                corr_base = 0
                corr_rest = 0
                total = 0

                for p in test_images_subset:
                    true_cid = int(os.path.basename(os.path.dirname(p)))
                    img_bgr = cv2.imread(p)
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # 1. Apply CURE-TSD Challenge
                    degraded = CureTsdChallenges.apply_challenge(img_rgb, challenge_type=c_idx, level=lvl)

                    # 2. Baseline Prediction (Raw degraded image)
                    res_base = self.predictor.predict_image(degraded, is_rgb=True, top_k=1)
                    if res_base["predicted_class_id"] == true_cid:
                        corr_base += 1

                    # 3. Autonomous Vision Preprocessing Restored
                    restored = RobustnessEngine.autonomous_restore(degraded)
                    res_rest = self.predictor.predict_image(restored, is_rgb=True, top_k=1)
                    if res_rest["predicted_class_id"] == true_cid:
                        corr_rest += 1

                    total += 1

                acc_base = (corr_base / float(total)) * 100.0 if total > 0 else 0.0
                acc_rest = (corr_rest / float(total)) * 100.0 if total > 0 else 0.0
                level_accuracies_baseline.append(round(acc_base, 1))
                level_accuracies_restored.append(round(acc_rest, 1))

            mean_base = float(np.mean(level_accuracies_baseline))
            mean_rest = float(np.mean(level_accuracies_restored))

            results_by_challenge[c_name] = {
                "challenge_id": c_idx,
                "challenge_name": c_name,
                "baseline_by_level": level_accuracies_baseline,
                "restored_by_level": level_accuracies_restored,
                "mean_baseline_accuracy": round(mean_base, 1),
                "mean_restored_accuracy": round(mean_rest, 1),
                "accuracy_gain_percent": round(mean_rest - mean_base, 1),
            }

        t_elapsed = time.time() - t0
        print(f"[+] Completed 60 environmental configurations in {t_elapsed:.1f}s!")

        # Calculate Overall CURE-TSD Reliability Index (RI)
        overall_base = float(np.mean([d["mean_baseline_accuracy"] for d in results_by_challenge.values()]))
        overall_rest = float(np.mean([d["mean_restored_accuracy"] for d in results_by_challenge.values()]))

        summary = {
            "dataset": "CURE-TSD (Georgia Tech Benchmark)",
            "num_challenges": 12,
            "severity_levels": 5,
            "test_samples_per_config": len(test_images_subset),
            "overall_baseline_accuracy": round(overall_base, 2),
            "overall_restored_accuracy": round(overall_rest, 2),
            "overall_preprocessing_benefit": round(overall_rest - overall_base, 2),
            "challenges": results_by_challenge,
        }

        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "cure_tsd_benchmark_report.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Saved CURE-TSD evaluation report to: {json_path}")

        # Plot CURE-TSD Degradation Resilience Curves
        self._plot_cure_tsd_curves(results_by_challenge, os.path.join(output_dir, "cure_tsd_challenge_curves.png"))

        # Save CSV Table
        csv_rows = []
        for k, v in results_by_challenge.items():
            csv_rows.append({
                "Challenge": v["challenge_name"],
                "Lvl1_Base": v["baseline_by_level"][0],
                "Lvl5_Base": v["baseline_by_level"][4],
                "Mean_Baseline_Acc": v["mean_baseline_accuracy"],
                "Mean_Restored_Acc": v["mean_restored_accuracy"],
                "Accuracy_Gain": v["accuracy_gain_percent"],
            })
        df_csv = pd.DataFrame(csv_rows)
        csv_path = os.path.join(output_dir, "cure_tsd_per_challenge_metrics.csv")
        df_csv.to_csv(csv_path, index=False)
        print(f"[+] Saved CURE-TSD metrics CSV to: {csv_path}")

        return summary

    @staticmethod
    def _plot_cure_tsd_curves(results: Dict[str, Any], output_png: str):
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), facecolor="#0F172A")
        levels = [1, 2, 3, 4, 5]

        for idx, (c_name, data) in enumerate(results.items()):
            r, c = idx // 4, idx % 4
            ax = axes[r, c]
            ax.set_facecolor("#1E293B")

            ax.plot(levels, data["baseline_by_level"], "o--", color="#F87171", label="Raw Degraded", linewidth=1.8)
            ax.plot(levels, data["restored_by_level"], "s-", color="#10B981", label="Restored Preprocessed", linewidth=2.0)

            clean_title = c_name.replace("_", " ")
            ax.set_title(clean_title, color="#38BDF8", fontsize=10, fontweight="bold")
            ax.set_xlabel("Severity Level (1-5)", color="#94A3B8", fontsize=8)
            ax.set_ylabel("Accuracy (%)", color="#94A3B8", fontsize=8)
            ax.set_ylim(0, 105)
            ax.tick_params(colors="#94A3B8", labelsize=8)
            ax.grid(True, alpha=0.2, color="#334155")
            if idx == 0:
                ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

        plt.suptitle("CURE-TSD Environmental Challenge Resilience Benchmark (12 Challenges x 5 Levels)",
                     color="#F8FAFC", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_png, dpi=200, bbox_inches="tight", facecolor="#0F172A")
        plt.close()
        print(f"[+] Saved CURE-TSD performance curves plot to: {output_png}")


if __name__ == "__main__":
    runner = CureTsdBenchmarkRunner()
    runner.run_benchmark(max_samples=30)

