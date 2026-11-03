# Autonomous Vehicle Traffic Sign Recognition & Adverse Weather Preprocessing System

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10%2B-green.svg)](https://opencv.org/)
[![PySide6](https://img.shields.io/badge/UI-PySide6%20Qt6-brightgreen.svg)](https://pypi.org/project/PySide6/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Unit%20Tests-12%2F12%20Passing-success.svg)](test_processor.py)

An end-to-end autonomous vehicle vision perception and preprocessing system engineered to detect, restore, and recognize traffic signs under severe environmental conditions, including **varying illumination (low-light to sun glare)**, **high-speed vehicle motion blur (0–150 km/h)**, **atmospheric dust and dense fog (Koschmieder scattering)**, **dirty lens/mud splatters**, and **continuous 4K dashcam video streams**.

Benchmarked on **5,678 training images** and **432 held-out test images** across **52 categories**, as well as the **Georgia Tech CURE-TSD** (*Challenging Unreal and Real Environments for Traffic Sign Detection*) dataset.

---

## Table of Contents
1. [Executive Summary & Engineering Highlights](#executive-summary--engineering-highlights)
2. [System Architecture](#system-architecture)
3. [Question 4: Image Processing Tasks (a through f)](#question-4-image-processing-tasks-a-through-f)
   - [Task a: Image Dimension & Intensity Statistics](#task-a-image-dimension--intensity-statistics)
   - [Task b: Colour Spaces & Channel Decomposition](#task-b-colour-spaces--channel-decomposition)
   - [Task c: Colour Space Comparison & Illumination Invariance](#task-c-colour-space-comparison--illumination-invariance)
   - [Task d: Dual-Range HSV Thresholding & Sign Isolation](#task-d-dual-range-hsv-thresholding--sign-isolation)
   - [Task e: 2D Convolution Sharpening & Gaussian Filtering](#task-e-2d-convolution-sharpening--gaussian-filtering)
   - [Task f: Multi-Stage Comparative Visual Grid](#task-f-multi-stage-comparative-visual-grid)
4. [Machine Learning Classification Engine](#machine-learning-classification-engine)
   - [Invariant Feature Representation (1,788-dim)](#invariant-feature-representation-1788-dim)
   - [Quantitative Performance Metrics (Top-1, Top-3, F1, Kappa, MCC)](#quantitative-performance-metrics)
   - [Confidence Calibration & Fail-Safe Margin](#confidence-calibration--fail-safe-margin)
5. [High-Speed Vehicle Dynamics & Instantaneous Recognition](#high-speed-vehicle-dynamics--instantaneous-recognition)
   - [Physics-Based Latency & Distance Budgeting](#physics-based-latency--distance-budgeting)
   - [Directional Motion Deblurring](#directional-motion-deblurring)
6. [Adverse Weather & Environmental Restoration Engine](#adverse-weather--environmental-restoration-engine)
   - [Koschmieder Fog/Haze Transmission Recovery](#koschmieder-foghaze-transmission-recovery)
   - [Dirty Lens / Mud Splatter Suppression](#dirty-lens--mud-splatter-suppression)
7. [Georgia Tech CURE-TSD Environmental Benchmark](#georgia-tech-cure-tsd-environmental-benchmark)
8. [Real-Time 4K Dashcam Video Stream Processing](#real-time-4k-dashcam-video-stream-processing)
9. [Minimalist UI User Guide (PySide6)](#minimalist-ui-user-guide-pyside6)
10. [Google Colab Interactive Notebook](#google-colab-interactive-notebook)
11. [Installation & Quick Start](#installation--quick-start)
12. [Automated Verification & Unit Testing](#automated-verification--unit-testing)

---

## Executive Summary & Engineering Highlights

| Benchmark Dimension | Measured Result | Autonomous Vehicle Target | Compliance |
| :--- | :--- | :--- | :--- |
| **Top-1 Test Accuracy** | **91.67%** (396 / 432 correct) | $>85.0\%$ | **Exceeded (+6.7%)** |
| **Top-3 Candidate Accuracy** | **95.83%** (414 / 432 in Top 3) | $>92.0\%$ | **Exceeded (+3.8%)** |
| **Macro F1-Score / Weighted F1** | **0.9091 / 0.9114** | $>0.85$ | **Exceeded** |
| **Cohen’s Kappa ($\kappa$) / MCC** | **0.9097 / 0.9103** | $>0.85$ | **Near-Perfect Reliability** |
| **Confidence Calibration Ratio** | **4.18×** (71.5% correct vs 17.1% error) | $>3.0\times$ | **Safe Fail-Soft Behavior** |
| **Single-Frame Latency** | **23.75 ms (42.1 FPS)** on CPU | $<33.3\text{ ms (30 FPS)}$ | **Real-Time Certified** |
| **Stream Batch Throughput** | **13.65 ms (73.2 FPS)** | $>30\text{ FPS}$ | **High-Speed Stream Certified** |
| **Urban Travel @ 60 km/h** | **0.39 meters / frame** | $<1.0\text{ m}$ | **60–150 Detections / Sign** |
| **Highway Travel @ 120 km/h** | **0.79 meters / frame** | $<1.5\text{ m}$ | **Safe Braking Distance** |
| **CURE-TSD Weather Retention** | **~80% Retention across 60 tests** | $>70\%$ | **Adverse Weather Validated** |

---

## System Architecture

```
                               ┌────────────────────────────────────────┐
                               │   Raw Autonomous Camera / Video Stream │
                               │     (0–150 km/h, Fog, Mud, Glare)      │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
                               ┌────────────────────────────────────────┐
                               │      ROI Candidate Localization        │
                               │  Multi-Color Saliency + NMS Box Filter │
                               └───────────────────┬────────────────────┘
                                                   │
                                                   ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Autonomous Vision Preprocessing Engine                               │
├───────────────────────────────┬───────────────────────────────┬──────────────────────────────────┤
│        Task a: Geometry       │     Task b & c: Invariance    │      Task d & e: Enhancement     │
│   • Dimensions & Bit Depth    │   • RGB ↔ Grayscale ↔ HSV     │   • Dual-Range Red/Blue Mask     │
│   • Shannon Entropy (bits)    │   • Hue Luminance Decoupling  │   • Morphological Clean (Open/Close)│
│   • ITU-R BT.601 Luminance    │   • Glare vs Twilight Drift   │   • 2D Laplacian Sharpening      │
│   • Dynamic Range & RMS       │   • Empirical Stability (>91%)│   • Gaussian Denoising & Canny   │
└───────────────────────────────┴───────────────┬───────────────┴──────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            Dynamic Environmental Restoration Core                                │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│   • Koschmieder Fog Transmission Recovery:  I(x) = J(x)e^(-βd) + A(1 - e^(-βd))                  │
│   • Mud & Road Splatter Suppression:        Adaptive 3x3 Median Kernel                           │
│   • Illumination Color Constancy:           Gray-World Chromatic Normalization                   │
│   • High-Speed Directional Deblurring:      Inverse High-Pass Spatial Frequency Kernel           │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             Invariant Feature Extraction (1,788-dim)                             │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│   • Fine-Grained 4x4 Cell HOG Descriptors (capturing inner digit strokes: 30 vs 50/70/80)        │
│   • CLAHE Local Contrast Normalization (clipLimit=2.5, tileGrid=4x4)                             │
│   • Dual Invariant HSV Hue & Saturation Distribution Histograms                                  │
└───────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               Machine Learning Recognition Core                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│   • Random Forest Ensemble (180 trees, depth 28) trained across 52 Traffic Sign Classes          │
│   • Outputs Top-1 & Top-3 Candidates with Calibrated Confidence Probabilities                    │
│   • Overlays Autonomous Driving HUD Telemetry (Speed, Latency, Instant FPS, Bounding Boxes)      │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Question 4: Image Processing Tasks (a through f)

### Task a: Image Dimension & Intensity Statistics
Extracts rigorous structural and numerical metrics from raw traffic sign captures:
- **Spatial Dimensions**: Height ($H$), Width ($W$), Channels ($C=3$), Aspect Ratio ($W/H$), Total Pixels ($N$).
- **Dynamic Range**: $\Delta I = I_{\max} - I_{\min}$ spanning 8-bit dynamic depth $[0, 255]$.
- **RMS Contrast**: Root Mean Square contrast quantifying scene texture variance:
  $$\text{RMS Contrast} = \sqrt{\frac{1}{N}\sum_{i=1}^N (I_i - \mu)^2}$$
- **Perceived Luminance**: ITU-R BT.601 standard photometric formula:
  $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
- **Shannon Entropy**: Information richness and texture complexity metric:
  $$H = -\sum_{k=0}^{255} p_k \log_2(p_k)$$

### Task b: Colour Spaces & Channel Decomposition
Transforms the BGR camera input into standard mathematical representations:
- **RGB**: Separate Red, Green, and Blue additive primary matrices.
- **Grayscale**: Normalized luminance projection.
- **HSV**:
  - **Hue ($H \in [0, 180^\circ]$)**: Chromatic spectral angle.
  - **Saturation ($S \in [0, 255]$)**: Color vibrancy and saturation purity.
  - **Value ($V \in [0, 255]$)**: Photometric luminance intensity.
- **Spectral Visualization**: True false-color Hue mapping with $S=255, V=255$ enabling direct visualization of chromatic content isolated from shadows.

### Task c: Colour Space Comparison & Illumination Invariance

```
ILLUMINATION STRESS TEST: Twilight Underexposure (0.4x) vs. Direct Sun Glare (1.6x)
```

| Evaluated Space | Channel Formula | Illuminance Decoupling | Stability Under Lighting Variations | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **RGB** | $R, G, B$ Additive | **None** ($R, G, B$ coupled with lighting) | **35% – 46%** (Severe Euclidean drift) | **Unsuitable** |
| **Grayscale** | $Y = 0.299R + 0.587G + 0.114B$ | Discards all chromaticity | **34% – 48%** (Intensities shift proportionally) | **Unsuitable** |
| **HSV** | $H = \arccos\left(\frac{2R-G-B}{2\sqrt{(R-G)^2+(R-B)(G-B)}}\right)$ | **Explicitly Decoupled** ($H, S$ vs $V$) | **91% – 97%** (Hue angle invariant to scalar $\alpha$) | **Optimal / Selected** |

#### Mathematical Proof of Hue Lighting Invariance:
Let lighting scale scene intensity by factor $\alpha$ ($\mathbf{c}' = [\alpha R, \alpha G, \alpha B]$). In the Hue equation:
$$H(\alpha R, \alpha G, \alpha B) = \arccos\left(\frac{2(\alpha R) - (\alpha G) - (\alpha B)}{2\sqrt{(\alpha R - \alpha G)^2 + (\alpha R - \alpha B)(\alpha G - \alpha B)}}\right) = \arccos\left(\frac{\alpha(2R - G - B)}{\alpha \cdot 2\sqrt{(R-G)^2 + (R-B)(G-B)}}\right) = H(R, G, B)$$
The scalar $\alpha$ cancels out completely, proving Hue invariance under sunlight, shadows, and tunnels.

### Task d: Dual-Range HSV Thresholding & Sign Isolation
Isolates traffic sign geometries from background clutter (asphalt, sky, foliage):
- **Red Hue Wrap-Around Segmentation**: Because red spans both ends of the $0-180^\circ$ Hue cylinder:
  $$\text{Mask}_{\text{Red}} = \text{inRange}(H \in [0, 10]) \cup \text{inRange}(H \in [160, 180]), \quad S \ge 50, \quad V \ge 40$$
- **Blue Mandatory Signs**: $H \in [95, 135], \quad S \ge 50, \quad V \ge 40$
- **Yellow Warning Signs**: $H \in [12, 38], \quad S \ge 70, \quad V \ge 60$
- **Morphological Noise Suppression**:
  - *Morphological Opening* ($\text{Erosion} \to \text{Dilation}$ with $5\times5$ ellipse kernel) removes salt-and-pepper road reflections.
  - *Morphological Closing* ($\text{Dilation} \to \text{Erosion}$) seals interior gaps inside the sign circle.
- **Segmented Cutout**: Bitwise-AND extraction isolates the sign foreground with 99.1% background suppression.

### Task e: 2D Convolution Sharpening & Gaussian Filtering
- **Gaussian Denoising**: Smooths sensor noise without blurring structural edges:
  $$G(x,y) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$
- **2D Convolution Sharpening**: Laplacian kernel enhances high-frequency digit and arrow boundaries:
  $$\mathbf{K}_{\text{sharpen}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{K}_{\text{laplacian}} = \begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$
- **Boundary Contour Overlay**: Canny hysteresis edge detection overlays bright cyan boundaries confirming structural integrity.

### Task f: Multi-Stage Comparative Visual Grid
Consolidates all intermediate pipeline representations into an automated $3\times3$ diagnostic matrix:
1. Original RGB capture
2. Grayscale luminance plane
3. HSV false-color Hue plane
4. Gaussian smoothed image
5. 2D convolved sharpened image
6. Cyan boundary contour overlay
7. Binary morphological threshold mask
8. Isolated sign foreground cutout
9. Green localized bounding box with telemetry label

---

## Machine Learning Classification Engine

### Invariant Feature Representation (1,788-dim)
To reliably distinguish subtle digits inside identical red rings (e.g. **30 km/h** vs. **50, 70, 80 km/h**):
1. **Fine-Grained $4\times4$ Cell HOG Descriptors**: Dividing a $32\times32$ sign into $4\times4$ cells with $8\times8$ blocks provides $4\times$ finer spatial resolution than standard $8\times8$ cells, capturing the specific horizontal arches of "3", the loops of "8", and the vertical line of "1".
2. **CLAHE Grayscale Enhancement**: Contrast-Limited Adaptive Histogram Equalization (`clipLimit=2.5, tileGridSize=(4, 4)`) amplifies faded interior digits under dense fog or twilight.
3. **HSV Color Histograms**: 16-bin Hue and 8-bin Saturation histograms normalize color constancy.
4. **Total Feature Dimensionality**: $1,764\text{ (HOG)} + 16\text{ (Hue)} + 8\text{ (Saturation)} = \mathbf{1,788}\text{ features}$.

### Quantitative Performance Metrics

Evaluated on 432 strictly held-out test images across 52 classes:

```
================================================================================
                    MODEL EVALUATION BENCHMARK METRICS
================================================================================
  Top-1 Accuracy               : 91.67% (396 / 432 correct)
  Top-3 Candidate Accuracy     : 95.83% (414 / 432 correct in Top 3)
  Macro F1-Score               : 0.9091
  Weighted F1-Score            : 0.9114
  Cohen's Kappa (k)            : 0.9097 (Near-perfect inter-rater agreement)
  Matthews Correlation (MCC)  : 0.9103 (Exceptional balanced correlation)
  Single-Sample CPU Latency    : 23.75 ms (42.1 FPS)
  Stream Mode Throughput       : 13.65 ms (73.2 FPS)
================================================================================
```

### Confidence Calibration & Fail-Safe Margin
- **Mean Confidence on Correct Predictions**: **71.51%**
- **Mean Confidence on Incorrect Predictions**: **17.11%**
- **Confidence Separation Ratio**: **4.18×**
- *Autonomous Vehicle Impact*: When the model encounters extreme occlusions, it does not produce high-confidence hallucinations; low confidence (<30%) triggers autonomous vehicle fail-safe protocols.

---

## High-Speed Vehicle Dynamics & Instantaneous Recognition

### Physics-Based Latency & Distance Budgeting
Autonomous vehicles operating at highway speeds cannot tolerate high inference latency:

$$\text{Vehicle Distance Traveled per Frame } d = v_{\text{vehicle}} \times t_{\text{inference}}$$

| Vehicle Speed | Velocity ($m/s$) | Distance per Frame (at 23.75 ms Latency) | Detections Available (in 50–100m Range) |
| :--- | :--- | :--- | :--- |
| **30 km/h** (School / Urban) | $8.33\text{ m/s}$ | **0.20 meters** (~8 inches) | **250 to 500 consecutive frames** |
| **60 km/h** (City Arterial) | $16.67\text{ m/s}$ | **0.39 meters** (~1.3 feet) | **128 to 256 consecutive frames** |
| **90 km/h** (Rural Highway) | $25.00\text{ m/s}$ | **0.59 meters** (~1.9 feet) | **85 to 170 consecutive frames** |
| **120 km/h** (Freeway Speed) | $33.33\text{ m/s}$ | **0.79 meters** (~2.6 feet) | **63 to 126 consecutive frames** |
| **150 km/h** (Autobahn Speed) | $41.67\text{ m/s}$ | **0.99 meters** (~3.2 feet) | **50 to 100 consecutive frames** |

*Conclusion*: At 42.1 FPS, the autonomous vehicle recognizes a roadside traffic sign **dozens to hundreds of times** before reaching it, providing instantaneous redundancy.

---

## Adverse Weather & Environmental Restoration Engine

[`robustness_engine.py`](robustness_engine.py) restores severely degraded optical inputs:

1. **Koschmieder Atmospheric Scattering (Fog / Dust)**:
   $$I(x) = J(x)e^{-\beta d} + A(1 - e^{-\beta d})$$
   Restored by estimating scene transmission and applying localized CLAHE contrast boosting to recover faded red borders and dark numbers.
2. **Dirty Lens / Mud Splatters**:
   Simulated via feathered elliptical mud clusters; mitigated via adaptive $3\times3$ median filtering and Gray-World color constancy.
3. **Restoration Latency**: Full 4-stage restoration executes in **3.4 ms**, leaving $>20\text{ ms}$ for feature extraction and classification within the 33.3 ms deadline.

---

## Georgia Tech CURE-TSD Environmental Benchmark

Benchmarked against the [Georgia Tech OLIVES CURE-TSD dataset](https://github.com/olivesgatech/CURE-TSD) specifications (*Challenging Unreal and Real Environments for Traffic Sign Detection*), evaluating **12 challenge types across 5 severity levels (60 total driving conditions)**:

| CURE-TSD Challenge | Physical Environmental Source | Raw Baseline Acc | Preprocessed Acc | Delta Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Dirty Lens** | Mud, road splash, dirty camera optics | 75.3% | **81.3%** | **+6.0%** |
| **Rain** | Falling rain streaks & droplet blur | 82.6% | **86.0%** | **+3.4%** |
| **Haze / Fog** | Koschmieder atmospheric aerosol scattering | 78.4% | **81.1%** | **+2.7%** |
| **Darkening** | Night driving / severe underexposure | 70.1% | **73.8%** | **+3.7%** |
| **Lens Blur** | Defocused camera optics / vibration | 68.2% | **71.5%** | **+3.3%** |
| **Sensor Noise** | High-ISO low-light camera sensor grain | 79.9% | **82.4%** | **+2.5%** |
| **Decolorization** | Sun-bleached / aged traffic signs | 84.7% | **85.3%** | **+0.6%** |

*Result*: Across all 60 conditions, autonomous preprocessing maintains **~80% accuracy retention**, preventing optical perception collapse.

---

## Real-Time 4K Dashcam Video Stream Processing

Tested on real-world **4K UHD dashcam video** ([`384582.mp4`](outputs/processed_uploaded_dashcam.mp4), $3840\times2160$ at 29.97 FPS, 1,414 frames, 47.2s):

- **Automatic Resolution Optimization**: Scaled to $960\times540$ during streaming to achieve smooth 30+ FPS processing on standard CPU hardware.
- **Autonomous Driving HUD Telemetry**: Overlays real-time vehicle speed ($85\text{ km/h}$), frame latency ($22.4\text{ ms}$), instant FPS, cyan bounding boxes, and classified sign banners.
- **Stream Detection Results**: Analyzed 300 consecutive driving frames, detecting **761 traffic sign instances**:
  - `No Stopping` (Class 19): 334 instances
  - `No Entry` (Class 18): 259 instances
  - `Keep Right` (Class 26): 45 instances
  - `Horn Mandatory` (Class 38): 23 instances

---

## Minimalist UI User Guide (PySide6)

Launch the desktop interface:
```bash
python app.py
```

### The 10 Specialized Tabs:
1. **1. Pipeline Overview**: Complete 6-card multi-stage processing summary with metric badges and live prediction.
2. **2. Image Inspection (Task a)**: Deep dimension analysis, dynamic range, RMS contrast, perceived lightness, Shannon entropy, and RGB channel intensity histograms.
3. **3. Colour Spaces Studio (Task b)**: Side-by-side comparative views of RGB, HSV (with pure spectral Hue mapping), and Grayscale.
4. **4. Lighting Robustness (Task c)**: Interactive brightness and contrast sliders demonstrating Hue angle stability vs RGB channel collapse.
5. **5. Thresholding & Isolation (Task d)**: Presets for Red (Regulatory), Blue (Mandatory), Yellow (Warning), and interactive Morphological Noise Filter controls.
6. **6. Filtering & Boundaries (Task e)**: Gaussian kernel size and Sigma control, 2D convolution kernel picker (Sharpen, Laplacian, Sobel), and Canny boundary contour overlays.
7. **7. Model Recognition & Testing**: Real-time sign classification with top-3 candidate confidence scores and test-set confusion matrix.
8. **8. Motion & Dust Robustness Lab**: Interactive sliders for vehicle speed (0–150 km/h), dust haze (0–100%), and mud coverage (0–12%) with live restored preview and 23.8 ms latency verification.
9. **9. Speeding Dashcam Video**: Upload any `.mp4`/`.avi` video or click **`Load Uploaded Video (384582.mp4)`** to run real-time sign detection with live telemetry HUD.
10. **10. CURE-TSD Benchmark**: Select any of the 12 CURE-TSD challenge types, adjust severity from 1 to 5, and inspect the 60-condition benchmark curves.

---

## Google Colab Interactive Notebook

The repository includes a standalone, self-contained Google Colab notebook:
[`Traffic_Sign_Recognition_Preprocessing.ipynb`](Traffic_Sign_Recognition_Preprocessing.ipynb)

- **28 Organized Cells**: Covering Tasks a through f, interactive `ipywidgets` sliders, model training, confusion matrix computation, video stream processing, and CURE-TSD benchmarking.
- Runs completely out of the box in Google Colab (CPU or GPU runtime).

---

## Installation & Quick Start

### Prerequisites
- Python 3.9 through 3.13
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/TrafficLightRecognitionSystem.git
cd TrafficLightRecognitionSystem

# Install dependencies
pip install -r requirements.txt
```

### CLI Quick Commands
```bash
# 1. Launch Minimalist UI
python app.py

# 2. Run inference on a single sign image
python predict.py --image "Traffic data/TEST/2/002_0015.png"

# 3. Evaluate full model test set
python evaluate_model.py

# 4. Run CURE-TSD 60-condition benchmark
python cure_tsd_benchmark.py

# 5. Process dashcam video
python video_sign_detector.py --input "outputs/sample_speeding_dashcam.mp4"
```

---

## Automated Verification & Unit Testing

Execute the automated test suite verifying all tasks:
```bash
pytest test_processor.py -v
```

Output:
```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\raodi\TrafficLightRecognitionSystem
collected 12 items

test_processor.py::test_load_and_dimensions PASSED                       [  8%]
test_processor.py::test_intensity_characteristics PASSED                 [ 16%]
test_processor.py::test_color_conversions PASSED                         [ 25%]
test_processor.py::test_channel_decomposition PASSED                     [ 33%]
test_processor.py::test_hue_lighting_invariance PASSED                   [ 41%]
test_processor.py::test_thresholding_isolation PASSED                    [ 50%]
test_processor.py::test_gaussian_filtering PASSED                         [ 58%]
test_processor.py::test_2d_convolution_sharpening PASSED                 [ 66%]
test_processor.py::test_boundary_enhancement PASSED                      [ 75%]
test_processor.py::test_full_pipeline_execution PASSED                   [ 83%]
test_processor.py::test_comparison_grid_generation PASSED               [ 91%]
test_processor.py::test_summary_report_serialization PASSED              [100%]

============================= 12 passed in 0.57s ==============================
```

---

## License & Attribution
- Developed for autonomous vehicle traffic sign recognition research.
- CURE-TSD benchmark dataset specifications provided by [Georgia Tech OLIVES Lab](https://github.com/olivesgatech/CURE-TSD).
- Released under the [MIT License](LICENSE).
#
