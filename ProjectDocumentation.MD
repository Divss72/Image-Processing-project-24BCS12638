# AUTONOMOUS VEHICLE TRAFFIC SIGN RECOGNITION SYSTEM
## Technical Project Documentation & Engineering Report

**Project Scope:** Autonomous Driving Perception Module | Computer Vision Preprocessing & Classification  
**Target Platform:** Embedded Autonomous Driving Systems (ADAS / L4 Automation)  
**Engineering Standards:** ITU-R BT.601 Photometric Luminance, Real-Time Hardware Budget ($<33.3\text{ ms}$ / $30\text{ FPS}$)  
**Datasets:** International Traffic Sign Dataset (5,678 Train, 432 Test, 52 Classes), Georgia Tech CURE-TSD Benchmark  

---

## 1. Executive Summary

This project delivers a complete, production-ready vision perception and preprocessing module for an autonomous vehicle manufacturer. Traffic sign recognition systems deployed in real-world self-driving vehicles must operate reliably under non-ideal optical conditions:
- Rapidly changing illumination (direct sunlight glare, tree shadows, dark tunnels).
- High-speed vehicle motion blur ($0\text{ to }150\text{ km/h}$).
- Adverse weather degradations (dense atmospheric fog, dust haze, falling rain, lens mud splatters).
- High-resolution dashcam video feeds ($4\text{K UHD}$ at $30\text{ FPS}$).

The developed solution couples an **illumination-invariant image preprocessing pipeline** (Tasks a through f) with a **fine-grained $1,788\text{-dimensional}$ invariant feature representation** and an **ensemble classifier**, achieving:
- **91.67% Top-1 Accuracy** and **95.83% Top-3 Candidate Accuracy** on held-out test data.
- **23.75 ms CPU Latency (42.1 FPS)** on single frames and **13.65 ms (73.2 FPS)** in video stream mode, well within standard camera latency budgets ($33.3\text{ ms}$).
- **4.18× Confidence Calibration Ratio** ($71.5\%$ on correct vs. $17.1\%$ on errors), providing a safe fail-soft margin.
- **~80% Average Accuracy Retention** across the **60 environmental stress configurations** of the Georgia Tech CURE-TSD benchmark.

---

## 2. Core Image Processing Tasks (Tasks a – f)

### Task a: Dimension Extraction & Mathematical Intensity Profiling
When a camera captures an image, spatial and statistical characteristics are computed:
1. **Geometric Structure**: Height ($H$), Width ($W$), Channels ($C=3$), Aspect Ratio ($W/H$), Bit Depth ($8\text{-bit}$), Total Pixels ($N$).
2. **Mean ($\mu$) & Median**: Measuring central tendency of scene brightness.
3. **RMS Contrast**: Root Mean Square contrast quantifying scene texture variance:
   $$\text{RMS Contrast} = \sqrt{\frac{1}{N}\sum_{i=1}^N (I_i - \mu)^2}$$
4. **Perceived Luminance**: Formulated according to the ITU-R BT.601 photometric standard:
   $$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
5. **Shannon Entropy**: Measures information richness and texture detail:
   $$H = -\sum_{k=0}^{255} p_k \log_2(p_k)$$

### Task b: Colour Spaces & Complete Channel Decomposition
Raw BGR camera images are decomposed into:
- **RGB Space**: Additive color model ($R, G, B$ individual channels).
- **Grayscale Space**: Normalized luminance plane.
- **HSV Space**: Cylindrical color model:
  - *Hue ($H \in [0, 180^\circ]$)*: Pure chromatic spectral angle.
  - *Saturation ($S \in [0, 255]$)*: Color saturation and vibrancy.
  - *Value ($V \in [0, 255]$)*: Photometric lightness.

### Task c: Comparative Analysis & Mathematical Proof of Illumination Invariance

```
EMPIRICAL STABILITY: Twilight Underexposure (0.4x) vs. Glaring Sunlight (1.6x)
```

| Dimension | RGB Representation | Grayscale Representation | HSV Representation |
| :--- | :--- | :--- | :--- |
| **Coupling** | Tightly coupled with luminance | Eliminates chromaticity | **Explicitly decoupled ($H, S$ vs $V$)** |
| **Stability Under Shadows** | Low ($35\%\text{--}46\%$) | Low ($34\%\text{--}48\%$) | **Optimal ($91\%\text{--}97\%$)** |
| **Recommendation** | Inflexible for thresholding | Cannot separate red/blue | **Selected Architecture Standard** |

#### Mathematical Derivation of Hue Lighting Invariance:
Let environmental lighting scale scene radiance by scalar $\alpha$:
$$\mathbf{c}' = [\alpha R, \alpha G, \alpha B]$$
Substituting into the geometric Hue angle equation:
$$H(\alpha R, \alpha G, \alpha B) = \arccos\left(\frac{2(\alpha R) - (\alpha G) - (\alpha B)}{2\sqrt{(\alpha R - \alpha G)^2 + (\alpha R - \alpha B)(\alpha G - \alpha B)}}\right) = \arccos\left(\frac{\alpha(2R - G - B)}{\alpha \cdot 2\sqrt{(R-G)^2 + (R-B)(G-B)}}\right) = H(R, G, B)$$
The intensity scalar $\alpha$ cancels out completely in both numerator and denominator. Consequently, Hue values remain identical whether under shadows or bright sun.

### Task d: Dual-Range HSV Thresholding & Morphological Segmentation
Traffic signs are isolated from background textures (asphalt, foliage, sky):
- **Red Hue Wrap-Around Segmentation**: Because red spans both extremes of the cylindrical Hue spectrum:
  $$\text{Mask}_{\text{Red}} = \text{inRange}(H \in [0, 10]) \cup \text{inRange}(H \in [160, 180]), \quad S \ge 50, \quad V \ge 40$$
- **Blue Mandatory Signs**: $H \in [95, 135], \quad S \ge 50, \quad V \ge 40$
- **Yellow Warning Signs**: $H \in [12, 38], \quad S \ge 70, \quad V \ge 60$
- **Morphological Cleaning**:
  - *Opening* ($\text{Erode} \to \text{Dilate}$ with $5\times5$ ellipse) eliminates salt-and-pepper asphalt noise.
  - *Closing* ($\text{Dilate} \to \text{Erode}$) fills interior holes within the sign symbol.
- **Segmented Cutout**: Bitwise-AND extraction isolates the sign with $99.1\%$ background suppression.

### Task e: 2D Convolution Sharpening & Boundary Enhancement
- **Gaussian Noise Filtering**: Suppresses camera sensor noise while maintaining contour edges:
  $$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
- **2D Convolution Sharpening**: Laplacian spatial filtering sharpens high-frequency digit and arrow boundaries:
  $$\mathbf{K}_{\text{sharpen}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$
- **Boundary Contour Overlay**: Canny hysteresis edge detection highlights structural sign boundaries with a cyan geometric overlay.

### Task f: Multi-Stage Comparative Visual Matrix
Consolidates all processing phases into an automated $3\times3$ diagnostic verification grid and JSON inspection report.

---

## 3. Machine Learning Classification & Benchmark

### 1,788-Dimensional Scale- & Illumination-Invariant Representation
To resolve subtle digits inside identical red rings (e.g., distinguishing **Speed Limit 30** from **50, 70, 80 km/h**):
1. **Fine-Grained $4\times4$ Cell HOG Descriptors**: Dividing a $32\times32$ sign into $4\times4$ cells with $8\times8$ blocks provides $4\times$ finer spatial resolution than standard $8\times8$ cells, capturing the interior digit contours.
2. **CLAHE Local Contrast Enhancement**: Contrast-Limited Adaptive Histogram Equalization (`clipLimit=2.5, tileGridSize=(4, 4)`) amplifies faded interior numbers under fog or dusk.
3. **HSV Distribution Histograms**: 16-bin Hue and 8-bin Saturation histograms normalize color distributions.
4. **Total Dimensionality**: $1,764\text{ (HOG)} + 16\text{ (Hue)} + 8\text{ (Saturation)} = \mathbf{1,788}\text{ features}$.

### Quantitative Performance Metrics (432 Held-Out Test Images)
- **Top-1 Accuracy**: **91.67%** (396 / 432 correct)
- **Top-3 Candidate Accuracy**: **95.83%** (414 / 432 in Top 3)
- **Macro F1-Score**: **0.9091** | **Weighted F1-Score**: **0.9114**
- **Cohen’s Kappa ($\kappa$)**: **0.9097** (Near-perfect reliability)
- **Matthews Correlation Coefficient (MCC)**: **0.9103**
- **Confidence Calibration Ratio**: **4.18×** ($71.51\%$ confidence on correct vs. $17.11\%$ on errors).

---

## 4. High-Speed Vehicle Motion Dynamics

### Latency Budgeting & Distance Traveled
In an autonomous vehicle, recognition latency dictates the vehicle's safe stopping distance:
$$\text{Distance Traveled per Frame } d = v_{\text{vehicle}} \times t_{\text{inference}}$$

At our measured latency of **23.75 ms (42.1 FPS)**:
- At **60 km/h** ($16.67\text{ m/s}$): The vehicle travels only **0.39 meters** per frame (~1.3 feet).
- At **120 km/h** ($33.33\text{ m/s}$): The vehicle travels only **0.79 meters** per frame (~2.6 feet).
- A sign detected at $50\text{ to }100\text{ meters}$ provides **60 to 150 consecutive detection frames** before the vehicle passes it.

---

## 5. Adverse Weather & CURE-TSD Environmental Benchmark

### Georgia Tech CURE-TSD Benchmark Results (60 Test Configurations)
Evaluated across **12 challenge types and 5 severity levels**:
- **Dirty Lens (Mud & Road Splatter)**: Baseline $75.3\% \to \mathbf{81.3\%}$ (**+6.0% Gain**) via adaptive $3\times3$ median filtering and Gray-World color constancy.
- **Rain Streaks**: Baseline $82.6\% \to \mathbf{86.0\%}$ (**+3.4% Gain**).
- **Atmospheric Fog / Haze**: Baseline $78.4\% \to \mathbf{81.1\%}$ (**+2.7% Gain**) via Koschmieder optical transmission recovery.
- **Night Darkening**: Baseline $70.1\% \to \mathbf{73.8\%}$ (**+3.7% Gain**).
- **Sensor Noise**: Baseline $79.9\% \to \mathbf{82.4\%}$ (**+2.5% Gain**).

---

## 6. Real-Time 4K Dashcam Video Stream Processing

Tested on real-world 4K UHD footage (`384582.mp4`, $3840\times2160$ at $29.97\text{ FPS}$, 1,414 frames, 47.2s):
- **Processing Resolution**: Optimized to $960\times540$ during streaming, achieving smooth real-time performance.
- **HUD Telemetry Display**: Overlays vehicle speed ($85\text{ km/h}$), frame latency ($22.4\text{ ms}$), instant FPS, bounding boxes, and label banners.
- **Detections**: In a 300-frame driving window, **761 traffic signs** were detected and classified (334 No Stopping, 259 No Entry, 45 Keep Right).

---

## 7. Deliverables & User Guide

1. **Minimalist Desktop UI (PySide6)**:
   ```bash
   python app.py
   ```
   Features 10 specialized tabs covering Tasks a–f, confusion matrix viewer, interactive speed/dust sliders, dashcam video player, and CURE-TSD benchmark explorer.

2. **Standalone Google Colab Notebook**:
   [`Traffic_Sign_Recognition_Preprocessing.ipynb`](Traffic_Sign_Recognition_Preprocessing.ipynb) — 28 self-contained cells covering all tasks, model training, evaluation, video detection, and CURE-TSD benchmarking.

3. **Automated Unit Test Suite**:
   ```bash
   pytest test_processor.py -v
   ```
   12/12 unit tests passing in 0.57s.

