# Autonomous Vehicle Traffic Sign Recognition - Preprocessing Module

A high-performance, modular image processing and preprocessing system designed for autonomous vehicle traffic sign recognition under diverse environmental lighting conditions, complex road backgrounds, and multi-colour textures.

Built in **Python** with OpenCV, NumPy, and a **Minimalist PySide6 Desktop GUI**.

---

## Table of Contents
1. [Project Overview & Tasks](#project-overview--tasks)
2. [Key Findings & Scientific Analysis (Task c)](#key-findings--scientific-analysis-task-c)
3. [Architecture & Module Organization](#architecture--module-organization)
4. [Installation & Requirements](#installation--requirements)
5. [How to Run](#how-to-run)
   - [Desktop Minimalist UI (PySide6)](#desktop-minimalist-ui-pyside6)
   - [CLI / Batch Automation](#cli--batch-automation)
   - [Running the Test Suite](#running-the-test-suite)
6. [Task Implementation Details](#task-implementation-details)
   - [Task a: Image Loading & Intensity Characteristics](#task-a-image-loading--intensity-characteristics)
   - [Task b: Colour Spaces & Channel Decomposition](#task-b-colour-spaces--channel-decomposition)
   - [Task c: Colour Space Comparison & Lighting Invariance](#task-c-colour-space-comparison--lighting-invariance)
   - [Task d: Colour-Channel Analysis & Sign Isolation](#task-d-colour-channel-analysis--sign-isolation)
   - [Task e: 2D Convolution & Gaussian Filtering](#task-e-2d-convolution--gaussian-filtering)
   - [Task f: Multi-Stage Comparative Display](#task-f-multi-stage-comparative-display)
7. [Experimental Results & Verification](#experimental-results--verification)

---

## Project Overview & Tasks

Autonomous vehicle vision systems encounter severe environmental challenges:
- Shadows cast by bridges and trees
- High-beam glare and direct sun exposure
- Twilight, overcast skies, rain, and nighttime driving
- Complex backgrounds (asphalt, foliage, buildings, other vehicles)

This project addresses **Question 4: Image Processing for Traffic Sign Recognition** by providing an end-to-end preprocessing module fulfilling all specified tasks:

- **Task a**: Load traffic sign images and extract spatial dimensions, channel profiles, and mathematical intensity statistics.
- **Task b**: Convert images into **RGB**, **HSV**, and **Grayscale** colour spaces, including full channel decomposition.
- **Task c**: Statistically and experimentally compare the three colour representations to establish the optimal representation for robust sign detection under varying lighting conditions.
- **Task d**: Apply colour-channel analysis and dual-range HSV thresholding with morphological filtering to cleanly isolate the traffic sign from background clutter.
- **Task e**: Implement 2D spatial convolution and Gaussian filtering to eliminate sensor noise, suppress artifacts, and sharpen boundary contours.
- **Task f**: Generate comprehensive comparative grids and interactive visual representations contrasting all intermediate and final outputs.

---

## Key Findings & Scientific Analysis (Task c)

### Colour Representation Comparison

| Property | RGB Colour Space | Grayscale Space | HSV Colour Space |
| :--- | :--- | :--- | :--- |
| **Channels** | Red ($R$), Green ($G$), Blue ($B$) | Luminance ($Y$) | Hue ($H$), Saturation ($S$), Value ($V$) |
| **Luminance Coupling** | **Tightly Coupled** across all 3 channels | Discards all chromaticity | **Explicitly Decoupled** ($V$ = Lightness, $H$ = Color) |
| **Shadow / Glare Impact** | Severe: $R, G, B$ all shift simultaneously | Moderate-Severe: pixel intensities shift | **Minimal**: Hue angle remains constant |
| **Empirical Stability Score** | **~35% – 45%** | **~34% – 48%** | **~91% – 95%** |
| **Suitability for Sign Rec.** | Poor to Moderate | Ineffective for colored borders | **Optimal / Recommended** |

### Mathematical & Engineering Conclusion
$$\text{Hue } H = \arccos\left(\frac{2R - G - B}{2\sqrt{(R-G)^2 + (R-B)(G-B)}}\right)$$

When illumination intensity varies by a scalar factor $\alpha$ (e.g. going from bright sunlight $\alpha > 1$ into a dark tunnel or shadow $\alpha < 1$):
$$\mathbf{c}' = \alpha \cdot [R, G, B]$$
In RGB and Grayscale, every pixel value shifts proportionally by $\alpha$, destroying any static threshold boundary. 
However, in the Hue formula, the scalar $\alpha$ cancels out in both numerator and denominator! Consequently:
$$H(\alpha R, \alpha G, \alpha B) = H(R, G, B)$$

**Conclusion**: **HSV is definitively the most suitable colour space** for autonomous vehicle traffic sign recognition under varying illumination. It enables robust segmentation of regulatory red, warning yellow, and mandatory blue signs using fixed Hue intervals regardless of whether the sign is in bright sunlight or deep shadow.

---

## Architecture & Module Organization

```
TrafficLightRecognitionSystem/
├── Traffic_Sign_Recognition_Preprocessing.ipynb # Standalone Google Colab notebook (Tasks a-f + Model + Motion/Dust Lab)
├── traffic_sign_processor.py    # Core image processing library (Tasks a - f)
├── robustness_engine.py         # Autonomous vehicle motion blur & dust restoration engine
├── train_classifier.py          # Machine learning model trainer (91.2% Test Accuracy)
├── evaluate_model.py            # Comprehensive evaluation tool (F1-score, confusion matrix, calibration)
├── predict.py                   # Real-time inference & interactive testing CLI (<25ms latency)
├── app.py                       # Minimalist PySide6 Desktop GUI (8 tabs with live sliders & benchmarks)
├── run_cli.py                   # Preprocessing CLI & batch processor
├── test_processor.py            # Automated pytest test suite (12 unit tests)
├── requirements.txt             # Project dependencies
├── README.md                    # Documentation & technical report
├── models/                      # Saved model artifact & metadata
│   ├── traffic_sign_classifier.pkl
│   └── model_metadata.json
├── Traffic data/                # Dataset directory (DATA, TEST, labels.csv)
│   ├── DATA/                    # 52 class subdirectories (5,678 images)
│   ├── TEST/                    # Test dataset (432 images)
│   └── labels.csv               # Ground truth label mapping (52 classes)
└── outputs/                     # Generated comparison grids, reports & confusion matrix
    ├── confusion_matrix_annotated.png
    ├── confidence_calibration.png
    ├── full_evaluation_report.json
    ├── per_class_metrics.csv
    └── comparison_000_0001.png
```

---

## Installation & Requirements

Ensure Python 3.9+ is installed. Install required packages:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `opencv-python`: Image manipulation, colour space conversions, filtering, contour extraction.
- `numpy`: High-speed vectorised matrix operations and numerical analysis.
- `PySide6`: Modern, cross-platform Qt GUI framework for the minimalist interface.
- `pytest`: Unit testing framework.

---

## How to Run

### Desktop Minimalist UI (PySide6)
Launch the interactive desktop interface:

```bash
python app.py
```

#### UI Features:
1. **Pipeline Overview Tab**: Real-time 6-card workflow (Original $\to$ HSV Hue $\to$ Gaussian Filtering $\to$ HSV Threshold Mask $\to$ 2D Convolution Sharpening $\to$ Isolated Sign Cutout) with quick metric badges.
2. **Image Inspection Tab (Task a)**: Deep dimension analysis, dynamic range, RMS contrast, perceived lightness, Shannon entropy, and per-channel distribution tables.
3. **Colour Spaces Studio (Task b)**: Side-by-side comparative views of RGB, HSV, and Grayscale, with one-click decomposition into individual channel planes ($R, G, B$ and $H, S, V$).
4. **Lighting Robustness Lab (Task c)**: Interactive brightness and contrast simulation sliders to visually demonstrate Hue stability vs RGB/Grayscale collapse, complete with empirical stability percentages.
5. **Thresholding & Isolation Tab (Task d)**: Presets for Red (Regulatory), Blue (Mandatory), Yellow (Warning), Auto-detect, and interactive Morphological Noise Filter sliders.
6. **Filtering & Boundaries Tab (Task e)**: Gaussian kernel size ($3\times3$ to $11\times11$), Sigma ($\sigma$) control, 2D convolution kernel picker (Sharpen, Laplacian, Sobel X, Sobel Y, Ridge, Emboss), and Canny edge overlays.
7. **One-Click Comparison Export**: Saves 9-panel publication-grade composite grids directly to disk.

---

### CLI / Batch Automation
Run single-image processing:
```bash
python run_cli.py --image "Traffic data/DATA/0/000_0001.png"
```

Run multi-class automated demo:
```bash
python run_cli.py --demo
```

Run batch processing on $N$ dataset images:
```bash
python run_cli.py --batch 10 --output outputs/batch
```

---

### Running the Test Suite
Execute the automated test suite verifying all tasks:

```bash
pytest test_processor.py -v
```

Expected output: `12 passed in < 1.0s`.

---

## Task Implementation Details

### Task a: Image Loading & Intensity Characteristics
- Dimensions extracted: Height, Width, Channels, Aspect Ratio, Total Pixels, Bit Depth ($8\text{-bit}$), Memory Size.
- Statistical intensity metrics:
  - **Mean ($\mu$) & Median**: Central tendency of scene illumination.
  - **Standard Deviation ($\sigma$) & RMS Contrast**: $RMS = \sqrt{\frac{1}{N}\sum (I_i - \mu)^2}$.
  - **Dynamic Range**: Span between $\min(I)$ and $\max(I)$.
  - **Perceived Luminance**: $Y = 0.299R + 0.587G + 0.114B$ (ITU-R BT.601).
  - **Shannon Entropy**: $H = -\sum p_i \log_2(p_i)$, measuring texture detail and information density.

### Task b: Colour Spaces & Channel Decomposition
- OpenCV BGR input converted to standard RGB, HSV, and Grayscale.
- Complete channel separation:
  - **RGB**: $R, G, B$ extracted as individual single-channel arrays and color-isolated projections.
  - **HSV**: Hue ($0-180^\circ$ spectral mapping), Saturation (Viridis purity map), Value (Inferno lightness map).
  - **Grayscale**: Normalized luminance channel.

### Task c: Colour Space Comparison & Lighting Invariance
- Simulated illumination tests spanning $0.4\times$ (twilight / heavy shadow) to $1.6\times$ (intense direct sun / glare).
- Measured stability via Coefficient of Variation:
  $$CV = \frac{\sigma}{\mu}$$
- Hue demonstrated $CV \approx 0.08$ (Stability $> 91\%$), whereas RGB channels demonstrated $CV \approx 0.54$ (Stability $< 46\%$), validating HSV superiority.

### Task d: Colour-Channel Analysis & Sign Isolation
- Dual-range HSV thresholding:
  - **Red Signs**: Red spans the wrap-around boundary of the Hue cylinder:
    $$H \in [0, 10] \cup [160, 180], \quad S \in [50, 255], \quad V \in [40, 255]$$
  - **Blue Signs**: $H \in [95, 135], \quad S \in [50, 255], \quad V \in [40, 255]$
  - **Yellow Signs**: $H \in [12, 38], \quad S \in [70, 255], \quad V \in [60, 255]$
- Morphological noise cleaning:
  - **Morphological Opening** ($\text{Erosion} \to \text{Dilation}$) removes isolated salt-and-pepper road reflections.
  - **Morphological Closing** ($\text{Dilation} \to \text{Erosion}$) fills internal gaps and holes in the sign boundary.
- Background isolation via Bitwise AND: $\text{Foreground} = \text{Image} \odot \text{Mask}$.
- Spatial localization: Contours extracted to generate green bounding boxes and region of interest coordinates.

### Task e: 2D Convolution & Gaussian Filtering
- **Gaussian Noise Reduction**:
  $$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$
  Suppresses high-frequency camera noise and rain/dust artifacts while maintaining geometric integrity.
- **2D Convolution Sharpening**:
  Applies spatial kernels via `cv2.filter2D`:
  $$\mathbf{K}_{\text{sharpen}} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}, \quad \mathbf{K}_{\text{laplacian}} = \begin{bmatrix} -1 & -1 & -1 \\ -1 & 8 & -1 \\ -1 & -1 & -1 \end{bmatrix}$$
- **Boundary Enhancement**:
  Canny edge detector with hysteresis thresholds combined with Sobel gradient magnitude overlays cyan sign contours directly onto the original image.

### Task f: Multi-Stage Comparative Display
Generates structured multi-panel visual representations:
1. Original RGB capture
2. Grayscale luminance
3. HSV Hue chrominance
4. Gaussian smoothed image
5. 2D convolved sharpened image
6. Boundary contour overlay
7. Cleaned binary threshold mask
8. Isolated sign foreground cutout
9. Sign localization & bounding box

---

## Experimental Results & Verification

All benchmarks executed on the real traffic sign dataset (`Traffic data/DATA/`):

| Metric | Class 0 (Speed 5km/h) | Class 14 (No Overtake) | Class 21 (Go Straight) | Class 34 (Danger Ahead) |
| :--- | :--- | :--- | :--- | :--- |
| **Dimensions** | $134 \times 128$ | $147 \times 131$ | $292 \times 232$ | $124 \times 117$ |
| **Detected Sign Type** | Red (Auto) | Red (Auto) | Blue (Auto) | Yellow/Red (Auto) |
| **Sign Pixel Ratio** | 18.80% | 15.29% | 51.65% | 13.39% |
| **RMS Contrast** | 34.98 | 32.55 | 67.81 | 36.07 |
| **Shannon Entropy** | 6.476 bits | 6.574 bits | 3.172 bits | 5.756 bits |
| **HSV Stability** | **91.45%** | **92.21%** | **96.84%** | **94.79%** |
| **RGB Stability** | 45.39% | 46.12% | 41.20% | 35.59% |

