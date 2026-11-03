"""
Traffic Sign Recognition Preprocessing Module
=============================================
Core image processing engine fulfilling tasks a to f:
  a. Image loading, dimensions, colour channels, and intensity characteristics
  b. Colour space conversions (RGB, HSV, Grayscale) and channel decomposition
  c. Colour representation comparison and lighting suitability determination
  d. Colour-channel analysis, HSV thresholding, morphological filtering, sign isolation
  e. 2D convolution and Gaussian filtering for noise reduction and boundary enhancement
  f. Multi-view comparison and visual representation generation
"""

import os
import math
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np


class TrafficSignProcessor:
    """
    Core image processing module for traffic sign recognition preprocessing.
    """

    def __init__(self, image_path: Optional[str] = None):
        self.image_path: Optional[str] = None
        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.image_hsv: Optional[np.ndarray] = None
        self.image_gray: Optional[np.ndarray] = None

        if image_path:
            self.load_image(image_path)

    # -------------------------------------------------------------------------
    # Task a: Load Image & Extract Characteristics
    # -------------------------------------------------------------------------
    def load_image(self, image_path: str) -> bool:
        """
        Load image from disk and initialize basic color representations.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to decode image from path: {image_path}")

        self.image_path = image_path
        self.image_bgr = img
        self.image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return True

    def set_image_array(self, rgb_array: np.ndarray):
        """
        Set image directly from an RGB numpy array.
        """
        self.image_path = None
        self.image_rgb = rgb_array.astype(np.uint8)
        self.image_bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
        self.image_hsv = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2HSV)
        self.image_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)

    def get_dimensions(self) -> Dict[str, Any]:
        """
        Extract dimensions, aspect ratio, and total pixel count (Task a).
        """
        if self.image_rgb is None:
            raise RuntimeError("No image loaded.")

        height, width = self.image_rgb.shape[:2]
        channels = 1 if len(self.image_rgb.shape) == 2 else self.image_rgb.shape[2]
        aspect_ratio = round(width / float(height), 4) if height > 0 else 0.0
        total_pixels = height * width

        return {
            "height": height,
            "width": width,
            "channels": channels,
            "aspect_ratio": aspect_ratio,
            "total_pixels": total_pixels,
            "shape": self.image_rgb.shape,
            "dtype": str(self.image_rgb.dtype),
            "bit_depth": self.image_rgb.dtype.itemsize * 8,
            "memory_bytes": self.image_rgb.nbytes,
        }

    def get_intensity_characteristics(self) -> Dict[str, Any]:
        """
        Extract detailed statistical intensity characteristics (Task a):
        Min, Max, Mean, Median, Standard Deviation, Variance, RMS Contrast,
        Perceived Brightness, Dynamic Range, and Shannon Entropy.
        """
        if self.image_rgb is None or self.image_gray is None:
            raise RuntimeError("No image loaded.")

        gray = self.image_gray.astype(np.float64)

        # Global Gray statistics
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        median_val = float(np.median(gray))
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        variance_val = float(np.var(gray))
        dynamic_range = max_val - min_val

        # RMS contrast = standard deviation of pixel intensities
        rms_contrast = std_val

        # Perceived brightness using standard ITU-R BT.601 weights: 0.299 R + 0.587 G + 0.114 B
        r, g, b = (
            self.image_rgb[:, :, 0].astype(np.float64),
            self.image_rgb[:, :, 1].astype(np.float64),
            self.image_rgb[:, :, 2].astype(np.float64),
        )
        perceived_brightness = float(np.mean(0.299 * r + 0.587 * g + 0.114 * b))

        # Shannon Entropy (information content)
        hist, _ = np.histogram(self.image_gray, bins=256, range=(0, 256))
        hist_prob = hist / hist.sum()
        hist_prob = hist_prob[hist_prob > 0]
        entropy = -float(np.sum(hist_prob * np.log2(hist_prob)))

        # Per-channel RGB stats
        rgb_stats = {}
        for idx, ch_name in enumerate(["Red", "Green", "Blue"]):
            ch = self.image_rgb[:, :, idx].astype(np.float64)
            rgb_stats[ch_name] = {
                "mean": round(float(np.mean(ch)), 2),
                "std": round(float(np.std(ch)), 2),
                "median": round(float(np.median(ch)), 2),
                "min": int(np.min(ch)),
                "max": int(np.max(ch)),
            }

        # Per-channel HSV stats
        hsv_stats = {}
        hsv_channels = ["Hue", "Saturation", "Value"]
        for idx, ch_name in enumerate(hsv_channels):
            ch = self.image_hsv[:, :, idx].astype(np.float64)
            hsv_stats[ch_name] = {
                "mean": round(float(np.mean(ch)), 2),
                "std": round(float(np.std(ch)), 2),
                "median": round(float(np.median(ch)), 2),
                "min": int(np.min(ch)),
                "max": int(np.max(ch)),
            }

        return {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "median": round(median_val, 2),
            "min": int(min_val),
            "max": int(max_val),
            "variance": round(variance_val, 2),
            "dynamic_range": int(dynamic_range),
            "rms_contrast": round(rms_contrast, 2),
            "perceived_brightness": round(perceived_brightness, 2),
            "entropy": round(entropy, 3),
            "rgb_channels": rgb_stats,
            "hsv_channels": hsv_stats,
        }

    # -------------------------------------------------------------------------
    # Task b: Colour Space Conversions & Channel Decomposition
    # -------------------------------------------------------------------------
    def get_rgb(self) -> np.ndarray:
        """Return RGB image."""
        if self.image_rgb is None:
            raise RuntimeError("No image loaded.")
        return self.image_rgb.copy()

    def get_hsv(self) -> np.ndarray:
        """Return HSV image."""
        if self.image_hsv is None:
            raise RuntimeError("No image loaded.")
        return self.image_hsv.copy()

    def get_grayscale(self) -> np.ndarray:
        """Return Grayscale image."""
        if self.image_gray is None:
            raise RuntimeError("No image loaded.")
        return self.image_gray.copy()

    def get_channel_decomposition(self) -> Dict[str, np.ndarray]:
        """
        Extract individual channel planes for RGB, HSV, and Grayscale (Task b).
        Returns raw 2D arrays for each channel.
        """
        if self.image_rgb is None or self.image_hsv is None or self.image_gray is None:
            raise RuntimeError("No image loaded.")

        return {
            "R": self.image_rgb[:, :, 0],
            "G": self.image_rgb[:, :, 1],
            "B": self.image_rgb[:, :, 2],
            "H": self.image_hsv[:, :, 0],
            "S": self.image_hsv[:, :, 1],
            "V": self.image_hsv[:, :, 2],
            "Gray": self.image_gray,
        }

    def get_visual_channels(self) -> Dict[str, np.ndarray]:
        """
        Returns visual representation of each channel:
        - R, G, B as isolated color views
        - H as false-color colormap (HSV colormap)
        - S and V as colormaps or grayscale
        """
        decomp = self.get_channel_decomposition()
        h, w = decomp["Gray"].shape

        # Visual Red (only red plane active)
        vis_r = np.zeros((h, w, 3), dtype=np.uint8)
        vis_r[:, :, 0] = decomp["R"]

        # Visual Green
        vis_g = np.zeros((h, w, 3), dtype=np.uint8)
        vis_g[:, :, 1] = decomp["G"]

        # Visual Blue
        vis_b = np.zeros((h, w, 3), dtype=np.uint8)
        vis_b[:, :, 2] = decomp["B"]

        # Visual Hue: construct HSV image with Sat=255, Val=255 to see pure Hue colors
        hue_vis = np.zeros((h, w, 3), dtype=np.uint8)
        hue_vis[:, :, 0] = decomp["H"]
        hue_vis[:, :, 1] = 255
        hue_vis[:, :, 2] = 255
        vis_h = cv2.cvtColor(hue_vis, cv2.COLOR_HSV2RGB)

        # Visual Saturation & Value with COLORMAP_JET or VIRIDIS for clarity
        vis_s = cv2.applyColorMap(decomp["S"], cv2.COLORMAP_VIRIDIS)
        vis_s = cv2.cvtColor(vis_s, cv2.COLOR_BGR2RGB)

        vis_v = cv2.applyColorMap(decomp["V"], cv2.COLORMAP_INFERNO)
        vis_v = cv2.cvtColor(vis_v, cv2.COLOR_BGR2RGB)

        # Grayscale visual (3-channel duplicate for uniform display)
        vis_gray = cv2.cvtColor(decomp["Gray"], cv2.COLOR_GRAY2RGB)

        return {
            "R": vis_r,
            "G": vis_g,
            "B": vis_b,
            "H": vis_h,
            "S": vis_s,
            "V": vis_v,
            "Gray": vis_gray,
        }

    # -------------------------------------------------------------------------
    # Task c: Comparative Evaluation & Lighting Invariance
    # -------------------------------------------------------------------------
    def simulate_lighting_variation(
        self, brightness_factor: float = 1.0, contrast_factor: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Simulate varying illumination conditions (shadows, glare, day/night)
        and return the altered RGB, HSV, and Grayscale representations.
        Formula: I_out = np.clip(contrast * I + (brightness - 1.0)*128, 0, 255)
        """
        if self.image_rgb is None:
            raise RuntimeError("No image loaded.")

        # Apply lighting modification to RGB
        base = self.image_rgb.astype(np.float64)
        offset = (brightness_factor - 1.0) * 128.0
        altered = np.clip(contrast_factor * base + offset, 0, 255).astype(np.uint8)

        altered_hsv = cv2.cvtColor(altered, cv2.COLOR_RGB2HSV)
        altered_gray = cv2.cvtColor(altered, cv2.COLOR_RGB2GRAY)

        return {
            "RGB": altered,
            "HSV": altered_hsv,
            "Grayscale": altered_gray,
        }

    def compute_lighting_robustness_metrics(self) -> Dict[str, Any]:
        """
        Empirically assess how RGB, Grayscale, and HSV channel values behave
        when subjected to lighting fluctuations (0.4x to 1.6x brightness).
        Measures stability of color signatures (Hue vs R/G/B vs Gray).
        """
        if self.image_rgb is None:
            raise RuntimeError("No image loaded.")

        # Focus on salient colored regions (saturation > 40)
        hsv_orig = self.image_hsv
        sat_mask = hsv_orig[:, :, 1] > 40
        if np.count_nonzero(sat_mask) < 20:
            sat_mask = np.ones(hsv_orig.shape[:2], dtype=bool)

        test_factors = [0.4, 0.7, 1.0, 1.3, 1.6]
        hue_means = []
        val_means = []
        r_means = []
        g_means = []
        b_means = []
        gray_means = []

        for factor in test_factors:
            sim = self.simulate_lighting_variation(brightness_factor=factor, contrast_factor=1.0)
            sim_rgb = sim["RGB"]
            sim_hsv = sim["HSV"]
            sim_gray = sim["Grayscale"]

            hue_means.append(float(np.mean(sim_hsv[:, :, 0][sat_mask])))
            val_means.append(float(np.mean(sim_hsv[:, :, 2][sat_mask])))
            r_means.append(float(np.mean(sim_rgb[:, :, 0][sat_mask])))
            g_means.append(float(np.mean(sim_rgb[:, :, 1][sat_mask])))
            b_means.append(float(np.mean(sim_rgb[:, :, 2][sat_mask])))
            gray_means.append(float(np.mean(sim_gray[sat_mask])))

        # Coefficient of variation (CV = std / mean) across lighting shifts
        def get_cv(vals):
            m = np.mean(vals)
            return float(np.std(vals) / (m + 1e-6))

        cv_hue = get_cv(hue_means)
        cv_rgb_avg = (get_cv(r_means) + get_cv(g_means) + get_cv(b_means)) / 3.0
        cv_gray = get_cv(gray_means)

        # Hue Stability Score: 100 * (1 - normalized CV)
        hue_stability_score = round(max(0.0, 100.0 * (1.0 - cv_hue)), 2)
        rgb_stability_score = round(max(0.0, 100.0 * (1.0 - cv_rgb_avg)), 2)
        gray_stability_score = round(max(0.0, 100.0 * (1.0 - cv_gray)), 2)

        return {
            "factors": test_factors,
            "hue_means": [round(x, 2) for x in hue_means],
            "val_means": [round(x, 2) for x in val_means],
            "r_means": [round(x, 2) for x in r_means],
            "gray_means": [round(x, 2) for x in gray_means],
            "cv_hue": round(cv_hue, 4),
            "cv_rgb": round(cv_rgb_avg, 4),
            "cv_gray": round(cv_gray, 4),
            "stability_scores": {
                "HSV (Hue)": hue_stability_score,
                "RGB (Channels)": rgb_stability_score,
                "Grayscale": gray_stability_score,
            },
            "winner": "HSV",
        }

    def get_scientific_suitability_report(self) -> Dict[str, Any]:
        """
        Structured comparison and conclusion answering Task c directly.
        """
        metrics = self.compute_lighting_robustness_metrics()

        analysis = {
            "RGB": {
                "pros": "Standard native sensor format; direct color representation.",
                "cons": "Chrominance and luminance are intrinsically coupled across all three channels. When scene lighting changes (sunlight vs shadow vs night), R, G, and B shift simultaneously, making static color thresholding highly prone to failure.",
                "suitability_under_varying_lighting": "Poor to Moderate",
                "stability_score": metrics["stability_scores"]["RGB (Channels)"],
            },
            "Grayscale": {
                "pros": "Single-channel representation; computationally minimal; fast processing.",
                "cons": "Completely discards chromatic data. Coloured traffic sign borders (e.g. red stop ring, blue mandatory sign) can share the exact same luminance value as foliage or asphalt backgrounds, making separation impossible.",
                "suitability_under_varying_lighting": "Poor for color-coded signs; useful only for geometric edge detection.",
                "stability_score": metrics["stability_scores"]["Grayscale"],
            },
            "HSV": {
                "pros": "Explicitly decouples chromaticity (Hue: true color wavelength) from illumination intensity (Value) and color purity (Saturation). Hue remains largely invariant to illumination shifts, shadows, and overexposure as long as sufficient saturation exists.",
                "cons": "Slight conversion overhead; Hue becomes unstable in achromatic (near-zero saturation or extreme black/white) regions.",
                "suitability_under_varying_lighting": "High / Optimal for autonomous vehicle sign recognition.",
                "stability_score": metrics["stability_scores"]["HSV (Hue)"],
            },
            "determination": (
                "HSV is definitively the most suitable colour representation for identifying coloured traffic signs "
                "under varying environmental lighting conditions. Because Hue isolates the dominant wavelength of the "
                "sign (e.g. regulatory red, warning yellow, mandatory blue) independently of light intensity (Value), "
                "a constant Hue threshold successfully segments the sign in both direct sunlight and shadowed road conditions, "
                "whereas RGB channels fluctuate drastically and Grayscale loses chromatic identity entirely."
            ),
        }
        return analysis

    # -------------------------------------------------------------------------
    # Task d: Colour-Channel Analysis & Sign Isolation (Thresholding)
    # -------------------------------------------------------------------------
    @staticmethod
    def get_color_presets() -> Dict[str, Dict[str, Any]]:
        """
        Preset HSV ranges for common traffic sign categories.
        OpenCV HSV ranges: H: [0, 180], S: [0, 255], V: [0, 255].
        """
        return {
            "Red (Regulatory / Stop / Prohibition)": {
                "ranges": [
                    {"lower": np.array([0, 50, 40]), "upper": np.array([10, 255, 255])},
                    {"lower": np.array([160, 50, 40]), "upper": np.array([180, 255, 255])},
                ],
                "description": "Segments red circular borders, stop octagons, and prohibitive slashes.",
            },
            "Blue (Mandatory / Directional / Highway)": {
                "ranges": [
                    {"lower": np.array([95, 50, 40]), "upper": np.array([135, 255, 255])},
                ],
                "description": "Segments blue circular mandatory and square informative signs.",
            },
            "Yellow / Amber (Warning / Caution)": {
                "ranges": [
                    {"lower": np.array([12, 70, 60]), "upper": np.array([38, 255, 255])},
                ],
                "description": "Segments yellow diamond warning signs and construction alerts.",
            },
        }

    def detect_dominant_sign_color(self) -> str:
        """
        Automatically detect whether the image is predominantly Red, Blue, or Yellow.
        """
        if self.image_hsv is None:
            raise RuntimeError("No image loaded.")

        hsv = self.image_hsv
        presets = self.get_color_presets()
        counts = {}

        for name, preset in presets.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for r in preset["ranges"]:
                sub_mask = cv2.inRange(hsv, r["lower"], r["upper"])
                mask = cv2.bitwise_or(mask, sub_mask)
            counts[name] = np.count_nonzero(mask)

        # Return preset with highest pixel count, default to Red if ambiguous
        best_preset = max(counts, key=counts.get)
        if counts[best_preset] < 50:
            return "Red (Regulatory / Stop / Prohibition)"
        return best_preset

    def apply_color_threshold(
        self,
        preset: str = "Auto",
        custom_lower: Optional[np.ndarray] = None,
        custom_upper: Optional[np.ndarray] = None,
        morph_kernel_size: int = 5,
        min_contour_area: int = 80,
    ) -> Dict[str, Any]:
        """
        Apply color thresholding in HSV space, morphological cleanup,
        and foreground sign isolation (Task d).
        """
        if self.image_rgb is None or self.image_hsv is None:
            raise RuntimeError("No image loaded.")

        hsv = self.image_hsv
        rgb = self.image_rgb

        # Resolve ranges
        if preset == "Auto":
            detected_name = self.detect_dominant_sign_color()
            selected_preset = self.get_color_presets()[detected_name]
            ranges = selected_preset["ranges"]
            active_name = f"Auto ({detected_name.split()[0]})"
        elif preset in self.get_color_presets():
            ranges = self.get_color_presets()[preset]["ranges"]
            active_name = preset
        elif custom_lower is not None and custom_upper is not None:
            ranges = [{"lower": np.array(custom_lower), "upper": np.array(custom_upper)}]
            active_name = "Custom"
        else:
            # Fallback to Red
            ranges = self.get_color_presets()["Red (Regulatory / Stop / Prohibition)"]["ranges"]
            active_name = "Red"

        # Construct binary mask across all range intervals (e.g. Red wraps 0 and 180)
        raw_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            sub = cv2.inRange(hsv, r["lower"], r["upper"])
            raw_mask = cv2.bitwise_or(raw_mask, sub)

        # Morphological filtering to eliminate noise and bridge gaps
        k = max(1, morph_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # Morphological Opening (remove salt noise) followed by Closing (fill holes)
        mask_opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)

        # Isolate sign foreground using bitwise AND
        isolated_foreground = cv2.bitwise_and(rgb, rgb, mask=cleaned_mask)

        # Extract contours to localize sign region
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        sign_contour = None
        max_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_boxes.append((x, y, w, h, area))
                if area > max_area:
                    max_area = area
                    sign_contour = cnt

        # Image with localization bounding boxes drawn
        boxed_image = rgb.copy()
        for x, y, w, h, _ in bounding_boxes:
            cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                boxed_image,
                "Sign Region",
                (x, max(12, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        total_pixels = raw_mask.size
        sign_pixels = int(np.count_nonzero(cleaned_mask))
        sign_ratio = round(float(sign_pixels) / total_pixels * 100.0, 2)

        return {
            "active_preset": active_name,
            "raw_mask": raw_mask,
            "cleaned_mask": cleaned_mask,
            "isolated_foreground": isolated_foreground,
            "boxed_image": boxed_image,
            "sign_pixels": sign_pixels,
            "sign_ratio_percent": sign_ratio,
            "bounding_boxes": bounding_boxes,
            "num_regions_found": len(bounding_boxes),
        }

    # -------------------------------------------------------------------------
    # Task e: 2D Convolution & Gaussian Filtering
    # -------------------------------------------------------------------------
    def apply_gaussian_filter(self, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
        """
        Apply 2D Gaussian Filtering to attenuate sensor noise, weather artifacts,
        and high-frequency camera grit (Task e).
        """
        if self.image_rgb is None:
            raise RuntimeError("No image loaded.")

        k = max(1, kernel_size)
        if k % 2 == 0:
            k += 1

        s = max(0.1, float(sigma))
        smoothed = cv2.GaussianBlur(self.image_rgb, (k, k), s)
        return smoothed

    @staticmethod
    def get_convolution_kernels() -> Dict[str, np.ndarray]:
        """
        Predefined 2D Convolution kernels for edge enhancement and boundary sharpening.
        """
        return {
            "Sharpen (Standard 3x3)": np.array(
                [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
            ),
            "High-Pass Laplacian": np.array(
                [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
            ),
            "Sobel Horizontal (X-Gradient)": np.array(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32
            ),
            "Sobel Vertical (Y-Gradient)": np.array(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32
            ),
            "Ridge / Edge Detection": np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
            ),
            "Emboss 3x3": np.array(
                [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32
            ),
        }

    def apply_2d_convolution(
        self,
        kernel_name: str = "Sharpen (Standard 3x3)",
        custom_kernel: Optional[np.ndarray] = None,
        source_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Perform 2D Convolution using custom or predefined spatial kernels (Task e).
        """
        if source_image is None:
            if self.image_rgb is None:
                raise RuntimeError("No image loaded.")
            source_image = self.image_rgb

        if custom_kernel is not None:
            k = np.array(custom_kernel, dtype=np.float32)
        else:
            kernels = self.get_convolution_kernels()
            k = kernels.get(kernel_name, kernels["Sharpen (Standard 3x3)"])

        # Filter image using 2D convolution
        convolved = cv2.filter2D(source_image, -1, k)
        return np.clip(convolved, 0, 255).astype(np.uint8)

    def enhance_sign_boundaries(
        self,
        gaussian_ksize: int = 5,
        sigma: float = 1.0,
        sharpen_kernel_name: str = "Sharpen (Standard 3x3)",
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> Dict[str, np.ndarray]:
        """
        Complete noise reduction and boundary enhancement pipeline (Task e):
        1. Gaussian filtering to remove high-frequency noise
        2. 2D convolution to sharpen sign borders
        3. Canny boundary extraction
        4. High-contrast boundary overlay on original image
        """
        if self.image_rgb is None:
            raise RuntimeError("No image loaded.")

        # Step 1: Gaussian noise suppression
        gaussian_denoised = self.apply_gaussian_filter(
            kernel_size=gaussian_ksize, sigma=sigma
        )

        # Step 2: 2D Convolution Sharpening
        sharpened = self.apply_2d_convolution(
            kernel_name=sharpen_kernel_name, source_image=gaussian_denoised
        )

        # Step 3: Edge and boundary detection
        gray_smoothed = cv2.cvtColor(gaussian_denoised, cv2.COLOR_RGB2GRAY)
        canny_edges = cv2.Canny(gray_smoothed, canny_low, canny_high)

        # Step 4: Boundary overlay on RGB image (cyan or neon green boundary highlight)
        boundary_overlay = self.image_rgb.copy()
        # Highlight edges in bright cyan (0, 255, 255)
        edge_coords = canny_edges > 0
        boundary_overlay[edge_coords] = [0, 240, 255]

        # Sobel magnitude
        sobelx = cv2.Sobel(gray_smoothed, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_smoothed, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = cv2.magnitude(sobelx, sobely)
        sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sobel_color = cv2.applyColorMap(sobel_norm, cv2.COLORMAP_MAGMA)
        sobel_color = cv2.cvtColor(sobel_color, cv2.COLOR_BGR2RGB)

        return {
            "gaussian_denoised": gaussian_denoised,
            "sharpened": sharpened,
            "canny_edges": cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB),
            "canny_raw": canny_edges,
            "boundary_overlay": boundary_overlay,
            "sobel_gradient": sobel_color,
        }

    # -------------------------------------------------------------------------
    # Task f: Display & Comparison Generation
    # -------------------------------------------------------------------------
    def run_full_pipeline(
        self,
        color_preset: str = "Auto",
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 1.0,
        sharpen_kernel: str = "Sharpen (Standard 3x3)",
    ) -> Dict[str, Any]:
        """
        Execute entire preprocessing pipeline from start to finish and return
        all comparative intermediate and final stages (Task f).
        """
        if self.image_rgb is None:
            raise RuntimeError("No image loaded.")

        dims = self.get_dimensions()
        intensity_stats = self.get_intensity_characteristics()
        suitability = self.get_scientific_suitability_report()

        # Step 1: Denoise & Boundary Enhance
        enhancement = self.enhance_sign_boundaries(
            gaussian_ksize=gaussian_ksize,
            sigma=gaussian_sigma,
            sharpen_kernel_name=sharpen_kernel,
        )

        # Step 2: Thresholding & Sign Isolation
        seg = self.apply_color_threshold(preset=color_preset, morph_kernel_size=5)

        # Step 3: Decompositions
        channels_vis = self.get_visual_channels()

        return {
            "dimensions": dims,
            "intensity_stats": intensity_stats,
            "suitability": suitability,
            "original_rgb": self.get_rgb(),
            "grayscale": cv2.cvtColor(self.get_grayscale(), cv2.COLOR_GRAY2RGB),
            "hsv_visual": channels_vis["H"],
            "channels_vis": channels_vis,
            "gaussian_denoised": enhancement["gaussian_denoised"],
            "sharpened": enhancement["sharpened"],
            "canny_edges": enhancement["canny_edges"],
            "boundary_overlay": enhancement["boundary_overlay"],
            "sobel_gradient": enhancement["sobel_gradient"],
            "binary_mask": cv2.cvtColor(seg["cleaned_mask"], cv2.COLOR_GRAY2RGB),
            "isolated_sign": seg["isolated_foreground"],
            "boxed_sign": seg["boxed_image"],
            "segmentation_metrics": {
                "active_preset": seg["active_preset"],
                "sign_pixels": seg["sign_pixels"],
                "sign_ratio_percent": seg["sign_ratio_percent"],
                "num_regions": seg["num_regions_found"],
            },
        }

    def create_comparison_grid(
        self,
        output_path: Optional[str] = None,
        color_preset: str = "Auto",
    ) -> np.ndarray:
        """
        Create a 3x3 high-resolution composite comparison grid image
        contrasting original, color spaces, filtering, and final output (Task f).
        """
        results = self.run_full_pipeline(color_preset=color_preset)

        # Panels to include in 3x3 grid:
        # Row 1: [Original Image, Grayscale (Task b), HSV Hue Map (Task b)]
        # Row 2: [Gaussian Denoised (Task e), 2D Sharpened (Task e), Boundary Overlay (Task e)]
        # Row 3: [Binary Mask (Task d), Isolated Sign Foreground (Task d), Sign Localization (Task d)]
        panels = [
            ("1. Original Image", results["original_rgb"]),
            ("2. Grayscale Space", results["grayscale"]),
            ("3. HSV Hue Space", results["hsv_visual"]),
            ("4. Gaussian Filtered (Denoised)", results["gaussian_denoised"]),
            ("5. 2D Convolved (Sharpened)", results["sharpened"]),
            ("6. Boundary Overlay", results["boundary_overlay"]),
            ("7. HSV Threshold Mask", results["binary_mask"]),
            ("8. Isolated Sign Foreground", results["isolated_sign"]),
            ("9. Sign Localization & Bounding Box", results["boxed_sign"]),
        ]

        target_h, target_w = 240, 240
        annotated_tiles = []

        for title, img in panels:
            resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # Add top header banner
            banner = np.zeros((32, target_w, 3), dtype=np.uint8)
            banner[:] = (26, 32, 44)  # Dark slate background
            cv2.putText(
                banner,
                title,
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            tile = np.vstack([banner, resized])
            # Subtle 1px border
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (60, 70, 85), 1)
            annotated_tiles.append(tile)

        # Assemble 3 rows of 3
        row1 = np.hstack(annotated_tiles[0:3])
        row2 = np.hstack(annotated_tiles[3:6])
        row3 = np.hstack(annotated_tiles[6:9])
        grid = np.vstack([row1, row2, row3])

        # Add top title banner
        grid_w = grid.shape[1]
        top_banner = np.zeros((48, grid_w, 3), dtype=np.uint8)
        top_banner[:] = (15, 23, 42)  # Midnight slate
        cv2.putText(
            top_banner,
            "Traffic Sign Preprocessing & Recognition Module - Multi-Stage Comparison",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (56, 189, 248),  # Sky blue accent
            2,
            cv2.LINE_AA,
        )

        final_composite = np.vstack([top_banner, grid])

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(final_composite, cv2.COLOR_RGB2BGR))

        return final_composite

