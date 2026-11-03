"""
Unit test suite for TrafficSignProcessor (Tasks a to f).
Run with: pytest test_processor.py -v
"""

import os
import glob
import pytest
import numpy as np
from traffic_sign_processor import TrafficSignProcessor


@pytest.fixture(scope="session")
def sample_image_path():
    """Find an existing sample traffic sign image from the dataset."""
    candidates = glob.glob("Traffic data/TEST/*/*.png")
    assert len(candidates) > 0, "No sample images found in Traffic data/DATA"
    return candidates[0]


@pytest.fixture
def processor(sample_image_path):
    return TrafficSignProcessor(sample_image_path)


# -------------------------------------------------------------------------
# Task a Tests: Image Loading, Dimensions, Intensity Characteristics
# -------------------------------------------------------------------------
def test_load_image_and_dimensions(processor):
    dims = processor.get_dimensions()
    assert "height" in dims and dims["height"] > 0
    assert "width" in dims and dims["width"] > 0
    assert dims["channels"] == 3
    assert dims["aspect_ratio"] > 0.0
    assert dims["total_pixels"] == dims["height"] * dims["width"]
    assert dims["dtype"] == "uint8"
    assert dims["bit_depth"] == 8


def test_intensity_characteristics(processor):
    stats = processor.get_intensity_characteristics()
    assert 0 <= stats["min"] <= stats["max"] <= 255
    assert 0 <= stats["mean"] <= 255
    assert stats["std"] >= 0
    assert stats["rms_contrast"] >= 0
    assert stats["perceived_brightness"] >= 0
    assert stats["entropy"] > 0
    assert "rgb_channels" in stats
    assert "hsv_channels" in stats
    assert "Red" in stats["rgb_channels"]
    assert "Hue" in stats["hsv_channels"]


# -------------------------------------------------------------------------
# Task b Tests: Colour Space Conversions & Decomposition
# -------------------------------------------------------------------------
def test_colour_space_conversions(processor):
    rgb = processor.get_rgb()
    hsv = processor.get_hsv()
    gray = processor.get_grayscale()

    assert rgb.shape == hsv.shape
    assert len(gray.shape) == 2
    assert rgb.shape[:2] == gray.shape

    decomp = processor.get_channel_decomposition()
    for ch in ["R", "G", "B", "H", "S", "V", "Gray"]:
        assert ch in decomp
        assert decomp[ch].shape == gray.shape

    vis = processor.get_visual_channels()
    for ch in ["R", "G", "B", "H", "S", "V", "Gray"]:
        assert ch in vis
        assert vis[ch].shape == rgb.shape


# -------------------------------------------------------------------------
# Task c Tests: Lighting Variation & Suitability Analysis
# -------------------------------------------------------------------------
def test_lighting_variation_simulation(processor):
    dim_sim = processor.simulate_lighting_variation(brightness_factor=0.5)
    bright_sim = processor.simulate_lighting_variation(brightness_factor=1.5)

    assert np.mean(bright_sim["RGB"]) > np.mean(dim_sim["RGB"])
    assert dim_sim["RGB"].shape == processor.get_rgb().shape


def test_lighting_robustness_metrics(processor):
    metrics = processor.compute_lighting_robustness_metrics()
    assert "stability_scores" in metrics
    assert "HSV (Hue)" in metrics["stability_scores"]
    assert "RGB (Channels)" in metrics["stability_scores"]
    assert "Grayscale" in metrics["stability_scores"]

    # HSV stability score should exceed RGB under varying lighting
    assert metrics["stability_scores"]["HSV (Hue)"] >= metrics["stability_scores"]["RGB (Channels)"]
    assert metrics["winner"] == "HSV"


def test_scientific_suitability_report(processor):
    report = processor.get_scientific_suitability_report()
    assert "RGB" in report
    assert "HSV" in report
    assert "Grayscale" in report
    assert "determination" in report
    assert "HSV is definitively the most suitable" in report["determination"]


# -------------------------------------------------------------------------
# Task d Tests: Thresholding & Sign Isolation
# -------------------------------------------------------------------------
def test_thresholding_and_isolation(processor):
    seg = processor.apply_color_threshold(preset="Auto", morph_kernel_size=5)

    assert "cleaned_mask" in seg
    assert "isolated_foreground" in seg
    assert "boxed_image" in seg
    assert seg["cleaned_mask"].shape == processor.get_grayscale().shape
    assert seg["isolated_foreground"].shape == processor.get_rgb().shape
    assert 0 <= seg["sign_ratio_percent"] <= 100.0


def test_presets(processor):
    for preset in ["Red (Regulatory / Stop / Prohibition)", "Blue (Mandatory / Directional / Highway)"]:
        res = processor.apply_color_threshold(preset=preset)
        assert res["cleaned_mask"] is not None


# -------------------------------------------------------------------------
# Task e Tests: 2D Convolution & Gaussian Filtering
# -------------------------------------------------------------------------
def test_gaussian_filter(processor):
    smooth_3 = processor.apply_gaussian_filter(kernel_size=3, sigma=1.0)
    smooth_7 = processor.apply_gaussian_filter(kernel_size=7, sigma=2.0)

    assert smooth_3.shape == processor.get_rgb().shape
    assert smooth_7.shape == processor.get_rgb().shape


def test_2d_convolution(processor):
    kernels = processor.get_convolution_kernels()
    for name in kernels:
        conv = processor.apply_2d_convolution(kernel_name=name)
        assert conv.shape == processor.get_rgb().shape


def test_boundary_enhancement(processor):
    enh = processor.enhance_sign_boundaries()
    assert "gaussian_denoised" in enh
    assert "sharpened" in enh
    assert "canny_edges" in enh
    assert "boundary_overlay" in enh
    assert "sobel_gradient" in enh
    assert enh["boundary_overlay"].shape == processor.get_rgb().shape


# -------------------------------------------------------------------------
# Task f Tests: Comparison Grid & Pipeline
# -------------------------------------------------------------------------
def test_full_pipeline_and_comparison_grid(processor, tmp_path):
    output_img = str(tmp_path / "comparison_grid.png")
    grid = processor.create_comparison_grid(output_path=output_img)

    assert os.path.exists(output_img)
    assert grid.ndim == 3
    assert grid.shape[2] == 3

    pipeline = processor.run_full_pipeline()
    assert "original_rgb" in pipeline
    assert "grayscale" in pipeline
    assert "hsv_visual" in pipeline
    assert "gaussian_denoised" in pipeline
    assert "sharpened" in pipeline
    assert "binary_mask" in pipeline
    assert "isolated_sign" in pipeline

