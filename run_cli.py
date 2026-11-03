"""
Traffic Sign Recognition Preprocessing CLI & Batch Runner
=========================================================
Runs tasks a to f from the command line, generating comparison figures
and scientific reports.

Usage:
  python run_cli.py --image "Traffic data/DATA/0/000_0001.png"
  python run_cli.py --demo
  python run_cli.py --batch 10
"""

import os
import sys
import glob
import json
import argparse
from typing import List
from traffic_sign_processor import TrafficSignProcessor


def print_banner():
    banner = """
========================================================================
    TRAFFIC SIGN PREPROCESSING & RECOGNITION SYSTEM (Tasks a - f)
========================================================================
Autonomous Vehicle Vision Preprocessing Module
Color Space Conversion | Noise Reduction | Boundary Enhancement
========================================================================
"""
    print(banner)


def process_single_image(image_path: str, output_dir: str = "outputs", preset: str = "Auto") -> dict:
    print(f"\n[*] Processing image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return {}

    processor = TrafficSignProcessor(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Task a: Dimensions & Intensity Characteristics
    # -----------------------------------------------------------------
    dims = processor.get_dimensions()
    stats = processor.get_intensity_characteristics()

    print("\n" + "-" * 60)
    print("TASK A: IMAGE DIMENSIONS & INTENSITY CHARACTERISTICS")
    print("-" * 60)
    print(f"Dimensions           : {dims['width']} x {dims['height']} (Width x Height)")
    print(f"Channels             : {dims['channels']} (Bit Depth: {dims['bit_depth']}-bit, {dims['dtype']})")
    print(f"Aspect Ratio         : {dims['aspect_ratio']}")
    print(f"Total Pixels         : {dims['total_pixels']:,} px")
    print(f"Memory Footprint     : {dims['memory_bytes'] / 1024:.2f} KB")
    print("\nIntensity Characteristics:")
    print(f"  Mean Intensity     : {stats['mean']}")
    print(f"  Std Deviation (RMS): {stats['std']}")
    print(f"  Median Intensity   : {stats['median']}")
    print(f"  Dynamic Range      : {stats['min']} to {stats['max']} (Range = {stats['dynamic_range']})")
    print(f"  Perceived Lightness: {stats['perceived_brightness']} (ITU-R BT.601 standard)")
    print(f"  Shannon Entropy    : {stats['entropy']} bits")
    print("  Channel Distributions:")
    for ch, ch_s in stats["rgb_channels"].items():
        print(f"    - {ch:5s} : Mean={ch_s['mean']}, Std={ch_s['std']}, Range=[{ch_s['min']}, {ch_s['max']}]")

    # -----------------------------------------------------------------
    # Task b: Colour Space Conversions
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TASK B: COLOUR SPACE CONVERSIONS (RGB, HSV, Grayscale)")
    print("-" * 60)
    rgb = processor.get_rgb()
    hsv = processor.get_hsv()
    gray = processor.get_grayscale()
    decomp = processor.get_channel_decomposition()
    print(f"RGB Shape            : {rgb.shape} (Red, Green, Blue 8-bit planes)")
    print(f"HSV Shape            : {hsv.shape} (Hue [0-180], Saturation [0-255], Value [0-255])")
    print(f"Grayscale Shape      : {gray.shape} (Luminance Y plane)")
    print(f"Individual Channels  : {list(decomp.keys())}")

    # -----------------------------------------------------------------
    # Task c: Comparative Suitability Analysis Under Varying Lighting
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TASK C: COLOUR REPRESENTATION COMPARISON & LIGHTING SUITABILITY")
    print("-" * 60)
    suitability = processor.get_scientific_suitability_report()
    robustness = processor.compute_lighting_robustness_metrics()

    print("Stability Scores Under Simulated Illumination Variations (0.4x - 1.6x brightness):")
    for space, score in robustness["stability_scores"].items():
        bar = "#" * int(score / 4)
        print(f"  {space:18s}: [{bar:<25}] {score}% stability")

    print(f"\nRecommended Space    : {robustness['winner']} (Winner)")
    print("\nScientific Determination:")
    print(f"  {suitability['determination']}")

    # -----------------------------------------------------------------
    # Task d: Colour-Channel Analysis & Sign Isolation (Thresholding)
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TASK D: COLOUR-CHANNEL ANALYSIS & SIGN ISOLATION")
    print("-" * 60)
    seg = processor.apply_color_threshold(preset=preset)
    print(f"Active Threshold Preset : {seg['active_preset']}")
    print(f"Sign Pixels Isolated    : {seg['sign_pixels']:,} px ({seg['sign_ratio_percent']}% of image)")
    print(f"Candidate Sign Regions  : {seg['num_regions_found']} contour region(s)")
    for idx, (x, y, w, h, area) in enumerate(seg["bounding_boxes"][:3], 1):
        print(f"  Region #{idx}: BoundingBox=(x={x}, y={y}, w={w}, h={h}), Area={area:.1f} px^2")

    # -----------------------------------------------------------------
    # Task e: 2D Convolution & Gaussian Filtering
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TASK E: 2D CONVOLUTION & GAUSSIAN FILTERING")
    print("-" * 60)
    enh = processor.enhance_sign_boundaries(
        gaussian_ksize=5, sigma=1.0, sharpen_kernel_name="Sharpen (Standard 3x3)"
    )
    print("Gaussian Filter      : Applied 5x5 kernel (sigma=1.0) -> Noise reduced")
    print("2D Convolution       : Applied 3x3 Laplacian Sharpening -> Boundaries sharpened")
    print("Boundary Extraction  : Canny edge detector + Sobel gradient magnitude overlay")

    # -----------------------------------------------------------------
    # Task f: Display & Compare (Generate Comparison Grid)
    # -----------------------------------------------------------------
    print("\n" + "-" * 60)
    print("TASK F: MULTI-VIEW COMPARISON GRID EXPORT")
    print("-" * 60)
    grid_filename = f"comparison_{base_name}.png"
    grid_path = os.path.join(output_dir, grid_filename)
    processor.create_comparison_grid(output_path=grid_path, color_preset=preset)
    print(f"[+] Saved high-resolution 3x3 comparison grid: {grid_path}")

    # Save summary report JSON
    report_data = {
        "image_path": image_path,
        "dimensions": dims,
        "intensity_characteristics": stats,
        "lighting_robustness": robustness,
        "suitability_determination": suitability["determination"],
        "segmentation": {
            "preset": seg["active_preset"],
            "sign_pixels": seg["sign_pixels"],
            "sign_ratio_percent": seg["sign_ratio_percent"],
            "bounding_boxes": [
                {"x": b[0], "y": b[1], "width": b[2], "height": b[3], "area": b[4]}
                for b in seg["bounding_boxes"]
            ],
        },
        "output_grid_path": grid_path,
    }
    json_path = os.path.join(output_dir, f"report_{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"[+] Saved JSON report: {json_path}")

    return report_data


def run_demo():
    """Run a rich demo across diverse classes (Red speed limit, Red danger, Blue direction)."""
    sample_targets = [
        ("Traffic data/DATA/0/000_0001.png", "Auto"),
        ("Traffic data/DATA/21/021.png", "Auto"),
        ("Traffic data/DATA/14/014_0001.png", "Auto"),
        ("Traffic data/DATA/34/034_0001.png", "Auto"),
    ]

    print("\nRunning Multi-Class Traffic Sign Recognition Preprocessing Demo...")
    results = []
    for path, preset in sample_targets:
        if os.path.exists(path):
            r = process_single_image(path, output_dir="outputs", preset=preset)
            results.append(r)
        else:
            # Fallback search
            cls_folder = os.path.dirname(path)
            found = glob.glob(os.path.join(cls_folder, "*.png"))
            if found:
                r = process_single_image(found[0], output_dir="outputs", preset=preset)
                results.append(r)

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Demo completed! Processed {len(results)} sample traffic signs.")
    print("All comparison grids and reports have been generated in ./outputs/")
    print("=" * 60)


def run_batch(count: int = 5):
    """Process a batch of images from the dataset."""
    all_images = glob.glob("Traffic data/DATA/*/*.png")
    selected = all_images[:count]
    print(f"\nRunning batch preprocessing on {len(selected)} traffic sign images...")

    for idx, img_path in enumerate(selected, 1):
        print(f"\n[{idx}/{len(selected)}] Processing {img_path}")
        process_single_image(img_path, output_dir="outputs/batch", preset="Auto")

    print(f"\n[+] Batch processing finished. Results saved to ./outputs/batch/")


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Preprocessing CLI")
    parser.add_argument("--image", "-i", type=str, help="Path to single traffic sign image")
    parser.add_argument("--preset", "-p", type=str, default="Auto", help="Color threshold preset (Auto, Red, Blue, Yellow)")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory for results")
    parser.add_argument("--demo", "-d", action="store_true", help="Run full multi-class demonstration")
    parser.add_argument("--batch", "-b", type=int, help="Run batch processing on N images")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.batch:
        run_batch(args.batch)
    elif args.image:
        process_single_image(args.image, output_dir=args.output, preset=args.preset)
    else:
        # Default to demo if no args provided
        run_demo()


if __name__ == "__main__":
    main()

