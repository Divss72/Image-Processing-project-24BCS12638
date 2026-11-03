import os
import cv2
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from video_sign_detector import VideoSignDetector

def main():
    input_video_path = r"C:\Users\raodi\OneDrive\Desktop\Traffic data\384582.mp4"
    output_video_path = "outputs/processed_uploaded_dashcam.mp4"

    if not os.path.exists(input_video_path):
        print(f"Error: {input_video_path} does not exist.")
        return

    print("=" * 70)
    print("PROCESSING USER UPLOADED VIDEO")
    print(f"Source: {input_video_path}")
    print(f"Target Output: {output_video_path}")
    print("=" * 70)

    detector = VideoSignDetector()

    # Process first 300 frames (~10 seconds) for fast verification, or full video
    # Let's check total frames
    cap = cv2.VideoCapture(input_video_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    max_frames_to_process = 300 # ~10 seconds of high-speed video
    print(f"Total video frames: {total_f} ({total_f/fps:.1f}s). Processing {max_frames_to_process} frames...")

    def progress(current, total):
        if current % 30 == 0 or current == total:
            pct = (current / total) * 100
            print(f"  -> Progress: frame {current}/{total} ({pct:.1f}%)")

    results = detector.process_video_file(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        speed_kmh=80.0,
        max_frames=max_frames_to_process,
        target_resolution=(960, 540), # 540p for optimal balance of speed and sharpness
        progress_callback=progress
    )

    print("\n" + "=" * 70)
    print("UPLOADED VIDEO PROCESSING COMPLETED")
    print("=" * 70)
    print(f"Frames Processed       : {results['total_frames_processed']}")
    print(f"Processed Resolution   : {results['processed_resolution']}")
    print(f"Mean Latency / Frame   : {results['mean_latency_ms']:.2f} ms")
    print(f"Processing Throughput  : {results['average_fps']:.1f} FPS")
    print(f"Signs Detected Count   : {results['total_signs_detected']}")
    print(f"Saved Output Video     : {output_video_path}")
    print("-" * 70)

    # Save detailed JSON report
    report_path = "outputs/uploaded_video_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved metrics report to: {report_path}")

    # Generate 4-frame visualization montage
    cap = cv2.VideoCapture(output_video_path)
    n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = [int(n_out * r) for r in [0.15, 0.40, 0.65, 0.90]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame_rgb)
            axes[i].set_title(f"Uploaded Video Frame #{idx} (HUD & Detections)", color="#38BDF8", fontsize=10)
            axes[i].axis("off")
    cap.release()

    montage_path = "outputs/uploaded_video_detections_montage.png"
    plt.suptitle("Autonomous Vehicle Traffic Sign Recognition on Uploaded Video", color="#F8FAFC", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(montage_path, dpi=180, facecolor="#0F172A")
    plt.close()
    print(f"[+] Saved detection montage to: {montage_path}")

if __name__ == "__main__":
    main()

