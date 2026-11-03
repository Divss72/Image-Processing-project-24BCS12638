import cv2
import os
import json
import time
from video_sign_detector import VideoSignDetector

def inspect_video():
    video_path = r"C:\Users\raodi\OneDrive\Desktop\Traffic data\384582.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print("=" * 65)
    print("UPLOADED VIDEO INSPECTION")
    print("=" * 65)
    print(f"File Path   : {video_path}")
    print(f"Resolution  : {w} x {h} (4K UHD)")
    print(f"Frame Rate  : {fps:.2f} FPS")
    print(f"Total Frames: {total_frames}")
    print(f"Duration    : {duration_sec:.2f} seconds ({duration_sec / 60:.2f} mins)")
    print("=" * 65)

    detector = VideoSignDetector()
    os.makedirs("outputs/uploaded_video_frames", exist_ok=True)

    # Sample across various sections of the video
    check_frames = [50, 200, 450, 700, 1000, 1250]
    results = []

    for f_idx in check_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Standard processing resolution: 960x540 (fast, high detail)
        proc_frame = cv2.resize(frame, (960, 540))
        t0 = time.time()
        annotated, dets, lat = detector.process_frame(proc_frame, estimated_speed_kmh=65.0)
        dt = (time.time() - t0) * 1000.0

        out_img_path = f"outputs/uploaded_video_frames/annotated_frame_{f_idx:04d}.jpg"
        cv2.imwrite(out_img_path, annotated)

        print(f"\nFrame #{f_idx} (Time: {f_idx/fps:.2f}s) - {len(dets)} sign candidates detected (Latency: {dt:.1f}ms):")
        for i, d in enumerate(dets[:5]):
            b = d["box"]
            print(f"  [{i+1}] {d['class_name']} (Class {d['class_id']}) | Conf: {d['confidence']:.1%} | Bounding Box: [x={b[0]}, y={b[1]}, w={b[2]}, h={b[3]}]")
        
        results.append({
            "frame": f_idx,
            "timestamp_sec": round(f_idx / fps, 2),
            "detections_count": len(dets),
            "latency_ms": round(dt, 2),
            "top_detections": dets[:5]
        })

    cap.release()

    with open("outputs/uploaded_video_frames/inspection_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "video_path": video_path,
            "resolution": f"{w}x{h}",
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": duration_sec,
            "sampled_frames": results
        }, f, indent=2)

    print("\n[+] Inspection summary saved to outputs/uploaded_video_frames/inspection_summary.json")

if __name__ == "__main__":
    inspect_video()

