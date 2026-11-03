"""
Autonomous Vehicle Vision - Real-Time Video Stream Processor
============================================================
Processes speeding vehicle dashboard/windshield video streams:
  1. Detects traffic sign candidates in each frame using color-space ROI localization.
  2. Preprocesses candidate regions (motion blur compensation, dust penetration, Gaussian filtering).
  3. Classifies signs in real-time using the trained machine learning model.
  4. Renders autonomous vehicle HUD overlays (Bounding boxes, telemetry, latency, FPS).
"""

import os
import time
from typing import Dict, Any, List, Tuple, Optional, Generator
import cv2
import numpy as np
from predict import TrafficSignPredictor
from robustness_engine import RobustnessEngine


class VideoSignDetector:
    """Real-time traffic sign detector and classifier for vehicle dashcam video streams."""

    def __init__(self, predictor: Optional[TrafficSignPredictor] = None):
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = TrafficSignPredictor()
            self.predictor.model.n_jobs = 1

        # Preset HSV ranges for candidate localization
        self.color_ranges = [
            # Red (Regulatory / Speed limits / Stop / Danger)
            {"name": "Red", "lower": np.array([0, 60, 50]), "upper": np.array([12, 255, 255])},
            {"name": "Red_wrap", "lower": np.array([160, 60, 50]), "upper": np.array([180, 255, 255])},
            # Blue (Mandatory / Informative)
            {"name": "Blue", "lower": np.array([95, 60, 50]), "upper": np.array([135, 255, 255])},
            # Yellow (Warning)
            {"name": "Yellow", "lower": np.array([14, 80, 70]), "upper": np.array([36, 255, 255])},
        ]

    def find_sign_candidates(self, frame_bgr: np.ndarray, min_area: int = 160) -> List[Tuple[int, int, int, int]]:
        """
        Extract regions of interest (ROI) where traffic signs are likely located.
        Filters by aspect ratio and geometric solidity.
        """
        h, w = frame_bgr.shape[:2]
        # Restrict search to upper-right and center-right quadrants where roadside signs appear
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Multi-color mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in self.color_ranges:
            sub = cv2.inRange(hsv, r["lower"], r["upper"])
            mask = cv2.bitwise_or(mask, sub)

        # Morphological noise cleanup
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > (h * w * 0.4):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / float(bh)
            if 0.55 <= aspect <= 1.8:
                # Add slight padding around sign
                pad_x = int(bw * 0.12)
                pad_y = int(bh * 0.12)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                candidates.append((x1, y1, x2 - x1, y2 - y1))

        # Merge overlapping bounding boxes
        return self._non_max_suppression(candidates)

    @staticmethod
    def _non_max_suppression(boxes: List[Tuple[int, int, int, int]], overlap_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []
        boxes_arr = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(areas)[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= overlap_thresh)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def process_frame(self, frame_bgr: np.ndarray, estimated_speed_kmh: float = 85.0) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Process a single video frame: detects, pre集中, classifies, and draws HUD.
        Returns: (annotated_frame, list_of_detections, latency_ms)
        """
        t0 = time.time()
        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]

        candidates = self.find_sign_candidates(frame_bgr)
        detections = []

        for x, y, bw, bh in candidates:
            crop = frame_bgr[y:y+bh, x:x+bw]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Apply motion and environmental restoration if moving fast
            if estimated_speed_kmh > 40:
                crop_restored = RobustnessEngine.autonomous_restore(crop_rgb)
            else:
                crop_restored = crop_rgb

            pred = self.predictor.predict_image(crop_restored, is_rgb=True, top_k=1)
            conf = pred["confidence_percent"]

            # Filter out low-confidence false positives
            if conf >= 30.0:
                det = {
                    "box": (x, y, bw, bh),
                    "class_id": pred["predicted_class_id"],
                    "class_name": pred["predicted_class_name"],
                    "confidence": conf,
                    "crop_rgb": crop_restored,
                }
                detections.append(det)

                # Draw bounding box
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 240, 255), 2)

                # Draw label banner
                label = f"{det['class_name'][:22]} ({conf:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x, max(0, y - th - 8)), (x + tw + 8, y), (15, 23, 42), -1)
                cv2.rectangle(annotated, (x, max(0, y - th - 8)), (x + tw + 8, y), (0, 240, 255), 1)
                cv2.putText(
                    annotated,
                    label,
                    (x + 4, max(th + 2, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )

        latency_ms = (time.time() - t0) * 1000.0
        fps = 1000.0 / max(0.1, latency_ms)

        # Render Autonomous Vehicle Telemetry HUD Header
        cv2.rectangle(annotated, (0, 0), (w, 36), (15, 23, 42), -1)
        cv2.line(annotated, (0, 36), (w, 36), (56, 189, 248), 1)

        hud_text = (
            f"AUTONOMOUS VISION | Speed: {estimated_speed_kmh:.0f} km/h | "
            f"Latency: {latency_ms:.1f}ms ({fps:.1f} FPS) | Signs in View: {len(detections)}"
        )
        cv2.putText(
            annotated,
            hud_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (56, 189, 248),
            1,
            cv2.LINE_AA,
        )

        return annotated, detections, latency_ms

    def process_video_file(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        speed_kmh: float = 85.0,
        max_frames: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Process an input video file and write an annotated output video.
        Supports automatic resolution scaling for 4K/UHD videos to maintain real-time performance.
        """
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Optimize resolution: if > 1280 wide and target_resolution not set, scale to 720p
        if target_resolution:
            out_w, out_h = target_resolution
        elif orig_w > 1280:
            out_w = 1280
            out_h = int(round(orig_h * (1280.0 / orig_w)))
        else:
            out_w, out_h = orig_w, orig_h

        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps_in, (out_w, out_h))

        all_detections = []
        latencies = []
        frame_idx = 0

        t_start = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            if (orig_w, orig_h) != (out_w, out_h):
                frame_to_proc = cv2.resize(frame, (out_w, out_h))
            else:
                frame_to_proc = frame

            ann, dets, lat = self.process_frame(frame_to_proc, estimated_speed_kmh=speed_kmh)
            latencies.append(lat)
            if dets:
                for d in dets:
                    d_copy = {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps_in, 2),
                        "box": d["box"],
                        "class_id": d["class_id"],
                        "class_name": d["class_name"],
                        "confidence": d["confidence"],
                    }
                    all_detections.append(d_copy)

            if writer:
                writer.write(ann)

            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, frames_to_process)

        cap.release()
        if writer:
            writer.release()

        t_total = time.time() - t_start
        mean_lat = float(np.mean(latencies)) if latencies else 0.0
        avg_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

        return {
            "input_video": input_video_path,
            "output_video": output_video_path,
            "original_resolution": f"{orig_w}x{orig_h}",
            "processed_resolution": f"{out_w}x{out_h}",
            "total_frames_processed": frame_idx,
            "total_signs_detected": len(all_detections),
            "mean_latency_ms": round(mean_lat, 2),
            "average_fps": round(avg_fps, 1),
            "total_time_sec": round(t_total, 2),
            "detections": all_detections,
        }

