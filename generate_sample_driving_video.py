"""
Generates a realistic speeding vehicle dashboard video (MP4)
with roadside traffic signs approaching and passing by at 85 km/h.
Used to test video stream detection immediately.
"""

import os
import glob
import cv2
import numpy as np


def generate_speeding_dashcam_video(
    output_path: str = "outputs/sample_speeding_dashcam.mp4",
    duration_sec: int = 10,
    fps: int = 30,
    width: int = 640,
    height: int = 360,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    total_frames = duration_sec * fps

    # Find sample traffic signs from dataset
    sign_paths = [
        "Traffic data/DATA/0/000_0001.png",   # Speed limit 5km/h
        "Traffic data/DATA/21/021.png",       # Go Straight (Blue)
        "Traffic data/DATA/34/034_0001.png",  # Danger Ahead (Warning)
    ]
    sign_imgs = []
    for p in sign_paths:
        if os.path.exists(p):
            sign_imgs.append(cv2.imread(p))
    if not sign_imgs:
        any_p = glob.glob("Traffic data/DATA/*/*.png")[:3]
        for p in any_p:
            sign_imgs.append(cv2.imread(p))

    # Horizon line and vanishing point
    vp_x, vp_y = width // 2, int(height * 0.45)

    # Road coordinates at bottom
    road_left_bottom = int(width * 0.15)
    road_right_bottom = int(width * 0.85)

    print(f"[*] Generating {duration_sec}s speeding vehicle dashcam video ({width}x{height} @ {fps}fps)...")

    # Sequence of 3 signs appearing along right roadside
    # Sign 1: frames 20 to 80
    # Sign 2: frames 100 to 170
    # Sign 3: frames 190 to 260
    sign_events = [
        {"img": sign_imgs[0] if len(sign_imgs) > 0 else None, "start": 20, "end": 85},
        {"img": sign_imgs[1] if len(sign_imgs) > 1 else None, "start": 105, "end": 175},
        {"img": sign_imgs[2] if len(sign_imgs) > 2 else None, "start": 195, "end": 265},
    ]

    for f_idx in range(total_frames):
        # 1. Base frame: Sky and terrain
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky gradient (top to horizon)
        for y in range(vp_y):
            ratio = y / float(vp_y)
            frame[y, :] = [int(190 - 40 * ratio), int(150 - 30 * ratio), int(100 - 20 * ratio)] # Twilight blue/gray
        # Ground (asphalt and roadside grass)
        for y in range(vp_y, height):
            frame[y, :] = [45, 65, 40] # Roadside foliage

        # 2. Road surface polygon
        road_pts = np.array([
            [vp_x - 15, vp_y],
            [vp_x + 15, vp_y],
            [road_right_bottom, height],
            [road_left_bottom, height]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [road_pts], (55, 55, 60)) # Dark asphalt

        # Road edges
        cv2.line(frame, (vp_x - 15, vp_y), (road_left_bottom, height), (220, 220, 220), 2)
        cv2.line(frame, (vp_x + 15, vp_y), (road_right_bottom, height), (220, 220, 220), 2)

        # 3. Moving dashed center line to simulate high speed
        dash_speed_offset = (f_idx * 16) % 90
        for d in range(0, 400, 90):
            d_pos = d + dash_speed_offset
            if d_pos > 300:
                continue
            ratio = (d_pos / 300.0) ** 1.8
            cy = int(vp_y + (height - vp_y) * ratio)
            dash_len = max(4, int(28 * ratio))
            if cy + dash_len < height:
                cv2.line(frame, (vp_x, cy), (vp_x, cy + dash_len), (245, 245, 245), max(1, int(3 * ratio)))

        # 4. Roadside Signs Approaching
        for ev in sign_events:
            if ev["img"] is not None and ev["start"] <= f_idx <= ev["end"]:
                progress = (f_idx - ev["start"]) / float(ev["end"] - ev["start"])
                # Non-linear perspective scaling: grows rapidly as it gets close
                scale = (progress ** 2.2)
                cur_size = int(18 + scale * 95) # 18px to 113px
                if cur_size < 12:
                    continue

                # Roadside right-hand coordinate
                sign_x = int(vp_x + 35 + progress * (width * 0.42))
                sign_y = int(vp_y + 10 + progress * (height * 0.38) - cur_size)

                # Draw signpost pole
                pole_bottom = int(vp_y + 10 + progress * (height * 0.38))
                cv2.line(frame, (sign_x + cur_size // 2, sign_y + cur_size), (sign_x + cur_size // 2, pole_bottom), (90, 90, 95), max(1, int(cur_size * 0.06)))

                # Resize sign image
                resized_sign = cv2.resize(ev["img"], (cur_size, cur_size), interpolation=cv2.INTER_AREA)

                # Add realistic directional speed motion blur at high progress
                if progress > 0.4:
                    mb_len = max(3, int(progress * 7))
                    if mb_len % 2 == 0:
                        mb_len += 1
                    k_mb = np.zeros((mb_len, mb_len), dtype=np.float32)
                    k_mb[mb_len // 2, :] = 1.0 / mb_len
                    resized_sign = cv2.filter2D(resized_sign, -1, k_mb)

                # Paste onto frame
                if 0 <= sign_y and sign_y + cur_size < height and 0 <= sign_x and sign_x + cur_size < width:
                    frame[sign_y:sign_y+cur_size, sign_x:sign_x+cur_size] = resized_sign

        # 5. Dashboard / Windshield Border at bottom
        cv2.rectangle(frame, (0, height - 16), (width, height), (20, 22, 25), -1)

        writer.write(frame)

    writer.release()
    print(f"[+] Successfully generated sample speeding vehicle dashcam video: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    return output_path


if __name__ == "__main__":
    generate_speeding_dashcam_video()

