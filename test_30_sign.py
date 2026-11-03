import cv2
import numpy as np
from predict import TrafficSignPredictor
from robustness_engine import RobustnessEngine

def test_crop():
    img = cv2.imread("Traffic data/image.png")
    # Coordinates of 30 sign in image.png
    # Let's inspect where it is exactly
    crop_30 = img[70:150, 290:370]
    cv2.imwrite("outputs/crop_manual_30.png", crop_30)

    predictor = TrafficSignPredictor()
    res = predictor.predict_image(crop_30, is_rgb=False, top_k=5)

    print("Raw Crop Prediction:")
    print(f"  Class: {res['predicted_class_id']} - {res['predicted_class_name']} ({res['confidence_percent']}%)")
    for c in res['top_candidates']:
        print(f"    Top: {c['class_id']} - {c['class_name']} ({c['confidence_percent']}%)")

    # Now let's apply autonomous vision restoration (fog dehaze + CLAHE contrast boost)
    crop_rgb = cv2.cvtColor(crop_30, cv2.COLOR_BGR2RGB)
    restored_rgb = RobustnessEngine.autonomous_restore(crop_rgb)
    cv2.imwrite("outputs/crop_restored_30.png", cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR))

    res_restored = predictor.predict_image(restored_rgb, is_rgb=True, top_k=5)
    print("\nRestored Crop Prediction (De-hazed & Enhanced):")
    print(f"  Class: {res_restored['predicted_class_id']} - {res_restored['predicted_class_name']} ({res_restored['confidence_percent']}%)")
    for c in res_restored['top_candidates']:
        print(f"    Top: {c['class_id']} - {c['class_name']} ({c['confidence_percent']}%)")

if __name__ == "__main__":
    test_crop()

