import cv2
from predict import TrafficSignPredictor

p = TrafficSignPredictor()
for fname in ["outputs/crop_tight_30.png", "outputs/crop_copy_30.png"]:
    img = cv2.imread(fname)
    res = p.predict_image(img, is_rgb=False, top_k=5)
    print(fname, "->", res["predicted_class_id"], res["predicted_class_name"], f"{res['confidence_percent']}%")
    for c in res["top_candidates"]:
        print("   top:", c["class_id"], c["class_name"], f"{c['confidence_percent']}%")

