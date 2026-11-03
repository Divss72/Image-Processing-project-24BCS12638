import glob
import os
import cv2
import numpy as np

def test_digit_features():
    # Let's inspect tight crop of 30 sign in image.png
    img = cv2.imread("Traffic data/image.png")
    # Circle at (x=329, y=116, r=27) -> tight box:
    c_x, c_y, r = 329, 116, 27
    crop_tight = img[c_y-r:c_y+r, c_x-r:c_x+r]
    cv2.imwrite("outputs/crop_tight_30.png", crop_tight)

    # Let's also inspect crop in image copy.png (the bottom circular sign)
    img_copy = cv2.imread("Traffic data/image copy.png")
    # In image copy.png, circular sign is at the bottom center:
    # Let's find circle in image copy.png
    gray_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray_copy, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80, param1=100, param2=30, minRadius=50, maxRadius=120)
    if circles is not None:
        c = np.uint16(np.around(circles))[0, 0]
        x, y, rad = c[0], c[1], c[2]
        crop_copy_30 = img_copy[max(0, y-rad):min(img_copy.shape[0], y+rad), max(0, x-rad):min(img_copy.shape[1], x+rad)]
        cv2.imwrite("outputs/crop_copy_30.png", crop_copy_30)
        print(f"Found 30 sign in image copy.png at x={x}, y={y}, r={rad}")

    print("Tight crop extracted successfully!")

if __name__ == "__main__":
    test_digit_features()

