import cv2
import os
import numpy as np

# Current file path
current_path = os.path.dirname(__file__)


cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(1)

print("Press 's' to save image")
print("Press 'ESC' to exit")

num = 0

while cap1.isOpened() and cap2.isOpened():
    success1, img1 = cap1.read()
    success2, img2 = cap2.read()

    # Rotate the image with rotation matrices
    h, w = img1.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 270, 1.0)
    img1 = cv2.warpAffine(img1, M, (w, h))

    h, w = img2.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -270, 1.0)
    img2 = cv2.warpAffine(img2, M, (w, h))

    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord("s"):
        cv2.imwrite(
            os.path.join(
                current_path, "images", "stereoLeft", "left" + str(num) + ".png"
            ),
            img1,
        )
        cv2.imwrite(
            os.path.join(
                current_path, "images", "stereoRight", "right" + str(num) + ".png"
            ),
            img2,
        )
        print("Save image " + str(num))
        num += 1

    cv2.imshow("left", img1)
    cv2.imshow("right", img2)

cap1.release()
cap2.release()
