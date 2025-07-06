import cv2
import random
import os
import pickle

cap_l = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap_r = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Camera L")
print(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH), cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Camera R")
print(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH), cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))


def undisort_frame(frame, cameraMatrix, dist):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, dist, (w, h), 1, (w, h)
    )

    # Undistort the image
    undistorted_frame = cv2.undistort(
        frame, cameraMatrix, dist, None, new_camera_matrix
    )

    # Crop the image
    # x, y, w, h = roi
    # undistorted_frame = undistorted_frame[y : y + h, x : x + w]
    # print(undistorted_frame.shape)
    return undistorted_frame


with open(os.path.join(os.path.dirname(__file__), "calibration_L.pkl"), "rb") as f:
    cameraMatrix_L, dist_L = pickle.load(f)
    print("Camera Matrix L")
    print(cameraMatrix_L)
    print("Distortion Coefficients L")
    print(dist_L)

with open(os.path.join(os.path.dirname(__file__), "calibration_R.pkl"), "rb") as f:
    cameraMatrix_R, dist_R = pickle.load(f)
    print("Camera Matrix R")
    print(cameraMatrix_R)
    print("Distortion Coefficients R")
    print(dist_R)

while True:
    ret_L, frame_L = cap_l.read()
    ret_R, frame_R = cap_r.read()

    # frame_L = undisort_frame(frame_L, cameraMatrix_L, dist_L)
    # frame_R = undisort_frame(frame_R, cameraMatrix_R, dist_R)

    if not (ret_L and ret_R):
        break

    # Write text on the frame1: frame1

    # Horizontal Concatenation
    frame = cv2.hconcat([frame_L, frame_R])

    cv2.imshow("Camera View", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap_l.release()
cap_r.release()
cv2.destroyAllWindows()
