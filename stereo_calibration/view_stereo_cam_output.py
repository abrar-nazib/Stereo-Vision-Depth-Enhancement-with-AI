import cv2
import os
import pickle

current_dir = os.path.dirname(__file__)

CAM_L_IND = 2
CAM_R_IND = 1

cap_l = cv2.VideoCapture(CAM_L_IND)
cap_r = cv2.VideoCapture(CAM_R_IND)

print(
    f"Camera {CAM_L_IND} resolution: {cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
)
print(
    f"Camera {CAM_R_IND} resolution: {cap_r.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
)

cv2.namedWindow("Left Camera")
cv2.namedWindow("Right Camera")
# Add two trackbars to the windows
cv2.createTrackbar("Left", "Left Camera", 0, 100, lambda x: None)
cv2.createTrackbar("Right", "Right Camera", 0, 100, lambda x: None)

# Global variables to store the line positions
line_pos_l = -1
line_pos_r = -1


def undisort_frame(frame, cameraMatrix, dist):
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        cameraMatrix, dist, (w, h), 1, (w, h)
    )

    # Undistort the image
    undistorted_frame = cv2.undistort(
        frame, cameraMatrix, dist, None, new_camera_matrix
    )

    return undistorted_frame


def draw_line(event, x, y, flags, param):
    global line_pos_l, line_pos_r
    if event == cv2.EVENT_LBUTTONDOWN:
        if param == "Left Camera":
            line_pos_l = y
        elif param == "Right Camera":
            line_pos_r = y


cv2.setMouseCallback("Left Camera", draw_line, "Left Camera")
cv2.setMouseCallback("Right Camera", draw_line, "Right Camera")

with open(os.path.join(os.path.dirname(__file__), "calibration_1.pkl"), "rb") as f:
    cameraMatrix1, dist1 = pickle.load(f)
    print("Camera Matrix 1: ", cameraMatrix1)
    print("Distortion Coefficients 1: ", dist1)

with open(os.path.join(os.path.dirname(__file__), "calibration_2.pkl"), "rb") as f:
    cameraMatrix2, dist2 = pickle.load(f)
    print("Camera Matrix 2: ", cameraMatrix2)
    print("Distortion Coefficients 2: ", dist2)

while True:
    ret_l, frame_l = cap_l.read()
    frame_l = undisort_frame(frame_l, cameraMatrix2, dist2)
    frame_l = cv2.rotate(frame_l, cv2.ROTATE_90_CLOCKWISE)
    ret_r, frame_r = cap_r.read()
    frame_r = undisort_frame(frame_r, cameraMatrix1, dist1)
    frame_r = cv2.rotate(frame_r, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if not (ret_l and ret_r):
        break

    # Get trackbar positions
    left_crop = cv2.getTrackbarPos("Left", "Left Camera")
    right_crop = cv2.getTrackbarPos("Right", "Right Camera")

    # Crop frames based on trackbar positions
    height_l, width_l = frame_l.shape[:2]
    height_r, width_r = frame_r.shape[:2]

    crop_l_w = int(width_l * left_crop / 100)
    crop_l_h = int(height_l * left_crop / 100)
    crop_r_w = int(width_r * right_crop / 100)
    crop_r_h = int(height_r * right_crop / 100)

    frame_l_cropped = frame_l[
        crop_l_h : height_l - crop_l_h, crop_l_w : width_l - crop_l_w
    ]
    frame_l_cropped = cv2.resize(frame_l_cropped, (width_l, height_l))
    frame_r_cropped = frame_r[
        crop_r_h : height_r - crop_r_h, crop_r_w : width_r - crop_r_w
    ]
    frame_r_cropped = cv2.resize(frame_r_cropped, (width_r, height_r))

    # Draw horizontal lines if positions are set
    if line_pos_l != -1:
        cv2.line(
            frame_l_cropped, (0, line_pos_l), (width_l, line_pos_l), (0, 255, 0), 2
        )
    if line_pos_r != -1:
        cv2.line(
            frame_r_cropped, (0, line_pos_r), (width_r, line_pos_r), (0, 255, 0), 2
        )

    cv2.imshow("Left Camera", frame_l_cropped)
    cv2.imshow("Right Camera", frame_r_cropped)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap_l.release()
cap_r.release()
