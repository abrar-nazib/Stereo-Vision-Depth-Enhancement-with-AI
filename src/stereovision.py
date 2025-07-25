import numpy as np
import cv2
import os

# Current file path
current_dir = os.path.dirname(__file__)

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open(os.path.join(current_dir, "stereoMap.xml"), cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode("stereoMapL_x").mat()
stereoMapL_y = cv_file.getNode("stereoMapL_y").mat()
stereoMapR_x = cv_file.getNode("stereoMapR_x").mat()
stereoMapR_y = cv_file.getNode("stereoMapR_y").mat()


cap_left = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the resolution of the cameras
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while cap_right.isOpened() and cap_left.isOpened():

    succes_left, frame_left = cap_left.read()
    succes_right, frame_right = cap_right.read()

    # Undistort and rectify images
    frame_left = cv2.remap(
        frame_left,
        stereoMapL_x,
        stereoMapL_y,
        cv2.INTER_LANCZOS4,
        cv2.BORDER_CONSTANT,
        0,
    )
    frame_right = cv2.remap(
        frame_right,
        stereoMapR_x,
        stereoMapR_y,
        cv2.INTER_LANCZOS4,
        cv2.BORDER_CONSTANT,
        0,
    )

    # Draw horizontal lines on theimage
    for i in range(0, 480, 50):
        cv2.line(frame_left, (0, i), (640, i), (0, 255, 0), 1)
        cv2.line(frame_right, (0, i), (640, i), (0, 255, 0), 1)

    # Show the frames
    cv2.imshow("frame left", frame_left)
    cv2.imshow("frame right", frame_right)

    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
