import cv2
import os
import numpy as np

CAM_L_IND = 4
CAM_R_IND = 2
current_directory = os.path.dirname(__file__)


def main():
    # Check the operating system
    if os.name == "nt":  # Windows
        cap_l = cv2.VideoCapture(CAM_L_IND, cv2.CAP_DSHOW)
        cap_r = cv2.VideoCapture(CAM_R_IND, cv2.CAP_DSHOW)
    elif os.name == "posix":  # Linux or macOS
        print("Using V4L2 for Linux")
        cap_l = cv2.VideoCapture(CAM_L_IND, cv2.CAP_V4L2)
        cap_r = cv2.VideoCapture(CAM_R_IND, cv2.CAP_V4L2)
    else:  # Fallback for other OS
        print("Unsupported operating system. Using default capture method.")
        cap_l = cv2.VideoCapture(CAM_L_IND)
        cap_r = cv2.VideoCapture(CAM_R_IND)
    print(
        f"Camera L resolution: {cap_l.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
    )
    print(
        f"Camera R resolution: {cap_r.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
    )

    cap_l.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    save_dir = os.path.join(current_directory, "data")
    os.makedirs(save_dir, exist_ok=True)
    counter = 0

    while True:
        ret_l, frame_l = cap_l.read()
        # frame1 = cv2.rotate(frame1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ret_r, frame_r = cap_r.read()
        # frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)
        if not (ret_l and ret_r):
            if not ret_l:
                # Make frame l black using numpy
                frame_l = np.zeros((480, 640, 3), dtype=np.uint8)
            if not ret_r:
                # Make frame r black using numpy
                frame_r = np.zeros((480, 640, 3), dtype=np.uint8)

        cv2.imshow(f"Camera L", frame_l)
        cv2.imshow(f"Camera R", frame_r)
        key = cv2.waitKey(1) & 0xFF

        if key in [27, ord("q")]:
            break
        elif key == ord("s"):
            filename_l = os.path.join(
                save_dir, "stereoL", f"captured_image_L_{counter}.jpg"
            )
            filename_r = os.path.join(
                save_dir, "stereoR", f"captured_image_R_{counter}.jpg"
            )
            cv2.imwrite(filename_l, frame_l)
            cv2.imwrite(filename_r, frame_r)
            print(f"Images saved: {filename_l}, {filename_r}")
            counter += 1

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
