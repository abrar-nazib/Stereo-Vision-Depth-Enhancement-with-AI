import cv2
import os

current_directory = os.path.dirname(__file__)
CAMERA_INDEX_L = 2
CAMERA_INDEX_R = 0
CAMERA_CHR_L = "L"
CAMERA_CHR_R = "R"


def main():
    cap_L = cv2.VideoCapture(CAMERA_INDEX_L, cv2.CAP_DSHOW)
    cap_R = cv2.VideoCapture(CAMERA_INDEX_R, cv2.CAP_DSHOW)

    print(
        f"Camera {CAMERA_INDEX_L} resolution: {cap_L.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_L.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
    )
    print(
        f"Camera {CAMERA_INDEX_R} resolution: {cap_R.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_R.get(cv2.CAP_PROP_FRAME_HEIGHT)}"
    )

    # Set cap props to 640x480
    cap_L.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_L.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_R.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_R.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Get the camera resolution

    save_dir = os.path.join(current_directory, "images")
    os.makedirs(save_dir, exist_ok=True)
    counter = 0

    while True:
        ret_L, frame_L = cap_L.read()
        ret_R, frame_R = cap_R.read()

        if not ret_L or not ret_R:
            break

        # Show camera frames (resize for easier viewing)
        cv2.imshow(f"Camera {CAMERA_CHR_L}", frame_L)
        cv2.imshow(f"Camera {CAMERA_CHR_R}", frame_R)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("s"):
            filename_L = os.path.join(
                save_dir, f"captured_image_{CAMERA_CHR_L}_{counter}.jpg"
            )
            filename_R = os.path.join(
                save_dir, f"captured_image_{CAMERA_CHR_R}_{counter}.jpg"
            )
            cv2.imwrite(filename_L, frame_L)
            cv2.imwrite(filename_R, frame_R)
            print(f"Image saved to {filename_L}")
            print(f"Image saved to {filename_R}")
            counter += 1

    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
