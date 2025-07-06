import cv2
import os

CAM_L_IND = 2
CAM_R_IND = 1
current_directory = os.path.dirname(__file__)


def main():
    cap_l = cv2.VideoCapture(CAM_L_IND, cv2.CAP_DSHOW)
    cap_r = cv2.VideoCapture(CAM_R_IND, cv2.CAP_DSHOW)

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
            break

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
