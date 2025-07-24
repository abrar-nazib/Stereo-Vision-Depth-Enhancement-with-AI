import cv2
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

cv_file = cv2.FileStorage(
    os.path.join(current_dir, "stereoMap.xml"), cv2.FILE_STORAGE_READ
)
Left_Stereo_Map_x = cv_file.getNode("stereoMapL_x").mat()
Left_Stereo_Map_y = cv_file.getNode("stereoMapL_y").mat()
Right_Stereo_Map_x = cv_file.getNode("stereoMapR_x").mat()
Right_Stereo_Map_y = cv_file.getNode("stereoMapR_y").mat()
Q = cv_file.getNode("Q").mat()
cv_file.release()


def compute_depth_map(disparity, Q):
    # Convert disparity to depth
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3D[:, :, 2]  # Z coordinate

    # Filter out invalid values
    mask = (disparity > disparity.min()) & (depth_map > 0) & (depth_map < 5000)
    depth_map[~mask] = 0

    return depth_map


# Read a disparity image and convert to depth map
img = cv2.imread(
    os.path.join(current_dir, "csre_stereo", "images", "disparity.png"),
    cv2.IMREAD_GRAYSCALE,
)

if img is not None:
    depth_map = compute_depth_map(img, Q)
    cv2.imwrite(
        os.path.join(current_dir, "csre_stereo", "images", "depth_image.png"), depth_map
    )
    # Save the raw depth map for further processing
    np.save(
        os.path.join(current_dir, "csre_stereo", "images", "depth_image.npy"), depth_map
    )
    print("Depth map computed and saved successfully.")
else:
    print("Error: Could not load the disparity image.")
