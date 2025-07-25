import numpy as np
import cv2
import os

current_dir = os.path.dirname(__file__)

# Camera setup
CamL_id = 2
CamR_id = 0


def compute_depth_map(disparity, Q):
    # Convert disparity to depth using the Q matrix
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3D[:, :, 2]  # Extract Z coordinate (depth)

    # Filter out invalid values (only remove negative/zero depths)
    mask = (disparity > disparity.min()) & (depth_map > 0)
    depth_map[~mask] = 0
    return depth_map


def visualize_depth(depth_map, min_depth=100, max_depth=5000):  # Depths in mm
    # Normalize depth map for visualization
    depth_map_normalized = np.clip(depth_map, min_depth, max_depth)
    depth_map_normalized = (
        (depth_map_normalized - min_depth) * 255 / (max_depth - min_depth)
    ).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_PLASMA)

    # Add depth range text
    cv2.putText(
        depth_colormap,
        f"Range: {min_depth}mm - {max_depth}mm",
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return depth_colormap


# Initialize cameras based on OS
if os.name == "nt":  # Windows
    print("Using DSHOW for Windows")
    CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
    CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)
elif os.name == "posix":  # Linux or macOS
    print("Using V4L2 for Linux")
    CamL = cv2.VideoCapture(CamL_id, cv2.CAP_V4L2)
    CamR = cv2.VideoCapture(CamR_id, cv2.CAP_V4L2)
else:  # Fallback
    print("Unsupported OS. Using default capture method.")
    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)

# Set camera resolution
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load calibration data
cv_file = cv2.FileStorage(
    os.path.join(current_dir, "stereoMap.xml"), cv2.FILE_STORAGE_READ
)
Left_Stereo_Map_x = cv_file.getNode("stereoMapL_x").mat()
Left_Stereo_Map_y = cv_file.getNode("stereoMapL_y").mat()
Right_Stereo_Map_x = cv_file.getNode("stereoMapR_x").mat()
Right_Stereo_Map_y = cv_file.getNode("stereoMapR_y").mat()
Q = cv_file.getNode("Q").mat()
cv_file.release()

# Create windows
cv2.namedWindow("Depth Map")
cv2.namedWindow("Controls")


# Create trackbars for depth visualization
def nothing(x):
    pass


cv2.createTrackbar("Min Depth (mm)", "Controls", 100, 1000, nothing)
cv2.createTrackbar("Max Depth (mm)", "Controls", 5000, 10000, nothing)

# Initialize StereoSGBM with tuned parameters
stereo = cv2.StereoSGBM_create(
    minDisparity=5,
    numDisparities=64,  # Reduced for closer objects
    blockSize=5,  # Smaller for finer details
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=20,  # Increased to filter ambiguous matches
    speckleWindowSize=50,  # Reduced for better noise handling
    speckleRange=1,  # Reduced for smaller consistent regions
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_HH,
)


# Mouse callback for depth measurement
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = depth_map[y, x]
        if depth > 0:
            print(f"Depth at ({x}, {y}): {depth:.1f} mm")
        else:
            print(f"Invalid depth at ({x}, {y})")


cv2.setMouseCallback("Depth Map", mouse_callback)

# Main loop
while True:
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    if retL and retR:
        # Convert to grayscale
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Apply stereo rectification
        rectL_gray = cv2.remap(
            grayL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rectR_gray = cv2.remap(
            grayR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rect_L_color = cv2.remap(
            imgL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rect_R_color = cv2.remap(
            imgR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4
        )

        # Compute disparity
        disparity = stereo.compute(rectL_gray, rectR_gray).astype(np.float32) / 16.0

        # Compute depth map
        depth_map = compute_depth_map(disparity, Q)

        # Get depth visualization range from trackbars
        min_depth = cv2.getTrackbarPos("Min Depth (mm)", "Controls")
        max_depth = cv2.getTrackbarPos("Max Depth (mm)", "Controls")
        if min_depth >= max_depth:
            min_depth = max_depth - 100
            cv2.setTrackbarPos("Min Depth (mm)", "Controls", min_depth)

        # Visualize depth
        depth_colormap = visualize_depth(depth_map, min_depth, max_depth)

        # Create disparity visualization
        disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

        # Stack rectified images with alignment lines
        rectified_images_gray = np.hstack((rectL_gray, rectR_gray))
        rectified_images_gray = cv2.cvtColor(rectified_images_gray, cv2.COLOR_GRAY2BGR)
        for y in range(0, rectL_gray.shape[0], 20):
            cv2.line(
                rectified_images_gray,
                (0, y),
                (rectified_images_gray.shape[1] // 2, y),
                (0, 255, 0),
                1,
            )
            cv2.line(
                rectified_images_gray,
                (rectified_images_gray.shape[1] // 2, y),
                (rectified_images_gray.shape[1], y),
                (0, 255, 0),
                1,
            )

        original_images = np.hstack((imgL, imgR))

        # Display results
        cv2.imshow("Original Images", original_images)
        cv2.imshow("Rectified images", rectified_images_gray)
        cv2.imshow("Depth Map", depth_colormap)
        cv2.imshow("Disparity Map", disparity_color)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == ord("s"):  # Save outputs
            cv2.imwrite(
                os.path.join(current_dir, "images", "rect_L_color.png"), rect_L_color
            )
            cv2.imwrite(
                os.path.join(current_dir, "images", "rect_R_color.png"), rect_R_color
            )
            cv2.imwrite(
                os.path.join(current_dir, "images", "rect_L_gray.png"), rectL_gray
            )
            cv2.imwrite(
                os.path.join(current_dir, "images", "rect_R_gray.png"), rectR_gray
            )
            cv2.imwrite(os.path.join(current_dir, "images", "img_l.png"), imgL)
            cv2.imwrite(os.path.join(current_dir, "images", "img_r.png"), imgR)
            cv2.imwrite(
                os.path.join(current_dir, "images", "depth_vis.png"), depth_colormap
            )
            cv2.imwrite(
                os.path.join(current_dir, "images", "disparity_vis.png"),
                disparity_color,
            )
            cv2.imwrite(
                os.path.join(current_dir, "images", "rectified_images_gray.png"),
                rectified_images_gray,
            )
            np.save(os.path.join(current_dir, "images", "disparity.npy"), disparity)
            np.save(os.path.join(current_dir, "images", "depth_raw.npy"), depth_map)
            print("Saved all outputs to 'images' directory.")
    else:
        # Reconnect cameras if frame capture fails
        if os.name == "nt":
            CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
            CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)
        elif os.name == "posix":
            CamL = cv2.VideoCapture(CamL_id, cv2.CAP_V4L2)
            CamR = cv2.VideoCapture(CamR_id, cv2.CAP_V4L2)
        else:
            CamL = cv2.VideoCapture(CamL_id)
            CamR = cv2.VideoCapture(CamR_id)

# Cleanup
CamL.release()
CamR.release()
cv2.destroyAllWindows()
