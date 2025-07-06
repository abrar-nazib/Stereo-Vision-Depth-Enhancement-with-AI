import numpy as np
import cv2
import os

current_dir = os.path.dirname(__file__)


def compute_depth_map(disparity, Q):
    # Convert disparity to depth
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3D[:, :, 2]  # Z coordinate

    # Filter out invalid values
    mask = (disparity > disparity.min()) & (depth_map > 0) & (depth_map < 5000)
    depth_map[~mask] = 0

    return depth_map


def save_depth_data(depth_map, color_image, timestamp):
    # Save the raw depth data (float32)
    depth_filename = os.path.join(current_dir, "images", f"depth_raw_{timestamp}.npy")
    np.save(depth_filename, depth_map)

    # Save the color image
    color_filename = f"color_{timestamp}.jpg"
    cv2.imwrite(os.path.join(current_dir, "images", color_filename), color_image)

    # Also save visualization
    normalized_depth = cv2.normalize(depth_map, None, 0, 65535, cv2.NORM_MINMAX)
    depth_uint16 = normalized_depth.astype(np.uint16)
    depth_vis_filename = f"depth_vis_{timestamp}.png"
    cv2.imwrite(os.path.join(current_dir, "images", depth_vis_filename), depth_uint16)

    print(f"Saved:\n{depth_filename}\n{color_filename}\n{depth_vis_filename}")


def visualize_depth(depth_map, min_depth=100, max_depth=2000):  # depths in mm
    # Normalize depth map for visualization
    depth_map_normalized = np.clip(depth_map, min_depth, max_depth)
    depth_map_normalized = (
        (depth_map_normalized - min_depth) * 255 / (max_depth - min_depth)
    ).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    # Add text showing depth range
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


# Camera setup
CamL_id = 2
CamR_id = 1
CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)

# Set resolution
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
cv2.createTrackbar("Max Depth (mm)", "Controls", 2000, 5000, nothing)

# Create SGBM object with the optimal parameters
stereo = cv2.StereoSGBM_create(
    minDisparity=5,
    numDisparities=128,
    blockSize=7,
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_HH,
)


# Mouse callback function for depth measurement
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth = depth_map[y, x]
        if depth > 0:
            print(f"Depth at ({x}, {y}): {depth:.1f} mm")
        else:
            print(f"Invalid depth measurement at ({x}, {y})")


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
        rectL = cv2.remap(
            grayL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rectR = cv2.remap(
            grayR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4
        )

        # Compute disparity
        disparity = stereo.compute(rectL, rectR).astype(np.float32) / 16.0

        # Compute depth map
        depth_map = compute_depth_map(disparity, Q)

        # Get current depth visualization range
        min_depth = cv2.getTrackbarPos("Min Depth (mm)", "Controls")
        max_depth = cv2.getTrackbarPos("Max Depth (mm)", "Controls")

        # Ensure min_depth is less than max_depth
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

        # Display results
        cv2.imshow("Original Images", np.hstack([imgL, imgR]))
        cv2.imshow("Depth Map", depth_colormap)
        cv2.imshow("Disparity Map", disparity_color)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord("s"):  # Save depth map
            timestamp = cv2.getTickCount()
            cv2.imwrite(f"depth_map_{timestamp}.png", depth_colormap)
            cv2.imwrite(f"disparity_{timestamp}.png", disparity_color)
            save_depth_data(depth_map, rectL, timestamp)
            print(f"Saved depth and disparity maps with timestamp {timestamp}")

    else:
        # Try to reconnect to cameras
        CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
        CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)

# Cleanup
CamL.release()
CamR.release()
cv2.destroyAllWindows()
