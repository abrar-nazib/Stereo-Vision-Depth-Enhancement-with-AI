import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# Approximate PSMNet architecture (simplified but closer to original)
class PSMNet(nn.Module):
    def __init__(self, maxdisp=192):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp

        # Basic feature extraction (approximated)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(64, 1, 3, padding=1)  # Dummy output for disparity

    def forward(self, left, right):
        left_feat = self.conv1(left)
        right_feat = self.conv1(right)
        disparity = self.conv2(left_feat - right_feat)  # Simplified disparity
        return disparity


current_dir = os.path.dirname(__file__)

# Camera setup
CamL_id = 2
CamR_id = 0


def compute_depth_map(disparity, Q):
    print("Computing depth map with disparity shape:", disparity.shape)
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map = points_3D[:, :, 2]  # Extract Z coordinate
    mask = (disparity > disparity.min()) & (depth_map > 0)
    depth_map[~mask] = 0
    print("Depth map shape:", depth_map.shape)
    return depth_map


def visualize_depth(depth_map, min_depth=100, max_depth=5000):
    print(f"Visualizing depth map with range {min_depth}mm - {max_depth}mm")
    depth_map_normalized = np.clip(depth_map, min_depth, max_depth)
    depth_map_normalized = (
        (depth_map_normalized - min_depth) * 255 / (max_depth - min_depth)
    ).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_PLASMA)
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


# Initialize cameras
if os.name == "nt":  # Windows
    print("Using DSHOW for Windows")
    CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
    CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)
elif os.name == "posix":  # Linux or macOS
    print("Using V4L2 for Linux")
    CamL = cv2.VideoCapture(CamL_id, cv2.CAP_V4L2)
    CamR = cv2.VideoCapture(CamR_id, cv2.CAP_V4L2)
else:
    print("Unsupported OS. Using default capture method.")
    CamL = cv2.VideoCapture(CamL_id)
    CamR = cv2.VideoCapture(CamR_id)

# Set resolution
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check actual resolution
print("Left camera actual width:", CamL.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Left camera actual height:", CamL.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Right camera actual width:", CamR.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Right camera actual height:", CamR.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

print("Left_Stereo_Map_x shape:", Left_Stereo_Map_x.shape)
print("Left_Stereo_Map_y shape:", Left_Stereo_Map_y.shape)
print("Right_Stereo_Map_x shape:", Right_Stereo_Map_x.shape)
print("Right_Stereo_Map_y shape:", Right_Stereo_Map_y.shape)

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PSMNet(maxdisp=192).to(device)

# Load 'psmnet_kitty' with handling for nested state_dict and DataParallel prefix
model_path = os.path.join(current_dir, "psmnet_kitty.tar")
try:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["state_dict"]

    # Remove 'module.' prefix for DataParallel compatibility
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)  # Allow partial loading
    print("Loaded 'psmnet_kitty' with 'module.' prefix removed.")
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("The architecture might not match. Inspecting keys...")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        print("Available keys in checkpoint:", checkpoint.keys())
        if "state_dict" in checkpoint:
            print("Nested state_dict keys:", checkpoint["state_dict"].keys())
    else:
        print("Checkpoint is not a dictionary:", type(checkpoint))
    print("Please verify the model file or provide the correct PSMNet architecture.")
    exit(1)

model.eval()

# Image preprocessing
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create windows
cv2.namedWindow("Depth Map")
cv2.namedWindow("Controls")


# Trackbars for depth visualization
def nothing(x):
    pass


cv2.createTrackbar("Min Depth (mm)", "Controls", 100, 1000, nothing)
cv2.createTrackbar("Max Depth (mm)", "Controls", 5000, 10000, nothing)


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
        print("imgL shape:", imgL.shape)
        print("imgR shape:", imgR.shape)
        if imgL.shape != (480, 640, 3) or imgR.shape != (480, 640, 3):
            print("Error: Input images do not match expected 640x480 resolution")
            break

        # Convert to grayscale
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization
        grayL = cv2.equalizeHist(grayL)
        grayR = cv2.equalizeHist(grayR)

        # Rectify images
        rectL_color = cv2.remap(
            imgL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rectR_color = cv2.remap(
            imgR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rectL_gray = cv2.remap(
            grayL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        rectR_gray = cv2.remap(
            grayR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4
        )
        print("rectL_color shape:", rectL_color.shape)
        print("rectR_color shape:", rectR_color.shape)
        print("rectL_gray shape:", rectL_gray.shape)
        print("rectR_gray shape:", rectR_gray.shape)
        if rectL_color.shape != (480, 640, 3) or rectR_color.shape != (480, 640, 3):
            print("Error: Rectified images do not match expected 640x480 resolution")
            break

        # Prepare images for PSMNet
        h, w = rectL_color.shape[:2]
        print(f"Extracted h, w: {h}, {w}")
        rectL_tensor = transform(rectL_color).unsqueeze(0).to(device)
        rectR_tensor = transform(rectR_color).unsqueeze(0).to(device)
        print("rectL_tensor shape:", rectL_tensor.shape)
        print("rectR_tensor shape:", rectR_tensor.shape)

        # PSMNet inference
        with torch.no_grad():
            disparity = model(rectL_tensor, rectR_tensor)
            disparity = disparity.squeeze().cpu().numpy()  # Ensure 2D (H, W)
            print("disparity shape after PSMNet:", disparity.shape)

        # Compute and visualize depth
        depth_map = compute_depth_map(disparity, Q)
        min_depth = cv2.getTrackbarPos("Min Depth (mm)", "Controls")
        max_depth = cv2.getTrackbarPos("Max Depth (mm)", "Controls")
        if min_depth >= max_depth:
            min_depth = max_depth - 100
            cv2.setTrackbarPos("Min Depth (mm)", "Controls", min_depth)
        depth_colormap = visualize_depth(depth_map, min_depth, max_depth)

        # Disparity visualization
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
        cv2.imshow("Rectified Images", rectified_images_gray)
        cv2.imshow("Depth Map", depth_colormap)
        cv2.imshow("Disparity Map", disparity_color)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == ord("s"):  # Save outputs
            os.makedirs(os.path.join(current_dir, "images"), exist_ok=True)
            cv2.imwrite(
                os.path.join(current_dir, "images", "depth_vis.png"), depth_colormap
            )
            cv2.imwrite(
                os.path.join(current_dir, "images", "disparity_vis.png"),
                disparity_color,
            )
            np.save(os.path.join(current_dir, "images", "disparity.npy"), disparity)
            np.save(os.path.join(current_dir, "images", "depth_raw.npy"), depth_map)
            print("Saved outputs to 'images' directory.")
    else:
        print("Failed to read frames")
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
