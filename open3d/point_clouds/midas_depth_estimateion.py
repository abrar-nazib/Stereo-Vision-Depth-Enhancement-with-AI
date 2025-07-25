import cv2
import torch
import time
import numpy as np

import os

current_directory = os.path.dirname(__file__)

# Load a MiDas model for depth estimation
model_type = (
    "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


# Open up the video capture from a webcam
cap = cv2.VideoCapture(2)

while cap.isOpened():

    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize the depth map
    output = cv2.normalize(
        depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )

    depth_map = cv2.normalize(
        depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F
    )

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    cv2.imshow("Image", img)
    cv2.imshow("Depth Map", depth_map)

    if cv2.waitKey(5) & 0xFF == 27:
        break

print("Saving images...")
images_directory = os.path.join(current_directory, "images")

cv2.imwrite(os.path.join(images_directory, "colorImg_upgraded.jpg"), img)
cv2.imwrite(os.path.join(images_directory, "depthImg_upgraded.png"), output)

print(f"Images saved to {images_directory}")
cap.release()
