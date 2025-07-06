import numpy as np
import cv2
import time
import os

print(cv2.getBuildInformation())

model_directory_name = "models"
model_name = "model-f6b98070.onnx"

currernt_directory = os.getcwd()
model_directory = os.path.join(currernt_directory, model_directory_name)
model_path = os.path.join(model_directory, model_name)
print(model_path)
# MiDaS v2.1 Large
# model_name = "model-small.onnx"; # MiDaS v2.1 Small


# Load the DNN model
model = cv2.dnn.readNet(model_path)


if model.empty():
    print("Could not load the neural net! - Check path")


# Set backend and target to CUDA to use GPU
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Webcam
cap = cv2.VideoCapture(1)


while cap.isOpened():

    # Read in the image
    success, img = cap.read()

    imgHeight, imgWidth, channels = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create Blob from Input Image
    # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    blob = cv2.dnn.blobFromImage(
        img, 1 / 255.0, (384, 384), (123.675, 116.28, 103.53), True, False
    )

    # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    # blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    output = model.forward()

    output = output[0, :, :]
    output = cv2.resize(output, (imgWidth, imgHeight))

    # Normalize the output
    output = cv2.normalize(
        output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
    )
    output_opencv = cv2.normalize(
        output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1
    )

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("image", img)
    cv2.imshow("Depth Map", output_opencv)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
open3d_directory = os.path.join(currernt_directory, "open3d")
point_clouds_directory = os.path.join(open3d_directory, "point_clouds")
images_directory = os.path.join(point_clouds_directory, "images")

cv2.imwrite(os.path.join(images_directory, "colorImg.jpg"), img)
cv2.imwrite(os.path.join(images_directory, "depthImg.png"), output)

cap.release()
cv2.destroyAllWindows()
