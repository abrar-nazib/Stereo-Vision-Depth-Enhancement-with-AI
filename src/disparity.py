import numpy as np
import cv2
import os

current_dir = os.path.dirname(__file__)

# Check for left and right camera IDs
# These values can change depending on the system
CamL_id = 2  # Camera ID for left camera
CamR_id = 1  # Camera ID for right camera

CamL = cv2.VideoCapture(CamL_id, cv2.CAP_DSHOW)
CamR = cv2.VideoCapture(CamR_id, cv2.CAP_DSHOW)

# Setting the resolution for the cameras
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage(
    os.path.join(current_dir, "stereoMap.xml"), cv2.FILE_STORAGE_READ
)
Left_Stereo_Map_x = cv_file.getNode("stereoMapL_x").mat()
Left_Stereo_Map_y = cv_file.getNode("stereoMapL_y").mat()
Right_Stereo_Map_x = cv_file.getNode("stereoMapR_x").mat()
Right_Stereo_Map_y = cv_file.getNode("stereoMapR_y").mat()
cv_file.release()


def nothing(x):
    pass


# Create window and trackbars
cv2.namedWindow("disp")
cv2.resizeWindow("disp", 600, 600)

# Create trackbars with optimized default values
cv2.createTrackbar("minDisparity", "disp", 5, 100, nothing)  # Default: 5
cv2.createTrackbar(
    "numDisparities", "disp", 8, 20, nothing
)  # Default: 8 (will be multiplied by 16 = 128)
cv2.createTrackbar("blockSize", "disp", 7, 50, nothing)  # Default: 7
cv2.createTrackbar("P1", "disp", 8, 200, nothing)  # Default: 8
cv2.createTrackbar("P2", "disp", 32, 400, nothing)  # Default: 32
cv2.createTrackbar("disp12MaxDiff", "disp", 1, 25, nothing)  # Default: 1
cv2.createTrackbar("uniquenessRatio", "disp", 15, 100, nothing)  # Default: 15
cv2.createTrackbar("speckleWindowSize", "disp", 100, 200, nothing)  # Default: 100
cv2.createTrackbar("speckleRange", "disp", 2, 50, nothing)  # Default: 2
cv2.createTrackbar("preFilterCap", "disp", 63, 100, nothing)  # Default: 63


while True:
    # Capturing and storing left and right camera images
    retL, imgL = CamL.read()
    retR, imgR = CamR.read()

    # Proceed only if the frames have been captured
    if retL and retR:
        imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

        # Applying stereo image rectification on the left image
        Left_nice = cv2.remap(
            imgL_gray,
            Left_Stereo_Map_x,
            Left_Stereo_Map_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )

        # Applying stereo image rectification on the right image
        Right_nice = cv2.remap(
            imgR_gray,
            Right_Stereo_Map_x,
            Right_Stereo_Map_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )

        # Get current positions of trackbars
        minDisparity = cv2.getTrackbarPos("minDisparity", "disp")
        numDisparities = cv2.getTrackbarPos("numDisparities", "disp") * 16
        blockSize = cv2.getTrackbarPos("blockSize", "disp")
        if blockSize % 2 == 0:
            blockSize += 1  # blockSize should be odd
        P1 = cv2.getTrackbarPos("P1", "disp") * 3 * blockSize**2
        P2 = cv2.getTrackbarPos("P2", "disp") * 3 * blockSize**2
        disp12MaxDiff = cv2.getTrackbarPos("disp12MaxDiff", "disp")
        uniquenessRatio = cv2.getTrackbarPos("uniquenessRatio", "disp")
        speckleWindowSize = cv2.getTrackbarPos("speckleWindowSize", "disp")
        speckleRange = cv2.getTrackbarPos("speckleRange", "disp")
        preFilterCap = cv2.getTrackbarPos("preFilterCap", "disp")

        # Create new stereo object with updated parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            preFilterCap=preFilterCap,
            mode=cv2.STEREO_SGBM_MODE_HH,
        )

        # Calculating disparity using the StereoBM algorithm
        disparity = stereo.compute(Left_nice, Right_nice).astype(np.float32) / 16.0

        # Normalize disparity for display
        disparity_normalized = cv2.normalize(
            disparity,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

        # Display the images
        fr = cv2.hconcat([imgL, imgR])
        cv2.imshow("Original Images", fr)
        cv2.imshow("disp", disparity_normalized)

        # Print current parameters (optional)
        print(f"Current Parameters:")
        print(f"minDisparity: {minDisparity}")
        print(f"numDisparities: {numDisparities}")
        print(f"blockSize: {blockSize}")
        print(f"P1: {P1}")
        print(f"P2: {P2}")
        print(f"disp12MaxDiff: {disp12MaxDiff}")
        print(f"uniquenessRatio: {uniquenessRatio}")
        print(f"speckleWindowSize: {speckleWindowSize}")
        print(f"speckleRange: {speckleRange}")
        print(f"preFilterCap: {preFilterCap}")
        print("------------------------")

        # Close window using esc key
        if cv2.waitKey(1) == 27:
            break

    else:
        CamL = cv2.VideoCapture(CamL_id)
        CamR = cv2.VideoCapture(CamR_id)

# Clean up
CamL.release()
CamR.release()
cv2.destroyAllWindows()
