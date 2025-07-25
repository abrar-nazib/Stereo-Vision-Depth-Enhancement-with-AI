import numpy as np
import cv2 as cv
import glob
import os

current_dir = os.path.dirname(__file__)

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (11, 7)
frameSize = (640, 480)

# termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
square_size_mm = 20.0  # size of each square in mm
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)
objp = objp * square_size_mm

# Arrays to store object points and image points
objpoints = []  # 3d points in real world space
imgpointsL = []  # 2d points in left image
imgpointsR = []  # 2d points in right image

# Load images
imagesLeft = sorted(glob.glob(os.path.join(current_dir, "data", "stereoL", "*.jpg")))
imagesRight = sorted(glob.glob(os.path.join(current_dir, "data", "stereoR", "*.jpg")))

if len(imagesLeft) != len(imagesRight):
    raise ValueError("Number of left and right images don't match")
if len(imagesLeft) < 10:
    print("Warning: At least 10 image pairs recommended for good calibration")

print(f"Found {len(imagesLeft)} image pairs. Starting calibration...")

# Process each image pair
for idx, (imgLeft, imgRight) in enumerate(zip(imagesLeft, imagesRight)):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)

    if imgL is None or imgR is None:
        print(f"Failed to load image pair {idx}")
        continue

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(
        grayL,
        chessboardSize,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
    )
    retR, cornersR = cv.findChessboardCorners(
        grayR,
        chessboardSize,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
    )

    # If found, refine and add points
    if retL and retR:
        print("Processed image pair:", idx)
        objpoints.append(objp)

        # Refine corners
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        # Check corner orientation for both cameras
        def check_corner_orientation(corners, camera_name):
            # First corner (index 0) and last corner in first row (index chessboardSize[0]-1)
            first_corner = corners[0][0]  # Top-left or bottom-left
            end_first_row = corners[chessboardSize[0] - 1][
                0
            ]  # Top-right or bottom-right

            # Check if we're going from top to bottom or bottom to top
            if first_corner[1] < end_first_row[1]:  # Y-coordinate comparison
                orientation = "Top-Left to Bottom-Right"
            else:
                orientation = "Bottom-Left to Top-Right"

            print(f"  {camera_name}: Corner pattern starts from {orientation}")
            print(
                f"    First corner at: ({first_corner[0]:.1f}, {first_corner[1]:.1f})"
            )
            return orientation

        orientationL = check_corner_orientation(cornersL, "Left Camera")
        orientationR = check_corner_orientation(cornersR, "Right Camera")

        # Check if both cameras have same orientation
        if orientationL == orientationR:
            print("  ✓ Both cameras have consistent orientation")
        else:
            print("  ⚠ Warning: Cameras have different orientations!")
            objpoints.pop()  # Remove this pair if inconsistent
            continue

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)

        # Display with smaller size for visualization
        display_size = (800, 600)
        imgL_small = cv.resize(imgL, display_size)
        imgR_small = cv.resize(imgR, display_size)

        # Stack images horizontally for display
        combined = np.hstack((imgL_small, imgR_small))
        cv.imshow("Chessboard Corners", combined)
        cv.waitKey(500)


print(f"Successfully processed {len(objpoints)} image pairs")
cv.destroyAllWindows()

############## CALIBRATION #######################################################

# Calibrate left camera (C30 - wider FOV)
print("Calibrating left camera...")
retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(
    objpoints,
    imgpointsL,
    frameSize,
    None,
    None,
    flags=cv.CALIB_RATIONAL_MODEL,  # Using rational model for better distortion correction
)

# Calibrate right camera (C31)
print("Calibrating right camera...")
retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(
    objpoints, imgpointsR, frameSize, None, None, flags=cv.CALIB_RATIONAL_MODEL
)

# Calculate optimal camera matrices
newMtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, frameSize, 1, frameSize)
newMtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, frameSize, 1, frameSize)

print("Individual camera calibration complete")

########## Stereo Vision Calibration #############################################

# Stereo calibration flags
stereo_flags = cv.CALIB_FIX_INTRINSIC
stereo_flags |= cv.CALIB_RATIONAL_MODEL
stereo_flags |= cv.CALIB_USE_INTRINSIC_GUESS

print("Performing stereo calibration...")
(
    retStereo,
    newMtxL,
    distL,
    newMtxR,
    distR,
    rot,
    trans,
    essentialMatrix,
    fundamentalMatrix,
) = cv.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    newMtxL,
    distL,
    newMtxR,
    distR,
    frameSize,
    criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=stereo_flags,
)

print(f"Stereo calibration RMS error: {retStereo}")

########## Stereo Rectification #################################################

print("Computing stereo rectification...")
rectifyScale = 1  # 0=full crop, 1=no crop
rectL, rectR, projMatrixL, projMatrixR, Q, validRoiL, validRoiR = cv.stereoRectify(
    newMtxL,
    distL,
    newMtxR,
    distR,
    frameSize,
    rot,
    trans,
    rectifyScale,
    (0, 0),
    flags=cv.CALIB_ZERO_DISPARITY,
    alpha=rectifyScale,
)

# Compute mapping for rectification
stereoMapL = cv.initUndistortRectifyMap(
    newMtxL, distL, rectL, projMatrixL, frameSize, cv.CV_16SC2
)
stereoMapR = cv.initUndistortRectifyMap(
    newMtxR, distR, rectR, projMatrixR, frameSize, cv.CV_16SC2
)

print("Saving calibration parameters...")
cv_file = cv.FileStorage(
    os.path.join(current_dir, "stereoMap.xml"), cv.FILE_STORAGE_WRITE
)

# Save the stereo mapping parameters
cv_file.write("stereoMapL_x", stereoMapL[0])
cv_file.write("stereoMapL_y", stereoMapL[1])
cv_file.write("stereoMapR_x", stereoMapR[0])
cv_file.write("stereoMapR_y", stereoMapR[1])

# Save additional calibration data for future use
cv_file.write("Q", Q)  # Disparity-to-depth mapping matrix
cv_file.write("validRoiL", np.array(validRoiL))
cv_file.write("validRoiR", np.array(validRoiR))
cv_file.write("cameraMatrixL", newMtxL)
cv_file.write("cameraMatrixR", newMtxR)
cv_file.write("distCoeffsL", distL)
cv_file.write("distCoeffsR", distR)
cv_file.write("rotationMatrix", rot)
cv_file.write("translationVector", trans)

cv_file.release()

print("Calibration complete! Testing rectification...")

# Test rectification on first image pair
if len(imagesLeft) > 0:
    imgL = cv.imread(imagesLeft[0])
    imgR = cv.imread(imagesRight[0])

    rectifiedL = cv.remap(imgL, stereoMapL[0], stereoMapL[1], cv.INTER_LANCZOS4)
    rectifiedR = cv.remap(imgR, stereoMapR[0], stereoMapR[1], cv.INTER_LANCZOS4)

    # Draw horizontal lines for checking rectification
    for i in range(0, rectifiedL.shape[0], 30):
        cv.line(rectifiedL, (0, i), (rectifiedL.shape[1], i), (0, 255, 0), 1)
        cv.line(rectifiedR, (0, i), (rectifiedR.shape[1], i), (0, 255, 0), 1)

    # Display side by side
    combined = np.hstack((rectifiedL, rectifiedR))
    cv.imshow("Rectified Images", combined)
    cv.waitKey(0)
    cv.destroyAllWindows()

print("Calibration and testing complete!")
