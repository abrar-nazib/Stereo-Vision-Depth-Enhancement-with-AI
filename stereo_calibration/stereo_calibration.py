import cv2
import numpy as np
from tqdm import tqdm
import os

current_dir = os.path.dirname(__file__)

# Set the path to the images captured by the left and right cameras
pathL = os.path.join(current_dir, "data", "stereoL")
pathR = os.path.join(current_dir, "data", "stereoR")

chessboardSize = (7, 7)
framesize = (480, 640)


# Termination criteria for refining the detected corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 134 / 6
objp = objp * size_of_chessboard_squares_mm

img_ptsL = []
img_ptsR = []
obj_pts = []

for i in tqdm(range(1, 18)):
    imgL = cv2.imread(os.path.join(pathL, f"captured_image_2_{i}.jpg"))
    imgR = cv2.imread(os.path.join(pathR, f"captured_image_1_{i}.jpg"))
    imgL_gray = cv2.imread(os.path.join(pathL, f"captured_image_2_{i}.jpg"), 0)
    imgR_gray = cv2.imread(os.path.join(pathR, f"captured_image_1_{i}.jpg"), 0)

    outputL = imgL.copy()
    outputR = imgR.copy()

    retR, cornersR = cv2.findChessboardCorners(outputR, chessboardSize, None)
    retL, cornersL = cv2.findChessboardCorners(outputL, chessboardSize, None)

    if retR and retL:
        obj_pts.append(objp)
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), criteria)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(outputR, chessboardSize, cornersR, retR)
        cv2.drawChessboardCorners(outputL, chessboardSize, cornersL, retL)
        cv2.imshow("cornersR", outputR)
        cv2.imshow("cornersL", outputL)
        cv2.waitKey(0)

        img_ptsL.append(cornersL)
        img_ptsR.append(cornersR)


# Calibrating left camera
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
    obj_pts, img_ptsL, imgL_gray.shape[::-1], None, None
)
hL, wL = imgL_gray.shape[:2]
new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

# Calibrating right camera
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
    obj_pts, img_ptsR, imgR_gray.shape[::-1], None, None
)
hR, wR = imgR_gray.shape[:2]
new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same

criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(
    obj_pts,
    img_ptsL,
    img_ptsR,
    new_mtxL,
    distL,
    new_mtxR,
    distR,
    imgL_gray.shape[::-1],
    criteria_stereo,
    flags,
)

rectify_scale = 1
rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(
    new_mtxL,
    distL,
    new_mtxR,
    distR,
    imgL_gray.shape[::-1],
    Rot,
    Trns,
    rectify_scale,
    (0, 0),
)


Left_Stereo_Map = cv2.initUndistortRectifyMap(
    new_mtxL, distL, rect_l, proj_mat_l, imgL_gray.shape[::-1], cv2.CV_16SC2
)
Right_Stereo_Map = cv2.initUndistortRectifyMap(
    new_mtxR, distR, rect_r, proj_mat_r, imgR_gray.shape[::-1], cv2.CV_16SC2
)
print("Saving parameters ...")
cv_file = cv2.FileStorage(
    os.path.join(current_dir, "improved_params2.xml"), cv2.FILE_STORAGE_WRITE
)
cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
cv_file.release()
