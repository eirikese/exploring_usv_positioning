import cv2
import os
import numpy as np

# Chess/checker board size, dimensions
CHESSBOARD_DIM = (9, 7)

# The size of squares in the checker board design
SQUARE_SIZE = 20  # millimeters (change it according to printed size)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

calib_data_path = "../calibration_data"

# saving the image / camera calibration data
if os.path.isdir(calib_data_path) == False:
    os.makedirs(calib_data_path)
    print(f'"{calib_data_path}" Directory is created')
else:
    print(f'"{calib_data_path}" Directory already Exists.')

# Preparing object points
obj_3D = np.zeros((CHESSBOARD_DIM[0] * CHESSBOARD_DIM[1], 3), np.float32)

obj_3D[:, :2] = np.mgrid[0 : CHESSBOARD_DIM[0], 0 : CHESSBOARD_DIM[1]].T.reshape(
    -1, 2
)
obj_3D *= SQUARE_SIZE
print(obj_3D)

# Arrays to store object points and image points from all the given images.
obj_points_3D = []  # 3d point in real world space
img_points_2D = []  # 2d points in image plane

# The images directory path
image_dir_path = "calibration_images"

# Making a list of all the files present in the images directory path
files = os.listdir(image_dir_path)  
for file in files:
    print(file)
    imagePath = os.path.join(image_dir_path, file)
    image = cv2.imread(imagePath)
    grayScale = cv2.cv2tColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(image, CHESSBOARD_DIM, None)
    if ret == True:
        obj_points_3D.append(obj_3D)
        corners2 = cv2.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), criteria)
        img_points_2D.append(corners2)

        img = cv2.drawChessboardCorners(image, CHESSBOARD_DIM, corners2, ret)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points_3D, img_points_2D, grayScale.shape[::-1], None, None
)
print("calibrated")

print("dumping the data into one files using numpy ")
np.savez(
    f"{calib_data_path}/CalibrationMatrix",
    camMatrix=mtx,
    distCoef=dist,
    rVector=rvecs,
    tVector=tvecs,
)

print("-------------------------------------------")

print("loading data stored using numpy savez function\n \n \n")

data = np.load(f"{calib_data_path}CalibrationMatrix.npz")

camMatrix = data["camMatrix"]
distCof = data["distCoef"]
rVector = data["rVector"]
tVector = data["tVector"]

print("loaded calibration data successfully")