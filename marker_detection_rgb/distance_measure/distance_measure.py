import cv2
from cv2 import aruco
import numpy as np
import csv

# Load in the calibration data
calib_data_path = "../calibration_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
# print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

# Marker size in centimeters. Needed for distance measurment
MARKER_SIZE_SMALL = 26.28  
MARKER_SIZE_BIG = 41.2  

# List of dictionaries to use for detection
dict_list = [
    aruco.DICT_4X4_50,
    aruco.DICT_5X5_50,
    aruco.DICT_6X6_50,
    aruco.DICT_7X7_50
]

param_markers = aruco.DetectorParameters()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    total_distance_cm = 0
    total_distance_m = 0
    marker_count = 0

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Iterate over the list of dictionaries for detection
    for dictionary in dict_list:
        marker_dict = aruco.getPredefinedDictionary(dictionary)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(gray_frame, marker_dict, parameters=param_markers)

        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(marker_corners, MARKER_SIZE_SMALL, cam_mat, dist_coef)
            total_markers = range(0, marker_IDs.size)

            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                cv2.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_left = corners[0].ravel()
                top_right = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()

                # Calculating the Euclidean distance
                distance_cm = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                distance_m = distance_cm * 0.01
                
                total_distance_cm += distance_cm
                total_distance_m += distance_m
                marker_count += 1
                
                # Draw the pose of the marker
                point = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
                # print(distance_m)
                
                cv2.putText(
                    frame, f"Dist [cm]: {round(distance_cm, 2)}", top_right, 
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"Dist [m]: {round(distance_m, 2)}", bottom_right, 
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 2, cv2.LINE_AA,
                )
                """
                cv2.putText(
                    frame, f"id: {ids[0]}", top_left, 
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame, f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                    bottom_left, cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
                )
                """
        
    # Compute and display the mean distance
    if marker_count > 0:
        mean_distance_cm = total_distance_cm / marker_count
        cv2.putText(frame, f"Mean Dist [cm]: {round(mean_distance_cm, 2)}", (10, 30), 
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
    if marker_count > 0:
        mean_distance_m = total_distance_m / marker_count
        cv2.putText(frame, f"Mean Dist [m]: {round(mean_distance_m, 2)}", (10, 60), 
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()