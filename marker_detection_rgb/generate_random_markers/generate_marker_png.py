import cv2
from cv2 import aruco
import random

# Specifies which dictionary we generate markers from
marker_dict_4x4_50 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
marker_dict_4x4_100 = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
marker_dict_4x4_250 = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
marker_dict_4x4_1000 = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

marker_dict_5x5_50 = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
marker_dict_5x5_100 = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
marker_dict_5x5_250 = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
marker_dict_5x5_1000 = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

marker_dict_6x6_50 = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
marker_dict_6x6_100 = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
marker_dict_6x6_250 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_dict_6x6_1000 = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

marker_dict_7x7_50 = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
marker_dict_7x7_100 = aruco.getPredefinedDictionary(aruco.DICT_7X7_100)
marker_dict_7x7_250 = aruco.getPredefinedDictionary(aruco.DICT_7X7_250)
marker_dict_7x7_1000 = aruco.getPredefinedDictionary(aruco.DICT_7X7_1000)


marker_size = 1000  # [pixels]
ant_markers = 5
for i in range(ant_markers):  # Generates 5 random markers
    
    id = random.randint(0, 49)
    
    marker_image = aruco.generateImageMarker(marker_dict_7x7_50, id, marker_size)
    # cv2.imshow("img", marker_image)
    cv2.imwrite(f"Markers/Markers_7x7/50/marker_{id}.png", marker_image)
    # cv2.waitKey(0)
    # break