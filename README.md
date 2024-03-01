# Exploring Auxiliary Sensors and Algorithms for Autonomous Surface Vessel Positioning
Project thesis NTNU, December 2023

![80x80_mr_stl_detected](https://github.com/eirikese/exploring_usv_positioning/assets/118887178/ce11d1c3-716a-43c6-8704-107e95e6fb40)



## Description
This repository contains all the code, documentation, and resources for our project thesis in collaboration with Maritime Robotics. Here, you'll find the algorithms and software developed for robust positioning of uncrewed surface vessels (USVs) in harbor environments, with a focus on sensor fusion using LiDAR and RGB cameras. This repo serves as a resource for the project thesis.

## Code overview 
The code is divided into three main parts, one for each positioning method.
* Relative Positioning using Fiducial Markers
* Point Cloud Object Detection and Ranging
* Point Cloud Map Fitting

The code for Point Cloud Object Detection is developed using ROS Noetic and C++, on Ubuntu 20.04. The PCL library is used for point cloud clustering and pose estimation. The lidar used for testing is Ouster OS1-64, with data transfer via ROS pointcloud2 messages.
Relative Positioning using Fiducial Markers utilizes OpenCV, together with ArUco markers for detection and relative positioning. Three different approaches have been tested for map fitting.

| Approach | Code |
|-|-|
| OpenCV homography and longside fitting | cv2_map |
| Fiducial marker detection using an RGB camera | marker_detection_rgb |
| Centroid-aligned, iterative rotation, ICP fitting | o3d_map |
| Point Cloud Cluster Detection | pointcloud_cluster_detection |
| Point Cloud Filtering | pointcloud_filter |
| RANSAC single-line fitting | ransac_map |
| Weather forecast using APIs | weather_forecasts |


More details on the functionality of each method can be found in the top section of the code files.
