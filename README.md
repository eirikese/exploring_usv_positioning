# Exploring Auxiliary Sensors and Algorithms for Autonomous Surface Vessel Positioning
Project thesis NTNU, December 2023

![80x80_mr_stl_detected](https://github.com/eirikese/exploring_usv_positioning/assets/118887178/ce11d1c3-716a-43c6-8704-107e95e6fb40)



## Description
This repository contains all the code, documentation, and resources for our project thesis in collaboration with Maritime Robotics. Here, you'll find the algorithms and software developed for robust positioning of uncrewed surface vessels (USVs) in harbor environments, with a focus on sensor fusion using LiDAR and RGB cameras. This repo serves as a comprehensive resource for anyone interested in maritime robotics, autonomous navigation, and sensor fusion technology.

## Code overview 
The code is divided into three main parts, one for each positioning method.
* Relative Positioning using Fiducial Markers
* Point Cloud Object Detection and Ranging
* Point Cloud Map Fitting

The code for Point Cloud Object Detection is developed using ROS Noetic and C++, on Ubuntu 20.04. The PCL library is used for point cloud clustering and pose estimation. The lidar used for testing is Ouster OS1-64, with data transfer via ROS pointcloud2 messages.
Relative Positioning using Fiducial Markers utilizes OpenCV, together with ArUco markers for detection and relative positioning.
The Map Fitting algorithm uses three different approaches to fit a source point cloud to a noisy target: 

| Approach | Code |
|-|-|
| Fiducial marker detection using an RGB camera | fiducial marker detection_RGB |
| Weather forecast using APIs | weather_forecasts |
| RANSAC single-line fitting | ransac_fit |
| OpenCV homography and longside fitting | cv2_map |
| Centroid-aligned, iterative rotation, ICP fitting | o3d_map |


More details on the functionality of each method can be found in the top section of the code files.


>**Acknowledgement**:
>*During the preparation of this work the authors used ChatGPT from OpenAI in order to improve readability of the code. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.*
