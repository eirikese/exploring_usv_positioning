cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_cluster_detection)

## Compile as C++11, supported in ROS Noetic and newer
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  sensor_msgs
  std_msgs
  pcl_conversions
  visualization_msgs
)

find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp pcl_ros sensor_msgs std_msgs pcl_conversions visualization_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


add_executable(pointcloud_cluster_detection_node src/pointcloud_cluster_detection_node.cpp)

target_link_libraries(pointcloud_cluster_detection_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS pointcloud_cluster_detection_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


add_executable(pcl_edge_detection_node src/pcl_edge_detection_node.cpp)

target_link_libraries(pcl_edge_detection_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS pcl_edge_detection_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_install_python(PROGRAMS scripts/position_plotter.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
