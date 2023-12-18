# PointCloud Filter ROS Package

The `pointcloud_filter` ROS package is designed to filter PointCloud2 data based on intensity thresholds. It subscribes to a PointCloud2 topic, filters the points based on intensity values within a specified range, and publishes the filtered point cloud to a new topic.

## Functionality

This package provides a ROS node called `pointcloud_filter_node` that performs the following tasks:

1. Subscribes to a PointCloud2 topic (e.g., "/ouster/points").
2. Filters the points based on intensity values within a specified range.
3. Publishes the filtered PointCloud2 data to a new topic (e.g., "/ouster/filtered").

## Usage

To use the `pointcloud_filter` package, follow these steps:

1. **Installation**:

   Make sure you have ROS installed, and the package is built within your Catkin workspace.

2. **Launch the Node**:

   Launch the `pointcloud_filter_node` using a ROS launch file. By default, it subscribes to the "/ouster/points" topic and filters points based on the intensity range of [0.0, 255.0]. You can modify these settings by editing the launch file.

   ```bash
   roslaunch pointcloud_filter pointcloud_filter.launch
