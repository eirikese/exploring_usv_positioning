#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense> // Include Eigen for matrix operations

class PointCloudClusterDetection
{
public:
  PointCloudClusterDetection(ros::NodeHandle nh, const std::string &csv_file_path, double cluster_tolerance, int min_cluster_size, int max_cluster_size, double dimension_threshold)
      : nh_(nh), csv_file_path_(csv_file_path), cluster_tolerance_(cluster_tolerance), min_cluster_size_(min_cluster_size), max_cluster_size_(max_cluster_size), dimension_threshold_(dimension_threshold)
  {
    loadObjectParameters(); // Load object dimensions and orientation from CSV

    // Subscribe to the filtered PointCloud2 topic
    cloud_sub_ = nh_.subscribe("/ouster/filtered", 1, &PointCloudClusterDetection::cloudCallback, this);

    // Publish bounding boxes
    bbox_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/cluster_bounding_boxes", 1);

    // Publish the object at the origin
    object_pub_ = nh_.advertise<visualization_msgs::Marker>("/object_at_origin", 1);
  }

  void loadObjectParameters()
  {
    // Read object parameters from the CSV file
    std::ifstream csv_file(csv_file_path_);
    if (csv_file.is_open())
    {
      std::string line;
      while (std::getline(csv_file, line))
      {
        std::istringstream iss(line);
        std::string shape;
        double width, height, depth, rotation;
        if (iss >> shape >> width >> height >> depth >> rotation)
        {
          if (shape == "box")
          {
            object_width_ = width;
            object_height_ = height;
            object_depth_ = depth;
            object_rotation_ = rotation;
            break; // Assuming only one object definition in the CSV
          }
        }
      }
      csv_file.close();
    }
    else
    {
      ROS_ERROR("Failed to open the CSV file: %s", csv_file_path_.c_str());
      ros::shutdown();
    }
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &input_cloud)
  {
    // Convert PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *cloud);

    // Create a matrix for the object shape transformation
    Eigen::Affine3f object_transform(Eigen::Affine3f::Identity());
    object_transform.translation() << 0.0, 0.0, 0.0; // Object at the origin
    object_transform.rotate(Eigen::AngleAxisf(object_rotation_, Eigen::Vector3f::UnitZ()));

    // Transform the object dimensions based on rotation
    Eigen::Vector3f object_dimensions(object_width_, object_height_, object_depth_);
    object_dimensions = object_transform * object_dimensions;

    // Define a threshold for matching object dimensions
    double dimension_threshold = dimension_threshold_; // Adjust as needed

    // Perform cluster extraction based on intensity
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setInputCloud(cloud);
    ec.setClusterTolerance(cluster_tolerance_); // Adjust as needed
    ec.setMinClusterSize(min_cluster_size_);    // Adjust as needed
    ec.setMaxClusterSize(max_cluster_size_);  // Adjust as needed
    ec.setSearchMethod(boost::make_shared<pcl::search::KdTree<pcl::PointXYZI>>());
    ec.extract(cluster_indices);

    // Create visualization markers for bounding boxes
    visualization_msgs::MarkerArray marker_array;
    int cluster_id = 0;

    for (const auto &indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::ExtractIndices<pcl::PointXYZI> extract;
      extract.setInputCloud(cloud);
      pcl::PointIndices::Ptr pcl_indices(new pcl::PointIndices(indices));
      extract.setIndices(pcl_indices);
      extract.filter(*cluster);

      // Calculate bounding box dimensions
      pcl::PointXYZI min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);

      // Check if the cluster dimensions match the object dimensions
      double cluster_width = max_pt.x - min_pt.x;
      double cluster_height = max_pt.y - min_pt.y;
      double cluster_depth = max_pt.z - min_pt.z;

      if (std::abs(cluster_width - object_dimensions.x()) < dimension_threshold &&
          std::abs(cluster_height - object_dimensions.y()) < dimension_threshold &&
          std::abs(cluster_depth - object_dimensions.z()) < dimension_threshold)
      {
        // Cluster dimensions match the object dimensions
        visualization_msgs::Marker bbox_marker;
        bbox_marker.header.frame_id = input_cloud->header.frame_id;
        bbox_marker.header.stamp = ros::Time::now();
        bbox_marker.ns = "cluster";
        bbox_marker.id = cluster_id;
        bbox_marker.type = visualization_msgs::Marker::CUBE;
        bbox_marker.action = visualization_msgs::Marker::ADD;
        bbox_marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
        bbox_marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
        bbox_marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
        bbox_marker.pose.orientation.x = 0.0;
        bbox_marker.pose.orientation.y = 0.0;
        bbox_marker.pose.orientation.z = 0.0;
        bbox_marker.pose.orientation.w = 1.0;
        bbox_marker.scale.x = cluster_width;
        bbox_marker.scale.y = cluster_height;
        bbox_marker.scale.z = cluster_depth;
        bbox_marker.color.r = 0.0;
        bbox_marker.color.g = 1.0;
        bbox_marker.color.b = 0.0;
        bbox_marker.color.a = 0.5;
        bbox_marker.lifetime = ros::Duration(1.0); // Adjust as needed

        marker_array.markers.push_back(bbox_marker);

        cluster_id++;
      }
    }

    // Publish the bounding boxes
    bbox_pub_.publish(marker_array);

    // Create a marker for the object at the origin
    visualization_msgs::Marker object_marker;
    object_marker.header.frame_id = input_cloud->header.frame_id;
    object_marker.header.stamp = ros::Time::now();
    object_marker.ns = "object";
    object_marker.id = 0;
    object_marker.type = visualization_msgs::Marker::CUBE;
    object_marker.pose.position.x = 0.0;
    object_marker.pose.position.y = 0.0;
    object_marker.pose.position.z = 0.0;
    object_marker.pose.orientation.x = 0.0;
    object_marker.pose.orientation.y = 0.0;
    object_marker.pose.orientation.z = 0.0;
    object_marker.pose.orientation.w = 1.0;
    object_marker.scale.x = object_dimensions.x();
    object_marker.scale.y = object_dimensions.y();
    object_marker.scale.z = object_dimensions.z();
    object_marker.color.r = 1.0;
    object_marker.color.g = 0.0;
    object_marker.color.b = 0.0;
    object_marker.color.a = 0.5;
    object_marker.lifetime = ros::Duration(0.0); //does not expire

    // Publish the object marker
    object_pub_.publish(object_marker);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher bbox_pub_;
  ros::Publisher object_pub_;
  std::string csv_file_path_;
  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;
  double dimension_threshold_;
  double object_width_;
  double object_height_;
  double object_depth_;
  double object_rotation_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pointcloud_cluster_detection");
  ros::NodeHandle nh;
  std::string csv_file_path = "/home/eirik18/lidar_ws/src/pointcloud_cluster_detection/csv/object_parameters.csv"; // Provide the correct path
  double cluster_tolerance = 0.2; // Set the cluster tolerance as needed
  int min_cluster_size = 10; // Set the minimum cluster size as needed
  int max_cluster_size = 5000; // Set the maximum cluster size as needed
  double dimension_threshold = 0.3; // Set the dimension threshold as needed

  PointCloudClusterDetection cluster_detection(nh, csv_file_path, cluster_tolerance, min_cluster_size, max_cluster_size, dimension_threshold);

  ros::spin();

  return 0;
}
