// Include necessary header files
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
#include <sstream>

// Define a class for the point cloud cluster detection
class PointCloudClusterDetection
{
public:
  PointCloudClusterDetection(ros::NodeHandle nh) : nh_(nh)
  {
    // Subscribe to the filtered PointCloud2 topic
    cloud_sub_ = nh_.subscribe("/ouster/filtered", 1, &PointCloudClusterDetection::cloudCallback, this);

    // Publish bounding boxes
    bbox_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/object_bounding_boxes", 1);

    // // Load object corners from CSV file
    // if (nh_.getParam("csv_file_path", csv_file_path_))
    // {
    //   loadObjectCorners();
    // }
    // else
    // {
    //   ROS_ERROR("Failed to load CSV file path from parameter server.");
    // }
  }

  void loadObjectCorners()
  {
    std::ifstream file("/home/eirik18/lidar_ws/src/pointcloud_cluster_detection/csv/object_parameters.csv");
    if (file.is_open())
    {
      std::string line;
      while (getline(file, line))
      {
        std::istringstream ss(line);
        pcl::PointXYZI corner;
        if (ss >> corner.x >> corner.y >> corner.z)
        {
          object_corners_.push_back(corner);
        }
      }
      file.close();
    }
    else
    {
      ROS_ERROR("Failed to open CSV file.");
    }
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
  {
    // Convert PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *cloud);

    // Initialize a flag to indicate object detection
    bool object_detected = false;

    // Define a tolerance for point inclusion in the object
    const double point_tolerance = 0.1; // Adjust as needed

    // Perform cluster extraction based on intensity
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setInputCloud(cloud);
    ec.setClusterTolerance(0.2);  // Adjust as needed
    ec.setMinClusterSize(10);    // Adjust as needed
    ec.setMaxClusterSize(1000);  // Adjust as needed
    ec.setSearchMethod(boost::make_shared<pcl::search::KdTree<pcl::PointXYZI>>());
    ec.extract(cluster_indices);

    // Loop through each cluster
    for (const auto& indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      pcl::ExtractIndices<pcl::PointXYZI> extract;
      extract.setInputCloud(cloud);
      pcl::PointIndices::Ptr pcl_indices(new pcl::PointIndices(indices));
      extract.setIndices(pcl_indices);
      extract.filter(*cluster);

      // Check if the cluster contains all object corners
      bool contains_all_corners = true;
      for (const auto& corner : object_corners_)
      {
        bool corner_found = false;
        for (const auto& point : cluster->points)
        {
          double distance = sqrt(pow(point.x - corner.x, 2) + pow(point.y - corner.y, 2) + pow(point.z - corner.z, 2)); // Calculate distance

          if (distance < point_tolerance)
          {
            corner_found = true;
            break;
          }
        }
        if (!corner_found)
        {
          contains_all_corners = false;
          break;
        }
      }

      // If all object corners are within the cluster, consider it as the detected object
      if (contains_all_corners)
      {
        object_detected = true;

        // Calculate bounding box dimensions
        pcl::PointXYZI min_pt, max_pt;
        pcl::getMinMax3D(*cluster, min_pt, max_pt);

        // Create a visualization marker for the bounding box
        visualization_msgs::Marker bbox_marker;
        bbox_marker.header.frame_id = input_cloud->header.frame_id;
        bbox_marker.header.stamp = ros::Time::now();
        bbox_marker.ns = "object";
        bbox_marker.id = 0;
        bbox_marker.type = visualization_msgs::Marker::CUBE;
        bbox_marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
        bbox_marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
        bbox_marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
        bbox_marker.pose.orientation.x = 0.0;
        bbox_marker.pose.orientation.y = 0.0;
        bbox_marker.pose.orientation.z = 0.0;
        bbox_marker.pose.orientation.w = 1.0;
        bbox_marker.scale.x = max_pt.x - min_pt.x;
        bbox_marker.scale.y = max_pt.y - min_pt.y;
        bbox_marker.scale.z = max_pt.z - min_pt.z;
        bbox_marker.color.r = 0.0;
        bbox_marker.color.g = 1.0;
        bbox_marker.color.b = 0.0;
        bbox_marker.color.a = 0.5;
        bbox_marker.lifetime = ros::Duration(0.0); // Adjust as needed

        // Create a MarkerArray and add the bounding box marker
        visualization_msgs::MarkerArray marker_array;
        marker_array.markers.push_back(bbox_marker);

        // Publish the marker array
        bbox_pub_.publish(marker_array);

        // Break out of the loop if you only want to detect one object
        break;
      }
    }

    // Publish the result or take further action based on object_detected
    if (object_detected)
    {
      ROS_INFO("Object detected!");
    }
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher bbox_pub_;
  std::string csv_file_path_;
  std::vector<pcl::PointXYZI> object_corners_;
};

// Main function
int main(int argc, char** argv)
{
  // Initialize ROS
  ros::init(argc, argv, "pointcloud_cluster_detection");
  ros::NodeHandle nh;

  // Create an instance of the PointCloudClusterDetection class
  PointCloudClusterDetection pcl_cluster_detection(nh);

  // Start ROS node
  ros::spin();

  return 0;
}
