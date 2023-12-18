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

class PointCloudClusterDetection
{
public:
  PointCloudClusterDetection(ros::NodeHandle nh) : nh_(nh)
  {
    // Subscribe to the filtered PointCloud2 topic
    cloud_sub_ = nh_.subscribe("/ouster/filtered", 1, &PointCloudClusterDetection::cloudCallback, this);

    // Publish bounding boxes
    bbox_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/cluster_bounding_boxes", 1);
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
  {
    // Convert PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *cloud);

    // Perform cluster extraction based on intensity
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setInputCloud(cloud);
    ec.setClusterTolerance(0.1);  // Adjust as needed
    ec.setMinClusterSize(10);    // Adjust as needed
    ec.setMaxClusterSize(1000);  // Adjust as needed
    ec.setSearchMethod(boost::make_shared<pcl::search::KdTree<pcl::PointXYZI>>());
    ec.extract(cluster_indices);

    // Create visualization markers for bounding boxes
    visualization_msgs::MarkerArray marker_array;
    int cluster_id = 0;

    for (const auto& indices : cluster_indices)
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

      visualization_msgs::Marker bbox_marker;
      bbox_marker.header.frame_id = input_cloud->header.frame_id;
      bbox_marker.header.stamp = ros::Time::now();
      bbox_marker.ns = "cluster";
      bbox_marker.id = cluster_id;
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
      bbox_marker.lifetime = ros::Duration(1.0);  // Adjust as needed
      marker_array.markers.push_back(bbox_marker);

      cluster_id++;
    }

    // Publish the marker array
    bbox_pub_.publish(marker_array);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher bbox_pub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointcloud_cluster_detection");
  ros::NodeHandle nh;

  PointCloudClusterDetection pcl_cluster_detection(nh);

  ros::spin();
  return 0;
}
