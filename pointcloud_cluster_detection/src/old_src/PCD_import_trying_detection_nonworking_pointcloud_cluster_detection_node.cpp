#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>

class PointCloudClusterDetection
{
public:
  PointCloudClusterDetection(ros::NodeHandle nh) : nh_(nh)
  {
    // Subscribe to the filtered PointCloud2 topic
    cloud_sub_ = nh_.subscribe("/ouster/filtered", 1, &PointCloudClusterDetection::cloudCallback, this);

    // Publish bounding boxes
    bbox_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/cluster_bounding_boxes", 1);

    // Publish object poses
    object_pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/object_poses", 1);

    // Initialize the object publisher
    object_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/loaded_object", 1);

    // Load the 3D object from the PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr temp_object(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/eirik18/lidar_ws/src/pointcloud_cluster_detection/objects/80x80free2.pcd", *temp_object) == -1)
    {
      ROS_ERROR("Failed to load PCD file.");
      loaded_object_.reset(); // Reset loaded_object_ to nullptr if loading fails
    }
    else
    {
      loaded_object_ = temp_object; // Assign the loaded object to loaded_object_
      ROS_INFO("Loaded PCD file successfully.");
    }
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
  {
    // Convert PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *cloud);

    // Perform feature matching to check if the cluster resembles the loaded object
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(loaded_object_);

    // Convert cloud to pcl::PointCloud<pcl::PointXYZ>
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *cloud_xyz);

    icp.setInputTarget(cloud_xyz);  // Use the converted cloud
    pcl::PointCloud<pcl::PointXYZ> aligned_cluster;
    icp.align(aligned_cluster);

    if (icp.hasConverged() && icp.getFitnessScore() < 0.1) // icp fitness, lower is more strict, default 0.1
    {
      // The cluster matches the loaded object
      visualization_msgs::Marker bbox_marker;
      bbox_marker.header.frame_id = input_cloud->header.frame_id;
      bbox_marker.header.stamp = ros::Time::now();
      bbox_marker.ns = "cluster";
      bbox_marker.id = 0;  // Using a single ID since we have only one object
      bbox_marker.type = visualization_msgs::Marker::CUBE;
      bbox_marker.pose.position.x = 0.0;  // Modify if needed
      bbox_marker.pose.position.y = 0.0;  // Modify if needed
      bbox_marker.pose.position.z = 0.0;  // Modify if needed
      bbox_marker.pose.orientation.x = 0.0;
      bbox_marker.pose.orientation.y = 0.0;
      bbox_marker.pose.orientation.z = 0.0;
      bbox_marker.pose.orientation.w = 1.0;
      bbox_marker.scale.x = 1.0;  // Modify if needed
      bbox_marker.scale.y = 1.0;  // Modify if needed
      bbox_marker.scale.z = 1.0;  // Modify if needed
      
      bbox_marker.color.r = 0.0;
      bbox_marker.color.g = 1.0;
      bbox_marker.color.b = 0.0;
      bbox_marker.color.a = 0.5;
      bbox_marker.lifetime = ros::Duration(1.0);  // Adjust as needed

      // Create PoseStamped message for the object pose
      geometry_msgs::PoseStamped object_pose_stamped;
      object_pose_stamped.header = input_cloud->header;
      object_pose_stamped.pose.position.x = 0.0;  // Modify if needed
      object_pose_stamped.pose.position.y = 0.0;  // Modify if needed
      object_pose_stamped.pose.position.z = 0.0;  // Modify if needed

      // Create a Quaternion with no rotation
      geometry_msgs::Quaternion bbox_quaternion_msg;
      bbox_quaternion_msg.x = 0.0;
      bbox_quaternion_msg.y = 0.0;
      bbox_quaternion_msg.z = 0.0;
      bbox_quaternion_msg.w = 1.0;

      // Assign the converted quaternion to object_pose_stamped
      object_pose_stamped.pose.orientation = bbox_quaternion_msg;

      // Add the object pose to the PoseArray
      geometry_msgs::PoseArray object_poses;
      object_poses.header = input_cloud->header;
      object_poses.poses.push_back(object_pose_stamped.pose);

      // Publish the marker array
      visualization_msgs::MarkerArray marker_array;
      marker_array.markers.push_back(bbox_marker);
      bbox_pub_.publish(marker_array);

      // Publish the object poses
      object_pose_pub_.publish(object_poses);

      // Publish the loaded object continuously
      publishLoadedObject(input_cloud->header.frame_id);
    }
  }

  void publishLoadedObject(const std::string& frame_id)
  {
    // Create a PointCloud2 message to publish the loaded object
    sensor_msgs::PointCloud2 loaded_object_msg;
    pcl::toROSMsg(*loaded_object_, loaded_object_msg);
    loaded_object_msg.header.frame_id = frame_id;
    loaded_object_msg.header.stamp = ros::Time::now();
    object_pub_.publish(loaded_object_msg);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher bbox_pub_;
  ros::Publisher object_pose_pub_;
  ros::Publisher object_pub_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr loaded_object_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointcloud_cluster_detection");
  ros::NodeHandle nh;

  PointCloudClusterDetection pcl_cluster_detection(nh);

  ros::spin();
  return 0;
}
