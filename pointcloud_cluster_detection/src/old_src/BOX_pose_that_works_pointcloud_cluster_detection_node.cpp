// Pointcloud cluster detection and bounding boxes with pose xyz. Works well with:
// "/home/eirik18/lidar_ws/src/pointcloud_filter/bags/troll_lidar_tag_test3.bag"
// and filter 100-150
// (adjust this in pointcloud_filter.launch)

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
#include <pcl/common/transforms.h>
#include <tf/transform_listener.h>
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

    // Read parameters from the launch file
    nh_.param("min_size_x", min_size_x_, 0.1);  // Default size in X
    nh_.param("min_size_y", min_size_y_, 1.0);  // Default size in Y
    nh_.param("min_size_z", min_size_z_, 1.0);  // Default size in Z
    nh_.param("tolerance",  tolerance_,  0.3);  // Default tolerance for size matching
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

    // Create PoseArray to store object poses
    geometry_msgs::PoseArray object_poses;
    object_poses.header = input_cloud->header;

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

      // Calculate cluster orientation using PCA
      Eigen::Vector4f pca_centroid;
      Eigen::Matrix3f covariance_matrix;
      pcl::compute3DCentroid(*cluster, pca_centroid);
      pcl::computeCovarianceMatrixNormalized(*cluster, pca_centroid, covariance_matrix);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
      Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();

      // The eigenvector with the smallest eigenvalue corresponds to the cluster's orientation
      Eigen::Vector3f cluster_orientation = eigen_vectors.col(0);

      // Calculate the cluster dimensions
      double cluster_size_x = max_pt.x - min_pt.x;
      double cluster_size_y = max_pt.y - min_pt.y;
      double cluster_size_z = max_pt.z - min_pt.z;

      // Create a transformation matrix to rotate the bounding box
      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.translation() << (min_pt.x + max_pt.x) / 2.0, (min_pt.y + max_pt.y) / 2.0, (min_pt.z + max_pt.z) / 2.0;
      transform.linear().col(0) = cluster_orientation;
      transform.linear().col(1) = Eigen::Vector3f::UnitY(); // Assuming Y-axis as the "up" direction
      transform.linear().col(2) = cluster_orientation.cross(Eigen::Vector3f::UnitY());

      // Apply the transformation to the bounding box
      pcl::transformPointCloud(*cluster, *cluster, transform);

      // Calculate the absolute differences
      double size_diff_x = std::abs(cluster_size_x - min_size_x_);
      double size_diff_y = std::abs(cluster_size_y - min_size_y_);
      double size_diff_z = std::abs(cluster_size_z - min_size_z_);

      // The cluster meets the size criteria
      if (size_diff_x <= tolerance_ &&
          size_diff_y <= tolerance_ &&
          size_diff_z <= tolerance_)
      {
        // Set the orientation of the bbox_marker using the quaternion that corresponds to the rotation matrix
        Eigen::Quaternionf bbox_quaternion(transform.linear());
        visualization_msgs::Marker bbox_marker;
        bbox_marker.header.frame_id = input_cloud->header.frame_id;
        bbox_marker.header.stamp = ros::Time::now();
        bbox_marker.ns = "cluster";
        bbox_marker.id = cluster_id;
        bbox_marker.type = visualization_msgs::Marker::CUBE;
        bbox_marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
        bbox_marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
        bbox_marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
        bbox_marker.pose.orientation.x = bbox_quaternion.x();
        bbox_marker.pose.orientation.y = bbox_quaternion.y();
        bbox_marker.pose.orientation.z = bbox_quaternion.z();
        bbox_marker.pose.orientation.w = bbox_quaternion.w();
        bbox_marker.scale.x = cluster_size_x;
        bbox_marker.scale.y = cluster_size_y;
        bbox_marker.scale.z = cluster_size_z;
        bbox_marker.color.r = 0.0;
        bbox_marker.color.g = 1.0;
        bbox_marker.color.b = 0.0;
        bbox_marker.color.a = 0.5;
        bbox_marker.lifetime = ros::Duration(1.0);  // Adjust as needed
        marker_array.markers.push_back(bbox_marker);

        // Create PoseStamped message for the object pose
        geometry_msgs::PoseStamped object_pose_stamped;
        object_pose_stamped.header = input_cloud->header;
        object_pose_stamped.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
        object_pose_stamped.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
        object_pose_stamped.pose.position.z = (min_pt.z + max_pt.z) / 2.0;

        // Convert the Eigen quaternion to geometry_msgs::Quaternion
        geometry_msgs::Quaternion bbox_quaternion_msg;
        bbox_quaternion_msg.x = bbox_quaternion.x();
        bbox_quaternion_msg.y = bbox_quaternion.y();
        bbox_quaternion_msg.z = bbox_quaternion.z();
        bbox_quaternion_msg.w = bbox_quaternion.w();

        // Assign the converted quaternion to object_pose_stamped
        object_pose_stamped.pose.orientation = bbox_quaternion_msg;

        // Add the object pose to the PoseArray
        object_poses.poses.push_back(object_pose_stamped.pose);

        cluster_id++;
      }
    }

    // Publish the marker array
    bbox_pub_.publish(marker_array);

    // Publish the object poses
    object_pose_pub_.publish(object_poses);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher bbox_pub_;
  ros::Publisher object_pose_pub_; 
  double min_size_x_;
  double min_size_y_;
  double min_size_z_;
  double tolerance_;
  tf::TransformListener tf_listener; 
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointcloud_cluster_detection");
  ros::NodeHandle nh;

  PointCloudClusterDetection pcl_cluster_detection(nh);

  ros::spin();
  return 0;
}
