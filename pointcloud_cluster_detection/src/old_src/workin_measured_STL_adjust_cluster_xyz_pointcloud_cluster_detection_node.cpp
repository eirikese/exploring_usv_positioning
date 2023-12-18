#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/common/transforms.h>
#include <tf/transform_listener.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <geometry_msgs/PoseStamped.h>

class PointCloudClusterDetection
{
public:
  PointCloudClusterDetection(ros::NodeHandle nh) : nh_(nh)
  {
    // Subscribe to the filtered PointCloud2 topic
    cloud_sub_ = nh_.subscribe("/ouster/filtered", 1, &PointCloudClusterDetection::cloudCallback, this);

    // Publish the mesh visualization
    stl_mesh_pub_ = nh_.advertise<visualization_msgs::Marker>("/stl_mesh", 1);

    // Initialize the object publisher
    stl_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/stl_cloud", 1);

    // Publish bounding boxes
    bbox_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/cluster_bounding_boxes", 1);

    // Publish object poses
    object_pose_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/detected_object_pose", 1);

    // Publish detected object mesh
    // detected_object_mesh_pub_ = nh_.advertise<visualization_msgs::Marker>("/detected_object_mesh", 1);

    os_sensor_frame_id_ = "os_sensor"; // Update with the correct frame name
    stl_file_ = "/home/mr_fusion/lidar_ws/src/pointcloud_cluster_detection/objects/80x80box.stl";
    // stl_file_ = "/home/mr_fusion/lidar_ws/src/pointcloud_cluster_detection/objects/31x37x39_box.STL";

    tolerance_ = 0.2; // tolerance for cluster fitting in meters ///////////////////////////////////////////////////////////////////

    // load stl object
    loadSTL();
  }

  void loadSTL()
  {
    // Load the STL mesh
    if (pcl::io::loadPolygonFileSTL(stl_file_, object_mesh_) == -1)
    {
      ROS_ERROR("Failed to load STL file: %s", stl_file_.c_str());
    }
    else
    {
      ROS_INFO("Successfully loaded STL file: %s", stl_file_.c_str());
    }

    // Convert the mesh to a point cloud for visualization
    pcl::PointCloud<pcl::PointXYZ>::Ptr loaded_object(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(object_mesh_.cloud, *loaded_object);

    // Apply scaling to the loaded object
    for (auto &point : loaded_object->points)
    {
      point.x *= millimeters_to_meters;
      point.y *= millimeters_to_meters;
      point.z *= millimeters_to_meters;
    }

    // Convert the PCL point cloud to a ROS point cloud message
    pcl::toROSMsg(*loaded_object, stl_object_msg);

    // Set the frame ID to "os_sensor" (or the appropriate frame name)
    stl_object_msg.header.frame_id = os_sensor_frame_id_;

    // Create a mesh visualization marker for the imported STL
    stl_mesh_marker.header.frame_id = os_sensor_frame_id_;
    stl_mesh_marker.header.stamp = ros::Time::now();
    stl_mesh_marker.ns = "mesh";
    stl_mesh_marker.id = 0;
    stl_mesh_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
    stl_mesh_marker.mesh_resource = "file://" + stl_file_; // Use the absolute path with "file://"
    stl_mesh_marker.action = visualization_msgs::Marker::ADD;
    stl_mesh_marker.pose.orientation.w = 1.0;
    stl_mesh_marker.scale.x = stl_mesh_marker.scale.y = stl_mesh_marker.scale.z = 1.0 * millimeters_to_meters;
    stl_mesh_marker.color.r = stl_mesh_marker.color.g = stl_mesh_marker.color.b = 1.0; // rgb color
    stl_mesh_marker.color.a = 0.6; // transparency

    calculatePointCloudDimensions(stl_object_msg, stl_size_x, stl_size_y, stl_size_z);

    // Print dimensions
    ROS_INFO("Dimensions of the imported STL object:");
    ROS_INFO("x: %.3f meters", stl_size_x);
    ROS_INFO("y: %.3f meters", stl_size_y);
    ROS_INFO("z: %.3f meters", stl_size_z);
  }

  void calculatePointCloudDimensions(const sensor_msgs::PointCloud2 input_cloud, double &width, double &length, double &height)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr stl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(input_cloud, *stl_cloud);

    double min_x = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_z = -std::numeric_limits<double>::max();

    for (const auto &point : stl_cloud->points)
    {
      min_x = std::min(min_x, static_cast<double>(point.x));
      max_x = std::max(max_x, static_cast<double>(point.x));
      min_y = std::min(min_y, static_cast<double>(point.y));
      max_y = std::max(max_y, static_cast<double>(point.y));
      min_z = std::min(min_z, static_cast<double>(point.z));
      max_z = std::max(max_z, static_cast<double>(point.z));
    }

    width = max_x - min_x;
    length = max_y - min_y;
    height = max_z - min_z;
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &input_cloud)
  {
    // Convert PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *cloud);

    // Perform cluster extraction based on intensity
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setInputCloud(cloud);
    ec.setClusterTolerance(0.1);  // Adjust as needed //////////////////////////////////////////////////////////////////////
    ec.setMinClusterSize(10);    // Adjust as needed
    ec.setMaxClusterSize(1000000);  // Adjust as needed
    ec.setSearchMethod(boost::make_shared<pcl::search::KdTree<pcl::PointXYZI>>());
    ec.extract(cluster_indices);

    // Create visualization markers for bounding boxes
    visualization_msgs::MarkerArray marker_array;
    int cluster_id = 0;

    // Create PoseArray to store object poses
    geometry_msgs::PoseArray object_poses;
    object_poses.header = input_cloud->header;

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

      // The cluster meets the size criteria
      if (std::abs(cluster_size_x - stl_size_x) <= tolerance_ &&
          std::abs(cluster_size_y - stl_size_y) <= tolerance_ &&
          std::abs(cluster_size_z - stl_size_z) <= tolerance_)
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

    // Publish
    stl_mesh_pub_.publish(stl_mesh_marker);
    stl_cloud_pub_.publish(stl_object_msg);
    bbox_pub_.publish(marker_array);
    object_pose_pub_.publish(object_poses);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Publisher bbox_pub_;
  ros::Publisher object_pose_pub_;
  ros::Publisher detected_object_mesh_pub_;
  ros::Publisher stl_mesh_pub_;
  ros::Publisher stl_cloud_pub_;
  pcl::PolygonMesh object_mesh_;
  std::string os_sensor_frame_id_;
  std::string stl_file_;
  visualization_msgs::Marker stl_mesh_marker;
  sensor_msgs::PointCloud2 stl_object_msg;

  const float millimeters_to_meters = 0.001;
  double stl_size_x, stl_size_y, stl_size_z;
  double tolerance_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "pointcloud_cluster_detection");
  ros::NodeHandle nh;

  PointCloudClusterDetection pcl_cluster_detection(nh);

  ros::spin();
  return 0;
}
