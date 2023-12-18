#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/Marker.h>
#include <pcl/common/transforms.h>
#include <tf/transform_listener.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/cloud_viewer.h>

class PointCloudClusterDetection
{
public:
  PointCloudClusterDetection(ros::NodeHandle nh) : nh_(nh)
  {
    // Subscribe to the filtered PointCloud2 topic
    lidar_cloud_sub_ = nh_.subscribe("/ouster/filtered", 1, &PointCloudClusterDetection::cloudCallback, this);

    // Publish the mesh visualization
    stl_mesh_pub_ = nh_.advertise<visualization_msgs::Marker>("/stl_mesh", 1);

    // Initialize the object publisher
    stl_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/stl_cloud", 1);

    // Initialize the detected object mesh publisher
    detected_object_mesh_pub_ = nh_.advertise<visualization_msgs::Marker>("/detected_object_mesh", 1);

    // Initialize the detected object pose publisher
    detected_object_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/detected_object_pose", 1);

    // Initialize the bounding box publisher
    bbox_pub_ = nh_.advertise<visualization_msgs::Marker>("/cluster_bounding_boxes", 1);
    
    // Set the STL file path
    // stl_file_ = "/home/eirik18/lidar_ws/src/pointcloud_cluster_detection/objects/Magic_Kitten.stl";
    stl_file_ = "/home/eirik18/lidar_ws/src/pointcloud_cluster_detection/objects/80x80box.stl";

    // Set the frame ID to "os_sensor" (or the appropriate frame name)
    os_sensor_frame_id_ = "os_sensor"; // Update with the correct frame name

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

    // Set the scaling factor for millimeters to meters (0.001)
    const float millimeters_to_meters = 0.001;

    // Apply scaling to the loaded object
    for (auto& point : loaded_object->points)
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
    stl_mesh_marker.color.r = stl_mesh_marker.color.g = stl_mesh_marker.color.b = stl_mesh_marker.color.a = 1.0;
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& input_cloud)
  {
    // Publish the STL object
    stl_mesh_pub_.publish(stl_mesh_marker);
    stl_cloud_pub_.publish(stl_object_msg);


    // Convert PointCloud2 to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr stl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*input_cloud, *stl_cloud);

    // Perform cluster extraction
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setInputCloud(stl_cloud);
    ec.setClusterTolerance(0.1);  // Adjust as needed
    ec.setMinClusterSize(10);    // Adjust as needed
    ec.setMaxClusterSize(1000);  // Adjust as needed
    ec.setSearchMethod(boost::make_shared<pcl::search::KdTree<pcl::PointXYZI>>());
    ec.extract(cluster_indices);

    int cluster_id = 0;

    for (const auto& indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      for (auto index : indices.indices)
      {
        cluster->push_back((*stl_cloud)[index]);
      }

      // Calculate bounding box dimensions
      pcl::PointXYZI min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);

      // Create a transformation matrix to rotate the bounding box
      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.translation() << (min_pt.x + max_pt.x) / 2.0, (min_pt.y + max_pt.y) / 2.0, (min_pt.z + max_pt.z) / 2.0;

      // Apply the transformation to the bounding box
      pcl::transformPointCloud(*cluster, *cluster, transform);

      // Create a blue bounding box marker
      visualization_msgs::Marker bbox_marker;
      bbox_marker.header.frame_id = input_cloud->header.frame_id;
      bbox_marker.header.stamp = ros::Time::now();
      bbox_marker.ns = "cluster";
      bbox_marker.id = cluster_id;
      bbox_marker.type = visualization_msgs::Marker::CUBE;
      bbox_marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
      bbox_marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
      bbox_marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
      bbox_marker.pose.orientation.w = 1.0;
      bbox_marker.scale.x = max_pt.x - min_pt.x;
      bbox_marker.scale.y = max_pt.y - min_pt.y;
      bbox_marker.scale.z = max_pt.z - min_pt.z;
      bbox_marker.color.r = 0.0;
      bbox_marker.color.g = 0.0;
      bbox_marker.color.b = 1.0;
      bbox_marker.color.a = 0.5;
      bbox_marker.lifetime = ros::Duration(1.0);  // Adjust as needed
      bbox_pub_.publish(bbox_marker);

      cluster_id++;
    }
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber lidar_cloud_sub_;
  ros::Publisher detected_object_mesh_pub_;
  ros::Publisher detected_object_pose_pub_;
  ros::Publisher stl_mesh_pub_;
  ros::Publisher stl_cloud_pub_;
  ros::Publisher bbox_pub_;
  pcl::PolygonMesh object_mesh_;
  std::string os_sensor_frame_id_;
  std::string stl_file_;
  visualization_msgs::Marker stl_mesh_marker;
  sensor_msgs::PointCloud2 stl_object_msg;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointcloud_cluster_detection");
  ros::NodeHandle nh;

  PointCloudClusterDetection pcl_cluster_detection(nh);

  // Set the rate at which you want to publish the object (e.g., every 1 second)
  ros::Rate rate(1.0); // 1 Hz

  while (ros::ok())
  {
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
