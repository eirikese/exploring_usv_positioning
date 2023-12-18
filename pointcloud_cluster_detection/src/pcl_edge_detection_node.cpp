#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/don.h>

class EdgeDetectionNode
{
public:
    EdgeDetectionNode()
    {
        sub_ = nh_.subscribe("/ouster/filtered", 1, &EdgeDetectionNode::cloudCallback, this);
        pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/ouster/edges", 1);
    }

        void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &input)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input, *cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());

        pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.1);
        ne.compute(*normals);

        // Extracting edges based on high curvature
        pcl::PointCloud<pcl::PointXYZ>::Ptr edges_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (size_t i = 0; i < normals->points.size(); i++) {
            if (normals->points[i].curvature > 0.03) {  // The threshold can be adjusted
                edges_cloud->points.push_back(cloud->points[i]);
            }
        }

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*edges_cloud, output);
        output.header.frame_id = input->header.frame_id;

        pub_.publish(output);
    }


private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher pub_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "edge_detection_node");
    EdgeDetectionNode node;

    ros::spin();

    return 0;
}
