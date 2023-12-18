#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

class PointCloudFilter
{
public:
    PointCloudFilter() : nh_("~")
    {
        nh_.param("min_intensity", min_intensity_, 0.0);
        nh_.param("max_intensity", max_intensity_, 255.0);

        // Subscribe to the "/ouster/points" topic
        cloud_sub_ = nh_.subscribe("/ouster/points", 1, &PointCloudFilter::cloudCallback, this);

        // Publish filtered point cloud to the "/ouster/filtered" topic
        filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/ouster/filtered", 1);
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
        pcl::fromROSMsg(*cloud_msg, pcl_cloud);

        pcl::PointCloud<pcl::PointXYZI> filtered_cloud;

        for (const auto& point : pcl_cloud.points)
        {
            if (point.intensity >= min_intensity_ && point.intensity <= max_intensity_)
            {
                filtered_cloud.push_back(point);
            }
        }

        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(filtered_cloud, filtered_cloud_msg);

        filtered_cloud_msg.header = cloud_msg->header;

        // Publish the filtered point cloud
        filtered_pub_.publish(filtered_cloud_msg);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher filtered_pub_;
    double min_intensity_;
    double max_intensity_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointcloud_filter_node");
    PointCloudFilter pointcloud_filter;

    ros::spin();
    return 0;
}
