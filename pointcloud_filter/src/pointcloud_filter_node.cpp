// Publish min and max values for intensity and euclidean distance with rqt



#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h> // For max and min intensity topics
#include <std_msgs/Float64MultiArray.h> // For max and min distance topics
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

class PointCloudFilter
{
public:
    PointCloudFilter() : nh_("~")
    {
        // Subscribe to the "/ouster/points" topic
        cloud_sub_ = nh_.subscribe("/ouster/points", 1, &PointCloudFilter::cloudCallback, this);

        // Subscribe to the topics for max and min intensity values
        max_intensity_sub_ = nh_.subscribe("/ouster/max_intensity", 1, &PointCloudFilter::maxIntensityCallback, this);
        min_intensity_sub_ = nh_.subscribe("/ouster/min_intensity", 1, &PointCloudFilter::minIntensityCallback, this);

        // Subscribe to the topics for max and min distance values
        max_distance_sub_ = nh_.subscribe("/ouster/max_distance", 1, &PointCloudFilter::maxDistanceCallback, this);
        min_distance_sub_ = nh_.subscribe("/ouster/min_distance", 1, &PointCloudFilter::minDistanceCallback, this);

        // Publish filtered point cloud to the "/ouster/filtered" topic
        filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/ouster/filter", 1);
        // filtered_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_points", 1);

        // Initialize min_intensity_, max_intensity_, min_distance_, and max_distance_ to default values
        min_intensity_ = 0.0;
        max_intensity_ = 255.0;
        min_distance_ = 0.0;
        max_distance_ = 100.0; // Set an arbitrary maximum distance
    }

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
        pcl::fromROSMsg(*cloud_msg, pcl_cloud);

        pcl::PointCloud<pcl::PointXYZI> filtered_cloud;

        for (const auto& point : pcl_cloud.points)
        {
            // Calculate the Euclidean distance from the sensor
            double distance = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);

            if (point.intensity >= min_intensity_ && point.intensity <= max_intensity_
                && distance >= min_distance_ && distance <= max_distance_)
            {
                filtered_cloud.push_back(point);
            }
            
        }
        point_counter_ += abs(double(sizeof(filtered_cloud)));
        iteration_counter_ += 1;

        sensor_msgs::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(filtered_cloud, filtered_cloud_msg);

        filtered_cloud_msg.header = cloud_msg->header;

        // Publish the filtered point cloud
        filtered_pub_.publish(filtered_cloud_msg);
    }

    void maxIntensityCallback(const std_msgs::Float64ConstPtr& max_intensity_msg)
    {
        max_intensity_ = max_intensity_msg->data;
    }

    void minIntensityCallback(const std_msgs::Float64ConstPtr& min_intensity_msg)
    {
        min_intensity_ = min_intensity_msg->data;
    }

    void maxDistanceCallback(const std_msgs::Float64ConstPtr& max_distance_msg)
    {
        max_distance_ = max_distance_msg->data;
    }

    void minDistanceCallback(const std_msgs::Float64ConstPtr& min_distance_msg)
    {
        min_distance_ = min_distance_msg->data;
    }
    double point_counter_;
    int iteration_counter_;

private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber max_intensity_sub_;
    ros::Subscriber min_intensity_sub_;
    ros::Subscriber max_distance_sub_;
    ros::Subscriber min_distance_sub_;
    ros::Publisher filtered_pub_;
    double min_intensity_;
    double max_intensity_;
    double min_distance_;
    double max_distance_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pointcloud_filter_node");
    PointCloudFilter pointcloud_filter;

    ros::spin();
    std::cout << std::endl << "Counted points: "<< pointcloud_filter.point_counter_ << std::endl;
    std::cout << "Number of samples: " << pointcloud_filter.iteration_counter_ << std::endl;
    std::cout << "Average number of points per sample: " << pointcloud_filter.point_counter_ / pointcloud_filter.iteration_counter_ << std::endl;
    return 0;
}
