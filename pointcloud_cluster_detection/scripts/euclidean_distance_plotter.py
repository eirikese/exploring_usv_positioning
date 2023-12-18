#!/usr/bin/env python

import rospy
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (non-interactive)
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray
import time
import math  # For calculating Euclidean distance

class PositionPlotterNode:
    def __init__(self):
        rospy.init_node('position_plotter_node', anonymous=True)
        self.distances = []  # Store Euclidean distances
        self.start_time = None
        self.duration = 2 # Duration to record data in seconds 
        self.data_received = False  # Flag to track if data has started being received

        # Subscribe to the "/detected_object_pose" topic
        rospy.Subscriber("/detected_object_pose", PoseArray, self.pose_callback)

        # Create a timer to periodically check if data has been received
        self.timer = rospy.Timer(rospy.Duration(5.0), self.check_data_received)

    def pose_callback(self, msg):
        current_time = rospy.Time.now()
        if self.start_time is None:
            self.start_time = current_time
            rospy.loginfo("Data reception started. Recording data for 10 seconds...")
            self.data_received = True
        elapsed_time = (current_time - self.start_time).to_sec()

        for pose in msg.poses:
            distance = math.sqrt(pose.position.x**2 + pose.position.y**2 + pose.position.z**2)
            self.distances.append(distance)

        if elapsed_time >= self.duration:
            self.plot_data()

    def check_data_received(self, event):
        if not self.data_received:
            rospy.loginfo("Waiting for data...")
        else:
            self.timer.shutdown()

    def plot_data(self):
        if not self.data_received:
            rospy.logwarn("No data received. Cannot generate plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        time_in_seconds = [t / 10.0 for t in range(len(self.distances))]
        ax.plot(time_in_seconds, self.distances, label='Euclidean Distance')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (meters)')
        ax.legend()
        ax.set_xlim(0, self.duration)

        plt.tight_layout()

        plot_filename = "/home/mr_fusion/lidar_ws/src/pointcloud_cluster_detection/plots/euclidean_distance_plot.pdf"
        plt.savefig(plot_filename)
        rospy.loginfo("Plot saved as {}".format(plot_filename))

        rospy.signal_shutdown("Data collection and plotting completed")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PositionPlotterNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
