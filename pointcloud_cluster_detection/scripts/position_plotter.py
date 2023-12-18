#!/usr/bin/env python

import rospy
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (non-interactive)
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseArray
import time

class PositionPlotterNode:
    def __init__(self):
        rospy.init_node('position_plotter_node', anonymous=True)
        self.pose_data = {'x': [], 'y': [], 'z': []}
        self.start_time = None
        self.duration = 10  # Duration to record data in seconds 
        self.data_received = False  # Flag to track if data has started being received

        # Subscribe to the "/object_poses" topic
        rospy.Subscriber("/detected_object_pose", PoseArray, self.pose_callback)

        # Create a timer to periodically check if data has been received
        self.timer = rospy.Timer(rospy.Duration(5.0), self.check_data_received)

    def pose_callback(self, msg):
        # Record x, y, and z values over time
        current_time = rospy.Time.now()
        if self.start_time is None:
            self.start_time = current_time
            rospy.loginfo("Data reception started. Recording data for 10 seconds...")
            self.data_received = True
        elapsed_time = (current_time - self.start_time).to_sec()

        for pose in msg.poses:
            self.pose_data['x'].append(pose.position.x)
            self.pose_data['y'].append(pose.position.y)
            self.pose_data['z'].append(pose.position.z)

        # Check if data collection duration is reached
        if elapsed_time >= self.duration:
            self.plot_data()

    def check_data_received(self, event):
        if not self.data_received:
            rospy.loginfo("Waiting for data...")
        else:
            self.timer.shutdown()  # Stop the timer once data is received

    def plot_data(self):
        if not self.data_received:
            rospy.logwarn("No data received. Cannot generate plot.")
            return

        # Create subplots for x, y, and z values
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        for ax, axis_label in zip(axes, ['x', 'y', 'z']):
            time_in_seconds = [t / 10.0 for t in range(len(self.pose_data[axis_label]))]  # Convert deciseconds to seconds
            ax.scatter(time_in_seconds, self.pose_data[axis_label], label=axis_label)
            ax.axhline(y=sum(self.pose_data[axis_label]) / len(self.pose_data[axis_label]), color='r', linestyle='--', label='Average')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('{} Value (meters)'.format(axis_label))  # Include units in the label
            ax.legend()

            # Set x-axis limits to display only the range from 0 to 10 seconds
            ax.set_xlim(0, 10)

        plt.tight_layout()
        # plt.show()

        # Save the plot
        plot_filename = "/home/mr_fusion/lidar_ws/src/pointcloud_cluster_detection/plots/position_plot.pdf"
        plt.savefig(plot_filename)
        rospy.loginfo("Plot saved as {}".format(plot_filename))  # Use .format() for string formatting

        # Shutdown the node
        rospy.signal_shutdown("Data collection and plotting completed, open plot here: /home/eirik18/lidar_ws/src/pointcloud_cluster_detection/plots/position_plot.png")


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = PositionPlotterNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
