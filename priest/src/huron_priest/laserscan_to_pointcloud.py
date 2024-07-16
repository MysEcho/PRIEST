#!/usr/bin/env python3
'''
This program turns raw laserscan from the huron dataset into point cloud data. 
Run this program along with obstacle_metadata_publisher.py before running PRIEST.
'''

import rospy
import math
from sensor_msgs.msg import LaserScan, PointCloud
from geometry_msgs.msg import Point32
from time import perf_counter

class Extractor:
    def __init__(self):
        rospy.init_node('extracting_node', anonymous=True)
        
        rospy.Subscriber('/laserscan', LaserScan, self.laser_callback)

        self.pointcloud_pub = rospy.Publisher('/pointcloud', PointCloud, queue_size=10)
        

    def laser_callback(self, scan_msg):
       
        pointcloud_msg = PointCloud()
        pointcloud_msg.header = scan_msg.header

        # LaserScan to PointCloud
        for i, range_value in enumerate(scan_msg.ranges):
            if range_value < scan_msg.range_min or range_value > scan_msg.range_max:
                continue  

            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            # polar coordinates to Cartesian
            x = range_value * math.cos(angle)
            y = range_value * math.sin(angle)

            point = Point32()
            point.x = x
            point.y = y
            point.z = 0.0  
            pointcloud_msg.points.append(point)

        self.pointcloud_pub.publish(pointcloud_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        converter = Extractor()
        converter.run()
    except rospy.ROSInterruptException:
        pass