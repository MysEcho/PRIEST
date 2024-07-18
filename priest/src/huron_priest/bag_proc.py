#!/usr/bin/env python
import rospy
import rosbag
import os
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from threading import Event
from visualization_msgs.msg import MarkerArray

class BagProcessor:
    def __init__(self, bag_directory):
        self.bag_directory = bag_directory
        self.done_event = Event()
        
        rospy.init_node('bag_processor', anonymous=True)
        
        self.pedestrian_pub = rospy.Publisher('/pedestrians_pose', MarkerArray, queue_size=10)
        self.laser_pub = rospy.Publisher('/laserscan', LaserScan, queue_size=10)
        self.odom_pub = rospy.Publisher('/odometry', Odometry, queue_size=10)
        
        rospy.Subscriber('/processing_done', Bool, self.done_callback)

    def done_callback(self, msg):
        if msg.data:
            self.done_event.set()

    def process_bags(self):

        bag_files = [f for f in os.listdir(self.bag_directory) if f.endswith('.bag')]
        
        for bag_file in bag_files:
            bag_path = os.path.join(self.bag_directory, bag_file)
            rospy.loginfo(f"Processing bag file: {bag_file}")
            
            try:
                with rosbag.Bag(bag_path, 'r') as bag:
                    # Create buffer for syncing
                    buffer = {'/pedestrians_pose': None, '/laserscan': None, '/odometry': None}
                    
                    for topic, msg, t in bag.read_messages(topics=['/pedestrians_pose', '/laserscan', '/odometry']):
                        buffer[topic] = msg
                        
                        if all(buffer.values()):  # Check for message for each topic

                            self.pedestrian_pub.publish(buffer['/pedestrians_pose'])
                            self.laser_pub.publish(buffer['/laserscan'])
                            self.odom_pub.publish(buffer['/odometry'])
                            
                            rospy.loginfo("Published messages on all topics. Waiting for done signal...")
                            self.done_event.wait()
                            self.done_event.clear()
                            rospy.loginfo("Received done signal, proceeding to next timestep.")
                            
                            # Clear buffer
                            buffer = {'/pedestrians_pose': None, '/laserscan': None, '/odometry': None}
                            # rospy.sleep(0.1)
                        
                        if rospy.is_shutdown():
                            return
                        
            except Exception as e:
                rospy.logerr(f"Error processing bag file {bag_file}: {str(e)}")
                
            print(f"Processed bag file: {bag_file}")
        
        rospy.loginfo("Finished processing all bag files.")

if __name__ == '__main__':
    try:
        bag_directory = '/home/themys/PRIEST/src/test_bags'  
        processor = BagProcessor(bag_directory)
        processor.process_bags()
    except rospy.ROSInterruptException:
        pass


# NOTES:
'''
On playing rosbag 1.bag manually: (All have been synced by using buffer)

i) /laserscan is being printed at every 10 rosbag timesteps.

ii) /pedestrians_pose is being printed at every 8 rosbag timesteps.

iii) /odometry is being printed at every 9 rosbag timesteps.

iv) No need to change bag_proc.py.

v) Use Approximate Time Synchronizer for PRIEST Planner. Use buffer to store all data and then process.

'''