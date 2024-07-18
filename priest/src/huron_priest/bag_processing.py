import os
from typing import  Dict, List, cast
import time  as time

from tqdm import tqdm
import numpy as np

import rospy
import rosbag
from std_msgs.msg import Int16
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray

class BagProc:
	def __init__(self) -> None:
		rospy.init_node("bags_proc")

		self.rate = rospy.Rate(10)
		bag_file_paths = self.get_bags('/home/wheelchair/bags')
		self.bags = [rosbag.Bag(bag_file_path, allow_unindexed=True) for bag_file_path in bag_file_paths]
		print(f"Opened {len(self.bags)} bags")

		self.topics = ['/laserscan', '/odometry', '/pedestrians_pose']
		self.scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=10)
		self.marker_pub = rospy.Publisher('/pedestrians', MarkerArray, queue_size=10)
		self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
		self.done_sub = rospy.Subscriber('/done', Int16, self.done_callback)

		self.done = 0

	def done_callback(self, msg: Int16):
		self.done = msg.data


	def get_bags(self, directory: str) -> List[str]:
		bag_file_path = []

		for root, dirs, files in os.walk(directory):
			for file in files:
				if file.endswith(".bag"):
					bag_file_path.append(os.path.join(root, file))

		return bag_file_path
	
	def publish_bag_message(self, bag: rosbag.Bag):
		
		data: Dict[str, Dict[str, np.ndarray]] = {
    		topic: None for topic in self.topics
		}			
		current_time = cast(float, bag.get_start_time())
		k = 0
		for topic, message, time in bag.read_messages(topics=self.topics):
			assert topic in self.topics, f"Unexpected topic: {topic}"

			data[topic] = message
			# Time Synchronization
			if (time.to_sec() - current_time) >= (1.0 / 12):
				current_time = time.to_sec()
				while self.done == 0:
					self.scan_pub.publish(data['/laserscan'])
					self.marker_pub.publish(data['/pedestrians_pose'])
					self.odom_pub.publish(data['/odometry'])
					print(f"Advertised Topics {k}")
					if k == 2:
						time.sleep(10)
					if self.done == 1:
						print("Processing Done")
						self.done = 0
						break
					
					self.rate.sleep()
					# Wait for done signal from PRIEST and then proceed to give the next message to PRIEST
					# print("Waiting for done signal...")
				# rospy.wait_for_message('/done', Int16, timeout=10)

		print(bag.filename, " Done")
		print("========================")
		# rospy.sleep(0.1)

	def run(self):
		# progress_bar = tqdm(total=len(self.bags), desc="Processing Bags", unit="Bag", position = 0) 
		
		for bag in self.bags:
			self.publish_bag_message(bag)
			# progress_bar.set_description(f"Processing {bag.filename}")
			bag.close()
			# progress_bar.update(1)
		
		# progress_bar.set_description("Done processing")


if __name__ == '__main__':
	BagProc().run()      
	# rospy.spin() 
