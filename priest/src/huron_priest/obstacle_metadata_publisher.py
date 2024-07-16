#!/usr/bin/env python3
'''
This program turns pedestrian pose from the huron dataset into their corresponfing velocities and prints both position and velocity. 
Run this program along with laserscan_to_pointcloud.py before running PRIEST.
'''

import rospy
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float64MultiArray
from time import perf_counter


class ObstacleVelocityPublisher:
    def __init__(self):
        rospy.init_node("obstacle_metadata_node")
        self.velocity_publisher = rospy.Publisher(
            "obstacle_metadata", Float64MultiArray, queue_size=10
        )
        rospy.Subscriber("/pedestrians_pose", MarkerArray, self.obstacle_callback)

        self.previous_data = {}

    def calculate_velocity(self, current_pos, previous_pos, dt):
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        velocity_x = dx / dt
        velocity_y = dy / dt
        return velocity_x, velocity_y

    def obstacle_callback(self, msg: MarkerArray):
        current_time = perf_counter()
        velocities = []

        for marker in msg.markers:
            if marker is not None:
                id = marker.id
                coord_x = marker.pose.position.x
                coord_y = marker.pose.position.y

                if id in self.previous_data:
                    prev_coord_x, prev_coord_y, prev_time = self.previous_data[id]
                    dt = current_time - prev_time

                    if dt > 0:
                        velocity_x, velocity_y = self.calculate_velocity(
                            (coord_x, coord_y), (prev_coord_x, prev_coord_y), dt
                        )
                        velocities.extend([float(id), coord_x,coord_y,velocity_x, velocity_y]) # ID, x, y, v_x, v_y

                self.previous_data[id] = (coord_x, coord_y, current_time)

            velocity_msg = Float64MultiArray()
            velocity_msg.data = velocities

            self.velocity_publisher.publish(velocity_msg)


def main():
    ObstacleVelocityPublisher()
    rospy.spin()


if __name__ == "__main__":
    main()
