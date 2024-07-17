#!/usr/bin/env python3

import rospy
import numpy as np
import jax.numpy as jnp
from jax import random
import time
import threading
import open3d
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import mpc_non_dy
import csv

class PlanningTraj:

    def __init__(self):
        rospy.init_node("MPC_planning")
        rospy.on_shutdown(self.shutdown)

        self.data_lock = threading.Lock()
        self.initialize_variables()
        self.setup_mpc_parameters()
        self.initialize_obstacle_data()
        self.prob = mpc_non_dy.batch_crowd_nav(
            self.a_obs_1, self.b_obs_1, self.a_obs_2, self.b_obs_2,
            self.v_max, self.v_min, self.a_max, 50, 10,
            self.t_fin, self.num, self.num_batch,
            self.maxiter, self.maxiter_cem,
            self.weight_smoothness, self.weight_track,
            1000, self.v_des
    )
        self.setup_subscribers()
        self.initialize_csv()

        self.key = random.PRNGKey(0)
        self.x_guess_per, self.y_guess_per = None, None

    def initialize_csv(self):
        # Initialize CSV file for storing coefficients
        self.csv_filename = 'coefficient_data_new.csv'
        with open(self.csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Timestamp', 'c_x_best', 'c_y_best']
            for i in range(self.num_dynamic_obstacles):
                header.extend([f'obs_{i}_x', f'obs_{i}_y', f'obs_{i}_vx', f'obs_{i}_vy'])
            writer.writerow(header)

        self.iteration_count = 0


    def initialize_variables(self):
        self.x_init, self.y_init = 1.0, 2.0
        self.vx_init, self.vy_init = 0.05, 0.1
        self.ax_init, self.ay_init = 0.0, 0.0
        self.x_fin, self.y_fin = 2.0, 15.0
        self.x_fin_t, self.y_fin_t = 2.0, 15.0
        self.theta_init = 0.0

    def setup_mpc_parameters(self):
        self.v_max, self.v_min = 1.0, 0.02
        self.a_max = 1.0
        self.maxiter, self.maxiter_cem = 1, 12
        self.weight_track, self.weight_smoothness = 0.001, 1
        self.a_obs_1, self.b_obs_1 = 0.5, 0.5
        self.a_obs_2, self.b_obs_2 = 0.68, 0.68
        self.t_fin, self.num, self.num_batch = 10, 100, 110
        self.v_des = 1.0
        self.maxiter_mpc = 300

    def initialize_obstacle_data(self):
        self.num_dynamic_obstacles = 10
        self.x_obs_init_dy = np.zeros(self.num_dynamic_obstacles)
        self.y_obs_init_dy = np.zeros(self.num_dynamic_obstacles)
        self.vx_obs_dy = np.zeros(self.num_dynamic_obstacles)
        self.vy_obs_dy = np.zeros(self.num_dynamic_obstacles)

        self.x_obs_init = np.ones(420) * 100
        self.y_obs_init = np.ones(420) * 100
        self.vx_obs = jnp.zeros(420)
        self.vy_obs = jnp.zeros(420)

        self.pcd = open3d.geometry.PointCloud()

    def setup_subscribers(self):
        rospy.Subscriber("/pointcloud", PointCloud, self.pointcloud_callback)
        rospy.Subscriber("/obstacle_metadata", Float64MultiArray, self.dynamic_obs_callback)
        rospy.Subscriber("/odometry", Odometry, self.odometry_callback)
        

    def odometry_callback(self, msg_odom):
        with self.data_lock:
            self.x_init = msg_odom.pose.pose.position.x
            self.y_init = msg_odom.pose.pose.position.y

            orientation_q = msg_odom.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (_, _, self.theta_init) = euler_from_quaternion(orientation_list)

        self.run_optimization()


    def pointcloud_callback(self, msg_pointcloud):
        x_obs_init_1 = np.array([p.x for p in msg_pointcloud.points])
        y_obs_init_1 = np.array([p.y for p in msg_pointcloud.points])

        mask = (x_obs_init_1 < 30) & (y_obs_init_1 < 30)
        x_obs_init_1 = x_obs_init_1[mask]
        y_obs_init_1 = y_obs_init_1[mask]

        xyz = np.column_stack((x_obs_init_1, y_obs_init_1, np.zeros_like(x_obs_init_1)))
        self.pcd.points = open3d.utility.Vector3dVector(xyz)
        downpcd = self.pcd.voxel_down_sample(voxel_size=0.16)
        downpcd_array = np.asarray(downpcd.points)

        num_down_samples = downpcd_array.shape[0]
        with self.data_lock:
            self.x_obs_init[:num_down_samples] = downpcd_array[:, 0]
            self.y_obs_init[:num_down_samples] = downpcd_array[:, 1]
            self.x_obs_init[num_down_samples:] = 100
            self.y_obs_init[num_down_samples:] = 100

    def dynamic_obs_callback(self, msg_dynamic_obs):
        dynamic_obs_data = np.array(msg_dynamic_obs.data)
        num_obstacles = len(dynamic_obs_data) // 5

        with self.data_lock:
            self.x_obs_init_dy.fill(1000.0) 
            self.y_obs_init_dy.fill(1000.0)
            self.vx_obs_dy.fill(0.0)
            self.vy_obs_dy.fill(0.0)

            id_to_index = {}
            current_index = 0

            for i in range(num_obstacles):
                idx = i * 5
                obstacle_id = int(dynamic_obs_data[idx])
            
                if obstacle_id not in id_to_index:
                    if current_index < self.num_dynamic_obstacles:
                        id_to_index[obstacle_id] = current_index
                        current_index += 1
                    else:
                        raise ValueError("Warning: More obstacles detected than can be handled.")
            
                array_idx = id_to_index[obstacle_id]
                self.x_obs_init_dy[array_idx] = dynamic_obs_data[idx + 1]
                self.y_obs_init_dy[array_idx] = dynamic_obs_data[idx + 2]
                self.vx_obs_dy[array_idx] = dynamic_obs_data[idx + 3]
                self.vy_obs_dy[array_idx] = dynamic_obs_data[idx + 4]
            

    def run_optimization(self):
        start_time = time.time()

        # prob = mpc_non_dy.batch_crowd_nav(
        #     self.a_obs_1, self.b_obs_1, self.a_obs_2, self.b_obs_2,
        #     self.v_max, self.v_min, self.a_max, 50, 10,
        #     self.t_fin, self.num, self.num_batch,
        #     self.maxiter, self.maxiter_cem,
        #     self.weight_smoothness, self.weight_track,
        #     1000, self.v_des,
        # )

        with self.data_lock:
            initial_state = jnp.array([self.x_init, self.y_init, self.vx_init, self.vy_init, self.ax_init, self.ay_init])
            x_obs_init_dy = self.x_obs_init_dy.copy()
            y_obs_init_dy = self.y_obs_init_dy.copy()
            x_obs_init = self.x_obs_init.copy()
            y_obs_init = self.y_obs_init.copy()
            vx_obs_dy = self.vx_obs_dy.copy()
            vy_obs_dy = self.vy_obs_dy.copy()
            vx_obs = self.vx_obs.copy()
            vy_obs = self.vy_obs.copy()

        x_waypoint = jnp.linspace(self.x_init, self.x_fin + 10.0 * jnp.cos(self.theta_init), 1000)
        y_waypoint = jnp.linspace(self.y_init, self.y_fin + 10.0 * jnp.sin(self.theta_init), 1000)

        arc_length, arc_vec, x_diff, y_diff = self.prob.path_spline(x_waypoint, y_waypoint)

        if self.x_guess_per is None or self.y_guess_per is None:
            self.x_guess_per, self.y_guess_per = self.prob.compute_warm_traj(
                initial_state, self.v_des, x_waypoint, y_waypoint, arc_vec, x_diff, y_diff,
            )

        x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj, x_obs_trajectory_dy, y_obs_trajectory_dy = self.prob.compute_obs_traj_prediction(
            jnp.asarray(x_obs_init_dy).flatten(), jnp.asarray(y_obs_init_dy).flatten(),
            vx_obs_dy, vy_obs_dy, jnp.asarray(x_obs_init).flatten(), jnp.asarray(y_obs_init).flatten(),
            vx_obs, vy_obs, initial_state[0], initial_state[1],
        )

        sol_x_bar, sol_y_bar, x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, c_mean, c_cov, x_fin, y_fin = self.prob.compute_traj_guess(
            initial_state, x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_dy, y_obs_trajectory_dy,
            self.v_des, x_waypoint, y_waypoint, arc_vec, self.x_guess_per, self.y_guess_per, x_diff, y_diff,
        )

        lamda_x = jnp.zeros((self.num_batch, self.prob.nvar))
        lamda_y = jnp.zeros((self.num_batch, self.prob.nvar))

        x, y, c_x_best, c_y_best, x_best, y_best, self.x_guess_per, self.y_guess_per = self.prob.compute_cem(
            self.key, initial_state, x_fin, y_fin, lamda_x, lamda_y,
            x_obs_trajectory, y_obs_trajectory, x_obs_trajectory_proj, y_obs_trajectory_proj,
            x_obs_trajectory_dy, y_obs_trajectory_dy, sol_x_bar, sol_y_bar,
            x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess,
            x_waypoint, y_waypoint, arc_vec, c_mean, c_cov,
        )

        print(f"Optimization time: {time.time() - start_time:.4f} seconds")

        # Save the best coefficients to CSV
        self.save_coefficients(c_x_best, c_y_best, self.x_obs_init_dy, self.y_obs_init_dy, self.vx_obs_dy, self.vy_obs_dy)

    def save_coefficients(self, c_x_best, c_y_best, x_obs_init_dy, y_obs_init_dy, vx_obs_dy, vy_obs_dy):
        timestamp = rospy.get_time()
        with open(self.csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [timestamp, c_x_best, c_y_best]
            for i in range(self.num_dynamic_obstacles):
                row.extend([x_obs_init_dy[i], y_obs_init_dy[i], vx_obs_dy[i], vy_obs_dy[i]])
            writer.writerow(row)
        
        self.iteration_count += 1
        print(f"Saved data for iteration {self.iteration_count}")

    # Update Robot State for MPC (Not used for Single Optimization step) {Check for change}
    def update_robot_state(self, x_best, y_best):
        with self.data_lock:
            self.vx_init = (x_best[1] - x_best[0]) / self.t_fin
            self.vy_init = (y_best[1] - y_best[0]) / self.t_fin

    def goal_reached(self):
        distance_to_goal = ((self.x_init - self.x_fin_t) ** 2 + (self.y_init - self.y_fin_t) ** 2) ** 0.5
        return distance_to_goal < 0.1

    def update_goal(self):
        pass

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        rospy.sleep(1)

if __name__ == "__main__":
    try:
        planner = PlanningTraj()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass