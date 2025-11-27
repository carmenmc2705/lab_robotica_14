#!/usr/bin/python3
# coding: utf-8

import math
import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
import numpy as np
import time

class TurtlebotController():
    def __init__(self, rate):

        # --- Parámetros ---
        self.goal_tol = 0.15
        self.angle_tol = 0.4
        self.rate = rate

        # Estados
        self.state = 'NAV'
        self.avoid_start_time = None
        self.avoid_timeout = 3.0
        self.avoid_direction = 1

        # Ganancias
        self.K_linear = 0.5
        self.K_angular = 1.5
        self.MAX_LINEAR = 0.8
        self.MAX_ANGULAR = 2.5

        # LIDAR thresholds
        self.OBSTACLE_DIST = 0.5
        self.SAFE_DIST = 0.7

        # Objetivo / Path
        self.goal = PoseStamped()
        self.path_poses = []
        self.path_received = False
        self.current_goal_index = 0
        self.goal_received = False

        # LIDAR
        self.lidar_data = None

        # ROS
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

        rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goalCallback)
        rospy.Subscriber("/scan", LaserScan, self.scanCallback)
        rospy.Subscriber("/path", Path, self.pathCallback)

        rospy.loginfo("TurtlebotController started (NAV + robust AVOID)")

    # --------------------------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------------------------
    def scanCallback(self, msg):
        self.lidar_data = msg

        # debug una vez
        if not hasattr(self, "scan_debugged"):
            self.scan_debugged = True
            rospy.loginfo("LIDAR: angle_min=%.2f, angle_max=%.2f, angle_inc=%.4f",
                          msg.angle_min, msg.angle_max, msg.angle_increment)

    def pathCallback(self, msg):
        if not self.path_received:
            rospy.loginfo("Path received with %d points.", len(msg.poses))
            self.path_poses = msg.poses
            self.current_goal_index = 0
            self.path_received = True

    def goalCallback(self, goal):
        rospy.loginfo("Manual Goal received!")
        self.goal = goal
        self.goal_received = True
        self.path_received = False

    # --------------------------------------------------------------------------
    # AUX LIDAR
    # --------------------------------------------------------------------------
    def _get_lidar_regions(self, ranges):
        if ranges is None:
            return {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}

        r = np.array(ranges)
        r = np.where(np.isfinite(r), r, 100.0)

        N = len(r)
        step = int(N / 360.0) if N >= 360 else 1

        def sector(center_deg, width_deg=20):
            c = int(center_deg * step)
            w = int(width_deg * step)
            return np.concatenate([r[c-w:c+w]])

        front_sector = sector(0, 15)        # -15º a +15º
        left_sector = sector(90, 30)        # 60º a 120º
        right_sector = sector(270, 30)      # 240º a 300º

        return {
            'front': np.min(front_sector) if front_sector.size else float('inf'),
            'left': np.min(left_sector) if left_sector.size else float('inf'),
            'right': np.min(right_sector) if right_sector.size else float('inf')
        }

    # --------------------------------------------------------------------------
    # STATE CHANGE
    # --------------------------------------------------------------------------
    def _start_avoid(self):
        self.state = 'AVOID'
        self.avoid_start_time = rospy.Time.now()
        rospy.loginfo(">>> ENTERING AVOID MODE")

    def _stop_avoid(self):
        self.state = 'NAV'
        self.avoid_start_time = None
        rospy.loginfo("<<< EXITING AVOID MODE, BACK TO NAV")

    # --------------------------------------------------------------------------
    # MAIN CONTROL LOOP
    # --------------------------------------------------------------------------
    def command(self):

        # Si no hay nada que perseguir:
        if not self.path_received and not self.goal_received:
            return

        # Path tiene prioridad
        if self.path_received:
            if self.current_goal_index >= len(self.path_poses):
                rospy.loginfo_throttle(5, "PATH FINISHED!")
                self.publish(0.0, 0.0)
                return

            self.goal = self.path_poses[self.current_goal_index]

        # Transformar a base_footprint
        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)
        except:
            return

        goal_x = pose_transformed.pose.position.x
        goal_y = pose_transformed.pose.position.y

        angle_to_goal = math.atan2(goal_y, goal_x)
        distance_to_goal = math.sqrt(goal_x**2 + goal_y**2)

        # Chequeo de llegada
        if self.goalReached():
            rospy.loginfo("Waypoint reached!")

            if self.path_received:
                self.current_goal_index += 1
            else:
                self.goal_received = False

            self.publish(0.0, 0.0)
            return

        # LIDAR
        regions = self._get_lidar_regions(self.lidar_data.ranges) if self.lidar_data else \
                  {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}

        obstacle_front = regions['front'] < self.OBSTACLE_DIST

        if self.state == 'NAV' and obstacle_front:
            self._start_avoid()

        if self.state == 'AVOID':
            if (rospy.Time.now() - self.avoid_start_time).to_sec() > self.avoid_timeout:
                rospy.logwarn("AVOID timeout exceeded! Going back to NAV")
                self._stop_avoid()

        # ----------------------------------------------------------------------
        # STATES
        # ----------------------------------------------------------------------
        linear = 0.0
        angular = 0.0

        if self.state == 'NAV':

            linear = self.K_linear * distance_to_goal
            angular = self.K_angular * angle_to_goal

            if regions['front'] < self.OBSTACLE_DIST:
                self._start_avoid()

        elif self.state == 'AVOID':

            left_clear = regions['left']
            right_clear = regions['right']

            # 1) Si sigue habiendo obstáculo delante → GIRAR
            if regions['front'] < self.SAFE_DIST:
                linear = 0.0

                if left_clear > right_clear:
                    angular = 0.8
                    self.avoid_direction = 1
                else:
                    angular = -0.8
                    self.avoid_direction = -1

            else:
                # 2) Si ya está despejado → AVANZAR recto para limpiar obstáculo
                linear = 0.25
                angular = 0.0

                if (rospy.Time.now() - self.avoid_start_time).to_sec() > 0.6:
                    self._stop_avoid()

        # Saturación
        linear = max(min(linear, self.MAX_LINEAR), -self.MAX_LINEAR)
        angular = max(min(angular, self.MAX_ANGULAR), -self.MAX_ANGULAR)

        self.publish(linear, angular)

    # --------------------------------------------------------------------------
    def goalReached(self):
        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)

            dx = pose_transformed.pose.position.x
            dy = pose_transformed.pose.position.y

            distance = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)

            return distance < self.goal_tol and abs(angle) < self.angle_tol

        except:
            return False

    # --------------------------------------------------------------------------
    def publish(self, lin_vel, ang_vel):
        move_cmd = Twist()
        move_cmd.linear.x = lin_vel
        move_cmd.angular.z = ang_vel
        self.cmd_vel_pub.publish(move_cmd)

    # --------------------------------------------------------------------------
    def shutdown(self):
        rospy.loginfo("Stopping TurtleBot...")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    rospy.init_node('TurtlebotController', anonymous=False)
    rospy.loginfo("To stop TurtleBot CTRL+C")

    rate = 10
    robot = TurtlebotController(rate)
    rospy.on_shutdown(robot.shutdown)

    r = rospy.Rate(rate)
    while not rospy.is_shutdown():
        robot.command()
        r.sleep()
