#!/usr/bin/python3
# coding: utf-8

import math
import rospy
import tf
import csv
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path

class TurtlebotController():
    def __init__(self, rate):
        self.rate = rate
        self.goal_tol = 0.15
        self.angle_tol = 0.4
        
        self.state = 'NAV'
        self.avoid_start_time = None
        self.avoid_timeout = 6.0
        self.avoid_direction = 1

        self.K_linear = 0.5
        self.K_angular = 1.5
        self.MAX_LINEAR = 0.6
        self.MAX_ANGULAR = 2.0

        self.OBSTACLE_DIST = 0.6
        self.SAFE_DIST = 0.9
        self.CLOSE_DIST = 0.4

        self.goal = PoseStamped()
        self.path_poses = []
        self.path_received = False
        self.current_goal_index = 0
        self.goal_received = False

        self.lidar_data = None
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

        # Buffer para datos: [timestamp, x, y]
        self.history = []

        rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goalCallback)
        rospy.Subscriber("/scan", LaserScan, self.scanCallback)
        rospy.Subscriber("/path", Path, self.pathCallback)

        rospy.loginfo("Controller started")

    def scanCallback(self, msg):
        self.lidar_data = msg

    def pathCallback(self, msg):
        if not self.path_received:
            self.path_poses = msg.poses
            self.current_goal_index = 0
            self.path_received = True

    def goalCallback(self, goal):
        self.goal = goal
        self.goal_received = True
        self.path_received = False

    def _index_from_angle(self, angle_deg):
        msg = self.lidar_data
        if msg is None: return 0
        N = len(msg.ranges)
        i = int((angle_deg % 360 / 360.0) * N)
        return i % N

    def _sector_data(self, center_deg, width_deg, ranges):
        if ranges is None: return []
        start = self._index_from_angle(center_deg - width_deg/2.0)
        end = self._index_from_angle(center_deg + width_deg/2.0)
        
        if start > end:
            sector = ranges[start:] + ranges[:end+1]
        else:
            sector = ranges[start:end+1]
            
        return [x for x in sector if x > 0.05 and math.isfinite(x)]

    def _get_lidar_regions(self, ranges):
        if ranges is None:
            return {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}
        
        def s_min(c, w, r):
            d = self._sector_data(c, w, r)
            return min(d) if d else float('inf')

        return {
            'front': s_min(0.0, 40.0, ranges),
            'left': s_min(60.0, 60.0, ranges),
            'right': s_min(-60.0, 60.0, ranges)
        }

    def _start_avoid(self):
        regions = self._get_lidar_regions(self.lidar_data.ranges) if self.lidar_data else {'left':0, 'right':0}
        self.avoid_direction = 1 if regions['left'] >= regions['right'] else -1
        self.state = 'AVOID_TURN'
        self.avoid_start_time = rospy.Time.now()

    def _stop_avoid(self):
        self.state = 'NAV'
        self.avoid_start_time = None

    def log_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('/odom', '/base_footprint', rospy.Time(0))
            self.history.append([rospy.Time.now().to_sec(), trans[0], trans[1]])
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def command(self):
        self.log_pose()

        if not self.path_received and not self.goal_received:
            return

        if self.path_received:
            if self.current_goal_index >= len(self.path_poses):
                self.publish(0.0, 0.0)
                return
            self.goal = self.path_poses[self.current_goal_index]

        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_tf = self.tf_listener.transformPose('base_footprint', self.goal)
        except:
            return

        gx, gy = pose_tf.pose.position.x, pose_tf.pose.position.y
        dist_goal = math.hypot(gx, gy)
        angle_goal = math.atan2(gy, gx)

        if dist_goal < self.goal_tol:
            if self.path_received:
                self.current_goal_index += 1
            else:
                self.goal_received = False
            self.publish(0.0, 0.0)
            return

        regions = self._get_lidar_regions(self.lidar_data.ranges) if self.lidar_data else {'front': float('inf'), 'left': 0, 'right': 0}
        
        if self.state == 'NAV' and regions['front'] < self.OBSTACLE_DIST:
            self._start_avoid()

        if self.state.startswith('AVOID'):
            if (rospy.Time.now() - self.avoid_start_time).to_sec() > self.avoid_timeout:
                self._stop_avoid()

        linear = 0.0
        angular = 0.0

        if self.state == 'NAV':
            scale = 1.0
            if regions['front'] < self.CLOSE_DIST: scale = 0.0
            elif regions['front'] < self.OBSTACLE_DIST: scale = 0.35

            linear = min(self.K_linear * dist_goal * scale, regions['front'] * 0.7)
            angular = self.K_angular * angle_goal
            
            if abs(angular) > self.MAX_ANGULAR:
                angular = math.copysign(self.MAX_ANGULAR, angular)

        elif self.state == 'AVOID_TURN':
            if regions['right'] < 0.4 and regions['left'] > (regions['right'] + 0.5): self.avoid_direction = 1
            if regions['left'] < 0.4 and regions['right'] > (regions['left'] + 0.5): self.avoid_direction = -1
            
            linear = 0.06 if regions['front'] < self.CLOSE_DIST else 0.12
            angular = 1.2 * self.avoid_direction

            if regions['front'] > self.SAFE_DIST:
                if abs(angle_goal) < 1.2 or (rospy.Time.now() - self.avoid_start_time).to_sec() > 1.2:
                    self.state = 'AVOID_ADVANCE'

        elif self.state == 'AVOID_ADVANCE':
            linear = 0.2
            angular = 0.9 * angle_goal * self.avoid_direction
            if regions['front'] > self.SAFE_DIST: self._stop_avoid()
            if regions['front'] < self.OBSTACLE_DIST: 
                self.state = 'AVOID_TURN'
                self.avoid_start_time = rospy.Time.now()

        linear = max(min(linear, self.MAX_LINEAR), -self.MAX_LINEAR)
        angular = max(min(angular, self.MAX_ANGULAR), -self.MAX_ANGULAR)
        self.publish(linear, angular)

    def publish(self, v, w):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.cmd_vel_pub.publish(msg)

    def shutdown(self):
        rospy.loginfo("Stopping...")
        self.cmd_vel_pub.publish(Twist())
        
        if self.history:
            with open('trajectory.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'x', 'y'])
                writer.writerows(self.history)
            rospy.loginfo(f"Data saved: {len(self.history)} points")
        rospy.sleep(1)

if __name__ == '__main__':
    rospy.init_node('TurtlebotController')
    rospy.sleep(1)
    
    rate = 10
    robot = TurtlebotController(rate)
    rospy.on_shutdown(robot.shutdown)
    
    r = rospy.Rate(rate)
    while not rospy.is_shutdown():
        robot.command()
        r.sleep()