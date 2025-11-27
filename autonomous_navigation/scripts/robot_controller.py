#!/usr/bin/python3
# This Python file uses the following encoding: utf-8

import math
import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
import numpy as np 

class TurtlebotController():
    
    def __init__(self, rate):
        
        # Read parameters
        self.goal_tol = 0.15
        self.rate = rate 
        
        # Initialize internal data 
        self.goal = PoseStamped()
        
        # Variables para el Path
        self.path_poses = []        # Lista para guardar el camino
        self.path_received = False  # Bandera para saber si tenemos camino
        self.current_goal_index = 0 # Índice para saber por qué punto vamos
        
        # Este lo mantenemos por si usas el 2D Nav Goal manual
        self.goal_received = False 
        self.lidar_data = None

        # Subscribers / publishers
        self.tf_listener = tf.TransformListener()

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        
        # Suscriptores
        rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goalCallback)
        rospy.Subscriber("/scan", LaserScan, self.scanCallback)
        rospy.Subscriber("/path", Path, self.pathCallback)
        
        rospy.loginfo("TurtlebotController started")
        

    def shutdown(self):
        rospy.loginfo("Stop TurtleBot")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)

    def scanCallback(self, msg):
        self.lidar_data = msg
    
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

    def command(self):
        # 1. COMPROBACIÓN DE DATOS
        if not self.path_received and not self.goal_received:
            return

        # 2. GESTIÓN DEL PATH
        if self.path_received:
            if self.current_goal_index >= len(self.path_poses):
                rospy.loginfo_throttle(5, "PATH FINISHED!!! Stopping!")
                self.publish(0.0, 0.0)
                return
            
            self.goal = self.path_poses[self.current_goal_index]

        # 3. COMPROBAR SI HEMOS LLEGADO AL PUNTO ACTUAL
        if self.goalReached():
            rospy.loginfo("Waypoint reached! Moving to next...")
            # No paramos (self.publish(0,0)) para que sea más fluido, 
            # pasamos directamente al siguiente
            
            if self.path_received:
                self.current_goal_index += 1 
            else:
                self.goal_received = False 
                self.publish(0.0, 0.0) # Si era manual y llegamos, paramos.
            return
        
        # 4. CONTROL LAW (Moverse hacia self.goal)
        try:
            self.goal.header.stamp = rospy.Time(0) 
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return    
            
        goal_x = pose_transformed.pose.position.x
        goal_y = pose_transformed.pose.position.y  
        
        # Calculamos el ángulo hacia el objetivo
        angle_to_goal = math.atan2(goal_y, goal_x)
        
        # --- AQUÍ ESTÁ EL CAMBIO PRINCIPAL (Lógica de tus amigos) ---
        
        # Si el robot no está alineado (error mayor a 0.05 rad / ~3 grados)
        if angle_to_goal > 0.05:
            linear = 0.0
            angular = 0.5  # Girar izquierda
        
        elif angle_to_goal < -0.05:
            linear = 0.0
            angular = -0.5 # Girar derecha
            
        else:
            # Si está alineado, avanzar rápido constante
            linear = 0.4
            angular = 0.0 
            
        # -----------------------------------------------------------
        
        # 5. EVASIÓN DE OBSTÁCULOS (Prioridad sobre lo anterior)
        if self.lidar_data:
            ranges = self.lidar_data.ranges
            front = ranges[0:20] + ranges[-20:]
            valid_front = [r for r in front if r > 0.1 and r < 10.0]
            
            # Si hay algo cerca, sobrescribimos las velocidades para evitar choque
            if valid_front and min(valid_front) < 0.5:
                rospy.logwarn_throttle(1, "Obstacle detected! Evading...")
                linear = 0.0
                angular = 0.6 
                
        # Publish velocity command
        self.publish(linear, angular)


    def goalReached(self):
        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)
            goal_distance = math.sqrt(pose_transformed.pose.position.x ** 2 + pose_transformed.pose.position.y ** 2)
            
            if goal_distance < self.goal_tol:
                return True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return False

        return False

    
    def publish(self, lin_vel, ang_vel):
        move_cmd = Twist()
        move_cmd.linear.x = lin_vel
        move_cmd.angular.z = ang_vel
        self.cmd_vel_pub.publish(move_cmd)


if __name__ == '__main__':
    rospy.init_node('TurtlebotController', anonymous=False)
    rospy.loginfo("To stop TurtleBot CTRL + C")

    rate = 10 
    robot = TurtlebotController(rate)
    rospy.on_shutdown(robot.shutdown)
        
    r = rospy.Rate(rate)
    while not rospy.is_shutdown():
        robot.command()
        r.sleep()