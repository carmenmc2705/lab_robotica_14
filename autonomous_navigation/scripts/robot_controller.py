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
        
        # Este lo mantenemos por si usas el 2D Nav Goal manual, aunque priorizaremos el path
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
        # Solo actualizamos si es la primera vez o si quieres reiniciar el path
        if not self.path_received:
            rospy.loginfo("Path received with %d points.", len(msg.poses))
            self.path_poses = msg.poses
            self.current_goal_index = 0
            self.path_received = True

    def goalCallback(self, goal):
        # Esto es por si usas el botón "2D Nav Goal" de RViz manualmente
        rospy.loginfo("Manual Goal received!")
        self.goal = goal  
        self.goal_received = True
        # Si recibimos un manual, desactivamos el path para que haga caso al manual
        self.path_received = False 

    def command(self):
        # 1. COMPROBACIÓN DE DATOS
        # Si no tenemos ni path ni goal manual, no hacemos nada
        if not self.path_received and not self.goal_received:
            return

        # 2. GESTIÓN DEL PATH
        # Si estamos siguiendo un path...
        if self.path_received:
            # Comprobar si hemos terminado la lista de puntos
            if self.current_goal_index >= len(self.path_poses):
                rospy.loginfo_throttle(5, "PATH FINISHED!!! Stopping!")
                self.publish(0.0, 0.0)
                return
            
            # Seleccionamos el objetivo actual de la lista
            self.goal = self.path_poses[self.current_goal_index]

        # 3. COMPROBAR SI HEMOS LLEGADO AL PUNTO ACTUAL
        if self.goalReached():
            rospy.loginfo("Waypoint reached! Moving to next...")
            
            if self.path_received:
                self.current_goal_index += 1 # Pasamos al siguiente punto del path
            else:
                self.goal_received = False # Si era manual, ya hemos terminado
                self.publish(0.0, 0.0)
            return
        
        # 4. CONTROL LAW (Moverse hacia self.goal)
        try:
            self.goal.header.stamp = rospy.Time(0) # Importante usar Time(0) para coger la última transformada disponible
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(1, "TF Error: No transform found") # Debug
            return    
            
        goal_x = pose_transformed.pose.position.x
        goal_y = pose_transformed.pose.position.y  
        
        angle_to_goal = math.atan2(goal_y, goal_x)
        distance_to_goal = math.sqrt(goal_x**2 + goal_y**2)
        
        # Proportional controller
        K_linear = 0.5
        K_angular = 1.5
        
        linear = K_linear * distance_to_goal
        angular = K_angular * angle_to_goal
        
        # 5. EVASIÓN DE OBSTÁCULOS
        if self.lidar_data:
            ranges = self.lidar_data.ranges
            # Cogemos un sector frontal (-20 grados a +20 grados aprox)
            front = ranges[0:20] + ranges[-20:]
            valid_front = [r for r in front if r > 0.1 and r < 10.0]
            
            if valid_front and min(valid_front) < 0.5:
                rospy.logwarn_throttle(1, "Obstacle detected! Evading...")
                linear = 0.0
                left_space = sum(ranges[20:80])
                right_space = sum(ranges[-80:-20])                
                if left_space > right_space:
                    angular = 0.6   # gira izquierda
                else:
                    angular = -0.6  # Girar a la izquierda
                
        # Saturate velocities
        linear = min(linear, 0.22) # Max velocidad real del robot
        linear = max(linear, 0.0)
        angular = max(min(angular, 1.0), -1.0)
        
        # Publish velocity command
        self.publish(linear, angular)


    def goalReached(self):
        # Usamos try-except porque TF puede fallar puntualmente
        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)
            dx = pose_transformed.pose.position.x
            dy = pose_transformed.pose.position.y

            distance = math.sqrt(dx**2 + dy**2)
            angle = math.atan2(dy, dx)

            # Condición mejorada → evita esperas innecesarias
            if distance < self.goal_tol:
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