#!/usr/bin/python3
# This Python file uses the following encoding: utf-8

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
        
        # Read parameters
        self.goal_tol = 0.15
        self.angle_tol = 0.4
        self.rate = rate 
        
                # Estados y temporizadores
        self.state = 'NAV'  # 'NAV' o 'AVOID'
        self.avoid_start_time = None
        self.avoid_timeout = 3.0  # segundos máximo en modo AVOID por seguridad

        # Velocidades / ganancias
        self.K_linear = 0.5
        self.K_angular = 1.5
        self.MAX_LINEAR = 0.8
        self.MAX_ANGULAR = 2.5

        # Umbrales LIDAR
        self.OBSTACLE_DIST = 0.5   # si el frontal < esto => obstáculo
        self.SAFE_DIST = 0.7   
        
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
        
 # ---------- NUEVAS FUNCIONES AUXILIARES ----------
    def _get_lidar_regions(self, ranges):
        """
        Devuelve mínimos en sectores: front, left, right.
        Asume un LaserScan típico con 0 index forward and increasing counterclockwise.
        Ajusta índices si tu LIDAR está configurado distinto.
        """
        if ranges is None:
            return {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}

        r = np.array(ranges)
        # Normalizar NaNs e infs
        r = np.where(np.isfinite(r), r, 100.0)

        # Definición de sectores (ajusta según resolución del LIDAR)
        # Aquí se asumen 360 medidas (1 deg), pero funciona también para otras resoluciones.
        front_sector = np.concatenate([r[0:10], r[-10:]])   # -10..+10 grados
        left_sector = r[60:120]   # ~+60 a +120 grados
        right_sector = r[-120:-60]  # ~-120 a -60 grados

        front_min = np.min(front_sector) if front_sector.size else float('inf')
        left_min = np.min(left_sector) if left_sector.size else float('inf')
        right_min = np.min(right_sector) if right_sector.size else float('inf')

        return {'front': float(front_min), 'left': float(left_min), 'right': float(right_min)}

    def _start_avoid(self):
        self.state = 'AVOID'
        self.avoid_start_time = rospy.Time.now()
        rospy.loginfo("Entering AVOID mode")

    def _stop_avoid(self):
        self.state = 'NAV'
        self.avoid_start_time = None
        rospy.loginfo("Returning to NAV mode")

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
        # Transformar goal a base_footprint
        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        goal_x = pose_transformed.pose.position.x
        goal_y = pose_transformed.pose.position.y

        angle_to_goal = math.atan2(goal_y, goal_x)
        distance_to_goal = math.sqrt(goal_x**2 + goal_y**2)
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
                if left_space > right_space+0.5:
                    angular = 0.6   # gira izquierda
                elif right_space > left_space + 0.5: # Solo si derecha es CLARAMENTE mejor
                    angular = -0.6 
                else:
                    angular = 0.6  # Girar a la izquierda
                
        # Saturate velocities
        #linear = min(linear, 0.22) 
        #angular = max(min(angular, 1.0), -1.0)
        
        # Publish velocity command
        self.publish(linear, angular)


    def goalReached(self):
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