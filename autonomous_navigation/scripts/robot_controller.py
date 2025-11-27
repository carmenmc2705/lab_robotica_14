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


        # Chequeo de llegada
        if self.goalReached():
            rospy.loginfo("Waypoint reached!")
            if self.path_received:
                self.current_goal_index += 1
            else:
                self.goal_received = False
            # parada breve
            self.publish(0.0, 0.0)
            return

        # Lectura LIDAR y regiones
        regions = self._get_lidar_regions(self.lidar_data.ranges) if self.lidar_data else {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}

        # Decide transición a AVOID si hay obstáculo cerca
        obstacle_front = regions['front'] < self.OBSTACLE_DIST

        # Timeout de seguridad en AVOID
        if self.state == 'AVOID':
            # Si timeout excedido -> volver a NAV para evitar quedar bloqueado
            if (rospy.Time.now() - self.avoid_start_time).to_sec() > self.avoid_timeout:
                rospy.logwarn("AVOID timeout exceeded; forcing NAV")
                self._stop_avoid()

        # Si hay obstáculo en frente y no estamos ya evitando -> entrar en AVOID
        if obstacle_front and self.state != 'AVOID':
            self._start_avoid()

        # ---------- COMPORTAMIENTO EN CADA ESTADO ----------
        linear = 0.0
        angular = 0.0

        if self.state == 'NAV':
            # Control proporcional hacia objetivo (cuando no hay obstáculo crítico)
            linear = self.K_linear * distance_to_goal
            angular = self.K_angular * angle_to_goal

            # Si durante NAV detectamos obstáculo cercano en lado con menos espacio vamos a AVOID
            if regions['front'] < self.OBSTACLE_DIST:
                self._start_avoid()

        elif self.state == 'AVOID':
            # En AVOID: girar hacia lado con más espacio (decisión discreta)
            linear = 0.0
            left_clear = regions['left']
            right_clear = regions['right']

            # Si ambos lados tienen espacio semejante, usar orientación del objetivo:
            # si objetivo está a la izquierda (goal_y > 0) preferir izquierda, sino derecha.
            # Preferencia por el lado con mayor distancia.
            if left_clear > right_clear:
                angular = 0.9
            elif right_clear > left_clear:
                angular = -0.9
            else:
                # empate -> preferir según señal del objetivo
                angular = 0.9 if goal_y > 0 else -0.9

            # Condición para dejar AVOID:
            # -> frente libre suficientemente y el ángulo al objetivo razonable
            if regions['front'] > self.SAFE_DIST and abs(angle_to_goal) < 0.7:
                self._stop_avoid()

        # Saturación de velocidades
        linear = max(min(linear, self.MAX_LINEAR), -self.MAX_LINEAR)
        angular = max(min(angular, self.MAX_ANGULAR), -self.MAX_ANGULAR)

        # Publicar comando
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
            if distance < self.goal_tol and abs(angle) < self.angle_tol:
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