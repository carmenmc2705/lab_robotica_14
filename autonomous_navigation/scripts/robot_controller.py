#!/usr/bin/python3
# coding: utf-8

import math
import rospy
import tf
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
import numpy as np

class TurtlebotController():
    def __init__(self, rate):

        # --- Parámetros ---
        self.goal_tol = 0.15
        self.angle_tol = 0.4
        self.rate = rate

        # Estados: NAV, AVOID_TURN, AVOID_ADVANCE
        self.state = 'NAV'
        self.avoid_start_time = None
        self.avoid_timeout = 6.0
        self.avoid_direction = 1   # 1 left, -1 right

        # Ganancias
        self.K_linear = 0.5
        self.K_angular = 1.5
        self.MAX_LINEAR = 0.6
        self.MAX_ANGULAR = 2.0

        # LIDAR thresholds (ajustables)
        self.OBSTACLE_DIST = 0.6   # distancia para considerar obstáculo cercano
        self.SAFE_DIST = 0.9       # distancia que consideramos "frente libre" para salir de AVOID
        self.CLOSE_DIST = 0.4     # muy cerca -> frenar aun más

        # Objetivo / Path (aun cuando eliminas path, conservamos soporte a goal manual)
        self.goal = PoseStamped()
        self.path_poses = []
        self.path_received = False
        self.current_goal_index = 0
        self.goal_received = False

        # LIDAR y TF
        self.lidar_data = None
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)

        # Suscriptores
        rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goalCallback)
        rospy.Subscriber("/scan", LaserScan, self.scanCallback)
        rospy.Subscriber("/path", Path, self.pathCallback)

        rospy.loginfo("TurtlebotController started (NAV + improved AVOID)")

    # ----------------------- Callbacks -----------------------
    def scanCallback(self, msg):
        self.lidar_data = msg

        # Debug una vez para conocer configuración del LIDAR
        if not hasattr(self, "scan_debugged"):
            self.scan_debugged = True
            rospy.loginfo("LIDAR: angle_min=%.3f, angle_max=%.3f, angle_inc=%.6f, ranges=%d",
                          msg.angle_min, msg.angle_max, msg.angle_increment, len(msg.ranges))

    def pathCallback(self, msg):
        # Mantengo soporte pero tu mencionaste que lo quitaste; no hará nada si no usas /path
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

    # ----------------------- LIDAR helpers -----------------------
    def _index_from_angle(self, angle_deg):
        """
        Devuelve índice aproximado del array de ranges para un ángulo en radianes
        (ángulo en el sistema del LaserScan: angle_min + i*angle_increment).
        Normaliza el índice dentro del rango.
        """
        msg = self.lidar_data
        if msg is None:
            return 0
        N = len(msg.ranges)
        angle_deg = angle_deg % 360
        i = int( (angle_deg / 360.0) * N )
        i = i % N
        return i

    def _sector_data(self, center_deg, width_deg, ranges):
        if ranges is None: return []
        N = len(ranges)
        
        # Calculamos inicio y fin en grados
        start_deg = center_deg - (width_deg / 2.0)
        end_deg   = center_deg + (width_deg / 2.0)
        
        # Convertimos a índices usando la función corregida
        i_start = self._index_from_angle(start_deg)
        i_end   = self._index_from_angle(end_deg)
        
        # Extraer datos del array circular
        if i_start > i_end:
            # Caso especial: El sector cruza el ángulo 0 (ej: de 350 a 10 grados)
            # Concatenamos el final del array con el principio
            sector = ranges[i_start:] + ranges[:i_end+1]
        else:
            # Caso normal
            sector = ranges[i_start:i_end+1]
            
        # Limpieza de datos (quitar infs y ceros erróneos)
        valid_data = [x for x in sector if x > 0.05 and math.isfinite(x)]
        return valid_data
    
    def _sector_min(self, center_deg, width_deg, ranges):
        data = self._sector_data(center_deg, width_deg, ranges)
        if not data: return float('inf')
        return min(data)

    def _sector_avg(self, center_deg, width_deg, ranges):
        data = self._sector_data(center_deg, width_deg, ranges)
        if not data: return float('inf')
        return sum(data) / len(data)

    def _get_lidar_regions(self, ranges):
        # Si no hay datos
        if ranges is None:
            return {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}

        # usamos centro 0 frente, 90 izq, -90 o 270 derecha
        front_min = self._sector_min(0.0, 40.0, ranges)    # -15..+15
        left_min = self._sector_min(60.0, 60.0, ranges)    # 60..120
        right_min = self._sector_min(-60.0, 60.0, ranges)  # -120..-60

        return {'front': front_min, 'left': left_min, 'right': right_min}

    # ----------------------- Estado AVOID -----------------------
    def _start_avoid(self):
        # Elegimos dirección: preferimos lado con más espacio
        regions = self._get_lidar_regions(self.lidar_data.ranges) if self.lidar_data else {'left': 0.0, 'right': 0.0}
        
        left_reg = regions['left']
        right_reg = regions['right']
        
        rospy.loginfo("AVOID CHECK: Left=%.2f vs Right=%.2f", left_reg, right_reg)
        if left_reg >= right_reg:
            self.avoid_direction = 1
        else:
            self.avoid_direction = -1

        self.state = 'AVOID_TURN'
        self.avoid_start_time = rospy.Time.now()
        side_str = "LEFT" if self.avoid_direction == 1 else "RIGHT"
        rospy.loginfo(">>> ENTERING AVOID MODE -> Turning %s", side_str)

    def _stop_avoid(self):
        self.state = 'NAV'
        self.avoid_start_time = None
        rospy.loginfo("<<< EXITING AVOID MODE - back to NAV")

    # ----------------------- Main control -----------------------
    def command(self):

        # No hay objetivo
        if not self.path_received and not self.goal_received:
            return

        # Prioridad path si existiera (en tu caso no)
        if self.path_received:
            if self.current_goal_index >= len(self.path_poses):
                rospy.loginfo_throttle(5, "PATH FINISHED!")
                self.publish(0.0, 0.0)
                return
            self.goal = self.path_poses[self.current_goal_index]

        # Transformar objetivo a base_footprint
        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        goal_x = pose_transformed.pose.position.x
        goal_y = pose_transformed.pose.position.y

        angle_to_goal = math.atan2(goal_y, goal_x)
        distance_to_goal = math.hypot(goal_x, goal_y)

        # Si llegamos
        if self.goalReached():
            rospy.loginfo("Waypoint reached!")
            if self.path_received:
                self.current_goal_index += 1
            else:
                self.goal_received = False
            self.publish(0.0, 0.0)
            return

        # LIDAR regions
        regions = self._get_lidar_regions(self.lidar_data.ranges) if self.lidar_data else \
                  {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}

        obstacle_front = regions['front'] < self.OBSTACLE_DIST

        # Si estamos en NAV y hay obstáculo en frente, arrancamos AVOID
        if self.state == 'NAV' and obstacle_front:
            self._start_avoid()

        # Timeout safety
        if self.state.startswith('AVOID'):
            if (rospy.Time.now() - self.avoid_start_time).to_sec() > self.avoid_timeout:
                rospy.logwarn("AVOID timeout exceeded -> forcing NAV")
                self._stop_avoid()

        # ---------- CONTROL POR ESTADOS ----------
        linear = 0.0
        angular = 0.0

        if self.state == 'NAV':
            # reducir velocidad si hay algo relativamente cerca para evitar choque
            speed_scale = 1.0
            if regions['front'] < self.CLOSE_DIST:
                speed_scale = 0.0
            elif regions['front'] < self.OBSTACLE_DIST:
                speed_scale = 0.35

            linear = self.K_linear * distance_to_goal * speed_scale
            angular = self.K_angular * angle_to_goal
            
            max_safe_speed = regions['front'] * 0.7  
            if linear > max_safe_speed:
                linear = max_safe_speed

            # limitar giro si vamos muy cerca
            if abs(angular) > self.MAX_ANGULAR:
                angular = math.copysign(self.MAX_ANGULAR, angular)

            # si en NAV detectamos obstáculo frontal muy cercano, arrancar avoid
            if regions['front'] < self.OBSTACLE_DIST:
                self._start_avoid()

        elif self.state == 'AVOID_TURN':
            # Girar hacia lado elegido, pero AVANZAR muy despacio para esquivar curva
            # Si el frontal está muy cerca, frenamos más y giramos más agresivo
            front = regions['front']
            left = regions['left']
            right = regions['right']
            # --- NUEVA LÓGICA: RE-EVALUACIÓN DINÁMICA ---
            # Si estamos girando a la IZQUIERDA (1), pero de repente la izquierda se cierra
            # y la derecha está MUCHO más libre, cambiamos de opinión.
            # Lo mismo si estamos girando a la DERECHA (-1)
            # Si derecha bloqueada  Y izquierda muy libre (> derecha + 0.5m)
            if right < 0.4 and left > (right + 0.5):
                self.avoid_direction = 1
                rospy.loginfo("Re-evaluating: Switching to LEFT")
            
            # Si izquierda bloqueada  Y derecha muy libre (> izquierda + 0.5m)
            if left < 0.4 and right > (left + 0.5):
                self.avoid_direction = -1
                rospy.loginfo("Re-evaluating: Switching to RIGHT")
            
            
            # --------------------------------------------
            # velocidad de avance reducida (permite sortear esquinas)
            linear = 0.06 if front < self.CLOSE_DIST else 0.12

            # angular fijo hacia el lado elegido, más fuerte si muy cerca
            ang_base = 1.2 #if front < self.CLOSE_DIST else 0.6
            angular = ang_base * self.avoid_direction

            # Condition para pasar a AVOID_ADVANCE: cuando el frente esté razonablemente libre en SAFE_DIST
            if front > self.SAFE_DIST:
                # nos aseguramos de que el ángulo al objetivo no sea muy grande
                if abs(angle_to_goal) < 1.2 or (rospy.Time.now() - self.avoid_start_time).to_sec() > 1.2:
                    self.state = 'AVOID_ADVANCE'
                    rospy.loginfo("AVOID -> ADVANCE (front %.2f, angle_to_goal %.2f)", front, angle_to_goal)

        elif self.state == 'AVOID_ADVANCE':
            # Avanzar pegado al obstáculo para "limpiar" la zona y volver a NAV después de un tiempo/espacio
            linear = 0.2
            angular=0.9* angle_to_goal*self.avoid_direction

            # Si el frente queda despejado y ángulo hacia objetivo pequeño -> salir
            if regions['front'] > self.SAFE_DIST:
                rospy.loginfo("Path clear! Exiting AVOID -> NAV")
                self._stop_avoid()

            # Safety: si volvemos a ver obstáculo frontal mientras avanzamos, volver a girar
            if regions['front'] < self.OBSTACLE_DIST:
                rospy.loginfo("Wall detected while advancing -> Back to TURN")
                self.state = 'AVOID_TURN'
                self.avoid_start_time = rospy.Time.now()
        # Saturación de velocidades
        linear = max(min(linear, self.MAX_LINEAR), -self.MAX_LINEAR)
        angular = max(min(angular, self.MAX_ANGULAR), -self.MAX_ANGULAR)

        # Publicar comando
        self.publish(linear, angular)

    # ----------------------- Utilidades -----------------------
    def goalReached(self):
        try:
            self.goal.header.stamp = rospy.Time(0)
            pose_transformed = self.tf_listener.transformPose('base_footprint', self.goal)

            dx = pose_transformed.pose.position.x
            dy = pose_transformed.pose.position.y

            distance = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)

            return distance < self.goal_tol
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return False

    def publish(self, lin_vel, ang_vel):
        move_cmd = Twist()
        move_cmd.linear.x = lin_vel
        move_cmd.angular.z = ang_vel
        self.cmd_vel_pub.publish(move_cmd)

    def shutdown(self):
        rospy.loginfo("Stopping TurtleBot...")
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)

# ----------------------- main -----------------------
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
