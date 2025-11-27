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
        self.OBSTACLE_DIST = 0.5   # distancia para considerar obstáculo cercano
        self.SAFE_DIST = 0.9       # distancia que consideramos "frente libre" para salir de AVOID
        self.CLOSE_DIST = 0.35     # muy cerca -> frenar aun más

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
    def _index_from_angle(self, angle_rad):
        """
        Devuelve índice aproximado del array de ranges para un ángulo en radianes
        (ángulo en el sistema del LaserScan: angle_min + i*angle_increment).
        Normaliza el índice dentro del rango.
        """
        msg = self.lidar_data
        if msg is None:
            return None
        ang_min = msg.angle_min
        ang_inc = msg.angle_increment
        N = len(msg.ranges)
        i = int(round((angle_rad - ang_min) / ang_inc))
        # clamp
        if i < 0:
            i = 0
        if i >= N:
            i = N - 1
        return i

    def _sector_min(self, center_deg, width_deg, ranges):
        """
        center_deg: 0 adelante, +90 izquierda, -90 derecha (en grados)
        width_deg: semiancho total (p.ej. 30 -> -15..+15 del centro)
        """
        if ranges is None:
            return float('inf')
        # convert degrees to radians (0 deg is front)
        center = math.radians(center_deg)
        half = math.radians(width_deg) / 2.0
        msg = self.lidar_data
        if msg is None:
            return float('inf')

        ang_min = msg.angle_min
        ang_inc = msg.angle_increment
        N = len(ranges)

        # generar índices
        idxs = []
        # muestreo denso: tomo índices entre center-half y center+half
        a_start = center - half
        a_end = center + half
        i_start = int(math.floor((a_start - ang_min) / ang_inc))
        i_end = int(math.ceil((a_end - ang_min) / ang_inc))
        i_start = max(0, i_start)
        i_end = min(N - 1, i_end)
        if i_start > i_end:
            return float('inf')
        sector = ranges[i_start:i_end + 1]
        # normalizar NaNs e infs
        sector = np.array(sector)
        sector = np.where(np.isfinite(sector), sector, 100.0)
        if sector.size == 0:
            return float('inf')
        return float(np.min(sector))

    def _get_lidar_regions(self, ranges):
        # Si no hay datos
        if ranges is None:
            return {'front': float('inf'), 'left': float('inf'), 'right': float('inf')}

        # usamos centro 0 frente, 90 izq, -90 o 270 derecha
        front_min = self._sector_min(0.0, 30.0, ranges)    # -15..+15
        left_min = self._sector_min(90.0, 60.0, ranges)    # 60..120
        right_min = self._sector_min(-90.0, 60.0, ranges)  # -120..-60

        return {'front': front_min, 'left': left_min, 'right': right_min}

    # ----------------------- Estado AVOID -----------------------
    def _start_avoid(self):
        # Elegimos dirección: preferimos lado con más espacio
        regions = self._get_lidar_regions(self.lidar_data.ranges) if self.lidar_data else {'left': float('inf'), 'right': float('inf')}
        if regions['left'] >= regions['right']:
            self.avoid_direction = 1
        else:
            self.avoid_direction = -1

        self.state = 'AVOID_TURN'
        self.avoid_start_time = rospy.Time.now()
        rospy.loginfo(">>> ENTERING AVOID MODE (dir=%s) front=%.2f left=%.2f right=%.2f",
                      "L" if self.avoid_direction == 1 else "R",
                      regions.get('front', float('inf')), regions.get('left', float('inf')), regions.get('right', float('inf')))

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
                speed_scale = 0.15
            elif regions['front'] < self.OBSTACLE_DIST:
                speed_scale = 0.35

            linear = self.K_linear * distance_to_goal * speed_scale
            angular = self.K_angular * angle_to_goal

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

            # velocidad de avance reducida (permite sortear esquinas)
            linear = 0.06 if front < self.CLOSE_DIST else 0.12

            # angular fijo hacia el lado elegido, más fuerte si muy cerca
            ang_base = 0.9 if front < self.CLOSE_DIST else 0.6
            angular = ang_base * float(self.avoid_direction)

            # Condition para pasar a AVOID_ADVANCE: cuando el frente esté razonablemente libre en SAFE_DIST
            if front > self.SAFE_DIST:
                # nos aseguramos de que el ángulo al objetivo no sea muy grande
                if abs(angle_to_goal) < 1.2 or (rospy.Time.now() - self.avoid_start_time).to_sec() > 1.2:
                    self.state = 'AVOID_ADVANCE'
                    rospy.loginfo("AVOID -> ADVANCE (front %.2f, angle_to_goal %.2f)", front, angle_to_goal)

        elif self.state == 'AVOID_ADVANCE':
            # Avanzar pegado al obstáculo para "limpiar" la zona y volver a NAV después de un tiempo/espacio
            linear = 0.18
            angular = 0.18 * float(self.avoid_direction)  # ligera corrección para seguir borde

            # Si el frente queda despejado y ángulo hacia objetivo pequeño -> salir
            if regions['front'] > self.SAFE_DIST and abs(angle_to_goal) < 0.6:
                rospy.loginfo("Conditions met to exit AVOID (front=%.2f, angle=%.2f)", regions['front'], angle_to_goal)
                self._stop_avoid()

            # Safety: si volvemos a ver obstáculo frontal mientras avanzamos, volver a girar
            if regions['front'] < self.OBSTACLE_DIST:
                rospy.loginfo("Obstacle re-detected during ADVANCE -> back to TURN")
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

            return distance < self.goal_tol and abs(angle) < self.angle_tol
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
